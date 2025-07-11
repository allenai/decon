// Reference dataset post-processing module
// This module handles various operations to refine and improve the quality of reference datasets,
// including deduplication and other preprocessing steps.

use anyhow::{Error, Result};
use dashmap::DashMap;
use rayon::prelude::*;
use serde_json::Value;
use std::collections::{HashMap, HashSet, VecDeque};
use std::fs::{create_dir_all, File};
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use flate2::read::GzDecoder;
use flate2::write::GzEncoder;
use flate2::Compression;

use mj_io::{build_pbar, expand_dirs};
use ndarray::Array1;
use sha2::{Digest, Sha256};
use crate::{clean_text, get_nested_json_val, OmniTokenizer};
use crate::minhash::{_expand_band_seeds};

// Constants for filtering
const MIN_LENGTH: usize = 15; // Minimum word count for text entries

// Constants for MinHash deduplication
const MINHASH_NUM_BANDS: usize = 7;
const MINHASH_BAND_SIZE: usize = 8;
const MINHASH_SIMILARITY_THRESHOLD: f32 = 0.98; // 98% similarity threshold for near-duplicates
const MINHASH_NGRAM_SIZE: usize = 5; // n-gram size for MinHash
const MINHASH_SEED: usize = 42; // Random seed for MinHash

// Type aliases for clarity
type LineHash = u64;
type LineOccurrences = Vec<(String, usize)>; // (filename, line_number)

// MinHash constants
const BIG_PRIME: u64 = 18446744073709551557;
const MAX_HASH: u64 = BIG_PRIME;

// MinHash type aliases
type MinHashSignature = Array1<u64>;
type BandSignature = Vec<u8>;
type MinHashBands = DashMap<BandSignature, Vec<(String, usize)>>; // Maps band signature to (filename, line_num)
type MinHashSignatures = DashMap<(String, usize), MinHashSignature>; // Maps (filename, line_num) to full signature

pub fn refine_reference_files(dry_run: bool) -> Result<(), Error> {
    println!("Starting reference file refinement...");

    // Fixed paths
    let reference_dir = PathBuf::from("fixtures/reference");
    let output_dir = PathBuf::from("fixtures/reference-refined");

    // Check if reference directory exists
    if !reference_dir.exists() {
        return Err(anyhow::anyhow!(
            "Reference directory not found: {:?}",
            reference_dir
        ));
    }

    // Find all reference files
    let reference_files = expand_dirs(
        vec![reference_dir.clone()],
        Some(vec![".jsonl", ".gz"].as_slice()),
    )?;

    if reference_files.is_empty() {
        println!("No reference files found in {:?}", reference_dir);
        return Ok(());
    }

    println!("Found {} reference files to process", reference_files.len());

    // Phase 1: Build hash map of all unique lines
    println!("\nPhase 1: Detecting exact duplicates...");
    let (duplicate_map, dedup_stats) = detect_exact_duplicates(&reference_files)?;

    // Display deduplication statistics
    display_duplicate_stats(&duplicate_map, &dedup_stats, dry_run);

    // Phase 2: Determine lines to keep after deduplication
    println!("\nPhase 2: Analyzing lines to keep after deduplication...");
    let lines_to_keep = build_lines_to_keep(&duplicate_map);
    let lines_after_dedup: usize = lines_to_keep.values().map(|set| set.len()).sum();
    println!("Lines remaining after deduplication: {} (removed {} duplicates)",
        lines_after_dedup,
        dedup_stats.total_lines - lines_after_dedup
    );

    // Phase 3: MinHash near-duplicate detection
    println!("\nPhase 3: Detecting near-duplicates with MinHash...");
    let (minhash_lines_to_keep, minhash_stats) = detect_minhash_duplicates(&reference_files, &lines_to_keep)?;
    display_minhash_stats(&minhash_stats, dry_run);
    let lines_after_minhash: usize = minhash_lines_to_keep.values().map(|set| set.len()).sum();
    println!("Lines remaining after MinHash deduplication: {} (removed {} near-duplicates)",
        lines_after_minhash,
        lines_after_dedup - lines_after_minhash
    );

    // Phase 4: Apply filters to analyze what would be removed
    println!("\nPhase 4: Analyzing filters...");
    let (filtered_lines_to_keep, filter_stats) = apply_filters(&reference_files, &minhash_lines_to_keep)?;
    display_filter_stats(&filter_stats);

    // Phase 5: Write files (only if not dry run)
    if !dry_run {
        println!("\nPhase 5: Writing refined files...");
        write_refined_files(&reference_files, &filtered_lines_to_keep, &output_dir)?;
    } else {
        println!("\n[DRY RUN] Skipping file writing phase.");
        let total_lines_final: usize = filtered_lines_to_keep.values().map(|set| set.len()).sum();
        println!("Would write {} refined files with {} lines total",
            reference_files.len(),
            total_lines_final
        );
        println!("Output directory would be: {:?}", output_dir);
    }

    // Final summary
    println!("\n=== OVERALL SUMMARY ===");
    println!("Original lines: {}", dedup_stats.total_lines);
    println!("After exact deduplication: {} (removed {} duplicates)", lines_after_dedup, dedup_stats.total_lines - lines_after_dedup);
    println!("After MinHash deduplication: {} (removed {} near-duplicates)", lines_after_minhash, lines_after_dedup - lines_after_minhash);
    println!("After filtering: {} (removed {} by filters)", filter_stats.total_lines_after_filters, filter_stats.total_lines_before_filters - filter_stats.total_lines_after_filters);
    println!("Total reduction: {:.1}% ({} lines removed)",
        ((dedup_stats.total_lines - filter_stats.total_lines_after_filters) as f64 / dedup_stats.total_lines as f64) * 100.0,
        dedup_stats.total_lines - filter_stats.total_lines_after_filters
    );

    Ok(())
}

#[derive(Default)]
struct DuplicateStats {
    total_files: usize,
    total_lines: usize,
    unique_lines: usize,
    duplicate_lines: usize,
}

#[derive(Default)]
struct FilterStats {
    total_lines_before_filters: usize,
    lines_removed_min_length: usize,
    total_lines_after_filters: usize,
}

fn detect_exact_duplicates(
    reference_files: &[PathBuf],
) -> Result<(DashMap<LineHash, LineOccurrences>, DuplicateStats), Error> {
    let line_occurrences: DashMap<LineHash, LineOccurrences> = DashMap::new();
    let total_lines = Arc::new(AtomicUsize::new(0));
    let duplicate_lines = Arc::new(AtomicUsize::new(0));

    let pbar = build_pbar(reference_files.len(), "Processing files");

    // Process files in parallel
    reference_files.par_iter().for_each(|file_path| {
        if let Err(e) = process_file_for_duplicates(
            file_path,
            &line_occurrences,
            &total_lines,
            &duplicate_lines,
        ) {
            eprintln!("Error processing file {:?}: {:?}", file_path, e);
        }
        pbar.inc(1);
    });

    pbar.finish_with_message("Detection complete");

    // Calculate statistics
    let stats = DuplicateStats {
        total_files: reference_files.len(),
        total_lines: total_lines.load(Ordering::Relaxed),
        unique_lines: line_occurrences.len(),
        duplicate_lines: duplicate_lines.load(Ordering::Relaxed),
    };

    Ok((line_occurrences, stats))
}

fn process_file_for_duplicates(
    file_path: &PathBuf,
    line_occurrences: &DashMap<LineHash, LineOccurrences>,
    total_lines: &Arc<AtomicUsize>,
    duplicate_lines: &Arc<AtomicUsize>,
) -> Result<(), Error> {
    // Extract filename for tracking
    let filename = file_path
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("unknown")
        .to_string();

    // Open file with appropriate decoder
    let file = File::open(file_path)?;
    let reader: Box<dyn BufRead> = if file_path.extension().and_then(|s| s.to_str()) == Some("gz") {
        Box::new(BufReader::new(GzDecoder::new(file)))
    } else {
        Box::new(BufReader::new(file))
    };

    // Process each line
    for (line_num, line) in reader.lines().enumerate() {
        let line = line?;
        total_lines.fetch_add(1, Ordering::Relaxed);

        // Parse JSON and extract text content
        let json_obj: Value = match serde_json::from_str(&line) {
            Ok(obj) => obj,
            Err(_) => continue, // Skip invalid JSON lines
        };

        // Try to extract text from common field names
        let text = get_text_from_json(&json_obj)?;

        // Clean and normalize the text
        let cleaned_text = clean_text(&text, "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~");
        let normalized_text = cleaned_text.to_lowercase();

        // Hash the normalized text
        let line_hash = hash_text(&normalized_text);

        // Track occurrence
        let mut is_duplicate = false;
        line_occurrences
            .entry(line_hash)
            .and_modify(|occurrences| {
                is_duplicate = true;
                occurrences.push((filename.clone(), line_num));
            })
            .or_insert_with(|| vec![(filename.clone(), line_num)]);

        if is_duplicate {
            duplicate_lines.fetch_add(1, Ordering::Relaxed);
        }
    }

    Ok(())
}

fn get_text_from_json(json_obj: &Value) -> Result<String, Error> {
    // Try common field names for text content
    let field_names = ["text", "content", "passage", "question", "answer"];

    for field in &field_names {
        if let Ok(text) = get_nested_json_val(json_obj, &field.to_string()) {
            return Ok(text);
        }
    }

    // If no common field found, try to get any string value
    if let Some(obj) = json_obj.as_object() {
        for (_, value) in obj {
            if let Some(text) = value.as_str() {
                return Ok(text.to_string());
            }
        }
    }

    Err(anyhow::anyhow!("No text content found in JSON object"))
}

fn hash_text(text: &str) -> LineHash {
    use std::hash::{Hash, Hasher};
    use std::collections::hash_map::DefaultHasher;

    let mut hasher = DefaultHasher::new();
    text.hash(&mut hasher);
    hasher.finish()
}

fn get_line_content(filename: &str, line_num: usize) -> Result<String, Error> {
    let file_path = PathBuf::from("fixtures/reference").join(filename);
    let file = File::open(&file_path)?;

    let reader: Box<dyn BufRead> = if file_path.extension().and_then(|s| s.to_str()) == Some("gz") {
        Box::new(BufReader::new(GzDecoder::new(file)))
    } else {
        Box::new(BufReader::new(file))
    };

    for (current_line, line) in reader.lines().enumerate() {
        if current_line == line_num {
            return line.map_err(|e| e.into());
        }
    }

    Err(anyhow::anyhow!("Line {} not found in file", line_num))
}

fn display_duplicate_stats(
    line_occurrences: &DashMap<LineHash, LineOccurrences>,
    stats: &DuplicateStats,
    dry_run: bool,
) {
    println!("\n=== DUPLICATE DETECTION SUMMARY ===");
    println!("Total files processed: {}", stats.total_files);
    println!("Total lines processed: {}", stats.total_lines);
    println!("Unique lines found: {}", stats.unique_lines);
    println!("Duplicate lines found: {}", stats.duplicate_lines);

    // Calculate lines that appear in multiple places
    let multi_occurrence_lines: Vec<_> = line_occurrences
        .iter()
        .filter(|entry| entry.value().len() > 1)
        .collect();

    println!(
        "Lines appearing in multiple locations: {}",
        multi_occurrence_lines.len()
    );

    if dry_run && !multi_occurrence_lines.is_empty() {
        println!("\nSample duplicates (showing first 10):");

        // Sort by number of occurrences (descending) and take first 10
        let mut sorted_duplicates: Vec<_> = multi_occurrence_lines
            .into_iter()
            .map(|entry| (*entry.key(), entry.value().clone()))
            .collect();
        sorted_duplicates.sort_by(|a, b| b.1.len().cmp(&a.1.len()));

        for (i, (_hash, occurrences)) in sorted_duplicates.iter().take(10).enumerate() {
            println!("\nDuplicate #{} (appears {} times):", i + 1, occurrences.len());

            // Try to read the actual content for this duplicate
            if let Some((first_file, first_line)) = occurrences.first() {
                if let Ok(content) = get_line_content(first_file, *first_line) {
                    // Parse JSON and extract text
                    if let Ok(json_obj) = serde_json::from_str::<Value>(&content) {
                        if let Ok(text) = get_text_from_json(&json_obj) {
                            // Show first 200 chars of the text
                            let preview = if text.len() > 200 {
                                format!("{}...", &text[..200])
                            } else {
                                text
                            };
                            println!("Content: \"{}\"", preview);
                        }
                    }
                }
            }

            println!("Found in:");
            for (file, line_num) in occurrences.iter().take(5) {
                println!("  - {}:{}", file, line_num + 1); // +1 for human-readable line numbers
            }
            if occurrences.len() > 5 {
                println!("  ... and {} more occurrences", occurrences.len() - 5);
            }
        }
    }
}

fn display_filter_stats(stats: &FilterStats) {
    println!("\n=== FILTER STATISTICS ===");
    println!("Lines after deduplication: {}", stats.total_lines_before_filters);
    println!("Lines removed by filters:");
    println!("  - min_length (<{} words): {}", MIN_LENGTH, stats.lines_removed_min_length);
    println!("Lines after all filters: {}", stats.total_lines_after_filters);

    let filter_reduction = stats.total_lines_before_filters - stats.total_lines_after_filters;
    if stats.total_lines_before_filters > 0 {
        println!("Filter reduction: {:.1}% ({} lines removed)",
            (filter_reduction as f64 / stats.total_lines_before_filters as f64) * 100.0,
            filter_reduction
        );
    }
}

fn build_lines_to_keep(
    line_occurrences: &DashMap<LineHash, LineOccurrences>,
) -> HashMap<String, HashSet<usize>> {
    let mut lines_to_keep: HashMap<String, HashSet<usize>> = HashMap::new();

    for entry in line_occurrences.iter() {
        let occurrences = entry.value();
        if let Some((first_file, first_line)) = occurrences.first() {
            lines_to_keep
                .entry(first_file.clone())
                .or_insert_with(HashSet::new)
                .insert(*first_line);
        }
    }

    lines_to_keep
}

fn apply_filters(
    reference_files: &[PathBuf],
    lines_to_keep: &HashMap<String, HashSet<usize>>,
) -> Result<(HashMap<String, HashSet<usize>>, FilterStats), Error> {
    let filtered_lines: Arc<DashMap<String, HashSet<usize>>> = Arc::new(DashMap::new());
    let total_lines_before = Arc::new(AtomicUsize::new(0));
    let lines_removed_min_length = Arc::new(AtomicUsize::new(0));
    let total_lines_after = Arc::new(AtomicUsize::new(0));

    let pbar = build_pbar(reference_files.len(), "Filtering files");

    reference_files.par_iter().for_each(|file_path| {
        if let Err(e) = filter_file(
            file_path,
            lines_to_keep,
            &filtered_lines,
            &total_lines_before,
            &lines_removed_min_length,
            &total_lines_after,
        ) {
            eprintln!("Error filtering file {:?}: {:?}", file_path, e);
        }
        pbar.inc(1);
    });

    pbar.finish_with_message("Filtering complete");

    // Convert DashMap to HashMap
    let mut filtered_lines_to_keep = HashMap::new();
    for entry in filtered_lines.iter() {
        filtered_lines_to_keep.insert(entry.key().clone(), entry.value().clone());
    }

    let stats = FilterStats {
        total_lines_before_filters: total_lines_before.load(Ordering::Relaxed),
        lines_removed_min_length: lines_removed_min_length.load(Ordering::Relaxed),
        total_lines_after_filters: total_lines_after.load(Ordering::Relaxed),
    };

    Ok((filtered_lines_to_keep, stats))
}

fn filter_file(
    file_path: &PathBuf,
    lines_to_keep: &HashMap<String, HashSet<usize>>,
    filtered_lines: &Arc<DashMap<String, HashSet<usize>>>,
    total_lines_before: &Arc<AtomicUsize>,
    lines_removed_min_length: &Arc<AtomicUsize>,
    total_lines_after: &Arc<AtomicUsize>,
) -> Result<(), Error> {
    let filename = file_path
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("unknown");

    let keep_lines = lines_to_keep.get(filename);

    // Open file
    let file = File::open(file_path)?;
    let reader: Box<dyn BufRead> = if file_path.extension().and_then(|s| s.to_str()) == Some("gz") {
        Box::new(BufReader::new(GzDecoder::new(file)))
    } else {
        Box::new(BufReader::new(file))
    };

    let mut kept_lines = HashSet::new();
    let mut lines_examined = 0;
    let mut kept_count = 0;

    for (line_num, line) in reader.lines().enumerate() {
        let line = line?;

        // First check if line should be kept (deduplication)
        if keep_lines.map_or(true, |set| set.contains(&line_num)) {
            lines_examined += 1;

            // Line passed deduplication, now apply filters
            // Parse JSON to check filters
            if let Ok(json_obj) = serde_json::from_str::<Value>(&line) {
                if let Ok(text) = get_text_from_json(&json_obj) {
                    // Apply minimum length filter (count words)
                    let word_count = text.split_whitespace().count();
                    if word_count < MIN_LENGTH {
                        lines_removed_min_length.fetch_add(1, Ordering::Relaxed);
                        continue;
                    }
                }
            }

            // Line passed all filters
            kept_lines.insert(line_num);
            kept_count += 1;
        }
    }

    // Update global statistics - only count lines that passed deduplication
    total_lines_before.fetch_add(lines_examined, Ordering::Relaxed);
    total_lines_after.fetch_add(kept_count, Ordering::Relaxed);

    // Store filtered lines for this file
    if !kept_lines.is_empty() {
        filtered_lines.insert(filename.to_string(), kept_lines);
    }

    Ok(())
}

fn write_refined_files(
    reference_files: &[PathBuf],
    lines_to_keep: &HashMap<String, HashSet<usize>>,
    output_dir: &Path,
) -> Result<(), Error> {
    // Create output directory
    create_dir_all(output_dir)?;

    let pbar = build_pbar(reference_files.len(), "Writing files");

    // Process each file
    reference_files.par_iter().for_each(|file_path| {
        if let Err(e) = write_refined_file(
            file_path,
            output_dir,
            &lines_to_keep,
        ) {
            eprintln!("Error writing refined file {:?}: {:?}", file_path, e);
        }
        pbar.inc(1);
    });

    pbar.finish_with_message("Writing complete");

    // Print summary
    let total_lines: usize = lines_to_keep.values().map(|set| set.len()).sum();
    println!(
        "\nWrote {} refined files with {} lines total",
        reference_files.len(),
        total_lines
    );
    println!("Output directory: {:?}", output_dir);

    Ok(())
}

fn write_refined_file(
    input_path: &PathBuf,
    output_dir: &Path,
    lines_to_keep: &HashMap<String, HashSet<usize>>,
) -> Result<(), Error> {
    let filename = input_path
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("unknown");

    let output_path = output_dir.join(filename);

    // Get the set of line numbers to keep for this file
    let keep_lines = lines_to_keep.get(filename);

    // Open input file
    let input_file = File::open(input_path)?;
    let reader: Box<dyn BufRead> = if input_path.extension().and_then(|s| s.to_str()) == Some("gz") {
        Box::new(BufReader::new(GzDecoder::new(input_file)))
    } else {
        Box::new(BufReader::new(input_file))
    };

    // Open output file with same compression as input
    let output_file = File::create(&output_path)?;
    let mut writer: Box<dyn Write> = if input_path.extension().and_then(|s| s.to_str()) == Some("gz") {
        Box::new(BufWriter::new(GzEncoder::new(
            output_file,
            Compression::default(),
        )))
    } else {
        Box::new(BufWriter::new(output_file))
    };

    // Copy only the lines we want to keep
    let mut kept_count = 0;
    let mut total_count = 0;

    for (line_num, line) in reader.lines().enumerate() {
        total_count += 1;
        let line = line?;

        // Keep line if it's in our keep set
        if keep_lines.map_or(false, |set| set.contains(&line_num)) {
            writeln!(writer, "{}", line)?;
            kept_count += 1;
        }
    }

    writer.flush()?;

    let removed_total = total_count - kept_count;
    if removed_total > 0 {
        println!(
            "  {} -> {} ({} kept, {} removed)",
            filename,
            output_path.display(),
            kept_count,
            removed_total
        );
    }

    Ok(())
}

// MinHash helper functions
fn calculate_jaccard_similarity(sig1: &Array1<u64>, sig2: &Array1<u64>) -> f32 {
    let matches = sig1.iter().zip(sig2.iter()).filter(|(a, b)| a == b).count();
    matches as f32 / sig1.len() as f32
}

fn get_hash_vals_from_tokens(
    tokens: Vec<usize>,
    perm_seeds: &Vec<u64>,
    ngram_size: usize,
) -> Array1<u64> {
    let a = init_permutations(perm_seeds);
    let n = perm_seeds.len();

    let mut hash_vals = Array1::ones(n) * MAX_HASH;
    let mut ngram: VecDeque<usize> = VecDeque::with_capacity(ngram_size);
    let mut ngram_count = 0;

    for token in tokens {
        ngram.push_back(token);
        if ngram.len() >= ngram_size {
            ngram_count += 1;
            hash_vals = update_hash_vals(hash_vals, &a, &ngram);
            ngram.pop_front();
        }
    }
    hash_vals = if ngram_count == 0 {
        update_hash_vals(hash_vals, &a, &ngram) // short document, still wanna hash it
    } else {
        hash_vals
    };

    hash_vals
}

fn init_permutations(seeds: &Vec<u64>) -> Array1<u128> {
    use rand::{Rng, SeedableRng};
    use rand_chacha::ChaCha20Rng;
    
    let n = seeds.len();
    let mut a = Array1::zeros(n);
    for (i, &seed) in seeds.iter().enumerate() {
        let mut rng = ChaCha20Rng::seed_from_u64(seed);
        a[i] = rng.gen::<u128>();
    }
    a
}

fn update_hash_vals(
    mut hash_vals: Array1<u64>,
    a: &Array1<u128>,
    ngram: &VecDeque<usize>,
) -> Array1<u64> {
    use ahash::RandomState;
    
    // hash the vecdeque as a u128
    let hash_a = RandomState::with_seed(123);
    let hash_b = RandomState::with_seed(456);
    let hash_val_a = hash_a.hash_one(ngram);
    let hash_val_b = hash_b.hash_one(ngram);
    let cur_hash = ((hash_val_a as u128) << 64) | (hash_val_b as u128);

    // then multiply by a (mod 2^128) and take top 64 most significant bits
    let phv: Array1<u64> = a.mapv(|x| (x.wrapping_mul(cur_hash) >> 64) as u64);
    hash_vals.zip_mut_with(&phv, |x, y| *x = std::cmp::min(*x, *y));

    hash_vals
}

// MinHash deduplication phase
fn detect_minhash_duplicates(
    reference_files: &[PathBuf],
    lines_to_keep: &HashMap<String, HashSet<usize>>,
) -> Result<(HashMap<String, HashSet<usize>>, MinHashStats), Error> {
    println!("Building MinHash signatures...");
    
    // Initialize MinHash structures
    let minhash_bands: MinHashBands = DashMap::new();
    let minhash_signatures: MinHashSignatures = DashMap::new();
    
    // Setup hashing parameters
    let band_seeds: Vec<u32> = _expand_band_seeds(&vec![MINHASH_SEED as u32], MINHASH_NUM_BANDS)
        .into_iter()
        .map(|x| x as u32)
        .collect();
    let perm_seeds = _expand_band_seeds(&band_seeds, MINHASH_BAND_SIZE);
    
    // Initialize tokenizer
    let tokenizer = OmniTokenizer::new("uniseg")?;
    
    // Track statistics
    let total_lines_processed = Arc::new(AtomicUsize::new(0));
    let near_duplicates_found = Arc::new(AtomicUsize::new(0));
    
    let pbar = build_pbar(reference_files.len(), "Computing MinHash signatures");
    
    // Process each file
    reference_files.par_iter().for_each(|file_path| {
        if let Err(e) = process_file_for_minhash(
            file_path,
            lines_to_keep,
            &band_seeds,
            &perm_seeds,
            &tokenizer,
            &minhash_bands,
            &minhash_signatures,
            &total_lines_processed,
        ) {
            eprintln!("Error processing file {:?} for MinHash: {:?}", file_path, e);
        }
        pbar.inc(1);
    });
    
    pbar.finish_with_message("MinHash signatures computed");
    
    // Store bands in the LSH index
    println!("Building LSH index...");
    for entry in minhash_signatures.iter() {
        let (filename, line_num) = entry.key();
        let signature = entry.value();
        
        let bands = signature.clone().into_shape((MINHASH_NUM_BANDS, MINHASH_BAND_SIZE)).unwrap();
        for row in bands.rows() {
            let mut hasher = Sha256::new();
            hasher.update(bytemuck::cast_slice(row.as_slice().unwrap()));
            let hash = hasher.finalize();
            let band_signature = hash[..8].to_vec();
            
            minhash_bands
                .entry(band_signature)
                .or_default()
                .push((filename.clone(), *line_num));
        }
    }
    
    // Now detect near-duplicates
    println!("Detecting near-duplicates with >{:.0}% similarity...", MINHASH_SIMILARITY_THRESHOLD * 100.0);
    
    let mut minhash_lines_to_keep: HashMap<String, HashSet<usize>> = HashMap::new();
    let near_duplicate_examples: DashMap<(String, usize), Vec<(String, usize, f32)>> = DashMap::new();
    
    // For each signature, check if it has near-duplicates
    for entry in minhash_signatures.iter() {
        let (filename, line_num) = entry.key();
        let signature = entry.value();
        
        // Generate bands for this signature
        let bands = signature.clone().into_shape((MINHASH_NUM_BANDS, MINHASH_BAND_SIZE)).unwrap();
        let mut potential_duplicates: HashMap<(String, usize), u32> = HashMap::new();
        
        // Check each band for collisions
        for row in bands.rows() {
            let mut hasher = Sha256::new();
            hasher.update(bytemuck::cast_slice(row.as_slice().unwrap()));
            let hash = hasher.finalize();
            let band_signature = hash[..8].to_vec();
            
            if let Some(matches) = minhash_bands.get(&band_signature) {
                for (other_file, other_line) in matches.value() {
                    if (other_file, other_line) != (filename, line_num) {
                        *potential_duplicates.entry((other_file.clone(), *other_line)).or_insert(0) += 1;
                    }
                }
            }
        }
        
        // Check Jaccard similarity for potential duplicates
        let mut is_near_duplicate = false;
        let mut similar_docs = Vec::new();
        
        for ((other_file, other_line), _band_matches) in potential_duplicates {
            if let Some(other_sig) = minhash_signatures.get(&(other_file.clone(), other_line)) {
                let similarity = calculate_jaccard_similarity(signature, other_sig.value());
                
                if similarity >= MINHASH_SIMILARITY_THRESHOLD {
                    // This is a near-duplicate
                    similar_docs.push((other_file.clone(), other_line, similarity));
                    
                    // Keep only the first occurrence (lexicographically by filename, then by line number)
                    if (other_file.as_str(), other_line) < (filename.as_str(), *line_num) {
                        is_near_duplicate = true;
                    }
                }
            }
        }
        
        if !similar_docs.is_empty() {
            near_duplicate_examples.insert((filename.clone(), *line_num), similar_docs);
            if is_near_duplicate {
                near_duplicates_found.fetch_add(1, Ordering::Relaxed);
            }
        }
        
        // Add to lines to keep if it's not a near-duplicate
        if !is_near_duplicate {
            minhash_lines_to_keep
                .entry(filename.clone())
                .or_insert_with(HashSet::new)
                .insert(*line_num);
        }
    }
    
    let stats = MinHashStats {
        total_lines_processed: total_lines_processed.load(Ordering::Relaxed),
        near_duplicates_found: near_duplicates_found.load(Ordering::Relaxed),
        near_duplicate_examples: near_duplicate_examples.into_iter().collect(),
    };
    
    Ok((minhash_lines_to_keep, stats))
}

#[derive(Default)]
struct MinHashStats {
    total_lines_processed: usize,
    near_duplicates_found: usize,
    near_duplicate_examples: Vec<((String, usize), Vec<(String, usize, f32)>)>,
}

fn process_file_for_minhash(
    file_path: &PathBuf,
    lines_to_keep: &HashMap<String, HashSet<usize>>,
    _band_seeds: &Vec<u32>,
    perm_seeds: &Vec<u64>,
    tokenizer: &OmniTokenizer,
    _minhash_bands: &MinHashBands,
    minhash_signatures: &MinHashSignatures,
    total_lines_processed: &Arc<AtomicUsize>,
) -> Result<(), Error> {
    let filename = file_path
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("unknown")
        .to_string();
    
    // Get lines to keep for this file
    let keep_lines = lines_to_keep.get(&filename);
    if keep_lines.is_none() || keep_lines.unwrap().is_empty() {
        return Ok(()); // No lines to process for this file
    }
    let keep_lines = keep_lines.unwrap();
    
    // Open file
    let file = File::open(file_path)?;
    let reader: Box<dyn BufRead> = if file_path.extension().and_then(|s| s.to_str()) == Some("gz") {
        Box::new(BufReader::new(GzDecoder::new(file)))
    } else {
        Box::new(BufReader::new(file))
    };
    
    // Process each line that passed exact deduplication
    for (line_num, line) in reader.lines().enumerate() {
        if !keep_lines.contains(&line_num) {
            continue; // Skip lines that were already exact duplicates
        }
        
        let line = line?;
        
        // Parse JSON and extract text
        let json_obj: Value = match serde_json::from_str(&line) {
            Ok(obj) => obj,
            Err(_) => continue,
        };
        
        let text = match get_text_from_json(&json_obj) {
            Ok(t) => t,
            Err(_) => continue,
        };
        
        // Clean text and tokenize
        let cleaned_text = clean_text(&text, "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~");
        let tokens = tokenizer.encode(&cleaned_text);
        
        // Generate MinHash signature
        let hash_vals = get_hash_vals_from_tokens(tokens, perm_seeds, MINHASH_NGRAM_SIZE);
        
        // Store signature
        minhash_signatures.insert((filename.clone(), line_num), hash_vals);
        total_lines_processed.fetch_add(1, Ordering::Relaxed);
    }
    
    Ok(())
}

fn display_minhash_stats(stats: &MinHashStats, dry_run: bool) {
    println!("\n=== MINHASH DEDUPLICATION SUMMARY ===");
    println!("Lines processed for MinHash: {}", stats.total_lines_processed);
    println!("Near-duplicates found (>{:.0}% similar): {}", 
        MINHASH_SIMILARITY_THRESHOLD * 100.0, 
        stats.near_duplicates_found
    );
    
    if dry_run && !stats.near_duplicate_examples.is_empty() {
        println!("\nDetailed near-duplicate comparisons (showing 20 examples):");
        println!("{}", "=".repeat(120));
        
        // Sort by similarity (highest first but not 100%) and take interesting examples
        let mut examples = stats.near_duplicate_examples.clone();
        examples.sort_by(|a, b| {
            let max_sim_a = a.1.iter().map(|(_, _, sim)| sim).max_by(|x, y| x.partial_cmp(y).unwrap()).unwrap_or(&0.0);
            let max_sim_b = b.1.iter().map(|(_, _, sim)| sim).max_by(|x, y| x.partial_cmp(y).unwrap()).unwrap_or(&0.0);
            max_sim_b.partial_cmp(max_sim_a).unwrap()
        });
        
        // Filter to show more interesting examples (not 100% similar)
        let mut shown = 0;
        for ((file, line_num), similar_docs) in examples.iter() {
            if shown >= 20 {
                break;
            }
            
            // Get the most similar match
            let mut sorted_similar = similar_docs.clone();
            sorted_similar.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap());
            
            if let Some((similar_file, similar_line, similarity)) = sorted_similar.first() {
                // Skip if similarity is exactly 100% (less interesting)
                if *similarity == 1.0 && shown > 5 {
                    continue;
                }
                
                // Try to load both texts for comparison
                if let Ok((text1, text2)) = load_texts_for_comparison(file, *line_num, similar_file, *similar_line) {
                    shown += 1;
                    println!("\n{}) {:.1}% similar", shown, similarity * 100.0);
                    println!("File 1: {}:{}", file, line_num + 1);
                    println!("File 2: {}:{}", similar_file, similar_line + 1);
                    println!("{}", "-".repeat(120));
                    
                    // Display side by side
                    display_side_by_side(&text1, &text2, 58);
                    println!("{}", "=".repeat(120));
                }
            }
        }
    }
}

fn load_texts_for_comparison(
    file1: &str,
    line1: usize,
    file2: &str,
    line2: usize,
) -> Result<(String, String), Error> {
    let text1 = get_line_content(file1, line1)?;
    let text2 = get_line_content(file2, line2)?;
    
    // Parse JSON and extract text
    let json1: Value = serde_json::from_str(&text1)?;
    let json2: Value = serde_json::from_str(&text2)?;
    
    let content1 = get_text_from_json(&json1)?;
    let content2 = get_text_from_json(&json2)?;
    
    Ok((content1, content2))
}

fn display_side_by_side(text1: &str, text2: &str, width: usize) {
    use textwrap::{wrap, Options};
    
    let options = Options::new(width)
        .break_words(false)
        .word_separator(textwrap::WordSeparator::AsciiSpace);
    
    let lines1 = wrap(text1, &options);
    let lines2 = wrap(text2, &options);
    
    let max_lines = std::cmp::max(lines1.len(), lines2.len());
    
    println!("{:<width$} │ {:<width$}", "TEXT 1", "TEXT 2", width = width);
    println!("{} │ {}", "-".repeat(width), "-".repeat(width));
    
    for i in 0..max_lines {
        let line1 = lines1.get(i).map(|s| s.as_ref()).unwrap_or("");
        let line2 = lines2.get(i).map(|s| s.as_ref()).unwrap_or("");
        
        // Highlight differences by checking if lines are exactly the same
        if line1 == line2 && !line1.is_empty() {
            println!("{:<width$} │ {:<width$}", line1, line2, width = width);
        } else {
            // Use different marker for different lines
            println!("{:<width$} ┃ {:<width$}", line1, line2, width = width);
        }
    }
}
