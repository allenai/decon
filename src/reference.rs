// Reference dataset post-processing module
// This module handles various operations to refine and improve the quality of reference datasets,
// including deduplication and other preprocessing steps.

use anyhow::{Error, Result};
use dashmap::DashMap;
use rayon::prelude::*;
use serde_json::Value;
use std::collections::{HashMap, HashSet};
use std::fs::{create_dir_all, File};
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use flate2::read::GzDecoder;
use flate2::write::GzEncoder;
use flate2::Compression;

use mj_io::{build_pbar, expand_dirs};

use crate::{clean_text, get_nested_json_val};

// Constants for filtering
const MIN_LENGTH: usize = 15; // Minimum word count for text entries

// Type aliases for clarity
type LineHash = u64;
type LineOccurrences = Vec<(String, usize)>; // (filename, line_number)

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

    // Phase 3: Apply filters to analyze what would be removed
    println!("\nPhase 3: Analyzing filters...");
    let (filtered_lines_to_keep, filter_stats) = apply_filters(&reference_files, &lines_to_keep)?;
    display_filter_stats(&filter_stats);

    // Phase 4: Write files (only if not dry run)
    if !dry_run {
        println!("\nPhase 4: Writing refined files...");
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
    println!("After deduplication: {} (removed {} duplicates)", lines_after_dedup, dedup_stats.total_lines - lines_after_dedup);
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
