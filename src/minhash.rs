// MinHash-specific contamination detection implementation

use ahash::RandomState;
use anyhow::{Error, Result};
use dashmap::DashMap;
use ndarray::Array1;
use rand::prelude::*;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha20Rng;
use rayon::prelude::*;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::collections::{HashMap, VecDeque};
use std::panic::catch_unwind;
use std::path::PathBuf;
use std::time::Instant;
use std::fs::create_dir_all;
use std::io::BufRead;
use unicode_segmentation::UnicodeSegmentation;

use mj_io::{expand_dirs, read_pathbuf_to_mem, write_mem_to_pathbuf, build_pbar};

use crate::{Config, OmniTokenizer, get_nested_json_val, preprocess_text, hash_object, get_results_filename};

const BIG_PRIME: u64 = 18446744073709551557;
const MAX_HASH: u64 = BIG_PRIME;

// Reference band storage: maps band signature to list of (eval_name, line_num)
type ReferenceBands = DashMap<Vec<u8>, Vec<(String, usize)>>;

// Reference signatures storage: maps (eval_name, line_num) to full signature
type ReferenceSignatures = DashMap<(String, usize), Array1<u64>>;

// Public type alias for MinHash index
pub type MinHashIndex = (ReferenceBands, ReferenceSignatures);

pub fn contamination_detect(config: &Config) -> Result<(), Error> {
    println!("Starting MinHash contamination detection...");
    let start_main = Instant::now();

    // Step 1: Process reference datasets and build in-memory band index
    println!("Processing reference datasets...");
    let (reference_bands, reference_signatures) = build_reference_index(config)?;
    println!("Built reference index with {} bands and {} signatures",
             reference_bands.len(), reference_signatures.len());

    // Step 2: Process training data and detect contamination
    println!("Processing training data for contamination detection...");
    detect_contamination_in_training_data(config, &reference_bands, &reference_signatures)?;

    println!("MinHash contamination detection completed in {:?} seconds", start_main.elapsed().as_secs());
    Ok(())
}

pub fn build_reference_index(config: &Config) -> Result<MinHashIndex, Error> {
    let reference_bands: ReferenceBands = DashMap::new();
    let reference_signatures: ReferenceSignatures = DashMap::new();

    // Set up hashing parameters
    let band_seeds: Vec<u32> = _expand_band_seeds(&vec![config.hash_seed as u32], config.num_bands)
        .into_iter()
        .map(|x| x as u32)
        .collect();
    let perm_seeds = _expand_band_seeds(&band_seeds, config.band_size);

    // Find all reference files
    let reference_files = expand_dirs(vec![config.reference_input.clone()], Some(vec![".jsonl", ".gz"].as_slice()))?;
    let pbar = build_pbar(reference_files.len(), "Reference files");

    reference_files.par_iter().for_each(|file_path| {
        if let Err(e) = process_reference_file(
            file_path,
            &band_seeds,
            &perm_seeds,
            config.band_size,
            config.ngram_size,
            &config.tokenizer_str,
            &config.content_key,
            &reference_bands,
            &reference_signatures,
            config.exact_override,
            &config.punctuation_chars
        ) {
            println!("Error processing reference file {:?}: {:?}", file_path, e);
        }
        pbar.inc(1);
    });

    Ok((reference_bands, reference_signatures))
}

fn process_reference_file(
    file_path: &PathBuf,
    band_seeds: &Vec<u32>,
    perm_seeds: &Vec<u64>,
    band_size: usize,
    ngram_size: usize,
    tokenizer_str: &str,
    content_key: &str,
    reference_bands: &ReferenceBands,
    reference_signatures: &ReferenceSignatures,
    exact_override: bool,
    punctuation_chars: &str
) -> Result<(), Error> {
    let data = read_pathbuf_to_mem(file_path)?;
    let tokenizer = OmniTokenizer::new(tokenizer_str)?;
    let num_bands = band_seeds.len();

    // Extract eval name from filename
    let eval_name = file_path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("unknown")
        .to_string();

    println!("Loading hashes for eval dataset: {}", eval_name);

    let mut lines_processed = 0;
    for (line_num, line) in data.lines().enumerate() {
        let line = line?;
        let json_obj: Value = serde_json::from_str(&line)?;
        let line_text = get_nested_json_val(&json_obj, &content_key.to_string())?;
        lines_processed += 1;

        let hash_vals = if exact_override {
            let Ok(tokens) = catch_unwind(|| preprocess_text(&line_text, &tokenizer, punctuation_chars)) else {
                println!("Tokenization failed on {:?} | line {:?}", file_path, line_num);
                continue;
            };
            get_hash_vals_from_tokens(tokens, perm_seeds, ngram_size)
        } else {
            let n = perm_seeds.len();
            let mut hash_vals: Array1<u64> = Array1::ones(n);
            hash_vals = hash_vals * (hash_object(&line_text) as u64);
            hash_vals
        };

        // Store full signature for Jaccard similarity calculation
        reference_signatures.insert((eval_name.clone(), line_num), hash_vals.clone());

        // Generate bands and store in reference index
        let bands = hash_vals.into_shape((num_bands, band_size))?;
        for (_band_idx, row) in bands.rows().into_iter().enumerate() {
            let mut hasher = Sha256::new();
            hasher.update(bytemuck::cast_slice(row.as_slice().unwrap()));
            let hash = hasher.finalize();
            let band_signature = hash[..8].to_vec(); // Truncate to 8 bytes for efficiency

            reference_bands
                .entry(band_signature)
                .or_default()
                .push((eval_name.clone(), line_num));
        }
    }

    println!("  → Processed {} lines from {}", lines_processed, eval_name);
    Ok(())
}

fn detect_contamination_in_training_data(
    config: &Config,
    reference_bands: &ReferenceBands,
    reference_signatures: &ReferenceSignatures
) -> Result<(), Error> {
    // Set up hashing parameters
    let band_seeds: Vec<u32> = _expand_band_seeds(&vec![config.hash_seed as u32], config.num_bands)
        .into_iter()
        .map(|x| x as u32)
        .collect();
    let perm_seeds = _expand_band_seeds(&band_seeds, config.band_size);

    // Find all training files
    let training_files = expand_dirs(vec![config.local_input.clone()], Some(vec![".jsonl", ".gz"].as_slice()))?;
    println!("Found {} training files to process", training_files.len());
    let pbar = build_pbar(training_files.len(), "Training files");

    let contamination_results: DashMap<String, Vec<(usize, String, usize, f32, Option<String>)>> = DashMap::new();

    training_files.par_iter().for_each(|file_path| {
        let file_name = if file_path.extension().and_then(|s| s.to_str()) == Some("gz") {
            // For .gz files, use file_stem to get the .jsonl name
            file_path
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("unknown")
                .to_string()
        } else {
            // For regular files, use the full filename
            file_path
                .file_name()
                .and_then(|s| s.to_str())
                .unwrap_or("unknown")
                .to_string()
        };

        if let Err(e) = process_training_file(
            file_path,
            &file_name,
            &band_seeds,
            &perm_seeds,
            config.band_size,
            config.ngram_size,
            &config.tokenizer_str,
            &config.content_key,
            reference_bands,
            reference_signatures,
            &contamination_results,
            config.exact_override,
            config.jaccard_similarity_threshold,
            &config.punctuation_chars,
            config
        ) {
            println!("Error processing training file {:?}: {:?}", file_path, e);
        }
        pbar.inc(1);
    });

    // Save contamination results
    save_contamination_results(&contamination_results, &config.output_dir)?;

    Ok(())
}

fn reconstruct_text_from_tokens(tokens: &[usize], original_text: &str, tokenizer: &OmniTokenizer) -> String {
    // For uniseg tokenizer, we need to reconstruct from the original text
    // This is a simplified approach - in practice we'd need better token mapping
    match tokenizer.tokenizer_name.as_str() {
        "uniseg" => {
            // For uniseg (word boundaries), try to reconstruct by taking a substring
            // This is an approximation since we don't have perfect token-to-text mapping
            let words: Vec<&str> = original_text.split_word_bounds().collect();
            let window_size = std::cmp::min(tokens.len(), words.len());
            words.iter().take(window_size).cloned().collect::<Vec<_>>().join("")
        }
        _ => {
            // For other tokenizers, return a placeholder
            format!("[Window of {} tokens]", tokens.len())
        }
    }
}

fn generate_document_windows(tokens: &[usize], window_size_increment: usize, num_windows: usize, window_step_size: usize) -> Vec<Vec<usize>> {
    let mut windows = Vec::new();
    
    if tokens.len() <= window_size_increment {
        // Document is smaller than the smallest window, return the full document
        windows.push(tokens.to_vec());
        return windows;
    }
    
    let mut start_pos = 0;
    
    // Generate windows at each step position
    while start_pos < tokens.len() {
        // Generate incrementally sized windows from this starting position
        for window_idx in 0..num_windows {
            let window_size = window_size_increment * (window_idx + 1);
            let end_pos = std::cmp::min(start_pos + window_size, tokens.len());
            
            if end_pos > start_pos {
                let window = tokens[start_pos..end_pos].to_vec();
                windows.push(window);
            }
        }
        
        // Move to next step position
        start_pos += window_step_size;
        if start_pos >= tokens.len() - window_size_increment {
            break; // Avoid generating tiny windows at the end
        }
    }
    
    windows
}

pub fn process_training_file(
    file_path: &PathBuf,
    file_name: &str,
    band_seeds: &Vec<u32>,
    perm_seeds: &Vec<u64>,
    band_size: usize,
    ngram_size: usize,
    tokenizer_str: &str,
    content_key: &str,
    reference_bands: &ReferenceBands,
    reference_signatures: &ReferenceSignatures,
    contamination_results: &DashMap<String, Vec<(usize, String, usize, f32, Option<String>)>>,
    exact_override: bool,
    jaccard_threshold: f32,
    punctuation_chars: &str,
    config: &Config
) -> Result<(), Error> {
    let data = read_pathbuf_to_mem(file_path)?;
    let tokenizer = OmniTokenizer::new(tokenizer_str)?;
    let num_bands = band_seeds.len();

    println!("Checking training file: {}", file_name);
    let mut contaminated_lines = 0;
    let mut _total_lines = 0;

    for (line_num, line) in data.lines().enumerate() {
        let line = line?;
        let json_obj: Value = serde_json::from_str(&line)?;
        let line_text = get_nested_json_val(&json_obj, &content_key.to_string())?;
        _total_lines += 1;

        // Check if windowing is enabled
        let use_windowing = config.window_size_increment.is_some() && 
                           config.num_windows.is_some() && 
                           config.window_step_size.is_some();

        if use_windowing && exact_override {
            // Generate document windows and check each one
            let Ok(tokens) = catch_unwind(|| preprocess_text(&line_text, &tokenizer, punctuation_chars)) else {
                continue;
            };
            
            let window_size_increment = config.window_size_increment.unwrap();
            let num_windows = config.num_windows.unwrap();
            let window_step_size = config.window_step_size.unwrap();
            
            let windows = generate_document_windows(&tokens, window_size_increment, num_windows, window_step_size);
            let mut best_line_matches: HashMap<(String, usize), (f32, Option<String>)> = HashMap::new();
            
            // Process each window and keep the best matches
            for window_tokens in windows {
                let hash_vals = get_hash_vals_from_tokens(window_tokens.clone(), perm_seeds, ngram_size);
                
                // Check for collisions with reference bands
                let bands = hash_vals.clone().into_shape((num_bands, band_size))?;
                
                for (_band_idx, row) in bands.rows().into_iter().enumerate() {
                    let mut hasher = Sha256::new();
                    hasher.update(bytemuck::cast_slice(row.as_slice().unwrap()));
                    let hash = hasher.finalize();
                    let band_signature = hash[..8].to_vec();

                    if let Some(matches) = reference_bands.get(&band_signature) {
                        // Found potential contamination - calculate Jaccard similarity
                        for (eval_name, eval_line_num) in matches.value() {
                            if let Some(ref_sig_entry) = reference_signatures.get(&(eval_name.clone(), *eval_line_num)) {
                                let jaccard_sim = calculate_jaccard_similarity(&hash_vals, ref_sig_entry.value());

                                // Store contamination result (threshold configurable)
                                if jaccard_sim > jaccard_threshold {
                                    let key = (eval_name.clone(), *eval_line_num);
                                    
                                    // Generate window content if debug mode is enabled
                                    let window_content = if config.debug {
                                        Some(reconstruct_text_from_tokens(&window_tokens, &line_text, &tokenizer))
                                    } else {
                                        None
                                    };
                                    
                                    // Keep the highest Jaccard similarity across all windows for each unique eval match
                                    best_line_matches.entry(key)
                                        .and_modify(|existing| {
                                            if jaccard_sim > existing.0 {
                                                *existing = (jaccard_sim, window_content.clone());
                                            }
                                        })
                                        .or_insert((jaccard_sim, window_content));
                                }
                            }
                        }
                    }
                }
            }
            
            // Add deduplicated matches to results
            if !best_line_matches.is_empty() {
                contaminated_lines += 1;
                for ((eval_name, eval_line_num), (jaccard_sim, window_content)) in best_line_matches {
                    contamination_results
                        .entry(file_name.to_string())
                        .or_default()
                        .push((line_num, eval_name, eval_line_num, jaccard_sim, window_content));
                }
            }
        } else {
            // Original processing logic (no windowing or non-exact mode)
            let hash_vals = if exact_override {
                let Ok(tokens) = catch_unwind(|| preprocess_text(&line_text, &tokenizer, punctuation_chars)) else {
                    continue;
                };
                get_hash_vals_from_tokens(tokens, perm_seeds, ngram_size)
            } else {
                let n = perm_seeds.len();
                let mut hash_vals: Array1<u64> = Array1::ones(n);
                hash_vals = hash_vals * (hash_object(&line_text) as u64);
                hash_vals
            };

            // Check for collisions with reference bands
            let bands = hash_vals.clone().into_shape((num_bands, band_size))?;
            let mut line_matches: HashMap<(String, usize), f32> = HashMap::new();

            for (_band_idx, row) in bands.rows().into_iter().enumerate() {
                let mut hasher = Sha256::new();
                hasher.update(bytemuck::cast_slice(row.as_slice().unwrap()));
                let hash = hasher.finalize();
                let band_signature = hash[..8].to_vec();

                if let Some(matches) = reference_bands.get(&band_signature) {
                    // Found potential contamination - calculate Jaccard similarity
                    for (eval_name, eval_line_num) in matches.value() {
                        if let Some(ref_sig_entry) = reference_signatures.get(&(eval_name.clone(), *eval_line_num)) {
                            let jaccard_sim = calculate_jaccard_similarity(&hash_vals, ref_sig_entry.value());

                            // Store contamination result (threshold configurable)
                            if jaccard_sim > jaccard_threshold {
                                let key = (eval_name.clone(), *eval_line_num);
                                // Keep the highest Jaccard similarity for each unique eval match
                                line_matches.entry(key)
                                    .and_modify(|existing_sim| *existing_sim = existing_sim.max(jaccard_sim))
                                    .or_insert(jaccard_sim);
                            }
                        }
                    }
                }
            }

            // Add deduplicated matches to results
            if !line_matches.is_empty() {
                contaminated_lines += 1;
                for ((eval_name, eval_line_num), jaccard_sim) in line_matches {
                    contamination_results
                        .entry(file_name.to_string())
                        .or_default()
                        .push((line_num, eval_name, eval_line_num, jaccard_sim, None));
                }
            }
        }
    }

    if contaminated_lines > 0 {
        // println!("  → Found {} contaminated lines out of {} total lines in {}",
        //         contaminated_lines, total_lines, file_name);
    } else {
        // println!("  → No contamination found in {} ({} lines)", file_name, total_lines);
    }

    Ok(())
}

fn get_hash_vals_from_tokens(tokens: Vec<usize>, perm_seeds: &Vec<u64>, ngram_size: usize) -> Array1<u64> {
    let a = _init_permutations(perm_seeds);
    let n = perm_seeds.len();

    let mut hash_vals = Array1::ones(n) * MAX_HASH;
    let mut ngram: VecDeque<usize> = VecDeque::with_capacity(ngram_size);
    let mut ngram_count = 0;

    for token in tokens {
        ngram.push_back(token);
        if ngram.len() >= ngram_size {
            ngram_count += 1;
            hash_vals = _update_hash_vals(hash_vals, &a, &ngram);
            ngram.pop_front();
        }
    }
    hash_vals = if ngram_count == 0 {
        _update_hash_vals(hash_vals, &a, &ngram) // short document, still wanna hash it
    } else {
        hash_vals
    };

    hash_vals
}

fn _init_permutations(seeds: &Vec<u64>) -> Array1<u128> {
    // Initialize the permutations needed for each minhash
    let n = seeds.len();
    let mut a = Array1::zeros(n);
    for (i, &seed) in seeds.iter().enumerate() {
        let mut rng = ChaCha20Rng::seed_from_u64(seed);
        a[i] = rng.gen::<u128>() as u128;
    }
    a
}

#[allow(dead_code)]
fn rand_u64s(seed: u64, output_size: usize) -> Vec<u64> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut output: Vec<u64> = Vec::new();
    for _i in 0..output_size {
        output.push(rng.gen::<u64>());
    }
    output
}

fn _update_hash_vals(mut hash_vals: Array1<u64>, a: &Array1<u128>, ngram: &VecDeque<usize>) -> Array1<u64> {
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

pub fn _expand_band_seeds(band_seeds: &Vec<u32>, band_size: usize) -> Vec<u64> {
    // Each "band seed" is expanded here to band_size random u64s, and flattened. (used to seed permutations)
    // Probably like no collisions here, so let's just not worry about that ;)

    let mut perm_seeds: Vec<u64> = Vec::new();
    for band_seed in band_seeds.iter() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(*band_seed as u64);
        for _i in 0..band_size {
            perm_seeds.push(rng.next_u64());
        }
    }
    perm_seeds
}

fn calculate_jaccard_similarity(sig1: &Array1<u64>, sig2: &Array1<u64>) -> f32 {
    let matches = sig1.iter()
        .zip(sig2.iter())
        .filter(|(a, b)| a == b)
        .count();

    matches as f32 / sig1.len() as f32
}

pub fn save_contamination_results(
    results: &DashMap<String, Vec<(usize, String, usize, f32, Option<String>)>>,
    output_dir: &PathBuf
) -> Result<PathBuf, Error> {
    save_contamination_results_with_filename(results, output_dir, None)
}

pub fn save_contamination_results_with_filename(
    results: &DashMap<String, Vec<(usize, String, usize, f32, Option<String>)>>,
    output_dir: &PathBuf,
    custom_filename: Option<&str>
) -> Result<PathBuf, Error> {
    create_dir_all(output_dir)?;
    let default_filename = get_results_filename("minhash");
    let filename = custom_filename.unwrap_or(&default_filename);
    let output_file = output_dir.join(filename);

    let mut output_data = Vec::new();
    let mut total_contaminations = 0;

    for entry in results.iter() {
        let training_file = entry.key();
        for (training_line, eval_name, eval_line, jaccard_sim, window_content) in entry.value() {
            let mut result = json!({
                "training_file": training_file,
                "training_line": training_line,
                "eval_dataset": eval_name,
                "eval_line": eval_line,
                "jaccard_similarity": jaccard_sim
            });
            
            // Add window content if available
            if let Some(content) = window_content {
                result["window_content"] = json!(content);
            }
            
            output_data.push(serde_json::to_vec(&result)?);
            total_contaminations += 1;
        }
    }

    let mut output_bytes = Vec::new();
    for line in output_data {
        output_bytes.extend(line);
        output_bytes.push(b'\n');
    }

    write_mem_to_pathbuf(&output_bytes, &output_file)?;

    if total_contaminations > 0 {
        println!("=== CONTAMINATION SUMMARY ===");
        println!("Found {} contamination instances across {} files",
                total_contaminations, results.len());
        println!("Results saved to: {:?}", output_file);
    } else {
        println!("=== NO CONTAMINATION DETECTED ===");
        println!("No contamination found in training data");
        println!("Empty results file saved to: {:?}", output_file);
    }

    Ok(output_file)
}
