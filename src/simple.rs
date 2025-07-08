use anyhow::{Error, Result};
use dashmap::DashMap;
use rayon::prelude::*;
use serde_json::{Value, json};
use std::collections::{HashMap, HashSet};
use std::fs::{create_dir_all, File};
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::panic::catch_unwind;
use std::path::PathBuf;
use std::time::Instant;

use mj_io::{expand_dirs, read_pathbuf_to_mem, write_mem_to_pathbuf, build_pbar};

use crate::{Config, get_nested_json_val, get_results_filename, get_purified_filename, OmniTokenizer, preprocess_text, clean_text};

// Simple mode structures
type NgramToIdMap = DashMap<u64, u64>; // Maps n-gram hash to unique ID
type IdToDocsMap = DashMap<u64, HashSet<u32>>; // Maps n-gram ID to set of document IDs
type EvalDocuments = DashMap<u32, (String, usize, usize, usize)>; // Maps doc_id to (eval_name, line_num, total_ngrams, unique_ngrams)
type IdToNgramTokens = DashMap<u64, Vec<usize>>; // Maps unique ID to token IDs for display

pub fn contamination_detect(config: &Config) -> Result<(), Error> {
    println!("Starting SIMPLE contamination detection...");
    let start_main = Instant::now();

    // Step 1: Process eval datasets and build n-gram mappings
    println!("Processing eval datasets and building n-gram index...");
    let index_start = Instant::now();
    let (ngram_to_id, id_to_docs, eval_documents, id_to_ngram_tokens, tokenizer) = build_simple_index(config)?;
    let index_time = index_start.elapsed();
    println!("Built simple index with {} unique n-grams in {:.2}s", ngram_to_id.len(), index_time.as_secs_f64());

    // Step 2: Process training data and detect contamination
    println!("Processing training data for contamination detection...");
    let detection_start = Instant::now();
    let total_contaminations = detect_simple_contamination(config, &ngram_to_id, &id_to_docs, &eval_documents, &id_to_ngram_tokens, &tokenizer)?;
    let detection_time = detection_start.elapsed();

    let total_time = start_main.elapsed();
    println!("\n=== SIMPLE Contamination Detection Results ===");
    println!("Index building time: {:.2}s", index_time.as_secs_f64());
    println!("Detection time: {:.2}s", detection_time.as_secs_f64());
    println!("Total time: {:.2}s", total_time.as_secs_f64());
    println!("Total contaminations found: {}", total_contaminations);
    Ok(())
}

// Public type for the index
pub type SimpleIndex = (NgramToIdMap, IdToDocsMap, EvalDocuments, IdToNgramTokens, OmniTokenizer);

pub fn build_simple_index(config: &Config) -> Result<SimpleIndex, Error> {
    let ngram_to_id: NgramToIdMap = DashMap::new();
    let id_to_docs: IdToDocsMap = DashMap::new();
    let eval_documents: EvalDocuments = DashMap::new();
    let id_to_ngram_tokens: IdToNgramTokens = DashMap::new();

    use std::sync::atomic::AtomicU64;
    let next_ngram_id = AtomicU64::new(0);
    
    // Initialize tokenizer
    let tokenizer = OmniTokenizer::new(&config.tokenizer_str)?;

    // Find all reference files
    let reference_files = expand_dirs(
        vec![config.reference_input.clone()],
        Some(vec![".jsonl", ".gz"].as_slice())
    )?;
    let pbar = build_pbar(reference_files.len(), "Reference files");

    // println!("Processing {} reference files for n-gram indexing...", reference_files.len()); //debug

    for file_path in reference_files.iter() {
        if let Err(e) = process_simple_reference_file(
            file_path,
            config,
            &ngram_to_id,
            &id_to_docs,
            &eval_documents,
            &next_ngram_id,
            &tokenizer,
            &id_to_ngram_tokens
        ) {
            println!("Error processing reference file {:?}: {:?}", file_path, e);
        }
        pbar.inc(1);
    }

    // println!("Finished processing reference files. Total unique n-grams: {}", ngram_to_id.len()); //debug
    // println!("Total eval documents indexed: {}", eval_documents.len()); //debug

    Ok((ngram_to_id, id_to_docs, eval_documents, id_to_ngram_tokens, tokenizer))
}

fn process_simple_reference_file(
    file_path: &PathBuf,
    config: &Config,
    ngram_to_id: &NgramToIdMap,
    id_to_docs: &IdToDocsMap,
    eval_documents: &EvalDocuments,
    next_ngram_id: &std::sync::atomic::AtomicU64,
    tokenizer: &OmniTokenizer,
    id_to_ngram_tokens: &IdToNgramTokens
) -> Result<(), Error> {
    let data = read_pathbuf_to_mem(file_path)?;

    // Extract eval name from filename
    let eval_name = file_path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("unknown")
        .to_string();

    // println!("Processing eval dataset: {}", eval_name); //debug

    let mut _lines_processed = 0;
    let mut _skipped_entries = 0;
    let min_word_count = config.ngram_size * 2;

    for (line_num, line) in data.lines().enumerate() {
        let line = line?;
        let json_obj: Value = serde_json::from_str(&line)?;
        let line_text = get_nested_json_val(&json_obj, &config.content_key.to_string())?;

        // Use preprocess_text to get token IDs
        let word_tokens = if config.tokenizer_str == "word" {
            // For word mode, build vocabulary from reference set
            let cleaned = clean_text(&line_text, &config.punctuation_chars);
            let words: Vec<String> = cleaned.split_whitespace()
                .map(|s| s.to_string())
                .filter(|s| !s.is_empty())
                .collect();
            // Add words to vocabulary and get IDs
            words.iter()
                .map(|word| tokenizer.add_word(word) as usize)
                .collect()
        } else {
            let Ok(tokens) = catch_unwind(|| preprocess_text(&line_text, tokenizer, &config.punctuation_chars)) else {
                println!("Tokenization failed on {:?} | line {:?}", file_path, line_num);
                _skipped_entries += 1;
                continue;
            };
            tokens
        };
        let word_count = word_tokens.len();

        // Skip entries with insufficient tokens for meaningful n-gram analysis
        if word_count < min_word_count {
            _skipped_entries += 1;
            continue;
        }

        _lines_processed += 1;

        // Calculate total n-grams for this document
        let total_ngrams = if word_tokens.len() < config.ngram_size {
            if word_tokens.is_empty() { 0 } else { 1 }
        } else {
            word_tokens.len() - config.ngram_size + 1
        };

        // Generate document ID (similar to toxic approach)
        let doc_id = generate_doc_id(&eval_name, line_num);

        // Track unique n-grams for this document
        let mut unique_ngrams_set = HashSet::new();

        // Process n-grams
        if word_tokens.len() < config.ngram_size {
            if !word_tokens.is_empty() {
                // For documents shorter than ngram_size, use all tokens
                let ngram_tokens = word_tokens.clone();
                let ngram_id = get_or_create_ngram_id(&ngram_tokens, ngram_to_id, next_ngram_id, id_to_ngram_tokens);
                unique_ngrams_set.insert(ngram_id);
                add_doc_to_ngram(ngram_id, doc_id, id_to_docs);

                // println!("Short doc ngram: '{:?}' -> ID: {}", ngram_tokens, ngram_id); //debug
            }
        } else {
            for i in 0..=word_tokens.len() - config.ngram_size {
                let ngram_slice = &word_tokens[i..i + config.ngram_size];
                let ngram_tokens = ngram_slice.to_vec();
                let ngram_id = get_or_create_ngram_id(&ngram_tokens, ngram_to_id, next_ngram_id, id_to_ngram_tokens);
                unique_ngrams_set.insert(ngram_id);
                add_doc_to_ngram(ngram_id, doc_id, id_to_docs);

                // if i < 3 { // Only log first few for debugging
                //     println!("Ngram {}: '{:?}' -> ID: {}", i, ngram_tokens, ngram_id); //debug
                // }
            }
        }

        let unique_ngrams = unique_ngrams_set.len();

        // Store document metadata
        eval_documents.insert(doc_id, (eval_name.clone(), line_num, total_ngrams, unique_ngrams));

        // if _lines_processed % 1000 == 0 {
        //     println!("Processed {} lines from {}, current n-gram vocab size: {}", _lines_processed, eval_name, ngram_to_id.len()); //debug
        // }
    }

    // println!("Finished {}: processed {} lines, skipped {} short entries", eval_name, _lines_processed, _skipped_entries); //debug
    Ok(())
}

// Helper functions
fn hash_ngram(tokens: &[usize]) -> u64 {
    use std::hash::{Hash, Hasher, DefaultHasher};
    let mut hasher = DefaultHasher::new();
    tokens.hash(&mut hasher);
    hasher.finish()
}

fn generate_doc_id(eval_name: &str, line_num: usize) -> u32 {
    // Simple hash-based document ID generation
    use std::hash::{Hash, Hasher, DefaultHasher};
    let mut hasher = DefaultHasher::new();
    eval_name.hash(&mut hasher);
    line_num.hash(&mut hasher);
    (hasher.finish() & 0xFFFFFFFF) as u32
}

fn get_or_create_ngram_id(ngram_tokens: &[usize], ngram_to_id: &NgramToIdMap, next_id: &std::sync::atomic::AtomicU64, id_to_ngram_tokens: &IdToNgramTokens) -> u64 {
    use std::sync::atomic::Ordering;
    
    let ngram_hash = hash_ngram(ngram_tokens);

    if let Some(existing_id) = ngram_to_id.get(&ngram_hash) {
        *existing_id
    } else {
        let new_id = next_id.fetch_add(1, Ordering::SeqCst);
        // Insert and handle potential race condition
        let id = ngram_to_id.entry(ngram_hash).or_insert(new_id).clone();
        // Store the tokens for display/lookup
        id_to_ngram_tokens.insert(id, ngram_tokens.to_vec());
        id
    }
}

fn add_doc_to_ngram(ngram_id: u64, doc_id: u32, id_to_docs: &IdToDocsMap) {
    id_to_docs.entry(ngram_id).or_insert_with(HashSet::new).insert(doc_id);
}

fn detect_simple_contamination(
    config: &Config,
    ngram_to_id: &NgramToIdMap,
    id_to_docs: &IdToDocsMap,
    eval_documents: &EvalDocuments,
    id_to_ngram_tokens: &IdToNgramTokens,
    tokenizer: &OmniTokenizer
) -> Result<usize, Error> {
    // println!("Starting training data contamination detection..."); //debug

    // Find all training files
    let training_files = expand_dirs(
        vec![config.local_input.clone()],
        Some(vec![".jsonl", ".gz"].as_slice())
    )?;

    // println!("Found {} training files to process", training_files.len()); //debug
    let pbar = build_pbar(training_files.len(), "Training files");

    // Results accumulator
    let contamination_results = DashMap::new();
    let total_lines_processed = std::sync::atomic::AtomicUsize::new(0);

    let processing_start = Instant::now();
    training_files.par_iter().for_each(|file_path| {
        match process_simple_training_file(
            file_path,
            config,
            ngram_to_id,
            id_to_docs,
            eval_documents,
            &contamination_results,
            tokenizer,
            id_to_ngram_tokens
        ) {
            Ok(lines_processed) => {
                total_lines_processed.fetch_add(lines_processed, std::sync::atomic::Ordering::SeqCst);
            }
            Err(e) => {
                println!("Error processing training file {:?}: {:?}", file_path, e);
            }
        }
        pbar.inc(1);
    });
    let processing_time = processing_start.elapsed();

    // Count total contaminations
    let total_contaminations: usize = contamination_results.iter()
        .map(|entry| entry.value().len())
        .sum();

    let lines_processed = total_lines_processed.load(std::sync::atomic::Ordering::SeqCst);
    if lines_processed > 0 {
        let total_micros = processing_time.as_micros() as f64;
        let micros_per_line = total_micros / lines_processed as f64;

        if micros_per_line >= 1000.0 {
            println!("Processed {} training lines in {:.2}s ({:.2} ms/line)",
                     lines_processed, processing_time.as_secs_f64(), micros_per_line / 1000.0);
        } else {
            println!("Processed {} training lines in {:.2}s ({:.0} μs/line)",
                     lines_processed, processing_time.as_secs_f64(), micros_per_line);
        }
    }

    // Save results
    save_contamination_results_toxic_format(config, &contamination_results)?;

    // Create purified files if requested
    if config.purify {
        create_purified_files(config, &contamination_results, &training_files)?;
    }

    // println!("Contamination detection completed. Results saved."); //debug
    Ok(total_contaminations)
}

pub type ContaminationResults = DashMap<String, Vec<SimpleContaminationEntry>>;

#[derive(Clone)]
pub struct SimpleContaminationEntry {
    pub training_line: usize,
    eval_name: String,
    eval_line: usize,
    overlap_ratio: f32,
    toxic_score: f32,
    matching_ngrams: Vec<String>,
    // Token indices for position recovery
    contamination_start_idx: Option<usize>, // Start index in token array
    contamination_end_idx: Option<usize>,   // End index in token array
}

#[derive(Clone)]
struct SimpleContaminationCluster {
    start_idx: usize,
    end_idx: usize,
    document_matches: HashMap<u32, HashSet<u64>>, // doc_id -> unique_ngram_ids
    matching_ngrams: Vec<String>,
}

pub fn process_simple_training_file(
    file_path: &PathBuf,
    config: &Config,
    ngram_to_id: &NgramToIdMap,
    id_to_docs: &IdToDocsMap,
    eval_documents: &EvalDocuments,
    contamination_results: &ContaminationResults,
    tokenizer: &OmniTokenizer,
    id_to_ngram_tokens: &IdToNgramTokens
) -> Result<usize, Error> {
    let data = read_pathbuf_to_mem(file_path)?;

    let file_name = if file_path.extension().and_then(|s| s.to_str()) == Some("gz") {
        file_path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("unknown")
            .to_string()
    } else {
        file_path
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("unknown")
            .to_string()
    };

    // println!("Processing training file: {}", file_name); //debug

    let mut lines_processed = 0;
    let min_word_count = config.ngram_size * 2;

    for (line_num, line) in data.lines().enumerate() {
        let line = line?;
        let json_obj: Value = serde_json::from_str(&line)?;
        let line_text = get_nested_json_val(&json_obj, &config.content_key.to_string())?;

        // Use preprocess_text to get token IDs
        let Ok(word_tokens) = catch_unwind(|| preprocess_text(&line_text, tokenizer, &config.punctuation_chars)) else {
            println!("Tokenization failed on {:?} | line {:?}", file_path, line_num);
            continue;
        };
        let word_count = word_tokens.len();

        // Skip entries with insufficient tokens
        if word_count < min_word_count {
            continue;
        }

        lines_processed += 1;

        // Process n-grams with sampling
        let clusters = process_ngrams_with_simple_sampling(
            &word_tokens,
            config,
            ngram_to_id,
            id_to_docs,
            id_to_ngram_tokens
        )?;

        // Convert clusters to contamination entries
        for cluster in clusters {
            let mut cluster_results = Vec::new();

            for (doc_id, matched_ngram_ids) in cluster.document_matches {
                if let Some(doc_info) = eval_documents.get(&doc_id) {
                    let (eval_name, eval_line, _total_ngrams, unique_ngrams) = doc_info.value();
                    let unique_matches = matched_ngram_ids.len();
                    let overlap_ratio = if *unique_ngrams > 0 {
                        unique_matches as f32 / *unique_ngrams as f32
                    } else {
                        0.0
                    };

                    // println!("DEBUG: eval={}:{}, unique_matches={}, unique_ngrams={}, ratio={:.3}", //debug
                    //          eval_name, eval_line, unique_matches, unique_ngrams, overlap_ratio);

                    // Calculate toxic score using IDF approach
                    let toxic_score = calculate_simple_toxic_score(
                        &cluster.matching_ngrams,
                        ngram_to_id,
                        id_to_docs,
                        id_to_ngram_tokens
                    );

                    // Only record results that exceed both thresholds
                    if overlap_ratio >= config.toxic_overlap_threshold && toxic_score >= config.toxic_score_threshold {
                        let entry = SimpleContaminationEntry {
                            training_line: line_num,
                            eval_name: eval_name.clone(),
                            eval_line: *eval_line,
                            overlap_ratio,
                            toxic_score,
                            matching_ngrams: cluster.matching_ngrams.clone(),
                            contamination_start_idx: Some(cluster.start_idx),
                            contamination_end_idx: Some(cluster.end_idx),
                        };

                        cluster_results.push(entry);

                        // println!("Found contamination: train_line={}, eval={}:{}, overlap={:.3} (>={:.3}), toxic_score={:.3} (>={:.3})", //debug
                        //          line_num, eval_name, eval_line, overlap_ratio, config.toxic_overlap_threshold, toxic_score, config.toxic_score_threshold);
                    } else {
                        // println!("Below threshold: train_line={}, eval={}:{}, overlap={:.3} < {:.3} OR toxic_score={:.3} < {:.3}", //debug
                        //          line_num, eval_name, eval_line, overlap_ratio, config.toxic_overlap_threshold, toxic_score, config.toxic_score_threshold);
                    }
                }
            }

            if !cluster_results.is_empty() {
                contamination_results.entry(file_name.clone()).or_insert_with(Vec::new).extend(cluster_results);
            }
        }

        // if lines_processed % 10000 == 0 {
        //     println!("Processed {} lines from {}", lines_processed, file_name); //debug
        // }
    }

    // println!("Finished processing {}: {} lines", file_name, lines_processed); //debug
    Ok(lines_processed)
}

/// Process n-grams with sampling: sample every M n-grams, then expand around hits
fn process_ngrams_with_simple_sampling(
    word_tokens: &[usize],
    config: &Config,
    ngram_to_id: &NgramToIdMap,
    id_to_docs: &IdToDocsMap,
    id_to_ngram_tokens: &IdToNgramTokens
) -> Result<Vec<SimpleContaminationCluster>, Error> {
    let mut clusters = Vec::new();
    let mut processed_indices = HashSet::new();
    let mut i = 0;

    // println!("Starting sampling with M={}, word_tokens len={}", config.sample_every_m_tokens, word_tokens.len()); //debug

    // Calculate total n-grams
    let total_ngrams = if word_tokens.len() < config.ngram_size {
        if word_tokens.is_empty() { 0 } else { 1 }
    } else {
        word_tokens.len() - config.ngram_size + 1
    };

    while i < total_ngrams {
        // Skip if we've already processed this n-gram as part of a cluster
        if processed_indices.contains(&i) {
            i += 1;
            continue;
        }

        // Sample every M n-grams (unless M=1, which means no sampling)
        if config.sample_every_m_tokens > 1 && i % config.sample_every_m_tokens != 0 {
            i += config.sample_every_m_tokens - (i % config.sample_every_m_tokens);
            continue;
        }

        // Check this n-gram for contamination
        if let Some(document_ids) = check_ngram_for_match(i, word_tokens, config, ngram_to_id, id_to_docs) {
            let ngram_tokens = if word_tokens.len() < config.ngram_size {
                word_tokens.to_vec()
            } else {
                word_tokens[i..i + config.ngram_size].to_vec()
            };

            // println!("HIT DETECTED at n-gram {}: '{:?}' with {} documents", i, ngram_tokens, document_ids.len()); //debug

            // Found contamination! Expand around this hit using intersection-based walking
            let cluster = expand_simple_contamination_cluster(
                i,
                word_tokens,
                config,
                ngram_to_id,
                id_to_docs,
                document_ids,
                &ngram_tokens,
                id_to_ngram_tokens
            )?;

            // Mark all indices in this cluster as processed
            for idx in cluster.start_idx..=cluster.end_idx {
                processed_indices.insert(idx);
            }

            clusters.push(cluster.clone());
            // println!("Cluster completed: indices {}-{}, {} document matches", //debug
            //          cluster.start_idx, cluster.end_idx, cluster.document_matches.len());

            // Jump past the processed region
            i = processed_indices.iter().max().copied().unwrap_or(i) + 1;
        } else {
            // No hit, continue sampling
            i += config.sample_every_m_tokens.max(1);
        }
    }

    // println!("Sampling completed, found {} clusters", clusters.len()); //debug
    Ok(clusters)
}

/// Check a single n-gram for matches, return document IDs that match
fn check_ngram_for_match(
    ngram_idx: usize,
    word_tokens: &[usize],
    config: &Config,
    ngram_to_id: &NgramToIdMap,
    id_to_docs: &IdToDocsMap
) -> Option<HashSet<u32>> {
    let ngram_tokens = if word_tokens.len() < config.ngram_size {
        word_tokens.to_vec()
    } else {
        word_tokens[ngram_idx..ngram_idx + config.ngram_size].to_vec()
    };
    
    // Skip n-grams containing unknown words (u32::MAX)
    if ngram_tokens.iter().any(|&token| token == u32::MAX as usize) {
        return None;
    }
    
    let ngram_hash = hash_ngram(&ngram_tokens);

    // Look up n-gram ID
    if let Some(ngram_id) = ngram_to_id.get(&ngram_hash) {
        // Look up documents containing this n-gram
        if let Some(doc_set) = id_to_docs.get(&ngram_id) {
            if !doc_set.is_empty() {
                // println!("N-gram '{:?}' found in {} documents", ngram_tokens, doc_set.len()); //debug
                return Some(doc_set.clone());
            }
        }
    }
    None
}

/// Expand contamination cluster using intersection-based left/right traversal
fn expand_simple_contamination_cluster(
    hit_idx: usize,
    word_tokens: &[usize],
    config: &Config,
    ngram_to_id: &NgramToIdMap,
    id_to_docs: &IdToDocsMap,
    initial_document_ids: HashSet<u32>,
    initial_training_ngram: &[usize],
    id_to_ngram_tokens: &IdToNgramTokens
) -> Result<SimpleContaminationCluster, Error> {
    let mut start_idx = hit_idx;
    let mut end_idx = hit_idx;
    let mut matching_ngrams = Vec::new();

    // Initialize document match tracking - track consecutive misses for each document
    let mut document_matches: HashMap<u32, HashSet<u64>> = HashMap::new();
    let mut document_misses: HashMap<u32, usize> = HashMap::new();
    let mut active_documents: HashSet<u32> = initial_document_ids.clone();

    // Get the initial n-gram ID for tracking
    let initial_ngram_hash = hash_ngram(initial_training_ngram);
    let initial_ngram_id = ngram_to_id.get(&initial_ngram_hash)
        .map(|id| *id)
        .unwrap_or(0);

    for doc_id in &initial_document_ids {
        let mut matched_ngrams = HashSet::new();
        matched_ngrams.insert(initial_ngram_id);
        document_matches.insert(*doc_id, matched_ngrams);
        document_misses.insert(*doc_id, 0);
    }

    // Store display format for the initial ngram
    if let Some(tokens) = id_to_ngram_tokens.get(&initial_ngram_id) {
        matching_ngrams.push(format!("{:?}", tokens.value()));
    } else {
        matching_ngrams.push(format!("{:?}", initial_training_ngram));
    }

    // println!("Starting intersection walking with {} documents", initial_document_ids.len()); //debug
    // println!("Initial n-gram: '{}'", initial_training_ngram); //debug

    let total_ngrams = if word_tokens.len() < config.ngram_size {
        if word_tokens.is_empty() { 0 } else { 1 }
    } else {
        word_tokens.len() - config.ngram_size + 1
    };

    // Expand backward (left traversal)
    // println!("Starting left traversal from index {}", hit_idx); //debug
    let mut left_idx = hit_idx;
    while left_idx > 0 && !active_documents.is_empty() {
        left_idx -= 1;

        if let Some(matched_docs) = check_ngram_for_match(left_idx, word_tokens, config, ngram_to_id, id_to_docs) {
            let ngram_tokens = if word_tokens.len() < config.ngram_size {
                word_tokens.to_vec()
            } else {
                word_tokens[left_idx..left_idx + config.ngram_size].to_vec()
            };

            // println!("Left traversal: checking n-gram {} '{:?}'", left_idx, ngram_tokens); //debug

            // Check intersection with active documents
            let intersection: Vec<u32> = active_documents.intersection(&matched_docs).cloned().collect();

            if !intersection.is_empty() {
                // println!("Left hit at {}: {} docs intersect", left_idx, intersection.len()); //debug
                start_idx = left_idx;

                // Get ngram_id before moving ngram_tokens
                let ngram_hash = hash_ngram(&ngram_tokens);
                let ngram_id = ngram_to_id.get(&ngram_hash).map(|id| *id).unwrap_or(0);
                // Store display format
                if let Some(tokens) = id_to_ngram_tokens.get(&ngram_id) {
                    matching_ngrams.insert(0, format!("{:?}", tokens.value()));
                } else {
                    matching_ngrams.insert(0, format!("{:?}", ngram_tokens));
                }

                // Update matches and reset misses for intersecting documents
                for doc_id in &intersection {
                    document_matches.entry(*doc_id).or_insert_with(HashSet::new).insert(ngram_id);
                    document_misses.insert(*doc_id, 0);
                }

                // Remove documents that didn't match this n-gram
                let to_remove: Vec<u32> = active_documents.difference(&matched_docs).cloned().collect();
                for doc_id in to_remove {
                    let miss_count = document_misses.entry(doc_id).or_insert(0);
                    *miss_count += 1;
                    if *miss_count >= config.max_consecutive_misses as usize {
                        active_documents.remove(&doc_id);
                        // println!("Removing doc {} after {} consecutive misses", doc_id, miss_count); //debug
                    }
                }
            } else {
                // println!("Left miss at {}: no intersection", left_idx); //debug
                // Increment miss count for all active documents
                let mut to_remove = Vec::new();
                for doc_id in &active_documents {
                    let miss_count = document_misses.entry(*doc_id).or_insert(0);
                    *miss_count += 1;
                    if *miss_count >= config.max_consecutive_misses as usize {
                        to_remove.push(*doc_id);
                    }
                }
                for doc_id in to_remove {
                    active_documents.remove(&doc_id);
                    // println!("Removing doc {} after {} consecutive misses", doc_id, document_misses[&doc_id]); //debug
                }
            }
        } else {
            // println!("Left miss at {}: n-gram not in eval", left_idx); //debug
            // Increment miss count for all active documents
            let mut to_remove = Vec::new();
            for doc_id in &active_documents {
                let miss_count = document_misses.entry(*doc_id).or_insert(0);
                *miss_count += 1;
                if *miss_count >= config.max_consecutive_misses as usize {
                    to_remove.push(*doc_id);
                }
            }
            for doc_id in to_remove {
                active_documents.remove(&doc_id);
                // println!("Removing doc {} after {} consecutive misses", doc_id, document_misses[&doc_id]); //debug
            }
        }
    }

    // Reset active documents and misses for right traversal
    active_documents = initial_document_ids.clone();
    for doc_id in &initial_document_ids {
        document_misses.insert(*doc_id, 0);
    }

    // Expand forward (right traversal)
    // println!("Starting right traversal from index {}", hit_idx); //debug
    let mut right_idx = hit_idx;
    while right_idx + 1 < total_ngrams && !active_documents.is_empty() {
        right_idx += 1;

        if let Some(matched_docs) = check_ngram_for_match(right_idx, word_tokens, config, ngram_to_id, id_to_docs) {
            let ngram_tokens = if word_tokens.len() < config.ngram_size {
                word_tokens.to_vec()
            } else {
                word_tokens[right_idx..right_idx + config.ngram_size].to_vec()
            };

            // println!("Right traversal: checking n-gram {} '{:?}'", right_idx, ngram_tokens); //debug

            // Check intersection with active documents
            let intersection: Vec<u32> = active_documents.intersection(&matched_docs).cloned().collect();

            if !intersection.is_empty() {
                // println!("Right hit at {}: {} docs intersect", right_idx, intersection.len()); //debug
                end_idx = right_idx;

                // Get ngram_id before moving ngram_tokens
                let ngram_hash = hash_ngram(&ngram_tokens);
                let ngram_id = ngram_to_id.get(&ngram_hash).map(|id| *id).unwrap_or(0);
                // Store display format
                if let Some(tokens) = id_to_ngram_tokens.get(&ngram_id) {
                    matching_ngrams.insert(0, format!("{:?}", tokens.value()));
                } else {
                    matching_ngrams.insert(0, format!("{:?}", ngram_tokens));
                }

                // Update matches and reset misses for intersecting documents
                for doc_id in &intersection {
                    document_matches.entry(*doc_id).or_insert_with(HashSet::new).insert(ngram_id);
                    document_misses.insert(*doc_id, 0);
                }

                // Remove documents that didn't match this n-gram
                let to_remove: Vec<u32> = active_documents.difference(&matched_docs).cloned().collect();
                for doc_id in to_remove {
                    let miss_count = document_misses.entry(doc_id).or_insert(0);
                    *miss_count += 1;
                    if *miss_count >= config.max_consecutive_misses as usize {
                        active_documents.remove(&doc_id);
                        // println!("Removing doc {} after {} consecutive misses", doc_id, miss_count); //debug
                    }
                }
            } else {
                // println!("Right miss at {}: no intersection", right_idx); //debug
                // Increment miss count for all active documents
                let mut to_remove = Vec::new();
                for doc_id in &active_documents {
                    let miss_count = document_misses.entry(*doc_id).or_insert(0);
                    *miss_count += 1;
                    if *miss_count >= config.max_consecutive_misses as usize {
                        to_remove.push(*doc_id);
                    }
                }
                for doc_id in to_remove {
                    active_documents.remove(&doc_id);
                    // println!("Removing doc {} after {} consecutive misses", doc_id, document_misses[&doc_id]); //debug
                }
            }
        } else {
            // println!("Right miss at {}: n-gram not in eval", right_idx); //debug
            // Increment miss count for all active documents
            let mut to_remove = Vec::new();
            for doc_id in &active_documents {
                let miss_count = document_misses.entry(*doc_id).or_insert(0);
                *miss_count += 1;
                if *miss_count >= config.max_consecutive_misses as usize {
                    to_remove.push(*doc_id);
                }
            }
            for doc_id in to_remove {
                active_documents.remove(&doc_id);
                // println!("Removing doc {} after {} consecutive misses", doc_id, document_misses[&doc_id]); //debug
            }
        }
    }

    // println!("Cluster expansion complete: indices {}-{}, {} matching n-grams", //debug
    //          start_idx, end_idx, matching_ngrams.len());

    Ok(SimpleContaminationCluster {
        start_idx,
        end_idx,
        document_matches,
        matching_ngrams,
    })
}

/// Calculate toxic score using inverse document frequency approach
/// Similar to toxic.rs, but for n-gram buckets instead of LSH buckets
fn calculate_simple_toxic_score(
    matching_ngrams: &[String],
    ngram_to_id: &NgramToIdMap,
    id_to_docs: &IdToDocsMap,
    _id_to_ngram_tokens: &IdToNgramTokens
) -> f32 {
    let mut unique_ngram_ids = HashSet::new();

    // For display ngrams (formatted as "[id1, id2, id3]"), we need to extract the actual ngram IDs
    // This is a bit inefficient but maintains compatibility with the display format
    for ngram_str in matching_ngrams {
        // Parse the string representation back to token IDs
        if let Ok(tokens) = parse_ngram_string(ngram_str) {
            let ngram_hash = hash_ngram(&tokens);
            if let Some(ngram_id) = ngram_to_id.get(&ngram_hash) {
                unique_ngram_ids.insert(*ngram_id);
            }
        }
    }

    // Calculate toxic score: sum of 1/ln(document_count) for each unique n-gram
    let toxic_score: f32 = unique_ngram_ids.iter()
        .map(|ngram_id| {
            if let Some(doc_set) = id_to_docs.get(ngram_id) {
                let document_count = doc_set.len() as f32;
                if document_count > 1.0 {
                    1.0 / document_count.ln()
                } else {
                    1.0 // For single-document n-grams, max weight
                }
            } else {
                0.0 // N-gram not found (shouldn't happen)
            }
        })
        .sum();

    toxic_score
}

fn parse_ngram_string(ngram_str: &str) -> Result<Vec<usize>, Error> {
    // Parse "[123, 456, 789]" format back to Vec<usize>
    let trimmed = ngram_str.trim_start_matches('[').trim_end_matches(']');
    trimmed.split(", ")
        .map(|s| s.parse::<usize>().map_err(|e| anyhow::anyhow!("Failed to parse token ID: {}", e)))
        .collect()
}

#[allow(dead_code)]
fn save_contamination_results(
    config: &Config,
    contamination_results: &ContaminationResults
) -> Result<(), Error> {
    // println!("Saving contamination results..."); //debug

    // Create output directory if it doesn't exist
    create_dir_all(&config.output_dir)?;

    for entry in contamination_results.iter() {
        let file_name = entry.key();
        let results = entry.value();

        if results.is_empty() {
            continue;
        }

        // println!("Saving results for {}: {} contamination entries", file_name, results.len()); //debug

        // Create output filename similar to toxic format
        let results_filename = config.output_dir.join(format!("{}_simple_contamination.jsonl", file_name));

        // Convert to JSON format similar to toxic results
        let json_results: Vec<Value> = results.iter().map(|entry| {
            json!({
                "training_line": entry.training_line,
                "eval_name": entry.eval_name,
                "eval_line": entry.eval_line,
                "overlap_ratio": entry.overlap_ratio,
                "matching_ngrams": entry.matching_ngrams,
                "detection_method": "simple"
            })
        }).collect();

        // Write results to file
        let mut file = BufWriter::new(File::create(&results_filename)?);
        for result in json_results {
            writeln!(file, "{}", serde_json::to_string(&result)?)?;
        }
        file.flush()?;

        // println!("Saved contamination results to: {:?}", results_filename); //debug
    }

    // println!("All contamination results saved successfully"); //debug
    Ok(())
}

pub fn save_contamination_results_toxic_format(
    config: &Config,
    contamination_results: &ContaminationResults
) -> Result<PathBuf, Error> {
    save_contamination_results_toxic_format_with_filename(config, contamination_results, None)
}

pub fn save_contamination_results_toxic_format_with_filename(
    config: &Config,
    contamination_results: &ContaminationResults,
    custom_filename: Option<&str>
) -> Result<PathBuf, Error> {
    // Create output directory if it doesn't exist
    create_dir_all(&config.output_dir)?;

    // Use custom filename if provided, otherwise use default
    let default_filename = get_results_filename("simple");
    let filename = custom_filename.unwrap_or(&default_filename);
    let output_file = config.output_dir.join(filename);

    let mut output_data = Vec::new();
    let mut total_contaminations = 0;

    for entry in contamination_results.iter() {
        let training_file = entry.key();
        for contamination_entry in entry.value() {
            let mut result = json!({
                "training_file": training_file,
                "training_line": contamination_entry.training_line,
                "eval_dataset": contamination_entry.eval_name,
                "eval_line": contamination_entry.eval_line,
                "overlap_ratio": contamination_entry.overlap_ratio,
                "toxic_score": contamination_entry.toxic_score,
                "method": "simple"
            });
            
            // Add token indices if available
            if let Some(start_idx) = contamination_entry.contamination_start_idx {
                result["contamination_start_idx"] = json!(start_idx);
            }
            if let Some(end_idx) = contamination_entry.contamination_end_idx {
                result["contamination_end_idx"] = json!(end_idx);
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
        println!("\n=== SIMPLE CONTAMINATION SUMMARY ===");
        println!("Found {} contamination instances across {} files",
                total_contaminations, contamination_results.len());
        println!("Results saved to: {:?}", output_file);
    } else {
        println!("\n=== NO CONTAMINATION DETECTED ===");
        println!("No contamination found in training data");
        println!("Empty results file saved to: {:?}", output_file);
    }

    Ok(output_file)
}

// Create purified versions of training files with contaminated lines removed
fn create_purified_files(
    config: &Config,
    contamination_results: &ContaminationResults,
    training_files: &[PathBuf],
) -> Result<(), Error> {
    println!("\nCreating purified files...");
    
    // Determine output directory for cleaned files
    let cleaned_dir = config.cleaned_file_output.as_ref().unwrap_or(&config.output_dir);
    create_dir_all(cleaned_dir)?;
    
    // Process each training file that has contamination
    for file_path in training_files {
        let file_name = file_path
            .file_name()
            .and_then(|f| f.to_str())
            .unwrap_or("unknown")
            .to_string();
        
        // Check if this file has any contamination
        if let Some(contaminations) = contamination_results.get(&file_name) {
            if contaminations.is_empty() {
                continue;
            }
            
            // Collect contaminated line numbers
            let mut contaminated_lines = HashSet::new();
            for entry in contaminations.iter() {
                contaminated_lines.insert(entry.training_line);
            }
            
            // Create purified file
            let purified_filename = get_purified_filename(file_path);
            let purified_path = cleaned_dir.join(&purified_filename);
            
            let input_file = BufReader::new(File::open(file_path)?);
            let mut output_file = BufWriter::new(File::create(&purified_path)?);
            
            let mut removed_count = 0;
            for (line_num, line) in input_file.lines().enumerate() {
                if !contaminated_lines.contains(&line_num) {
                    writeln!(output_file, "{}", line?)?;
                } else {
                    removed_count += 1;
                }
            }
            
            output_file.flush()?;
            println!("Created purified file: {:?} (removed {} contaminated lines)", 
                     purified_path, removed_count);
        }
    }
    
    println!("Purification complete.");
    Ok(())
}
