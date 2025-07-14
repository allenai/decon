use anyhow::{Error, Result};
use dashmap::DashMap;
use flate2::read::GzDecoder;
use rayon::prelude::*;
use serde_json::{json, Value};
use std::collections::{HashMap, HashSet};
use std::fs::{create_dir_all, File};
use std::io::{BufRead, BufWriter, Read, Write};
use std::panic::catch_unwind;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;
use std::time::Instant;
use zstd::stream::read::Decoder as ZstdDecoder;

use mj_io::{build_pbar, expand_dirs, read_pathbuf_to_mem, write_mem_to_pathbuf};

use crate::{
    clean_text, get_nested_json_val, get_results_filename, preprocess_text, write_purified_file,
    Config, OmniTokenizer,
};

// Helper function to read compressed files (supporting .gz and .zst)
fn read_compressed_file(path: &PathBuf) -> Result<Vec<u8>, Error> {
    let mut file = File::open(path)?;
    let mut buffer = Vec::new();

    match path.extension().and_then(|s| s.to_str()) {
        Some("gz") => {
            let mut decoder = GzDecoder::new(file);
            decoder.read_to_end(&mut buffer)?;
        }
        Some("zst") => {
            let mut decoder = ZstdDecoder::new(file)?;
            decoder.read_to_end(&mut buffer)?;
        }
        _ => {
            // No compression, read file directly
            file.read_to_end(&mut buffer)?;
        }
    }

    Ok(buffer)
}

// Simple mode structures
type NgramToIdMap = DashMap<u64, u64>; // Maps n-gram hash to unique ID
type IdToQuestionDocsMap = DashMap<u64, HashSet<u32>>; // Maps n-gram ID to set of document IDs (for questions)
type IdToShortAnswerMap = DashMap<u32, HashSet<usize>>; // Maps doc_id to set of token IDs (for answers)
type EvalDocuments = DashMap<u32, (String, usize, usize, usize)>; // Maps doc_id to (eval_name, line_num, total_ngrams, unique_ngrams)
type EvalTextSnippets = DashMap<(String, usize), String>; // Maps (eval_name, line_num) to text snippet (first 1000 words)
type IdToNgramTokens = DashMap<u64, Vec<usize>>; // Maps unique ID to token IDs for display


// Note: CONTAMINATION_SCORE_THRESHOLD is now configurable via Config.simple_contamination_score_threshold

// This is the 'k' in the formula P(L) = 1.0 - e^(-k*L^n).
// It controls how quickly the penalty rises from 0 toward 1.0 as length increases.
// A larger 'k' means the penalty reaches higher values faster (less penalty for shorter texts).
const LENGTH_PENALTY_DECAY_RATE: f32 = 0.6; // Tuned for L=10→0.80, L=36→0.88, L=40→0.89

// This is the 'n' in the formula P(L) = 1.0 - e^(-k*L^n).
// It controls the shape of the curve. When n < 1, the curve has a gentler initial rise
// and longer tail. When n > 1, the curve rises sharply at first then flattens.
const LENGTH_PENALTY_POWER_N: f32 = 0.4;


pub fn contamination_detect(config: &Config) -> Result<(), Error> {
    println!("Starting SIMPLE contamination detection...");
    let start_main = Instant::now();

    // Step 1: Process eval datasets and build n-gram mappings
    println!("Processing eval datasets and building n-gram index...");
    let index_start = Instant::now();
    let (
        ngram_to_id,
        id_to_question_docs,
        id_to_short_answer,
        eval_documents,
        id_to_ngram_tokens,
        tokenizer,
        eval_text_snippets,
    ) = build_simple_index(config)?;
    let index_time = index_start.elapsed();
    println!(
        "Built simple index with {} unique n-grams in {:.2}s",
        ngram_to_id.len(),
        index_time.as_secs_f64()
    );

    // Step 2: Process training data and detect contamination
    println!("Processing training data for contamination detection...");
    let detection_start = Instant::now();
    let total_contaminations = detect_simple_contamination(
        config,
        &ngram_to_id,
        &id_to_question_docs,
        &id_to_short_answer,
        &eval_documents,
        &id_to_ngram_tokens,
        &tokenizer,
        &eval_text_snippets,
    )?;
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
pub type SimpleIndex = (
    NgramToIdMap,
    IdToQuestionDocsMap,
    IdToShortAnswerMap,
    EvalDocuments,
    IdToNgramTokens,
    OmniTokenizer,
    EvalTextSnippets,
);

pub fn build_simple_index(config: &Config) -> Result<SimpleIndex, Error> {
    let ngram_to_id: NgramToIdMap = DashMap::new();
    let id_to_question_docs: IdToQuestionDocsMap = DashMap::new();
    let id_to_short_answer: IdToShortAnswerMap = DashMap::new();
    let eval_documents: EvalDocuments = DashMap::new();
    let id_to_ngram_tokens: IdToNgramTokens = DashMap::new();
    let eval_text_snippets: EvalTextSnippets = DashMap::new();

    use std::sync::atomic::AtomicU64;
    let next_ngram_id = AtomicU64::new(0);

    // Initialize tokenizer
    let tokenizer = OmniTokenizer::new(&config.tokenizer_str)?;

    // Find all reference files
    let reference_files = expand_dirs(
        vec![config.reference_input.clone()],
        Some(vec![".jsonl", ".gz"].as_slice()),
    )?;
    let pbar = build_pbar(reference_files.len(), "Reference files");

    // println!("Processing {} reference files for n-gram indexing...", reference_files.len()); //debug

    reference_files.par_iter().for_each(|file_path| {
        if let Err(e) = process_simple_reference_file(
            file_path,
            config,
            &ngram_to_id,
            &id_to_question_docs,
            &id_to_short_answer,
            &eval_documents,
            &next_ngram_id,
            &tokenizer,
            &id_to_ngram_tokens,
            &eval_text_snippets,
        ) {
            println!("Error processing reference file {:?}: {:?}", file_path, e);
        }
        pbar.inc(1);
    });

    // println!("Finished processing reference files. Total unique n-grams: {}", ngram_to_id.len()); //debug
    // println!("Total eval documents indexed: {}", eval_documents.len()); //debug

    Ok((
        ngram_to_id,
        id_to_question_docs,
        id_to_short_answer,
        eval_documents,
        id_to_ngram_tokens,
        tokenizer,
        eval_text_snippets,
    ))
}

fn process_simple_reference_file(
    file_path: &PathBuf,
    config: &Config,
    ngram_to_id: &NgramToIdMap,
    id_to_question_docs: &IdToQuestionDocsMap,
    id_to_short_answer: &IdToShortAnswerMap,
    eval_documents: &EvalDocuments,
    next_ngram_id: &std::sync::atomic::AtomicU64,
    tokenizer: &OmniTokenizer,
    id_to_ngram_tokens: &IdToNgramTokens,
    eval_text_snippets: &EvalTextSnippets,
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
    let _min_word_count = config.eval_min_word_count;

    for (line_num, line) in data.lines().enumerate() {
        let line = line?;

        // Skip comment lines
        if line.starts_with('#') {
            continue;
        }

        let json_obj: Value = serde_json::from_str(&line)?;

        // Check if this is a Q&A dataset
        let has_answer_fields = json_obj.get("answer").is_some();

        // Read document ID from JSON (generated by Python download script)
        let doc_id = json_obj
            .get("doc_id")
            .and_then(|v| v.as_u64())
            .ok_or_else(|| anyhow::anyhow!("Missing or invalid doc_id field in reference file"))?
            as u32;


        // Process question field
        if let Some(question) = json_obj.get("question").and_then(|v| v.as_str()) {
            process_question_field(
                question,
                doc_id,
                line_num,
                &eval_name,
                config,
                tokenizer,
                ngram_to_id,
                id_to_question_docs,
                eval_documents,
                &next_ngram_id,
                id_to_ngram_tokens,
                eval_text_snippets,
                &mut _lines_processed,
                &mut _skipped_entries,
            )?;
        }

        if has_answer_fields {
            // Process answer field
            if let Some(answer) = json_obj.get("answer").and_then(|v| v.as_str()) {
                process_answer_field(
                    answer,
                    doc_id,
                    config,
                    tokenizer,
                    id_to_short_answer,
                )?;
            }
        }
    }

    // println!("Finished {}: processed {} lines, skipped {} short entries", eval_name, _lines_processed, _skipped_entries); //debug
    Ok(())
}

// Helper functions
fn hash_ngram(tokens: &[usize]) -> u64 {
    use std::hash::{DefaultHasher, Hash, Hasher};
    let mut hasher = DefaultHasher::new();
    tokens.hash(&mut hasher);
    hasher.finish()
}

fn get_or_create_ngram_id(
    ngram_tokens: &[usize],
    ngram_to_id: &NgramToIdMap,
    next_id: &std::sync::atomic::AtomicU64,
    id_to_ngram_tokens: &IdToNgramTokens,
) -> u64 {
    use std::sync::atomic::Ordering;

    let ngram_hash = hash_ngram(ngram_tokens);

    if let Some(existing_id) = ngram_to_id.get(&ngram_hash) {
        *existing_id
    } else {
        // Use entry API to handle race condition properly
        let entry = ngram_to_id.entry(ngram_hash);
        match entry {
            dashmap::mapref::entry::Entry::Occupied(occupied) => {
                // Another thread inserted it first
                *occupied.get()
            }
            dashmap::mapref::entry::Entry::Vacant(vacant) => {
                // We're first, insert new ID
                let new_id = next_id.fetch_add(1, Ordering::SeqCst);
                vacant.insert(new_id);
                // Store the tokens for display/lookup
                id_to_ngram_tokens.insert(new_id, ngram_tokens.to_vec());
                new_id
            }
        }
    }
}

fn add_doc_to_ngram(ngram_id: u64, doc_id: u32, id_to_docs: &IdToQuestionDocsMap) {
    id_to_docs
        .entry(ngram_id)
        .or_insert_with(HashSet::new)
        .insert(doc_id);
}

// Helper function to process question field
fn process_question_field(
    question: &str,
    doc_id: u32,
    line_num: usize,
    eval_name: &str,
    config: &Config,
    tokenizer: &OmniTokenizer,
    ngram_to_id: &NgramToIdMap,
    id_to_question_docs: &IdToQuestionDocsMap,
    eval_documents: &EvalDocuments,
    next_ngram_id: &std::sync::atomic::AtomicU64,
    id_to_ngram_tokens: &IdToNgramTokens,
    eval_text_snippets: &EvalTextSnippets,
    lines_processed: &mut usize,
    skipped_entries: &mut usize,
) -> Result<(), Error> {
    // Clean text for both tokenizer types
    let cleaned = clean_text(question, &config.punctuation_chars);

    // Store eval text snippet (first 1000 words)
    let snippet_words: Vec<&str> = cleaned.split_whitespace().take(1000).collect();
    let text_snippet = snippet_words.join(" ");
    eval_text_snippets.insert((eval_name.to_string(), line_num), text_snippet);

    // Use preprocess_text to get token IDs
    let word_tokens = if config.tokenizer_str == "word" {
        // For word mode, build vocabulary from reference set
        let words: Vec<String> = cleaned
            .split_whitespace()
            .map(|s| s.to_string())
            .filter(|s| !s.is_empty())
            .collect();
        // Add words to vocabulary and get IDs
        words
            .iter()
            .map(|word| tokenizer.add_word(word) as usize)
            .collect()
    } else {
        let Ok(tokens) =
            catch_unwind(|| preprocess_text(question, tokenizer, &config.punctuation_chars))
        else {
            println!(
                "Tokenization failed on question for doc_id {} | line {}",
                doc_id, line_num
            );
            *skipped_entries += 1;
            return Ok(());
        };
        tokens
    };
    let word_count = word_tokens.len();

    // Skip entries with insufficient tokens for meaningful n-gram analysis
    if word_count < config.eval_min_word_count {
        *skipped_entries += 1;
        return Ok(());
    }

    *lines_processed += 1;

    // Calculate total n-grams for this document
    let total_ngrams = if word_tokens.len() < config.ngram_size {
        if word_tokens.is_empty() {
            0
        } else {
            1
        }
    } else {
        word_tokens.len() - config.ngram_size + 1
    };

    // Track unique n-grams for this document
    let mut unique_ngrams_set = HashSet::new();

    // Process n-grams
    if word_tokens.len() < config.ngram_size {
        if !word_tokens.is_empty() {
            // For documents shorter than ngram_size, use all tokens
            let ngram_tokens = word_tokens.clone();
            let ngram_id = get_or_create_ngram_id(
                &ngram_tokens,
                ngram_to_id,
                next_ngram_id,
                id_to_ngram_tokens,
            );
            unique_ngrams_set.insert(ngram_id);
            add_doc_to_ngram(ngram_id, doc_id, id_to_question_docs);
        }
    } else {
        for i in 0..=word_tokens.len() - config.ngram_size {
            let ngram_slice = &word_tokens[i..i + config.ngram_size];
            let ngram_tokens = ngram_slice.to_vec();
            let ngram_id = get_or_create_ngram_id(
                &ngram_tokens,
                ngram_to_id,
                next_ngram_id,
                id_to_ngram_tokens,
            );
            unique_ngrams_set.insert(ngram_id);
            add_doc_to_ngram(ngram_id, doc_id, id_to_question_docs);
        }
    }

    let unique_ngrams = unique_ngrams_set.len();

    // Store document metadata (no collision possible with sequential IDs)
    eval_documents.insert(
        doc_id,
        (eval_name.to_string(), line_num, total_ngrams, unique_ngrams),
    );

    Ok(())
}

// Helper function to process answer field
fn process_answer_field(
    answer: &str,
    doc_id: u32,
    config: &Config,
    tokenizer: &OmniTokenizer,
    id_to_short_answer: &IdToShortAnswerMap,
) -> Result<(), Error> {
    // Clean text for both tokenizer types
    let cleaned = clean_text(answer, &config.punctuation_chars);

    // Get token IDs
    let word_tokens = if config.tokenizer_str == "word" {
        let words: Vec<String> = cleaned
            .split_whitespace()
            .map(|s| s.to_string())
            .filter(|s| !s.is_empty())
            .collect();
        words
            .iter()
            .map(|word| tokenizer.add_word(word) as usize)
            .collect()
    } else {
        let Ok(tokens) =
            catch_unwind(|| preprocess_text(answer, tokenizer, &config.punctuation_chars))
        else {
            // Skip this answer if tokenization fails
            return Ok(());
        };
        tokens
    };

    // Store in short answer map
    let token_set: HashSet<usize> = word_tokens.into_iter().collect();
    id_to_short_answer.insert(doc_id, token_set);

    Ok(())
}

fn detect_simple_contamination(
    config: &Config,
    ngram_to_id: &NgramToIdMap,
    id_to_question_docs: &IdToQuestionDocsMap,
    id_to_short_answer: &IdToShortAnswerMap,
    eval_documents: &EvalDocuments,
    id_to_ngram_tokens: &IdToNgramTokens,
    tokenizer: &OmniTokenizer,
    eval_text_snippets: &EvalTextSnippets,
) -> Result<usize, Error> {
    // println!("Starting training data contamination detection..."); //debug

    // Calculate total documents once
    let total_docs = eval_documents.len() as f32;

    // Find all training files
    let training_files = expand_dirs(
        vec![config.local_input.clone()],
        Some(vec![".jsonl", ".gz", ".zst"].as_slice()),
    )?;

    // println!("Found {} training files to process", training_files.len()); //debug
    let pbar = build_pbar(training_files.len(), "Training files");

    // Create output directory
    create_dir_all(&config.report_output_dir)?;

    // Track statistics
    let total_lines_processed = Arc::new(std::sync::atomic::AtomicUsize::new(0));

    // Keep track of contaminated files for purification (if needed)
    let contaminated_files: Arc<DashMap<String, HashSet<usize>>> = Arc::new(DashMap::new());

    // Track total contaminations across all files
    let total_contaminations = Arc::new(AtomicU32::new(0));

    let processing_start = Instant::now();
    training_files.par_iter().for_each(|file_path| {
        // Create a contamination results map for this specific file
        let file_contamination_results = DashMap::new();

        match process_simple_training_file(
            file_path,
            config,
            ngram_to_id,
            id_to_question_docs,
            id_to_short_answer,
            eval_documents,
            &file_contamination_results,
            tokenizer,
            id_to_ngram_tokens,
            eval_text_snippets,
            total_docs,
        ) {
            Ok(lines_processed) => {
                total_lines_processed
                    .fetch_add(lines_processed, std::sync::atomic::Ordering::SeqCst);

                // Save results for this file if contamination was found
                if !file_contamination_results.is_empty() {
                    let unique_filename = crate::get_unique_results_filename(file_path, config);
                    if let Err(e) = save_contamination_results_toxic_format_with_filename_and_eval_text(
                        config,
                        &file_contamination_results,
                        Some(&unique_filename),
                        eval_text_snippets,
                    ) {
                        println!("Error saving results for {:?}: {:?}", file_path, e);
                    } else {
                        // Track contaminated files and update total count
                        let file_path_str = file_path.to_string_lossy().to_string();
                        let contamination_count = file_contamination_results
                            .iter()
                            .map(|entry| entry.value().len())
                            .sum::<usize>();

                        total_contaminations.fetch_add(contamination_count as u32, Ordering::Relaxed);

                        let contaminated_lines: HashSet<usize> = file_contamination_results
                            .iter()
                            .flat_map(|entry| {
                                entry.value().iter().map(|e| e.training_line).collect::<Vec<_>>()
                            })
                            .collect();
                        contaminated_files.insert(file_path_str, contaminated_lines);
                    }
                }
            }
            Err(e) => {
                println!("Error processing training file {:?}: {:?}", file_path, e);
            }
        }
        pbar.inc(1);
    });
    let processing_time = processing_start.elapsed();

    let total_contaminations_count = total_contaminations.load(Ordering::Relaxed) as usize;

    let lines_processed = total_lines_processed.load(std::sync::atomic::Ordering::SeqCst);

    if lines_processed > 0 {
        let total_micros = processing_time.as_micros() as f64;
        let micros_per_line = total_micros / lines_processed as f64;

        if micros_per_line >= 1000.0 {
            println!(
                "Processed {} training lines in {:.2}s ({:.2} ms/line)",
                lines_processed,
                processing_time.as_secs_f64(),
                micros_per_line / 1000.0
            );
        } else {
            println!(
                "Processed {} training lines in {:.2}s ({:.0} μs/line)",
                lines_processed,
                processing_time.as_secs_f64(),
                micros_per_line
            );
        }
    }

    // Create purified files if requested
    if config.purify {
        create_purified_files_streaming(config, &contaminated_files, &training_files)?;
    }

    // println!("Contamination detection completed. Results saved."); //debug
    Ok(total_contaminations_count)
}

pub type ContaminationResults = DashMap<String, Vec<SimpleContaminationEntry>>;

/// Extract overlapping text with context from cleaned text
fn extract_overlap_with_context(
    cleaned_text: &str,
    tokens: &[usize],
    start_idx: usize,
    end_idx: usize,
    tokenizer: &OmniTokenizer,
    context_words: usize,
) -> Option<String> {
    // For word tokenizer, tokens are just words split by whitespace
    if tokenizer.tokenizer_name == "word" {
        // Only iterate through the words we need
        let context_start = start_idx.saturating_sub(context_words);
        let needed_end = end_idx + context_words;

        let mut word_iter = cleaned_text.split_whitespace();
        let mut current_idx = 0;
        let mut words_buffer = Vec::new();

        // Skip to context_start
        while current_idx < context_start {
            if word_iter.next().is_none() {
                return None; // Invalid indices
            }
            current_idx += 1;
        }

        // Collect only the words we need
        while current_idx < needed_end {
            if let Some(word) = word_iter.next() {
                words_buffer.push(word);
                current_idx += 1;
            } else {
                break;
            }
        }

        // Check if we have enough words
        let relative_start = start_idx - context_start;
        let relative_end = end_idx - context_start;
        if relative_end > words_buffer.len() || relative_start >= relative_end {
            return None;
        }

        // Build the output with highlighted contamination
        let mut result = String::new();

        // Add leading context
        if relative_start > 0 {
            result.push_str("... ");
            result.push_str(&words_buffer[0..relative_start].join(" "));
            result.push(' ');
        }

        // Add contaminated section with highlighting
        result.push_str("【");
        result.push_str(&words_buffer[relative_start..relative_end].join(" "));
        result.push_str("】");

        // Add trailing context
        if relative_end < words_buffer.len() {
            result.push(' ');
            result.push_str(&words_buffer[relative_end..].join(" "));
            result.push_str(" ...");
        }

        return Some(result);
    }

    // For BPE tokenizers (c100k, p50k), decode from token array
    if let Some(inner) = tokenizer.inner.as_ref() {
        // Check bounds
        if start_idx >= tokens.len() || end_idx > tokens.len() || start_idx >= end_idx {
            return None;
        }

        let context_start = start_idx.saturating_sub(context_words);
        let context_end = (end_idx + context_words).min(tokens.len());

        let mut result = String::new();

        // Decode and add leading context
        if context_start < start_idx {
            if let Ok(prefix) = inner.decode(tokens[context_start..start_idx].to_vec()) {
                result.push_str("... ");
                result.push_str(&prefix);
                result.push_str(" ");
            }
        }

        // Decode and add contaminated section
        if let Ok(contaminated) = inner.decode(tokens[start_idx..end_idx].to_vec()) {
            result.push_str("【");
            result.push_str(&contaminated);
            result.push_str("】");
        }

        // Decode and add trailing context
        if end_idx < context_end {
            if let Ok(suffix) = inner.decode(tokens[end_idx..context_end].to_vec()) {
                result.push_str(" ");
                result.push_str(&suffix);
                result.push_str(" ...");
            }
        }

        if !result.is_empty() {
            return Some(result);
        }
    }

    // Fallback for other tokenizers
    None
}

#[derive(Clone)]
pub struct SimpleContaminationEntry {
    pub training_line: usize,
    eval_name: String,
    eval_line: usize,
    overlap_ratio: f32,
    idf_sum: f32,              // Sum of IDF scores for all matching n-grams
    max_idf: f32,              // Maximum IDF score among matching n-grams
    matching_ngrams: Vec<String>,
    // Token indices for position recovery
    contamination_start_idx: Option<usize>, // Start index in token array
    contamination_end_idx: Option<usize>,   // End index in token array
    training_overlap_text: Option<String>,
    ngram_match_cnt: usize,    // Number of unique n-gram matches
    eval_unique_ngrams: usize, // Total unique n-grams in the eval document
    length_penalty: Option<f32>, // Length penalty factor applied during scoring
    // Answer contamination fields
    answer_overlap_ratio: Option<f32>, // Overlap ratio for answer tokens
    matched_answer_tokens: Option<Vec<String>>, // Matched answer tokens as text
}

impl SimpleContaminationEntry {
    /// Calculate the length penalty factor for this entry
    pub fn calculate_length_penalty(&self, min_length_penalty: f32) -> f32 {
        let l = self.eval_unique_ngrams as f32;
        let mut length_penalty = 1.0 - (-LENGTH_PENALTY_DECAY_RATE * l.powf(LENGTH_PENALTY_POWER_N)).exp();
        length_penalty = length_penalty.max(min_length_penalty);
        length_penalty
    }

    /// Calculate contamination score based on overlap ratio and IDF, with length penalty
    /// Returns a score between 0.0 and 1.0, where higher scores indicate more likely contamination
    pub fn score_question_contamination(&self, min_length_penalty: f32) -> f32 {
        // Basically we reject anything with less than 80% overlap. Just on principle.
        // TODO review
        if self.overlap_ratio < 0.8 {
            return 0.0;
        }

        // Perfect matches always get maximum score
        if self.overlap_ratio >= 1.0 {
            return 1.0;
        }

        // Average IDF per n-gram
        let avg_idf_per_ngram = self.idf_sum / self.ngram_match_cnt.max(1) as f32;

        // Sigmoid normalization for IDF: maps (-∞,+∞) to (0,1)
        // Center around 0.0 (the natural threshold where doc_freq = N/e ≈ 36.8%)
        // Positive IDF = rare terms, Negative IDF = common terms
        // The 0.5 factor controls the steepness of the sigmoid
        let normalized_idf = 1.0 / (1.0 + (-0.5 * avg_idf_per_ngram).exp());

        // Create a combined score using linear combination
        // Weight overlap more heavily than IDF
        let overlap_weight = 0.4;
        let idf_weight = 0.6;

        // Base combined score
        let base_score = (overlap_weight * self.overlap_ratio) + (idf_weight * normalized_idf);

        // Apply length penalty - shorter texts need to have higher scores to survive.
        // They need to be more exact and have more salient tokens.
        let length_penalty = self.calculate_length_penalty(min_length_penalty);

        // Final score with length penalty applied
        base_score * length_penalty
    }

    /// Check if this entry represents contamination (score >= threshold)
    /// Returns (is_contaminated, answer_overlap_ratio, matched_answer_tokens)
    pub fn is_contaminated(
        &self,
        doc_id: u32,
        id_to_short_answer: &IdToShortAnswerMap,
        cluster: &SimpleContaminationCluster,
        training_tokens: &[usize],
        config: &Config,
        tokenizer: &OmniTokenizer,
    ) -> (bool, Option<f32>, Option<Vec<String>>) {
        let question_contam = self.score_question_contamination(config.simple_contamination_score_threshold) >= config.simple_contamination_score_threshold;

        if question_contam {
            // Check if this document has a short answer
            if let Some(answer_token_set) = id_to_short_answer.get(&doc_id) {
                // Get matching tokens first
                let matching_tokens = short_answer_tokens(
                    answer_token_set.value(),
                    cluster,
                    training_tokens,
                    config,
                );

                // Calculate overlap ratio
                let answer_overlap_ratio = if answer_token_set.is_empty() {
                    0.0
                } else {
                    matching_tokens.len() as f32 / answer_token_set.len() as f32
                };

                // Convert matching token IDs to text
                let matched_token_strings: Vec<String> = matching_tokens
                    .iter()
                    .filter_map(|&token_id| tokenizer.get_word(token_id as u32))
                    .collect();

                // Require both question and answer contamination
                let is_contaminated = question_contam && answer_overlap_ratio >= config.short_answer_contamination_threshold;
                return (is_contaminated, Some(answer_overlap_ratio), Some(matched_token_strings));
            }
        }

        (question_contam, None, None)
    }
}

fn short_answer_tokens(
    answer_token_set: &HashSet<usize>,
    cluster: &SimpleContaminationCluster,
    training_tokens: &[usize],
    config: &Config,
) -> HashSet<usize> {
    // Calculate window size as max(answer_length*2, min_short_answer_distance)
    let answer_length = answer_token_set.len();
    let window_size = std::cmp::max(answer_length * 2, config.min_short_answer_distance);

    if config.exclude_question_from_answer_sweep {
        // When excluding question tokens, search in prefix and suffix regions
        let prefix_search_start = cluster.start_idx.saturating_sub(window_size);
        let prefix_search_end = cluster.start_idx;
        let suffix_search_start = cluster.end_idx;
        let suffix_search_end = (cluster.end_idx + window_size).min(training_tokens.len());

        // Collect tokens from both regions
        let mut training_token_set = HashSet::new();

        // Add prefix tokens
        if prefix_search_start < prefix_search_end {
            training_token_set.extend(
                training_tokens[prefix_search_start..prefix_search_end].iter().copied()
            );
        }

        // Add suffix tokens
        if suffix_search_start < suffix_search_end {
            training_token_set.extend(
                training_tokens[suffix_search_start..suffix_search_end].iter().copied()
            );
        }

        // Find matching tokens
        answer_token_set
            .iter()
            .filter(|token| training_token_set.contains(token))
            .copied()
            .collect()
    } else {
        // Original behavior: search entire window including the question
        let search_start = cluster.start_idx.saturating_sub(window_size);
        let search_end = (cluster.end_idx + window_size).min(training_tokens.len());

        // Extract training tokens in the search window
        let training_window = &training_tokens[search_start..search_end];
        let training_token_set: HashSet<usize> = training_window.iter().copied().collect();

        // Find matching tokens
        answer_token_set
            .iter()
            .filter(|token| training_token_set.contains(token))
            .copied()
            .collect()
    }
}

#[derive(Clone)]
pub(crate) struct SimpleContaminationCluster {
    start_idx: usize,
    end_idx: usize,
    document_matches: HashMap<u32, HashSet<u64>>, // doc_id -> unique_ngram_ids
    matching_ngrams: Vec<String>,
}

pub fn process_simple_training_file(
    file_path: &PathBuf,
    config: &Config,
    ngram_to_id: &NgramToIdMap,
    id_to_question_docs: &IdToQuestionDocsMap,
    id_to_short_answer: &IdToShortAnswerMap,
    eval_documents: &EvalDocuments,
    contamination_results: &ContaminationResults,
    tokenizer: &OmniTokenizer,
    id_to_ngram_tokens: &IdToNgramTokens,
    _eval_text_snippets: &EvalTextSnippets,
    total_docs: f32,
) -> Result<usize, Error> {
    let data = read_compressed_file(file_path)?;

    let file_name = match file_path.extension().and_then(|s| s.to_str()) {
        Some("gz") | Some("zst") => file_path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("unknown")
            .to_string(),
        _ => file_path
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("unknown")
            .to_string(),
    };

    // println!("Processing training file: {}", file_name); //debug

    let mut lines_processed = 0;
    let min_word_count = config.ngram_size * 2; // Keep original for training data

    for (line_num, line) in data.lines().enumerate() {
        let line = line?;
        let json_obj: Value = serde_json::from_str(&line)?;
        let line_text = get_nested_json_val(&json_obj, &config.content_key.to_string())?;

        // Clean text once and get token IDs
        let cleaned_text = clean_text(&line_text, &config.punctuation_chars);
        let Ok(word_tokens) = catch_unwind(|| tokenizer.encode(&cleaned_text)) else {
            println!(
                "Tokenization failed on {:?} | line {:?}",
                file_path, line_num
            );
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
            id_to_question_docs,
            id_to_ngram_tokens,
        )?;

        // Convert clusters to contamination entries
        for cluster in clusters {
            let mut cluster_results = Vec::new();

            for (doc_id, matched_ngram_ids) in &cluster.document_matches {
                if let Some(doc_info) = eval_documents.get(&doc_id) {
                    let (eval_name, eval_line, _total_ngrams, unique_ngrams) = doc_info.value();
                    let unique_matches = matched_ngram_ids.len();
                    let overlap_ratio = if *unique_ngrams > 0 {
                        unique_matches as f32 / *unique_ngrams as f32
                    } else {
                        0.0
                    };

                    // Calculate IDF scores
                    let (idf_sum, max_idf) = calculate_simple_toxic_score(
                        &cluster.matching_ngrams,
                        ngram_to_id,
                        id_to_question_docs,
                        total_docs,
                        id_to_ngram_tokens,
                    );

                    // Create the contamination entry first
                    let mut entry = SimpleContaminationEntry {
                        training_line: line_num,
                        eval_name: eval_name.clone(),
                        eval_line: *eval_line,
                        overlap_ratio,
                        idf_sum,
                        max_idf,
                        matching_ngrams: cluster.matching_ngrams.clone(),
                        contamination_start_idx: Some(cluster.start_idx),
                        contamination_end_idx: Some(cluster.end_idx),
                        training_overlap_text: None, // Will be filled if contaminated
                        ngram_match_cnt: unique_matches,
                        eval_unique_ngrams: *unique_ngrams,
                        length_penalty: None, // Will be calculated and set below
                        answer_overlap_ratio: None,
                        matched_answer_tokens: None,
                    };

                    // Calculate and store the length penalty
                    entry.length_penalty = Some(entry.calculate_length_penalty(config.simple_contamination_score_threshold));

                    // Check if this entry represents contamination using score threshold
                    let (is_contaminated, answer_overlap_ratio, matched_answer_tokens) =
                        entry.is_contaminated(*doc_id, id_to_short_answer, &cluster, &word_tokens, config, tokenizer);

                    if is_contaminated {
                        // Extract the overlapping text with context
                        let training_overlap_text = extract_overlap_with_context(
                            &cleaned_text,
                            &word_tokens,
                            cluster.start_idx,
                            cluster.end_idx,
                            tokenizer,
                            60,
                        );

                        let mut entry_with_text = entry;
                        entry_with_text.training_overlap_text = training_overlap_text;
                        entry_with_text.answer_overlap_ratio = answer_overlap_ratio;
                        entry_with_text.matched_answer_tokens = matched_answer_tokens;
                        cluster_results.push(entry_with_text);
                    }
                }
            }

            if !cluster_results.is_empty() {
                contamination_results
                    .entry(file_name.clone())
                    .or_insert_with(Vec::new)
                    .extend(cluster_results);
            }
        }
    }

    Ok(lines_processed)
}


/// Process n-grams with sampling: sample every M n-grams, then expand around hits
fn process_ngrams_with_simple_sampling(
    word_tokens: &[usize],
    config: &Config,
    ngram_to_id: &NgramToIdMap,
    id_to_question_docs: &IdToQuestionDocsMap,
    id_to_ngram_tokens: &IdToNgramTokens,
) -> Result<Vec<SimpleContaminationCluster>, Error> {
    let mut clusters = Vec::new();
    let mut processed_indices = HashSet::new();
    let mut i = 0;

    // println!("Starting sampling with M={}, word_tokens len={}", config.sample_every_m_tokens, word_tokens.len()); //debug

    // Calculate total n-grams
    let total_ngrams = if word_tokens.len() < config.ngram_size {
        if word_tokens.is_empty() {
            0
        } else {
            1
        }
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
        if let Some(document_ids) =
            check_ngram_for_match(i, word_tokens, config, ngram_to_id, id_to_question_docs)
        {
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
                id_to_question_docs,
                document_ids,
                &ngram_tokens,
                id_to_ngram_tokens,
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
    id_to_question_docs: &IdToQuestionDocsMap,
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
        if let Some(doc_set) = id_to_question_docs.get(&ngram_id) {
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
    id_to_question_docs: &IdToQuestionDocsMap,
    initial_document_ids: HashSet<u32>,
    initial_training_ngram: &[usize],
    id_to_ngram_tokens: &IdToNgramTokens,
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
    let initial_ngram_id = ngram_to_id
        .get(&initial_ngram_hash)
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
        if word_tokens.is_empty() {
            0
        } else {
            1
        }
    } else {
        word_tokens.len() - config.ngram_size + 1
    };

    // Expand backward (left traversal)
    // println!("Starting left traversal from index {}", hit_idx); //debug
    let mut left_idx = hit_idx;
    while left_idx > 0 && !active_documents.is_empty() {
        left_idx -= 1;

        if let Some(matched_docs) =
            check_ngram_for_match(left_idx, word_tokens, config, ngram_to_id, id_to_question_docs)
        {
            let ngram_tokens = if word_tokens.len() < config.ngram_size {
                word_tokens.to_vec()
            } else {
                word_tokens[left_idx..left_idx + config.ngram_size].to_vec()
            };

            // println!("Left traversal: checking n-gram {} '{:?}'", left_idx, ngram_tokens); //debug

            // Check intersection with active documents
            let intersection: Vec<u32> = active_documents
                .intersection(&matched_docs)
                .cloned()
                .collect();

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
                    document_matches
                        .entry(*doc_id)
                        .or_insert_with(HashSet::new)
                        .insert(ngram_id);
                    document_misses.insert(*doc_id, 0);
                }

                // Remove documents that didn't match this n-gram
                let to_remove: Vec<u32> = active_documents
                    .difference(&matched_docs)
                    .cloned()
                    .collect();
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

        if let Some(matched_docs) =
            check_ngram_for_match(right_idx, word_tokens, config, ngram_to_id, id_to_question_docs)
        {
            let ngram_tokens = if word_tokens.len() < config.ngram_size {
                word_tokens.to_vec()
            } else {
                word_tokens[right_idx..right_idx + config.ngram_size].to_vec()
            };

            // println!("Right traversal: checking n-gram {} '{:?}'", right_idx, ngram_tokens); //debug

            // Check intersection with active documents
            let intersection: Vec<u32> = active_documents
                .intersection(&matched_docs)
                .cloned()
                .collect();

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
                    document_matches
                        .entry(*doc_id)
                        .or_insert_with(HashSet::new)
                        .insert(ngram_id);
                    document_misses.insert(*doc_id, 0);
                }

                // Remove documents that didn't match this n-gram
                let to_remove: Vec<u32> = active_documents
                    .difference(&matched_docs)
                    .cloned()
                    .collect();
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

/// Calculate IDF scores using inverse document frequency approach
/// Returns (idf_sum, max_idf) for all matching n-grams
fn calculate_simple_toxic_score(
    matching_ngrams: &[String],
    ngram_to_id: &NgramToIdMap,
    id_to_question_docs: &IdToQuestionDocsMap,
    total_docs: f32,
    _id_to_ngram_tokens: &IdToNgramTokens,
) -> (f32, f32) {  // Returns (idf_sum, max_idf)
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

    // Calculate IDF values for all unique n-grams
    let mut idf_sum = 0.0f32;
    let mut max_idf = 0.0f32;

    for ngram_id in &unique_ngram_ids {
        if let Some(doc_set) = id_to_question_docs.get(ngram_id) {
            let doc_freq = doc_set.len() as f32;
            // Standard IDF formula: ln(N/doc_freq)
            let idf = (total_docs / doc_freq).ln();
            idf_sum += idf;
            max_idf = max_idf.max(idf);
        }
    }

    (idf_sum, max_idf)
}

fn parse_ngram_string(ngram_str: &str) -> Result<Vec<usize>, Error> {
    // Parse "[123, 456, 789]" format back to Vec<usize>
    let trimmed = ngram_str.trim_start_matches('[').trim_end_matches(']');
    trimmed
        .split(", ")
        .map(|s| {
            s.parse::<usize>()
                .map_err(|e| anyhow::anyhow!("Failed to parse token ID: {}", e))
        })
        .collect()
}

#[allow(dead_code)]
fn save_contamination_results(
    config: &Config,
    contamination_results: &ContaminationResults,
) -> Result<(), Error> {
    // println!("Saving contamination results..."); //debug

    // Create output directory if it doesn't exist
    create_dir_all(&config.report_output_dir)?;

    for entry in contamination_results.iter() {
        let file_name = entry.key();
        let results = entry.value();

        if results.is_empty() {
            continue;
        }

        // println!("Saving results for {}: {} contamination entries", file_name, results.len()); //debug

        // Create output filename similar to toxic format
        let results_filename = config
            .report_output_dir
            .join(format!("{}_simple_contamination.jsonl", file_name));

        // Convert to JSON format similar to toxic results
        let json_results: Vec<Value> = results
            .iter()
            .map(|entry| {
                json!({
                    "training_line": entry.training_line,
                    "eval_name": entry.eval_name,
                    "eval_line": entry.eval_line,
                    "overlap_ratio": entry.overlap_ratio,
                    "matching_ngrams": entry.matching_ngrams,
                    "detection_method": "simple"
                })
            })
            .collect();

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

pub fn save_contamination_results_toxic_format_with_filename_and_eval_text(
    config: &Config,
    contamination_results: &ContaminationResults,
    custom_filename: Option<&str>,
    eval_text_snippets: &EvalTextSnippets,
) -> Result<PathBuf, Error> {
    // Create output directory if it doesn't exist
    create_dir_all(&config.report_output_dir)?;

    // Use custom filename if provided, otherwise use default
    let default_filename = get_results_filename("simple");
    let filename = custom_filename.unwrap_or(&default_filename);
    let output_file = config.report_output_dir.join(filename);

    let mut output_data = Vec::new();

    for entry in contamination_results.iter() {
        let training_file = entry.key();
        for contamination_entry in entry.value() {
            let mut result = json!({
                "training_file": training_file,
                "training_line": contamination_entry.training_line,
                "eval_dataset": contamination_entry.eval_name,
                "eval_line": contamination_entry.eval_line,
                "overlap_ratio": contamination_entry.overlap_ratio,
                "toxic_score": contamination_entry.idf_sum,  // Keep for backward compatibility
                "idf_sum": contamination_entry.idf_sum,
                "max_idf": contamination_entry.max_idf,
                "ngram_match_cnt": contamination_entry.ngram_match_cnt,
                "eval_unique_ngrams": contamination_entry.eval_unique_ngrams,
                "contamination_score": contamination_entry.score_question_contamination(config.simple_contamination_score_threshold),
                "length_penalty": contamination_entry.length_penalty.unwrap_or_else(|| contamination_entry.calculate_length_penalty(config.simple_contamination_score_threshold)),
                "method": "simple"
            });

            // Add token indices if available
            if let Some(start_idx) = contamination_entry.contamination_start_idx {
                result["contamination_start_idx"] = json!(start_idx);
            }
            if let Some(end_idx) = contamination_entry.contamination_end_idx {
                result["contamination_end_idx"] = json!(end_idx);
            }

            // Add overlapping text if available
            if let Some(ref overlap_text) = contamination_entry.training_overlap_text {
                result["training_overlap_text"] = json!(overlap_text);
            }

            // Look up eval text snippet
            let eval_key = (
                contamination_entry.eval_name.clone(),
                contamination_entry.eval_line,
            );
            let eval_text = eval_text_snippets
                .get(&eval_key)
                .map(|entry| entry.value().clone())
                .unwrap_or_else(|| String::new());
            result["eval_overlap_text"] = json!(eval_text);

            // Add answer contamination fields
            if let Some(ratio) = contamination_entry.answer_overlap_ratio {
                result["answer_overlap_ratio"] = json!(ratio);
            }
            if let Some(ref tokens) = contamination_entry.matched_answer_tokens {
                result["matched_answer_tokens"] = json!(tokens);
            }

            output_data.push(serde_json::to_vec(&result)?);
        }
    }

    let mut output_bytes = Vec::new();
    for line in output_data {
        output_bytes.extend(line);
        output_bytes.push(b'\n');
    }

    write_mem_to_pathbuf(&output_bytes, &output_file)?;

    Ok(output_file)
}


fn create_purified_files_streaming(
    config: &Config,
    contaminated_files: &Arc<DashMap<String, HashSet<usize>>>,
    training_files: &[PathBuf],
) -> Result<(), Error> {
    println!("\nCreating purified files...");

    // Determine output directory for cleaned files
    let cleaned_dir = config
        .cleaned_output_dir
        .as_ref()
        .unwrap_or(&config.report_output_dir);

    // Process each training file that has contamination
    for file_path in training_files {
        // Match the same logic used in process_training_file
        let file_name = match file_path.extension().and_then(|s| s.to_str()) {
            Some("gz") | Some("zst") => file_path
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("unknown")
                .to_string(),
            _ => file_path
                .file_name()
                .and_then(|s| s.to_str())
                .unwrap_or("unknown")
                .to_string(),
        };

        // Get contaminated lines for this file (if any)
        let contaminated_lines = if let Some(entry) = contaminated_files.get(&file_name) {
            entry.value().clone()
        } else {
            HashSet::new()
        };

        // Always create a purified file when purify mode is enabled
        write_purified_file(file_path, cleaned_dir, &contaminated_lines)?;

        if contaminated_lines.is_empty() {
            println!("Copied clean file: {}", file_name);
        } else {
            println!(
                "Created purified file for {} (removed {} lines)",
                file_name,
                contaminated_lines.len()
            );
        }
    }

    println!("Purification complete.");
    Ok(())
}
