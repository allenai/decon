use anyhow::{Error, Result};
use dashmap::DashMap;
use flate2::read::GzDecoder;
use rayon::prelude::*;
use serde_json::{json, Value};
use std::collections::{HashMap, HashSet};
use std::fs::{create_dir_all, File};
use std::io::{BufRead, Read};
use std::panic::catch_unwind;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU32, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Instant;
use zstd::stream::read::Decoder as ZstdDecoder;

use mj_io::{build_pbar, expand_dirs, read_pathbuf_to_mem, write_mem_to_pathbuf};

use crate::{
    clean_text, get_nested_json_val, get_results_filename, write_purified_file, Config,
    OmniTokenizer,
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
type EvalDocIdToAnswerTokenLengthMap = DashMap<u32, usize>; // Maps doc_id to actual answer token length (not unique count)
type EvalDocuments = DashMap<u32, (String, usize, usize, usize, usize)>; // Maps doc_id to (eval_name, line_num, total_ngrams, unique_ngrams, token_count)
type EvalTextSnippets = DashMap<(String, usize), String>; // Maps (eval_name, line_num) to text snippet (first 1000 words)
type IdToNgramTokens = DashMap<u64, Vec<usize>>; // Maps unique ID to token IDs for display
type EvalDocumentIdfCache = DashMap<u32, f32>; // Maps doc_id to total IDF sum for all ngrams in the eval document
type DocToNgramIdsMap = DashMap<u32, HashSet<u64>>; // Maps doc_id to set of ngram IDs for efficient IDF calculation
type TokenDocFreqMap = DashMap<usize, AtomicUsize>; // Maps token ID to number of documents containing that token

// Note: CONTAMINATION_SCORE_THRESHOLD is now configurable via Config.question_threshold

// This is the 'k' in the formula P(L) = 1.0 - e^(-k*L^n).
// It controls how quickly the penalty rises from 0 toward 1.0 as length increases.
// A larger 'k' means the penalty reaches higher values faster (less penalty for shorter texts).
const LENGTH_PENALTY_DECAY_RATE: f32 = 0.6; // Tuned for L=10→0.80, L=36→0.88, L=40→0.89

// This is the 'n' in the formula P(L) = 1.0 - e^(-k*L^n).
// It controls the shape of the curve. When n < 1, the curve has a gentler initial rise
// and longer tail. When n > 1, the curve rises sharply at first then flattens.
const LENGTH_PENALTY_POWER_N: f32 = 0.4;

// Global counters for traversal statistics
static LEFT_TRAVERSAL_COUNT: AtomicUsize = AtomicUsize::new(0); //DEBUGCOUNTER
static RIGHT_TRAVERSAL_COUNT: AtomicUsize = AtomicUsize::new(0); //DEBUGCOUNTER
static EXCLUDED_NO_ANSWER_MATCH: AtomicUsize = AtomicUsize::new(0); //DEBUGCOUNTER

// Helper function to format numbers with commas
fn format_number_with_commas(n: usize) -> String {
    let s = n.to_string();
    let mut result = String::new();
    let mut count = 0;

    for ch in s.chars().rev() {
        if count > 0 && count % 3 == 0 {
            result.push(',');
        }
        result.push(ch);
        count += 1;
    }

    result.chars().rev().collect()
}

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
        eval_doc_id_to_answer_token_length,
        eval_documents,
        id_to_ngram_tokens,
        tokenizer,
        eval_text_snippets,
        eval_document_idf_cache,
        doc_to_ngram_ids,
        token_doc_freq,
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
        &eval_doc_id_to_answer_token_length,
        &eval_documents,
        &id_to_ngram_tokens,
        &tokenizer,
        &eval_text_snippets,
        &eval_document_idf_cache,
        &doc_to_ngram_ids,
        &token_doc_freq,
    )?;
    let detection_time = detection_start.elapsed();

    let total_time = start_main.elapsed();
    println!("\n=== SIMPLE Contamination Detection Results ===");
    println!("Index building time: {:.2}s", index_time.as_secs_f64());
    println!("Detection time: {:.2}s", detection_time.as_secs_f64());
    println!("Total time: {:.2}s", total_time.as_secs_f64());
    println!("Total contaminations found: {}", total_contaminations);

    // Print traversal statistics
    let left_traversals = LEFT_TRAVERSAL_COUNT.load(Ordering::Relaxed);
    let right_traversals = RIGHT_TRAVERSAL_COUNT.load(Ordering::Relaxed);
    let excluded_no_answer = EXCLUDED_NO_ANSWER_MATCH.load(Ordering::Relaxed);
    println!("\n=== Traversal Statistics ===");
    println!(
        "Left traversals: {:>15}",
        format_number_with_commas(left_traversals)
    );
    println!(
        "Right traversals: {:>15}",
        format_number_with_commas(right_traversals)
    );
    println!(
        "Total traversals: {:>15}",
        format_number_with_commas(left_traversals + right_traversals)
    );
    println!("\n=== Contamination Exclusion Statistics ===");
    println!(
        "Excluded (no answer match): {:>15}",
        format_number_with_commas(excluded_no_answer)
    );

    Ok(())
}

// Public type for the index
pub type SimpleIndex = (
    NgramToIdMap,
    IdToQuestionDocsMap,
    IdToShortAnswerMap,
    EvalDocIdToAnswerTokenLengthMap,
    EvalDocuments,
    IdToNgramTokens,
    OmniTokenizer,
    EvalTextSnippets,
    EvalDocumentIdfCache,
    DocToNgramIdsMap,
    TokenDocFreqMap,
);

pub fn build_simple_index(config: &Config) -> Result<SimpleIndex, Error> {
    let ngram_to_id: NgramToIdMap = DashMap::new();
    let id_to_question_docs: IdToQuestionDocsMap = DashMap::new();
    let id_to_short_answer: IdToShortAnswerMap = DashMap::new();
    let eval_doc_id_to_answer_token_length: EvalDocIdToAnswerTokenLengthMap = DashMap::new();
    let eval_documents: EvalDocuments = DashMap::new();
    let id_to_ngram_tokens: IdToNgramTokens = DashMap::new();
    let eval_text_snippets: EvalTextSnippets = DashMap::new();
    let eval_document_idf_cache: EvalDocumentIdfCache = DashMap::new();
    let doc_to_ngram_ids: DocToNgramIdsMap = DashMap::new();
    let token_doc_freq: TokenDocFreqMap = DashMap::new();

    use std::sync::atomic::AtomicU64;
    let next_ngram_id = AtomicU64::new(0);

    // Initialize tokenizer
    let tokenizer = OmniTokenizer::new(&config.tokenizer_str)?;

    // Create dedup map if deduplication is enabled
    // Now stores (eval_name, line_num) of first occurrence for debug tracking
    // Key is (question_hash, answer_hash) to handle cases where same question has different answers
    let dedup_map: Option<DashMap<(u64, u64), (String, usize)>> = if config.eval_dedup {
        Some(DashMap::new())
    } else {
        None
    };

    // Find all reference files
    let reference_files = expand_dirs(
        vec![config.reference_input.clone()],
        Some(vec![".jsonl", ".gz"].as_slice()),
    )?;

    // Check if any reference files were found and verify they exist
    let existing_files: Vec<PathBuf> = reference_files
        .into_iter()
        .filter(|path| path.exists())
        .collect();

    if existing_files.is_empty() {
        return Err(anyhow::anyhow!(
            "\nReference files not found at {}. Please run 'make evals-s3' to download evaluation datasets.",
            config.reference_input.display()
        ));
    }

    let reference_files = existing_files;

    let pbar = build_pbar(reference_files.len(), "Reference files");

    // Global statistics for deduplication and filtering
    let total_skipped_duplicates = Arc::new(AtomicUsize::new(0));
    let total_skipped_min_length = Arc::new(AtomicUsize::new(0));
    let total_skipped_min_unique_words = Arc::new(AtomicUsize::new(0));
    let total_lines_processed = Arc::new(AtomicUsize::new(0));

    // println!("Processing {} reference files for n-gram indexing...", reference_files.len()); //debug

    reference_files.par_iter().for_each(|file_path| {
        if let Err(e) = process_simple_reference_file(
            file_path,
            config,
            &ngram_to_id,
            &id_to_question_docs,
            &id_to_short_answer,
            &eval_doc_id_to_answer_token_length,
            &eval_documents,
            &next_ngram_id,
            &tokenizer,
            &id_to_ngram_tokens,
            &eval_text_snippets,
            &doc_to_ngram_ids,
            &token_doc_freq,
            &dedup_map,
            &total_skipped_duplicates,
            &total_skipped_min_length,
            &total_skipped_min_unique_words,
            &total_lines_processed,
        ) {
            println!("Error processing reference file {:?}: {:?}", file_path, e);
        }
        pbar.inc(1);
    });

    pbar.finish_with_message("Index building complete");
    
    // Display summary statistics if any preprocessing was done
    let total_lines = total_lines_processed.load(Ordering::Relaxed);
    let skipped_duplicates = total_skipped_duplicates.load(Ordering::Relaxed);
    let skipped_min_length = total_skipped_min_length.load(Ordering::Relaxed);
    let skipped_min_unique_words = total_skipped_min_unique_words.load(Ordering::Relaxed);
    let total_skipped = skipped_duplicates + skipped_min_length + skipped_min_unique_words;
    
    if config.eval_dedup || config.eval_min_cleaned_char_length > 0 || config.eval_min_unique_word_count > 0 {
        println!("\nReference preprocessing summary:");
        println!("  Total lines examined: {}", total_lines + total_skipped);
        println!("  Lines indexed: {}", total_lines);
        if total_skipped > 0 {
            println!("  Lines skipped: {} total", total_skipped);
            if skipped_duplicates > 0 {
                println!("    - Duplicates: {}", skipped_duplicates);
            }
            if skipped_min_length > 0 {
                println!("    - Below minimum length: {}", skipped_min_length);
            }
            if skipped_min_unique_words > 0 {
                println!("    - Below minimum unique words: {}", skipped_min_unique_words);
            }
        }
    }

    Ok((
        ngram_to_id,
        id_to_question_docs,
        id_to_short_answer,
        eval_doc_id_to_answer_token_length,
        eval_documents,
        id_to_ngram_tokens,
        tokenizer,
        eval_text_snippets,
        eval_document_idf_cache,
        doc_to_ngram_ids,
        token_doc_freq,
    ))
}

fn process_simple_reference_file(
    file_path: &PathBuf,
    config: &Config,
    ngram_to_id: &NgramToIdMap,
    id_to_question_docs: &IdToQuestionDocsMap,
    id_to_short_answer: &IdToShortAnswerMap,
    eval_doc_id_to_answer_token_length: &EvalDocIdToAnswerTokenLengthMap,
    eval_documents: &EvalDocuments,
    next_ngram_id: &std::sync::atomic::AtomicU64,
    tokenizer: &OmniTokenizer,
    id_to_ngram_tokens: &IdToNgramTokens,
    eval_text_snippets: &EvalTextSnippets,
    doc_to_ngram_ids: &DocToNgramIdsMap,
    token_doc_freq: &TokenDocFreqMap,
    dedup_map: &Option<DashMap<(u64, u64), (String, usize)>>,
    global_skipped_duplicates: &Arc<AtomicUsize>,
    global_skipped_min_length: &Arc<AtomicUsize>,
    global_skipped_min_unique_words: &Arc<AtomicUsize>,
    global_lines_processed: &Arc<AtomicUsize>,
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
    let mut _skipped_min_length = 0;
    let mut _skipped_min_unique_words = 0;
    let mut _skipped_duplicates = 0;

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

        // Get question and answer text for filtering
        let question_text = json_obj.get("question").and_then(|v| v.as_str()).unwrap_or("");
        let answer_text = json_obj.get("answer").and_then(|v| v.as_str()).unwrap_or("");
        
        // Clean texts early for filtering
        let cleaned_question = clean_text(question_text, &config.punctuation_chars);
        let cleaned_answer = clean_text(answer_text, &config.punctuation_chars);
        
        // For filtering and deduplication, use combined cleaned question + answer (or just question if no answer)
        let combined_cleaned_text = if has_answer_fields && !cleaned_answer.is_empty() {
            format!("{} {}", cleaned_question, cleaned_answer)
        } else {
            cleaned_question.clone()
        };
        
        // Apply minimum length filter on cleaned text
        if config.eval_min_cleaned_char_length > 0 && combined_cleaned_text.len() < config.eval_min_cleaned_char_length {
            _skipped_entries += 1;
            _skipped_min_length += 1;
            global_skipped_min_length.fetch_add(1, Ordering::Relaxed);
            continue;
        }
        
        // Apply minimum unique word count filter
        if config.eval_min_unique_word_count > 0 {
            let words: HashSet<&str> = combined_cleaned_text.split_whitespace().collect();
            if words.len() < config.eval_min_unique_word_count {
                _skipped_entries += 1;
                _skipped_min_unique_words += 1;
                global_skipped_min_unique_words.fetch_add(1, Ordering::Relaxed);
                continue;
            }
        }
        
        // Apply deduplication if enabled
        if let Some(ref dedup_map) = dedup_map {
            // Already have cleaned texts from above, just normalize for deduplication
            let normalized_question = cleaned_question.to_lowercase();
            let question_hash = hash_text_for_dedup(&normalized_question);
            
            let normalized_answer = cleaned_answer.to_lowercase();
            let answer_hash = if answer_text.is_empty() {
                0 // Special value for no answer
            } else {
                hash_text_for_dedup(&normalized_answer)
            };
            
            let dedup_key = (question_hash, answer_hash);
            
            // Check if we've seen this exact question+answer combination before
            if let Some(first_occurrence) = dedup_map.get(&dedup_key) {
                let (first_file, first_line) = first_occurrence.value();
                
                // Debug print if DECON_DEBUG_DEDUP is set
                if std::env::var("DECON_DEBUG_DEDUP").is_ok() {
                    println!("\n=== DUPLICATE DETECTED ===");
                    println!("Current: {} line {}", eval_name, line_num);
                    println!("Duplicate of: {} line {}", first_file, first_line);
                    println!("Question: {}", question_text);
                    println!("Answer: {}", answer_text);
                    println!("Cleaned question: {}", cleaned_question);
                    println!("Normalized question: {}", normalized_question);
                    println!("Question hash: {}", question_hash);
                    println!("Cleaned answer: {}", cleaned_answer);
                    println!("Normalized answer: {}", normalized_answer);
                    println!("Answer hash: {}", answer_hash);
                    println!("========================\n");
                }
                
                _skipped_entries += 1;
                _skipped_duplicates += 1;
                global_skipped_duplicates.fetch_add(1, Ordering::Relaxed);
                continue;
            }
            
            // Mark as seen with file and line info
            dedup_map.insert(dedup_key, (eval_name.clone(), line_num));
        }

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
                doc_to_ngram_ids,
                token_doc_freq,
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
                    eval_doc_id_to_answer_token_length,
                    token_doc_freq,
                )?;
            }
        }
    }

    global_lines_processed.fetch_add(_lines_processed, Ordering::Relaxed);
    
    // Log filtering statistics per file if verbose logging is enabled
    if std::env::var("DECON_VERBOSE").is_ok() && (config.eval_dedup || config.eval_min_cleaned_char_length > 0 || config.eval_min_unique_word_count > 0) {
        println!("  {}: processed {} lines, skipped {} total ({} min_length, {} min_unique_words, {} duplicates)", 
            eval_name, _lines_processed, _skipped_entries, _skipped_min_length, _skipped_min_unique_words, _skipped_duplicates);
    }
    Ok(())
}

// Helper functions
fn hash_ngram(tokens: &[usize]) -> u64 {
    use std::hash::{DefaultHasher, Hash, Hasher};
    let mut hasher = DefaultHasher::new();
    tokens.hash(&mut hasher);
    hasher.finish()
}

// Helper function to hash text for deduplication
fn hash_text_for_dedup(text: &str) -> u64 {
    use std::hash::{DefaultHasher, Hash, Hasher};
    let mut hasher = DefaultHasher::new();
    text.hash(&mut hasher);
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
    doc_to_ngram_ids: &DocToNgramIdsMap,
    token_doc_freq: &TokenDocFreqMap,
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
        // For BPE tokenizers, clean text first (matching training data processing)
        // then add padding since questions are text fragments that likely appear
        // mid-document rather than at the beginning
        let padded_question = format!(" {}", cleaned);
        let Ok(tokens) = catch_unwind(|| tokenizer.encode(&padded_question)) else {
            println!(
                "Tokenization failed on question for doc_id {} | line {}",
                doc_id, line_num
            );
            *skipped_entries += 1;
            return Ok(());
        };
        tokens
    };
    let token_count = word_tokens.len();

    *lines_processed += 1;

    // Track unique tokens for this document for IDF calculation
    let unique_tokens: HashSet<usize> = word_tokens.iter().copied().collect();
    for token in &unique_tokens {
        token_doc_freq
            .entry(*token)
            .or_insert_with(|| AtomicUsize::new(0))
            .fetch_add(1, Ordering::Relaxed);
    }

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

    // Store the ngram IDs for this document
    doc_to_ngram_ids.insert(doc_id, unique_ngrams_set);

    // Store document metadata (no collision possible with sequential IDs)
    eval_documents.insert(
        doc_id,
        (
            eval_name.to_string(),
            line_num,
            total_ngrams,
            unique_ngrams,
            token_count,
        ),
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
    eval_doc_id_to_answer_token_length: &EvalDocIdToAnswerTokenLengthMap,
    token_doc_freq: &TokenDocFreqMap,
) -> Result<(), Error> {
    // Clean text for both tokenizer types
    let cleaned = clean_text(answer, &config.punctuation_chars);

    // Debug logging for answer processing
    if std::env::var("DEBUG_ANSWER").is_ok() && doc_id == 454571 {
        println!("\n=== Processing Answer (doc_id: {}) ===", doc_id);
        println!("Raw answer: \"{}\"", answer);
        println!("Cleaned answer: \"{}\"", cleaned);
    }

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
        // For BPE tokenizers, use cleaned text (matching training data processing)
        // then add padding since answers are text fragments that likely appear
        // mid-document rather than at the beginning
        let padded_answer = format!(" {}", cleaned);

        // Since unpadded would only matter if the match is at the very beginning
        // of the document, which is unlikely, we go with padded only
        let Ok(padded_tokens) = catch_unwind(|| tokenizer.encode(&padded_answer)) else {
            return Ok(());
        };

        padded_tokens
    };

    // Store the actual token length before converting to set
    let token_length = word_tokens.len();
    eval_doc_id_to_answer_token_length.insert(doc_id, token_length);

    // Store in short answer map
    let token_set: HashSet<usize> = word_tokens.into_iter().collect();

    // Track unique tokens for this document for IDF calculation
    for token in &token_set {
        token_doc_freq
            .entry(*token)
            .or_insert_with(|| AtomicUsize::new(0))
            .fetch_add(1, Ordering::Relaxed);
    }

    // Debug logging for stored tokens
    if std::env::var("DEBUG_ANSWER").is_ok() && doc_id == 454571 {
        println!("Token IDs stored: {:?}", token_set);
        println!(
            "Actual token length: {}, Unique token count: {}",
            token_length,
            token_set.len()
        );
        if let Some(inner) = tokenizer.inner.as_ref() {
            let token_vec: Vec<usize> = token_set.iter().copied().collect();
            if let Ok(decoded) = inner.decode(token_vec) {
                println!("Tokens decode to: \"{}\"", decoded);
            }
        }
    }

    id_to_short_answer.insert(doc_id, token_set);

    Ok(())
}

fn detect_simple_contamination(
    config: &Config,
    ngram_to_id: &NgramToIdMap,
    id_to_question_docs: &IdToQuestionDocsMap,
    id_to_short_answer: &IdToShortAnswerMap,
    eval_doc_id_to_answer_token_length: &EvalDocIdToAnswerTokenLengthMap,
    eval_documents: &EvalDocuments,
    id_to_ngram_tokens: &IdToNgramTokens,
    tokenizer: &OmniTokenizer,
    eval_text_snippets: &EvalTextSnippets,
    eval_document_idf_cache: &EvalDocumentIdfCache,
    doc_to_ngram_ids: &DocToNgramIdsMap,
    token_doc_freq: &TokenDocFreqMap,
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
            eval_doc_id_to_answer_token_length,
            eval_documents,
            &file_contamination_results,
            tokenizer,
            id_to_ngram_tokens,
            eval_text_snippets,
            total_docs,
            eval_document_idf_cache,
            doc_to_ngram_ids,
            token_doc_freq,
        ) {
            Ok(lines_processed) => {
                total_lines_processed
                    .fetch_add(lines_processed, std::sync::atomic::Ordering::SeqCst);

                // Save results for this file if contamination was found
                if !file_contamination_results.is_empty() {
                    let unique_filename = match crate::get_unique_results_filename(file_path, config, &config.local_input) {
                        Ok(filename) => filename,
                        Err(e) => {
                            println!("Error generating unique filename for {:?}: {:?}", file_path, e);
                            return;
                        }
                    };
                    if let Err(e) =
                        save_contamination_results_toxic_format_with_filename_and_eval_text(
                            config,
                            &file_contamination_results,
                            Some(&unique_filename),
                            eval_text_snippets,
                        )
                    {
                        println!("Error saving results for {:?}: {:?}", file_path, e);
                    } else {
                        // Track contaminated files and update total count
                        // Use the same filename extraction logic as in create_purified_files_streaming
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
                        let contamination_count = file_contamination_results
                            .iter()
                            .map(|entry| entry.value().len())
                            .sum::<usize>();

                        total_contaminations
                            .fetch_add(contamination_count as u32, Ordering::Relaxed);

                        let contaminated_lines: HashSet<usize> = file_contamination_results
                            .iter()
                            .flat_map(|entry| {
                                entry
                                    .value()
                                    .iter()
                                    .map(|e| e.training_line)
                                    .collect::<Vec<_>>()
                            })
                            .collect();
                        contaminated_files.insert(file_name, contaminated_lines);
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
        if relative_end >= words_buffer.len() || relative_start > relative_end {
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
        result.push_str(&words_buffer[relative_start..relative_end + 1].join(" "));
        result.push_str("】");

        // Add trailing context
        if relative_end + 1 < words_buffer.len() {
            result.push(' ');
            result.push_str(&words_buffer[relative_end + 1..].join(" "));
            result.push_str(" ...");
        }

        return Some(result);
    }

    // For BPE tokenizers (c100k, p50k), decode from token array
    if let Some(inner) = tokenizer.inner.as_ref() {
        // Check bounds
        if start_idx >= tokens.len() || end_idx >= tokens.len() || start_idx > end_idx {
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
        if let Ok(contaminated) = inner.decode(tokens[start_idx..end_idx + 1].to_vec()) {
            result.push_str("【");
            result.push_str(&contaminated);
            result.push_str("】");
        }

        // Decode and add trailing context
        if end_idx + 1 < context_end {
            if let Ok(suffix) = inner.decode(tokens[end_idx + 1..context_end].to_vec()) {
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
    idf_overlap: Option<f32>, // IDF overlap ratio: matched_idf_sum / eval_total_idf_sum
    // Token indices for position recovery
    contamination_start_idx: Option<usize>, // Start index in token array
    contamination_end_idx: Option<usize>,   // End index in token array
    training_overlap_text: Option<String>,
    ngram_match_cnt: usize,    // Number of unique n-gram matches
    eval_unique_ngrams: usize, // Total unique n-grams in the eval document
    #[allow(dead_code)]
    length_penalty: Option<f32>, // Length penalty factor applied during scoring
    // Answer contamination fields
    answer_overlap_ratio: Option<f32>, // Overlap ratio for answer tokens
    answer_idf_overlap: Option<f32>,   // IDF overlap ratio for answer tokens
    matched_answer_tokens: Option<Vec<String>>, // Matched answer tokens as text
    // Token length tracking fields
    cluster_token_length: Option<usize>, // Length of the matched cluster in tokens
    eval_token_length: Option<usize>,    // Total length of the eval document in tokens
}

impl SimpleContaminationEntry {
    /// Calculate the length penalty factor for this entry
    #[allow(dead_code)]
    pub fn calculate_length_penalty(&self, min_length_penalty: f32) -> f32 {
        let l = self.eval_unique_ngrams as f32;
        let mut length_penalty =
            1.0 - (-LENGTH_PENALTY_DECAY_RATE * l.powf(LENGTH_PENALTY_POWER_N)).exp();
        length_penalty = length_penalty.max(min_length_penalty);
        length_penalty
    }

    /// Calculate the required threshold based on eval token length
    pub fn get_required_threshold(&self, config: &Config) -> f32 {
        let eval_len = self.eval_token_length.unwrap_or(usize::MAX);
        
        match (config.perfect_match_eval_token_length_start, config.threshold_match_eval_token_length_end) {
            (Some(perfect_start), Some(threshold_end)) => {
                if eval_len <= perfect_start {
                    1.0  // Perfect match required
                } else if eval_len >= threshold_end {
                    config.question_threshold  // Normal threshold
                } else if perfect_start == threshold_end {
                    // Step function: immediate transition at the threshold
                    config.question_threshold
                } else {
                    // Linear interpolation between 1.0 and question_threshold
                    let range = (threshold_end - perfect_start) as f32;
                    let position = (eval_len - perfect_start) as f32;
                    let ratio = position / range;
                    // Interpolate: start at 1.0, end at question_threshold
                    1.0 - (1.0 - config.question_threshold) * ratio
                }
            }
            (Some(perfect_start), None) => {
                // Original behavior - hard cutoff
                if eval_len <= perfect_start {
                    1.0
                } else {
                    config.question_threshold
                }
            }
            _ => config.question_threshold
        }
    }

    /// Calculate contamination score based on overlap ratio and IDF, with length penalty
    /// Returns a score between 0.0 and 1.0, where higher scores indicate more likely contamination
    pub fn score_question_contamination(&self, _min_length_penalty: f32) -> f32 {
        //let length_penalty = self.calculate_length_penalty(min_length_penalty);

        // Final score with length penalty applied
        //base_score * length_penalty
        self.idf_overlap.unwrap_or(0.0)
    }

    /// Check if this entry represents contamination (score >= threshold)
    /// Returns (is_contaminated, answer_overlap_ratio, matched_answer_tokens, answer_idf_overlap)
    pub fn is_contaminated(
        &self,
        doc_id: u32,
        id_to_short_answer: &IdToShortAnswerMap,
        eval_doc_id_to_answer_token_length: &EvalDocIdToAnswerTokenLengthMap,
        cluster: &SimpleContaminationCluster,
        training_tokens: &[usize],
        config: &Config,
        tokenizer: &OmniTokenizer,
        token_doc_freq: &TokenDocFreqMap,
        total_docs: f32,
    ) -> (bool, Option<f32>, Option<Vec<String>>, Option<f32>) {
        let required_threshold = self.get_required_threshold(config);
        let question_contam = self.score_question_contamination(required_threshold)
            >= required_threshold;

        if question_contam {
            // If require_answer_when_eval_has_answer is false, return question contamination only
            if !config.require_answer_when_eval_has_answer {
                return (true, None, None, None);
            }

            // Check if this document has a short answer
            if let Some(answer_token_set) = id_to_short_answer.get(&doc_id) {
                // Use the new answer matching method
                let (answer_found, answer_overlap_ratio, matched_token_strings, answer_idf_overlap) =
                    has_matching_answer(
                        answer_token_set.value(),
                        eval_doc_id_to_answer_token_length,
                        cluster,
                        doc_id,
                        training_tokens,
                        config,
                        tokenizer,
                        token_doc_freq,
                        total_docs,
                    );

                // Require both question and answer contamination
                let is_contaminated = question_contam && answer_found;
                return (
                    is_contaminated,
                    Some(answer_overlap_ratio),
                    Some(matched_token_strings),
                    Some(answer_idf_overlap),
                );
            }
        }

        (question_contam, None, None, None)
    }
}

/// Check if the answer matches - exact same behavior as before
/// Returns (is_contaminated, overlap_ratio, matched_tokens_text, answer_idf_overlap)
fn has_matching_answer(
    answer_token_set: &HashSet<usize>,
    eval_doc_id_to_answer_token_length: &EvalDocIdToAnswerTokenLengthMap,
    cluster: &SimpleContaminationCluster,
    doc_id: u32,
    training_tokens: &[usize],
    config: &Config,
    tokenizer: &OmniTokenizer,
    token_doc_freq: &TokenDocFreqMap,
    total_docs: f32,
) -> (bool, f32, Vec<String>, f32) {
    // Get matching tokens separately for prefix and suffix
    let (prefix_matches, suffix_matches) = short_answer_tokens(
        answer_token_set,
        eval_doc_id_to_answer_token_length,
        cluster,
        doc_id,
        training_tokens,
        config,
        tokenizer,
    );

    // Calculate overlap ratios for prefix and suffix
    let prefix_overlap_ratio = if answer_token_set.is_empty() {
        0.0
    } else {
        prefix_matches.len() as f32 / answer_token_set.len() as f32
    };

    let suffix_overlap_ratio = if answer_token_set.is_empty() {
        0.0
    } else {
        suffix_matches.len() as f32 / answer_token_set.len() as f32
    };

    // Use the better overlap ratio and corresponding matches
    let (answer_overlap_ratio, matching_tokens) = if prefix_overlap_ratio >= suffix_overlap_ratio {
        (prefix_overlap_ratio, prefix_matches)
    } else {
        (suffix_overlap_ratio, suffix_matches)
    };

    // Convert matching token IDs to text
    let matched_token_strings: Vec<String> = matching_tokens
        .iter()
        .filter_map(|&token_id| tokenizer.get_word(token_id as u32))
        .collect();

    // Calculate IDF overlap using the better match set
    let answer_idf_overlap = calculate_answer_idf_overlap(
        &matching_tokens,
        answer_token_set,
        token_doc_freq,
        total_docs,
    );

    // Check if meets threshold using IDF overlap
    let is_contaminated = answer_idf_overlap >= config.answer_threshold;

    (
        is_contaminated,
        answer_overlap_ratio,
        matched_token_strings,
        answer_idf_overlap,
    )
}

fn short_answer_tokens(
    answer_token_set: &HashSet<usize>,
    eval_doc_id_to_answer_token_length: &EvalDocIdToAnswerTokenLengthMap,
    cluster: &SimpleContaminationCluster,
    doc_id: u32,
    training_tokens: &[usize],
    config: &Config,
    tokenizer: &OmniTokenizer,
) -> (HashSet<usize>, HashSet<usize>) {
    // Calculate window size as max(answer_length*2, min_short_answer_distance)
    // Use the actual token length, not the unique token count
    let answer_length = eval_doc_id_to_answer_token_length
        .get(&doc_id)
        .map(|len| *len)
        .unwrap_or_else(|| answer_token_set.len()); // Fallback to unique count if not found
    let window_size = std::cmp::max(answer_length * 2, config.min_short_answer_distance);

    // Get document-specific boundaries
    let (doc_start_idx, doc_end_idx) = cluster
        .document_boundaries
        .get(&doc_id)
        .copied()
        .expect("Document boundaries should exist for all matched documents");

    if config.exclude_question_from_answer_sweep {
        // When excluding question tokens, search in prefix and suffix regions
        let prefix_search_start = doc_start_idx.saturating_sub(window_size);
        let prefix_search_end = doc_start_idx;
        // Fix: doc_end_idx is the last n-gram position, but we need the last token position
        let suffix_search_start = doc_end_idx + config.ngram_size - 1 + 1;
        let suffix_search_end = (suffix_search_start + window_size).min(training_tokens.len());

        // Collect tokens from both regions
        let mut training_token_set = HashSet::new();

        // Add prefix tokens and decode for debug display
        let mut prefix_tokens = Vec::new();
        if prefix_search_start < prefix_search_end {
            prefix_tokens.extend(&training_tokens[prefix_search_start..prefix_search_end]);
            training_token_set.extend(prefix_tokens.iter().copied());
        }

        // Add suffix tokens and decode for debug display
        let mut suffix_tokens = Vec::new();
        if suffix_search_start < suffix_search_end && suffix_search_start < training_tokens.len() {
            let end = suffix_search_end.min(training_tokens.len());
            suffix_tokens.extend(&training_tokens[suffix_search_start..end]);
            training_token_set.extend(suffix_tokens.iter().copied());
        }

        // Debug logging - decode and display the prefix and suffix text
        if std::env::var("DEBUG_ANSWER").is_ok() {
            let prefix_text = if tokenizer.tokenizer_name == "word" {
                // For word tokenizer, convert token IDs back to words
                prefix_tokens
                    .iter()
                    .filter_map(|&token_id| tokenizer.get_word(token_id as u32))
                    .collect::<Vec<_>>()
                    .join(" ")
            } else if let Some(inner) = tokenizer.inner.as_ref() {
                // For BPE tokenizers, decode token array
                inner
                    .decode(prefix_tokens.clone())
                    .unwrap_or_else(|_| "[decode error]".to_string())
            } else {
                "[no decoder]".to_string()
            };

            let suffix_text = if tokenizer.tokenizer_name == "word" {
                // For word tokenizer, convert token IDs back to words
                suffix_tokens
                    .iter()
                    .filter_map(|&token_id| tokenizer.get_word(token_id as u32))
                    .collect::<Vec<_>>()
                    .join(" ")
            } else if let Some(inner) = tokenizer.inner.as_ref() {
                // For BPE tokenizers, decode token array
                inner
                    .decode(suffix_tokens.clone())
                    .unwrap_or_else(|_| "[decode error]".to_string())
            } else {
                "[no decoder]".to_string()
            };

            // Decode answer tokens to see what we're looking for
            let answer_text = if tokenizer.tokenizer_name == "word" {
                answer_token_set
                    .iter()
                    .filter_map(|&token_id| tokenizer.get_word(token_id as u32))
                    .collect::<Vec<_>>()
                    .join(" ")
            } else if let Some(inner) = tokenizer.inner.as_ref() {
                let answer_vec: Vec<usize> = answer_token_set.iter().copied().collect();
                inner
                    .decode(answer_vec)
                    .unwrap_or_else(|_| "[decode error]".to_string())
            } else {
                "[no decoder]".to_string()
            };

            println!("\n=== Answer Detection Debug (doc_id: {}) ===", doc_id);
            println!(
                "Answer text to find: \"{}\" ({} tokens)",
                answer_text,
                answer_token_set.len()
            );
            println!("Answer token IDs: {:?}", answer_token_set);
            println!("Prefix ({} tokens): {}", prefix_tokens.len(), prefix_text);
            println!("Suffix ({} tokens): {}", suffix_tokens.len(), suffix_text);

            // Show what tokens we found in common
            let found_in_training: HashSet<usize> = answer_token_set
                .iter()
                .filter(|token| training_token_set.contains(token))
                .copied()
                .collect();
            println!(
                "Found {} of {} answer tokens in training text",
                found_in_training.len(),
                answer_token_set.len()
            );
            println!("Found token IDs: {:?}", found_in_training);
            let missing_tokens: HashSet<usize> = answer_token_set
                .difference(&found_in_training)
                .copied()
                .collect();
            println!("Missing token IDs: {:?}", missing_tokens);

            // Debug: Show the last few tokens of prefix and first few of suffix
            if prefix_tokens.len() >= 4 {
                let last_prefix_tokens = &prefix_tokens[prefix_tokens.len() - 4..];
                println!("Last 4 prefix tokens: {:?}", last_prefix_tokens);
            }
            if suffix_tokens.len() >= 4 {
                let first_suffix_tokens = &suffix_tokens[..4.min(suffix_tokens.len())];
                println!("First 4 suffix tokens: {:?}", first_suffix_tokens);
            }
        }

        // Find matching tokens separately for prefix and suffix
        let prefix_token_set: HashSet<usize> = prefix_tokens.iter().copied().collect();
        let suffix_token_set: HashSet<usize> = suffix_tokens.iter().copied().collect();

        let prefix_matches: HashSet<usize> = answer_token_set
            .iter()
            .filter(|token| prefix_token_set.contains(token))
            .copied()
            .collect();

        let suffix_matches: HashSet<usize> = answer_token_set
            .iter()
            .filter(|token| suffix_token_set.contains(token))
            .copied()
            .collect();

        (prefix_matches, suffix_matches)
    } else {
        // Original behavior: search entire window including the question
        // Split into prefix (before question) and suffix (after question) parts
        let prefix_start = doc_start_idx.saturating_sub(window_size);
        let prefix_end = doc_start_idx;
        let suffix_start = doc_end_idx + config.ngram_size - 1 + 1;
        let suffix_end = suffix_start
            .saturating_add(window_size)
            .min(training_tokens.len());

        // Extract prefix tokens
        let prefix_token_set: HashSet<usize> = if prefix_start < prefix_end {
            training_tokens[prefix_start..prefix_end]
                .iter()
                .copied()
                .collect()
        } else {
            HashSet::new()
        };

        // Extract suffix tokens
        let suffix_token_set: HashSet<usize> =
            if suffix_start < suffix_end && suffix_start < training_tokens.len() {
                training_tokens[suffix_start..suffix_end.min(training_tokens.len())]
                    .iter()
                    .copied()
                    .collect()
            } else {
                HashSet::new()
            };

        // Find matching tokens separately
        let prefix_matches: HashSet<usize> = answer_token_set
            .iter()
            .filter(|token| prefix_token_set.contains(token))
            .copied()
            .collect();

        let suffix_matches: HashSet<usize> = answer_token_set
            .iter()
            .filter(|token| suffix_token_set.contains(token))
            .copied()
            .collect();

        (prefix_matches, suffix_matches)
    }
}

#[derive(Clone)]
pub(crate) struct SimpleContaminationCluster {
    document_matches: HashMap<u32, HashSet<u64>>, // doc_id -> unique_ngram_ids that matched eval
    document_boundaries: HashMap<u32, (usize, usize)>, // doc_id -> (start_idx, end_idx)
    end_idx: usize,
}

pub fn process_simple_training_file(
    file_path: &PathBuf,
    config: &Config,
    ngram_to_id: &NgramToIdMap,
    id_to_question_docs: &IdToQuestionDocsMap,
    id_to_short_answer: &IdToShortAnswerMap,
    eval_doc_id_to_answer_token_length: &EvalDocIdToAnswerTokenLengthMap,
    eval_documents: &EvalDocuments,
    contamination_results: &ContaminationResults,
    tokenizer: &OmniTokenizer,
    id_to_ngram_tokens: &IdToNgramTokens,
    _eval_text_snippets: &EvalTextSnippets,
    total_docs: f32,
    eval_document_idf_cache: &EvalDocumentIdfCache,
    doc_to_ngram_ids: &DocToNgramIdsMap,
    token_doc_freq: &TokenDocFreqMap,
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

    let mut lines_processed = 0;
    let min_token_count = config.ngram_size * 2; // Minimum tokens needed for meaningful n-gram analysis

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
        let token_count = word_tokens.len();

        // Skip entries with insufficient tokens
        if token_count < min_token_count {
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
                    let (eval_name, eval_line, _total_ngrams, unique_ngrams, eval_token_count) =
                        doc_info.value();
                    let unique_matches = matched_ngram_ids.len();
                    let overlap_ratio = if *unique_ngrams > 0 {
                        unique_matches as f32 / *unique_ngrams as f32
                    } else {
                        0.0
                    };

                    // IMPORTANT BUT SUBTLE!!!!
                    // We kind of have a hard coded short circuit here.
                    // Basically we reject anything with less than this much raw overlap.
                    // TODO move this to even prevent cluster creation at some point.
                    if overlap_ratio < config.question_threshold - 0.1 {
                        continue;
                    }

                    // Get document-specific boundaries
                    let (doc_start_idx, doc_end_idx) = cluster
                        .document_boundaries
                        .get(doc_id)
                        .copied()
                        .expect("Document boundaries should exist for all matched documents");

                    // Calculate cluster token length
                    let cluster_token_length =
                        (doc_end_idx + config.ngram_size - 1) - doc_start_idx + 1;

                    // Create the contamination entry first
                    let mut entry = SimpleContaminationEntry {
                        training_line: line_num,
                        eval_name: eval_name.clone(),
                        eval_line: *eval_line,
                        overlap_ratio,
                        idf_overlap: None, // Will be calculated below
                        contamination_start_idx: Some(doc_start_idx),
                        contamination_end_idx: Some(doc_end_idx + config.ngram_size - 1),
                        training_overlap_text: None, // Will be filled if contaminated
                        ngram_match_cnt: unique_matches,
                        eval_unique_ngrams: *unique_ngrams,
                        length_penalty: None, // Will be calculated and set below
                        answer_overlap_ratio: None,
                        answer_idf_overlap: None,
                        matched_answer_tokens: None,
                        cluster_token_length: Some(cluster_token_length),
                        eval_token_length: Some(*eval_token_count),
                    };

                    // Calculate IDF overlap ratio
                    entry.idf_overlap = calculate_idf_overlap(
                        matched_ngram_ids,
                        *doc_id,
                        doc_to_ngram_ids,
                        id_to_question_docs,
                        eval_document_idf_cache,
                        total_docs,
                    );

                    // Calculate and store the length penalty
                    // entry.length_penalty = Some(
                    //     entry.calculate_length_penalty(config.question_threshold),
                    // );

                    // Check if this entry represents contamination using score threshold
                    let (
                        is_contaminated,
                        answer_overlap_ratio,
                        matched_answer_tokens,
                        answer_idf_overlap,
                    ) = entry.is_contaminated(
                        *doc_id,
                        id_to_short_answer,
                        eval_doc_id_to_answer_token_length,
                        &cluster,
                        &word_tokens,
                        config,
                        tokenizer,
                        token_doc_freq,
                        total_docs,
                    );

                    // Track if question was contaminated but excluded due to no answer match
                    let required_threshold = entry.get_required_threshold(config);
                    let question_contam = entry
                        .score_question_contamination(required_threshold)
                        >= required_threshold;
                    if question_contam && !is_contaminated {
                        EXCLUDED_NO_ANSWER_MATCH.fetch_add(1, Ordering::Relaxed);
                    }

                    if is_contaminated {
                        // Extract the overlapping text with context
                        // Note: cluster indices are n-gram positions, but we need token positions
                        // An n-gram at position i covers tokens from i to i+ngram_size-1
                        let token_end_idx = doc_end_idx + config.ngram_size - 1;
                        let training_overlap_text = extract_overlap_with_context(
                            &cleaned_text,
                            &word_tokens,
                            doc_start_idx,
                            token_end_idx,
                            tokenizer,
                            60,
                        );

                        let mut entry_with_text = entry;
                        entry_with_text.training_overlap_text = training_overlap_text;
                        entry_with_text.answer_overlap_ratio = answer_overlap_ratio;
                        entry_with_text.answer_idf_overlap = answer_idf_overlap;
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
    let mut i = 0;

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

            let cluster_end = cluster.end_idx;
            clusters.push(cluster);
            i = cluster_end + 1;
        } else {
            i += config.sample_every_m_tokens.max(1); // No hit, continue sampling
        }
    }

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

    // Look up documents containing this n-gram
    if let Some(ngram_id) = ngram_to_id.get(&ngram_hash) {
        if let Some(doc_set) = id_to_question_docs.get(&ngram_id) {
            if !doc_set.is_empty() {
                return Some(doc_set.clone());
            }
        }
    }
    None
}

/// Expand contamination cluster using intersection-based left/right traversal
/// Note that we don't care if we go over a little, because we are solving a containment
/// problem and using overlap scores. The real problem is when a cluster is shorter than
/// the eval document.
fn expand_simple_contamination_cluster(
    hit_idx: usize,
    word_tokens: &[usize],
    config: &Config,
    ngram_to_id: &NgramToIdMap,
    id_to_question_docs: &IdToQuestionDocsMap,
    initial_document_ids: HashSet<u32>,
    initial_training_ngram: &[usize],
    _id_to_ngram_tokens: &IdToNgramTokens,
) -> Result<SimpleContaminationCluster, Error> {
    // Initialize document match tracking - track consecutive misses for each document
    let mut document_matches: HashMap<u32, HashSet<u64>> = HashMap::new();
    let mut document_misses: HashMap<u32, usize> = HashMap::new();
    let mut document_boundaries: HashMap<u32, (usize, usize)> = HashMap::new();
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
        document_boundaries.insert(*doc_id, (hit_idx, hit_idx));
    }

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
    let mut left_idx = hit_idx;
    while left_idx > 0 && !active_documents.is_empty() {
        left_idx -= 1;
        LEFT_TRAVERSAL_COUNT.fetch_add(1, Ordering::Relaxed); //DEBUGCOUNTER

        // Get the n-gram tokens for hash calculation
        let ngram_tokens = if word_tokens.len() < config.ngram_size {
            word_tokens.to_vec()
        } else {
            word_tokens[left_idx..left_idx + config.ngram_size].to_vec()
        };

        // Calculate n-gram hash for match tracking
        let ngram_hash = hash_ngram(&ngram_tokens);

        if let Some(matched_docs) = check_ngram_for_match(
            left_idx,
            word_tokens,
            config,
            ngram_to_id,
            id_to_question_docs,
        ) {
            // Check intersection with active documents
            let intersection: Vec<u32> = active_documents
                .intersection(&matched_docs)
                .cloned()
                .collect();

            if !intersection.is_empty() {
                // Get ngram_id for tracking matches
                let ngram_id = ngram_to_id.get(&ngram_hash).map(|id| *id).unwrap_or(0);

                // Update matches and reset misses for intersecting documents
                for doc_id in &intersection {
                    let is_new_ngram = document_matches
                        .entry(*doc_id)
                        .or_insert_with(HashSet::new)
                        .insert(ngram_id);

                    // Only reset miss counter if this is a new n-gram for this document
                    if is_new_ngram {
                        document_misses.insert(*doc_id, 0);
                    }

                    // Always update boundary to show full contamination span
                    if let Some((doc_start, _doc_end)) = document_boundaries.get_mut(doc_id) {
                        *doc_start = left_idx;
                    }
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
                    }
                }
            } else {
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
                }
            }
        } else {
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
            }
        }
    }

    // Reset active documents and misses for right traversal
    active_documents = initial_document_ids.clone();
    for doc_id in &initial_document_ids {
        document_misses.insert(*doc_id, 0);
    }

    // Expand forward (right traversal)
    let mut right_idx = hit_idx;
    while right_idx + 1 < total_ngrams && !active_documents.is_empty() {
        right_idx += 1;
        RIGHT_TRAVERSAL_COUNT.fetch_add(1, Ordering::Relaxed); //DEBUGCOUNTER

        // Get the n-gram tokens for hash calculation
        let ngram_tokens = if word_tokens.len() < config.ngram_size {
            word_tokens.to_vec()
        } else {
            word_tokens[right_idx..right_idx + config.ngram_size].to_vec()
        };

        // Calculate n-gram hash for match tracking
        let ngram_hash = hash_ngram(&ngram_tokens);

        if let Some(matched_docs) = check_ngram_for_match(
            right_idx,
            word_tokens,
            config,
            ngram_to_id,
            id_to_question_docs,
        ) {
            // Check intersection with active documents
            let intersection: Vec<u32> = active_documents
                .intersection(&matched_docs)
                .cloned()
                .collect();

            if !intersection.is_empty() {
                // Get ngram_id for tracking matches
                let ngram_id = ngram_to_id.get(&ngram_hash).map(|id| *id).unwrap_or(0);

                // Update matches and reset misses for intersecting documents
                for doc_id in &intersection {
                    let is_new_ngram = document_matches
                        .entry(*doc_id)
                        .or_insert_with(HashSet::new)
                        .insert(ngram_id);

                    // Only reset miss counter if this is a new n-gram for this document
                    if is_new_ngram {
                        document_misses.insert(*doc_id, 0);
                    }

                    // Always update boundary to show full contamination span
                    if let Some((_doc_start, doc_end)) = document_boundaries.get_mut(doc_id) {
                        *doc_end = right_idx;
                    }
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
                    }
                }
            } else {
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
                }
            }
        } else {
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
            }
        }
    }

    Ok(SimpleContaminationCluster {
        document_matches,
        document_boundaries,
        end_idx: right_idx,
    })
}

/// Calculate total IDF sum for all n-grams in an eval document
/// This is cached to avoid recalculating for each contamination check
fn calculate_eval_document_idf_sum(
    doc_id: u32,
    doc_to_ngram_ids: &DocToNgramIdsMap,
    id_to_question_docs: &IdToQuestionDocsMap,
    total_docs: f32,
) -> f32 {
    let mut idf_sum = 0.0f32;

    // Get the n-gram IDs for this document
    if let Some(ngram_ids) = doc_to_ngram_ids.get(&doc_id) {
        // Iterate through this document's n-grams
        for ngram_id in ngram_ids.value() {
            if let Some(doc_set) = id_to_question_docs.get(ngram_id) {
                let doc_freq = doc_set.len() as f32;
                // Standard IDF formula: ln(N/doc_freq)
                let idf = (total_docs / doc_freq).ln();
                idf_sum += idf;
            }
        }
    }

    idf_sum
}

/// Calculate IDF overlap ratio for answer tokens
/// Returns the ratio of matched token IDF sum to total answer token IDF sum
fn calculate_answer_idf_overlap(
    matched_tokens: &HashSet<usize>,
    answer_tokens: &HashSet<usize>,
    token_doc_freq: &TokenDocFreqMap,
    total_docs: f32,
) -> f32 {
    // Calculate IDF sum for all answer tokens
    let mut answer_idf_sum = 0.0f32;
    for token in answer_tokens {
        if let Some(doc_freq) = token_doc_freq.get(token) {
            let freq = doc_freq.load(Ordering::Relaxed) as f32;
            if freq > 0.0 {
                let idf = (total_docs / freq).ln();
                answer_idf_sum += idf;
            }
        }
    }

    // Calculate IDF sum for matched tokens only
    let mut matched_idf_sum = 0.0f32;
    for token in matched_tokens {
        if let Some(doc_freq) = token_doc_freq.get(token) {
            let freq = doc_freq.load(Ordering::Relaxed) as f32;
            if freq > 0.0 {
                let idf = (total_docs / freq).ln();
                matched_idf_sum += idf;
            }
        }
    }

    // Calculate overlap ratio
    if answer_idf_sum > 0.0 {
        matched_idf_sum / answer_idf_sum
    } else {
        0.0
    }
}

/// Calculate IDF overlap ratio between matched n-grams and eval document
/// Returns the ratio of matched n-gram IDF sum to total eval document IDF sum
fn calculate_idf_overlap(
    matched_ngram_ids: &HashSet<u64>,
    doc_id: u32,
    doc_to_ngram_ids: &DocToNgramIdsMap,
    id_to_question_docs: &IdToQuestionDocsMap,
    eval_document_idf_cache: &EvalDocumentIdfCache,
    total_docs: f32,
) -> Option<f32> {
    // Calculate IDF sum for matched n-grams (shared between training and eval)
    let (matched_idf_sum, _max_idf) =
        calculate_idf_values(matched_ngram_ids, id_to_question_docs, total_docs);

    // Get or calculate the eval document's total IDF sum
    let eval_total_idf = if let Some(cached_idf_ref) = eval_document_idf_cache.get(&doc_id) {
        *cached_idf_ref
    } else {
        let total_idf = calculate_eval_document_idf_sum(
            doc_id,
            doc_to_ngram_ids,
            id_to_question_docs,
            total_docs,
        );
        eval_document_idf_cache.insert(doc_id, total_idf);
        total_idf
    };

    // Calculate IDF overlap ratio
    if eval_total_idf > 0.0 {
        Some(matched_idf_sum / eval_total_idf)
    } else {
        Some(0.0)
    }
}

/// Calculate IDF values for a set of n-gram IDs
/// Returns (idf_sum, max_idf) for all n-grams
fn calculate_idf_values(
    ngram_ids: &HashSet<u64>,
    id_to_question_docs: &IdToQuestionDocsMap,
    total_docs: f32,
) -> (f32, f32) {
    // Returns (idf_sum, max_idf)
    let mut idf_sum = 0.0f32;
    let mut max_idf = 0.0f32;

    for ngram_id in ngram_ids {
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

pub fn save_contamination_results_toxic_format_with_filename_and_eval_text(
    config: &Config,
    contamination_results: &ContaminationResults,
    custom_filename: Option<&str>,
    eval_text_snippets: &EvalTextSnippets,
) -> Result<PathBuf, Error> {
    // Use custom filename if provided, otherwise use default
    let default_filename = get_results_filename("simple");
    let filename = custom_filename.unwrap_or(&default_filename);
    let output_file = config.report_output_dir.join(filename);
    
    // Create parent directories if they don't exist (for preserving directory structure)
    if let Some(parent) = output_file.parent() {
        create_dir_all(parent)?;
    }

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
                "ngram_match_cnt": contamination_entry.ngram_match_cnt,
                "eval_unique_ngrams": contamination_entry.eval_unique_ngrams,
                "contamination_score": contamination_entry.score_question_contamination(config.question_threshold),
                //"length_penalty": contamination_entry.length_penalty.unwrap_or_else(|| contamination_entry.calculate_length_penalty(config.question_threshold)),
                "length_adjusted_question_threshold": contamination_entry.get_required_threshold(config),
                "method": "simple"
            });

            // Add IDF overlap if available
            if let Some(idf_overlap) = contamination_entry.idf_overlap {
                result["idf_overlap"] = json!(idf_overlap);
            }

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
            if let Some(idf_overlap) = contamination_entry.answer_idf_overlap {
                result["answer_idf_overlap"] = json!(idf_overlap);
            }
            if let Some(ref tokens) = contamination_entry.matched_answer_tokens {
                result["matched_answer_tokens"] = json!(tokens);
            }

            // Add token length information
            if let Some(cluster_len) = contamination_entry.cluster_token_length {
                result["cluster_token_length"] = json!(cluster_len);
            }
            if let Some(eval_len) = contamination_entry.eval_token_length {
                result["eval_token_length"] = json!(eval_len);
            }
            if let (Some(cluster_len), Some(eval_len)) = (
                contamination_entry.cluster_token_length,
                contamination_entry.eval_token_length,
            ) {
                let delta = cluster_len as i32 - eval_len as i32;
                result["token_length_delta"] = json!(delta);
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
        write_purified_file(file_path, cleaned_dir, &contaminated_lines, config, &config.local_input)?;

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
