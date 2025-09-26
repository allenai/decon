use anyhow::{Error, Result};
use dashmap::DashMap;
use rayon::prelude::*;
use serde_json::Value;
use std::collections::HashSet;
use std::fs::create_dir_all;
use std::io::BufRead;
use std::panic::catch_unwind;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;
use std::time::Instant;

use mj_io::expand_dirs;

use crate::common::{
    clean_text, get_nested_json_val, Config, OmniTokenizer
};
use crate::detect::{
    contamination_entry::SimpleContaminationEntry,
    reference_index::{
        SimpleReferenceIndex, hash_ngram,
        QuestionNgramToIdMap, QuestionNgramIdToEvalDocIdsMap,
    },
    cluster::{
        SimpleContaminationCluster, ExpansionContext,
        expand_simple_contamination_cluster,
    },
};
use crate::detect::utils::{
    build_pbar_quiet, display_timing_stats, read_compressed_file,
};
use crate::detect::reporting::{
    save_contamination_results,
    create_purified_files_streaming,
};
use crate::detect::scoring::{
    calculate_question_idf_overlap,
    interpolate_threshold,
};
use crate::detect::stats::{StatsContainer, AggregatedStats};

pub type ContaminationResults = DashMap<String, Vec<SimpleContaminationEntry>>;

/// Groups shared state needed for file processing
pub struct FileProcessingState<'a> {
    pub total_lines_processed: &'a Arc<std::sync::atomic::AtomicUsize>,
    pub total_contaminations: &'a Arc<AtomicU32>,
    pub contaminated_files: &'a Arc<DashMap<String, HashSet<usize>>>,
}

/// Context for contamination evaluation to reduce function parameters
pub(crate) struct ContaminationContext<'a> {
    pub training_tokens: &'a [usize],
    pub config: &'a Config,
    pub index: &'a SimpleReferenceIndex,
    pub stats: Option<&'a StatsContainer>,
}

/// Context for overlap extraction to reduce function parameters
struct OverlapExtractionContext<'a> {
    pub tokenizer: &'a OmniTokenizer,
    pub context_words_before: usize,
    pub context_words_after: usize,
}

/// Entrypoint to main contamination detection activity.
/// Builds an index, then iterates over training documents and records contamination.
pub fn contamination_detect(config: &Config) -> Result<(), Error> {
    let start_main = Instant::now();

    // Step 1: Process eval datasets and build n-gram mappings
    let index_start = Instant::now();
    let (index, index_stats) = super::reference_index::build_simple_index(config)?;
    let index_time = index_start.elapsed();

    super::display::display_index_building_results(&index_stats, &index, index_time);

    if index_stats.lines_indexed == 0 {
        return Err(create_no_data_indexed_error(&index_stats, config));
    }

    if config.verbose
        && let Some(ref hot_bucket_stats) = index.hot_bucket_stats {
            hot_bucket_stats.display();
        }

    // Step 2: Process training data and detect contamination
    let detection_start = Instant::now();
    let (total_contaminations, contaminated_documents, contaminated_lines, aggregated_stats, lines_processed) = detect_simple_contamination(config, &index)?;
    let detection_time = detection_start.elapsed();

    let total_time = start_main.elapsed();

    super::display::display_detection_results(
        index_time,
        detection_time,
        total_time,
        total_contaminations,
        contaminated_documents,
        contaminated_lines,
        lines_processed,
    );

    if config.verbose {
        display_timing_stats(&aggregated_stats);
    }

    super::display::display_completion_message(config, total_contaminations);

    Ok(())
}

/// Create an error when no evaluation data was indexed
fn create_no_data_indexed_error(
    stats: &super::reference_index::IndexBuildingStats,
    config: &Config,
) -> Error {
    anyhow::anyhow!(
        "No evaluation data was indexed. Processed {} files with {} lines, but found 0 conformant eval instance entries to index.\n\n\
        Possible causes:\n\
        - The evaluation files may not be in the expected format\n\
        - All entries were filtered out due to minimum length requirements (min {} tokens, min {} unique words)\n\
        - The files may be empty or contain only invalid JSON\n\n\
        Please check your evaluation directory: {}",
        stats.files_processed,
        stats.total_lines_examined,
        config.eval_min_token_length,
        config.eval_min_unique_word_count,
        config.evals_dir.display()
    )
}


/// Update the global contamination count with results from a file
fn update_contamination_count(
    file_contamination_results: &ContaminationResults,
    total_contaminations: &Arc<AtomicU32>,
) -> usize {
    let contamination_count = file_contamination_results
        .iter()
        .map(|entry| entry.value().len())
        .sum::<usize>();

    total_contaminations.fetch_add(contamination_count as u32, Ordering::Relaxed);

    contamination_count
}

/// Extract base filename from a path, handling compressed file extensions
fn extract_base_filename(file_path: &Path) -> String {
    match file_path.extension().and_then(|s| s.to_str()) {
        Some("gz") | Some("zst") | Some("zstd") => file_path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("unknown")
            .to_string(),
        _ => file_path
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("unknown")
            .to_string(),
    }
}


/// Track contaminated lines for the purification feature
fn track_contaminated_lines_for_purification(
    file_path: &Path,
    file_contamination_results: &ContaminationResults,
) -> (String, HashSet<usize>) {
    // Extract filename using the shared helper
    let file_name = extract_base_filename(file_path);

    // Collect all contaminated line numbers
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

    (file_name, contaminated_lines)
}

/// Update counters and statistics and write a contamination report file.
fn handle_successful_file_processing(
    file_path: &PathBuf,
    lines_processed: usize,
    file_contamination_results: &ContaminationResults,
    config: &Config,
    index: &SimpleReferenceIndex,
    state: &FileProcessingState,
    stats: Option<&StatsContainer>,
) {
    state.total_lines_processed.fetch_add(lines_processed, std::sync::atomic::Ordering::SeqCst);

    if let Some(stats) = stats {
        stats.increment_files_processed();
        stats.add_lines_processed(lines_processed);
    }

    if file_contamination_results.is_empty() {
        return;
    }

    let unique_filename = match crate::common::generate_report_filename(file_path, config, &config.training_dir) {
        Ok(filename) => filename,
        Err(e) => {
            eprintln!("Error generating unique filename for {:?}: {:?}", file_path, e);
            return;
        }
    };

    if let Err(e) = save_contamination_results(
        config,
        file_contamination_results,
        &unique_filename,
        &index.eval_text_snippets,
        &index.eval_answer_text_snippets,
        &index.eval_passage_text_snippets,
    ) {
        eprintln!("Error saving results for {:?}: {:?}", file_path, e);
        return;
    }

    update_contamination_count(file_contamination_results, state.total_contaminations);
    let (file_name, contaminated_lines) = track_contaminated_lines_for_purification(file_path, file_contamination_results);

    state.contaminated_files.insert(file_name, contaminated_lines);

    // Update contamination count in stats if available
    if let Some(stats) = stats {
        stats.add_contaminations_found(file_contamination_results.len());
    }
}

fn detect_simple_contamination(config: &Config, index: &SimpleReferenceIndex,) -> Result<(usize, usize, usize, AggregatedStats, usize), Error> {
    let stats_container = if config.verbose {
        Some(StatsContainer::new())
    } else {
        None
    };
    let training_files = expand_dirs(
        vec![config.training_dir.clone()],
        Some(crate::common::EXPAND_DIRS_EXTENSIONS),
    )?;

    create_dir_all(&config.report_output_dir)?;

    let total_lines_processed = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let contaminated_files: Arc<DashMap<String, HashSet<usize>>> = Arc::new(DashMap::new());
    let total_contaminations = Arc::new(AtomicU32::new(0));

    let pbar = build_pbar_quiet(training_files.len(), "Training files");

    let processing_start = Instant::now();
    training_files.par_iter().for_each(|file_path| {
        let file_contamination_results = DashMap::new();
        let stats = stats_container.as_ref();

        match process_simple_training_file(file_path, config, index, &file_contamination_results, stats) {
            Ok(lines_processed) => {
                let state = FileProcessingState {
                    total_lines_processed: &total_lines_processed,
                    total_contaminations: &total_contaminations,
                    contaminated_files: &contaminated_files,
                };
                handle_successful_file_processing(
                    file_path,
                    lines_processed,
                    &file_contamination_results,
                    config,
                    index,
                    &state,
                    stats,
                );
            }
            Err(e) => {
                eprintln!("Error processing training file {:?}: {:?}", file_path, e);
            }
        }
        pbar.inc(1);
    });
    let _processing_time = processing_start.elapsed();

    pbar.finish();

    let total_contaminations_count = total_contaminations.load(Ordering::Relaxed) as usize;
    let lines_processed = total_lines_processed.load(std::sync::atomic::Ordering::SeqCst);
    let contaminated_documents_count = contaminated_files.len();

    // Calculate total contaminated lines (sum of all line numbers across all files)
    let contaminated_lines_count: usize = contaminated_files
        .iter()
        .map(|entry| entry.value().len())
        .sum();

    // After detecting all contamination and record training files, line number of contaminated entries
    // we read all the files and write new copies with contaminated lines removed.
    if config.purify {
        create_purified_files_streaming(config, &contaminated_files, &training_files)?;
    }

    // Aggregate all statistics from thread-local storage if stats were collected
    let aggregated_stats = if let Some(container) = stats_container {
        container.aggregate()
    } else {
        AggregatedStats::default()
    };

    Ok((total_contaminations_count, contaminated_documents_count, contaminated_lines_count, aggregated_stats, lines_processed))
}

pub fn process_simple_training_file(file_path: &PathBuf, config: &Config, index: &SimpleReferenceIndex, contamination_results: &ContaminationResults, stats: Option<&StatsContainer>) -> Result<usize, Error> {
    let data = read_compressed_file(file_path)?;
    let file_name = extract_base_filename(file_path);

    let mut lines_processed = 0;
    let min_token_count = config.ngram_size * 2; // Minimum tokens needed for meaningful n-gram analysis. This is extremely conservative.

    for (line_num, line) in data.lines().enumerate() {
        let line = line?;

        // Track contamination entries to avoid duplicates within the same training document line
        // Key: (eval_key, eval_instance_index, question_start_idx)
        let mut processed_contaminations: HashSet<(String, usize, usize)> = HashSet::new();

        // Prepare line: parse JSON, extract text, clean (strip punctuation), and tokenize
        let (cleaned_text, training_tokens) = match prepare_line_for_processing(&line, line_num, file_path, config, &index.tokenizer) {
            Some(result) => result,
            None => continue,  // Skip lines that fail preparation
        };

        // Skip entries with insufficient tokens to make a meaningful detection.
        if training_tokens.len() < min_token_count {
            continue;
        }

        lines_processed += 1;

        let clusters = identify_contamination_clusters(&training_tokens, config, index, stats)?;

        // TODO: Consider optimization - call contamination inline with detection and set a boundary
        // for left expansion to prevent duplicate detection of the same contaminated content.
        // This would involve restructuring to process clusters one at a time and updating
        // a contamination boundary that constrains future left traversals.

        // Convert clusters to scored contamination entries
        for cluster in clusters {
            let mut cluster_results = Vec::new();

            for (doc_id, matched_ngram_ids) in &cluster.document_matches {
                // Evaluate if this document match represents contamination
                let eval_context = ContaminationContext {
                    training_tokens: &training_tokens,
                    config,
                    index,
                    stats,
                };
                let (entry, result) = match evaluate_document_match(
                    *doc_id,
                    matched_ngram_ids,
                    &cluster,
                    line_num,
                    &eval_context,
                ) {
                    Some(evaluation) => evaluation,
                    None => continue,  // Skip if not contaminated or below threshold
                };

                // Finalize the contaminated entry with all details
                let final_context = ContaminationContext {
                    training_tokens: &training_tokens,
                    config,
                    index,
                    stats,
                };
                let finalized_entry = finalize_contaminated_entry(
                    entry,
                    &result,
                    &cleaned_text,
                    *doc_id,
                    &final_context,
                );

                // Check for duplicate contamination based on eval_key, eval_instance_index, and question_start_idx
                let contamination_key = (
                    finalized_entry.eval_key.clone(),
                    finalized_entry.eval_instance_index,
                    finalized_entry.contamination_start_idx.unwrap_or(0),
                );

                if processed_contaminations.insert(contamination_key) {
                    // This is a new contamination, add it
                    cluster_results.push(finalized_entry);
                }
                // If insert returned false, this is a duplicate and we skip it
            }

            if !cluster_results.is_empty() {
                contamination_results
                    .entry(file_name.clone())
                    .or_default()
                    .extend(cluster_results);
            }
        }
    }

    Ok(lines_processed)
}

/// Prepare a line for processing: parse JSON, extract text, clean, and tokenize
fn prepare_line_for_processing(
    line: &str,
    line_num: usize,
    file_path: &PathBuf,
    config: &Config,
    tokenizer: &OmniTokenizer,
) -> Option<(String, Vec<usize>)> {
    let json_obj: Value = serde_json::from_str(line).ok()?;

    // Extract text from configured field
    let line_text = get_nested_json_val(&json_obj, &config.content_key.to_string()).ok()?;

    let cleaned_text = clean_text(&line_text, &config.punctuation_chars);

    // Tokenize with error recovery
    // For BPE tokenizers, add space padding. This is because most matches will not be at token index 0
    // in the document. As a result, we pad eval strings with a space, so bpe tokenization with leading
    // space chars naturally from training documents will match evals. Likewise, we pad the training document
    // in case the padded eval does align with index 0. We essentially omit any first char in document bpe issues.
    let padded_text = format!(" {}", cleaned_text);
    let Ok(training_tokens) = catch_unwind(move || tokenizer.encode(&padded_text)) else {
        eprintln!(
            "Tokenization failed on {:?} | line {:?}",
            file_path, line_num
        );
        return None;
    };

    Some((cleaned_text, training_tokens))
}

fn build_contamination_entry_from_match(
    doc_id: u32,
    matched_ngram_ids: &HashSet<u32>,
    cluster: &SimpleContaminationCluster,
    line_num: usize,
    config: &Config,
    index: &SimpleReferenceIndex,
    stats: Option<&StatsContainer>,
) -> Option<SimpleContaminationEntry> {
    // Get hot path data (used for scoring), but not contamination metadata for reporting.
    let doc_info = index.eval_documents.get(&doc_id)?;
    let (unique_ngrams, eval_token_count, question_idf_sum, answer_total_token_length, passage_total_token_length) = doc_info;

    let idf_overlap = calculate_question_idf_overlap(
        matched_ngram_ids,
        *question_idf_sum,
        &cluster.ngram_doc_frequencies,
        index.total_docs,
    );

    // short circuit sanity check, lowest question idf possible for a contamination call outcome.
    if idf_overlap.unwrap_or(0.0) < config.minimum_question_idf_threshold {
        if let Some(stats) = stats {
            stats.increment_excluded_low_idf_threshold();
        }
        return None;
    }

    // Get present document specific boundaries
    let (doc_start_idx, doc_end_idx) = cluster
        .document_boundaries
        .get(&doc_id)
        .copied()
        .expect("Document boundaries should exist for all matched documents");

    let cluster_token_length = (doc_end_idx + config.ngram_size - 1) - doc_start_idx + 1;
    let unique_matches = matched_ngram_ids.len();

    // default/blank values filled out after complete scoring.
    Some(SimpleContaminationEntry {
        training_line: line_num,
        eval_key: String::new(),
        eval_line: 0,
        eval_instance_index: 0,
        split: None,
        idf_overlap,
        contamination_start_idx: Some(doc_start_idx),
        contamination_end_idx: Some(doc_end_idx + config.ngram_size - 1),
        training_overlap_text: None,
        ngram_match_cnt: unique_matches,
        eval_unique_ngrams: *unique_ngrams,
        eval_has_answer: false,
        answer_overlap_ratio: None,
        answer_idf_overlap: None,
        answer_boundaries: None,
        answer_start_idx: None,
        answer_end_idx: None,
        eval_has_passage: false,
        passage_overlap_ratio: None,
        passage_idf_overlap: None,
        passage_start_idx: None,
        passage_end_idx: None,
        cluster_token_length: Some(cluster_token_length),
        eval_token_length: Some(*eval_token_count),
        answer_token_length: None,
        answer_total_token_count: Some(*answer_total_token_length),
        passage_total_token_count: Some(*passage_total_token_length),
        eval_question_text: None,
        eval_answer_text: None,
        eval_passage_text: None,
        fingerprint: None,
        is_correct: None,
        reference_file: None,
        contamination_score: None,
    })
}

/// Evaluate a document match to determine if it represents contamination
fn evaluate_document_match(
    doc_id: u32,
    matched_ngram_ids: &HashSet<u32>,
    cluster: &SimpleContaminationCluster,
    line_num: usize,
    context: &ContaminationContext,
) -> Option<(SimpleContaminationEntry, super::contamination_entry::ContaminationCheckResult)> {
    let entry: SimpleContaminationEntry = build_contamination_entry_from_match(
        doc_id,
        matched_ngram_ids,
        cluster,
        line_num,
        context.config,
        context.index,
        context.stats,
    )?;

    // Early exit: Check if question meets length-adjusted threshold.
    // This is our second early check in the hot path, and length thresholds only
    // increase the required threshold from the configured contamination_score_threshold
    // towards 1. So this is more strict, and our last early check before complete scoring.
    let question_idf = entry.idf_overlap.unwrap_or(0.0);
    let required_question_threshold = interpolate_threshold(
        entry.eval_token_length.unwrap_or(usize::MAX),
        context.config.perfect_match_decay_start,
        context.config.perfect_match_decay_end,
        context.config.minimum_question_idf_threshold,
    );
    if question_idf < required_question_threshold {
        return None;
    }

    // Check if this entry represents contamination using score threshold
    let result = entry.is_contaminated(
        doc_id,
        cluster,
        context.training_tokens,
        context.config,
        context.index,
        context.stats,
    );


    if result.is_contaminated {
        Some((entry, result))
    } else {
        None
    }
}

/// ContaminationEntry records that are deemed to be contaminated get modified in this method to include
/// information for detailed reporting. This includes text snippets for qualitative review, and other
/// details to display to the user. We delay filling out the contamination entry to minimize allocations
/// and lookups in the hot path, in which almost all candidate clusters considered are not actual contamination.
fn finalize_contaminated_entry(
    mut entry: SimpleContaminationEntry,
    result: &super::contamination_entry::ContaminationCheckResult,
    cleaned_text: &str,
    doc_id: u32,
    context: &ContaminationContext,
) -> SimpleContaminationEntry {
    // Load metadata for reports now that we know this is an actual contamination
    if let Some(metadata) = context.index.eval_document_metadata.get(&doc_id) {
        entry.eval_key = metadata.eval_key.clone();
        entry.eval_line = metadata.line_num;
        entry.eval_instance_index = metadata.eval_instance_index;
        entry.split = metadata.split.clone();
        entry.fingerprint = metadata.fingerprint.clone();
        entry.is_correct = metadata.is_correct;

        entry.reference_file = context.index.reference_filenames.get(metadata.reference_file_idx)
            .cloned();
    }

    let doc_start_idx = entry.contamination_start_idx.unwrap();
    let doc_end_idx = entry.contamination_end_idx.unwrap() - context.config.ngram_size + 1;
    let cluster_token_length = entry.cluster_token_length.unwrap();
    let eval_token_count = entry.eval_token_length.unwrap();

    let mut question_end_idx = doc_end_idx + context.config.ngram_size - 1;

    // Adjust question boundaries if they overlap with answer boundaries
    // This happens when the cluster is longer than the eval question
    if let Some(ref boundaries) = result.answer_boundaries
        && !boundaries.is_empty() {
            let first_answer_start = boundaries[0].0;
            // Check if answer starts before question ends AND cluster is longer than eval
            if first_answer_start <= question_end_idx && cluster_token_length > eval_token_count {
                // Truncate question to end right before answer starts
                question_end_idx = first_answer_start.saturating_sub(1);
            }
        }

    // Extract the overlapping text with both question and answer highlights
    let extract_context = OverlapExtractionContext {
        tokenizer: &context.index.tokenizer,
        context_words_before: 30,  // Fixed context before question/passage
        context_words_after: 60,  // Context words to add after answer/question
    };
    let training_overlap_text = extract_overlap_with_question_and_answer(
        cleaned_text,
        context.training_tokens,
        doc_start_idx,
        question_end_idx,
        result.passage_boundaries,
        result.answer_boundaries.clone(),
        &extract_context,
    );

    // Update the contamination_end_idx to match the adjusted question_end_idx
    entry.contamination_end_idx = Some(question_end_idx);
    entry.training_overlap_text = training_overlap_text;
    entry.contamination_score = Some(result.contamination_score);
    entry.eval_has_answer = result.eval_has_answer;
    entry.answer_overlap_ratio = result.answer_overlap_ratio;
    entry.answer_idf_overlap = result.answer_idf_overlap;
    entry.answer_token_length = result.answer_token_length;
    entry.answer_boundaries = result.answer_boundaries.clone();

    if let Some(ref boundaries) = result.answer_boundaries
        && !boundaries.is_empty() {
            entry.answer_start_idx = Some(boundaries[0].0);
            entry.answer_end_idx = Some(boundaries[boundaries.len()-1].1);
        }

    entry.eval_has_passage = result.eval_has_passage;
    entry.passage_overlap_ratio = result.passage_overlap_ratio;
    entry.passage_idf_overlap = result.passage_idf_overlap;
    if let Some((start, end)) = result.passage_boundaries {
        entry.passage_start_idx = Some(start);
        entry.passage_end_idx = Some(end);
    }

    // Question text from eval_text_snippets
    entry.eval_question_text = context.index.eval_text_snippets
        .get(&doc_id)
        .cloned();

    // Answer text from eval_answer_text_snippets
    entry.eval_answer_text = context.index.eval_answer_text_snippets
        .get(&doc_id)
        .cloned();

    // Passage text from eval_passage_text_snippets
    entry.eval_passage_text = context.index.eval_passage_text_snippets
        .get(&doc_id)
        .cloned();

    entry
}

/// Identify contamination clusters by sampling n-grams and expanding matches.
///
/// Contamination "clusters" are not necessarily contamination until they have been
/// scored. They are just collections of matched eval documents, along with information
/// for scoring, like the ids of matched tokens and eval idf information etc.
///
/// LIMITATION: The current approach has a known blind spot. When we expand a cluster from
/// an initial seed position, we only track documents that matched at that seed. When we begin
/// sampling again, we start from the rightmost index of the cluster expansion.
///
/// This means it is possible that we could, in an edge case, have a cluster that will not score
/// as contamination, but have a different eval that would have matched with a seed n-gram within
/// the cluster boundaries. Then we would miss contamination that overlaps with the cluster
/// but doesn't share the seed n-gram.
///
/// Example scenario where Document D is missed:
/// ```text
/// Position:    10  15  20  25  30  35  40  45
/// Doc A:       [===match===]
/// Doc B:       [======match======]
/// Doc C:       [=========match=========]         <- Cluster end, sampling resumes here
/// Doc D:                   [===match===]         <- Never detected!
/// Proximate cluster:
/// Doc E:                       [=====match====]  <- matches because expansion moves left from next seed
/// Initial hit at position 10: Matches A, B, C
/// Cluster expansion: Tracks only A, B, C
/// Cluster ends at position ~40
/// Next sample at position 41+: Misses D entirely
/// ```
///
/// This is a trade-off for efficiency - checking for new documents at every position during
/// expansion would be more thorough but significantly slower. We note that because cluster
/// expansion is bi-directional, this is somewhat mitigated, as the edge case must be a cluster that
/// fits completely within the right most boundary of a cluster or has no n-gram matches following
/// the cluster yet would score as contamination, and is essentially never observed in practice.
/// Nonetheless, we note this in the case that anyone returns to go for full exhaustive decontamation,
/// in which case we would essentially want to change this so that the cluster_end is reported as
/// the right most index of an entry that scores as actual contamination, in addition to setting sampling
/// rate to 1.
fn identify_contamination_clusters(training_tokens: &[usize], config: &Config, index: &SimpleReferenceIndex, stats: Option<&StatsContainer>) -> Result<Vec<SimpleContaminationCluster>, Error> {
    let mut clusters = Vec::new();

    // Early return for documents too short to form proper n-grams
    if training_tokens.len() < config.ngram_size {
        return Ok(clusters);
    }

    let total_ngrams = training_tokens.len() - config.ngram_size + 1;
    let mut i = 0;

    while i < total_ngrams {
        let document_ids = check_ngram_for_match(i, training_tokens, config, &index.question_ngram_to_id, &index.question_ngram_id_to_eval_doc_ids);

        if document_ids.is_none() {
            i += config.sample_every_m_tokens; // No match - advance by sampling interval
            continue;
        }

        let document_ids = document_ids.expect("Document IDs should exist after check_ngram_for_match returned Some");

        // Skip if this is a hot bucket (contains only the sentinel ID)
        // Hot buckets are very common n-grams, and a single presence is unlikely to be actual contamination, just an occurrence of
        // a common n-gram. We sample the next index, essentially stopping our normal n-gram sampling until we find a not-hot bucket
        // which would be reasonable to start a new cluster expansion on.
        if document_ids.len() == 1 && document_ids.contains(&index.hot_bucket_id) {
            i += 1;  // Move by 1 to continue scanning
            continue;
        }

        let ngram_tokens = training_tokens[i..i + config.ngram_size].to_vec();

        // Expand around this hit using intersection-based walking
        let context = ExpansionContext {
            question_ngram_to_id: &index.question_ngram_to_id,
            question_ngram_id_to_eval_doc_ids: &index.question_ngram_id_to_eval_doc_ids,
            ngram_size: config.ngram_size,
            question_max_consecutive_misses: config.question_max_consecutive_misses as u32,
            hot_bucket_id: index.hot_bucket_id,
            question_ngram_id_to_hot_bucket_doc_ids: &index.question_ngram_id_to_hot_bucket_doc_ids,
        };

        if let Some(cluster) = expand_simple_contamination_cluster(
            i,
            training_tokens,
            &context,
            document_ids,
            &ngram_tokens,
            stats,
        ) {
            let cluster_end = cluster.end_idx;
            clusters.push(cluster);
            i = i.max(cluster_end + 1); // start sampling again at the index after the cluster. See method level discussion for implications.
        } else {
            // If expansion failed (not enough tokens), just advance by sampling interval
            i += config.sample_every_m_tokens;
        }
    }

    Ok(clusters)
}

/// Check a single n-gram for matches, return document IDs that match
fn check_ngram_for_match(
    ngram_idx: usize,
    training_tokens: &[usize],
    config: &Config,
    question_ngram_to_id: &QuestionNgramToIdMap,
    question_ngram_id_to_eval_doc_ids: &QuestionNgramIdToEvalDocIdsMap,
) -> Option<HashSet<u32>> {
    // Extract n-gram
    let ngram_tokens = training_tokens[ngram_idx..ngram_idx + config.ngram_size].to_vec();

    // Check if this n-gram exists in reference index
    let ngram_hash = hash_ngram(&ngram_tokens);
    if let Some(ngram_id) = question_ngram_to_id.get(&ngram_hash) {
        // Found a match, get the document IDs
        if let Some(doc_ids) = question_ngram_id_to_eval_doc_ids.get(ngram_id) {
            return Some(doc_ids.clone());
        }
    }

    None
}

/// Helper function to decode a range of tokens from u32 to usize
fn decode_token_range(
    inner: &tiktoken_rs::CoreBPE,
    tokens: &[u32],
    start: usize,
    end: usize,
) -> Option<String> {
    let tokens_usize: Vec<usize> = tokens[start..end]
        .iter()
        .map(|&t| t as usize)
        .collect();
    inner.decode(tokens_usize).ok()
}

/// Append prefix context (text before the first highlight)
fn append_prefix_context(
    result: &mut String,
    inner: &tiktoken_rs::CoreBPE,
    tokens: &[u32],
    context_start: usize,
    question_start: usize,
    passage_boundaries: Option<(usize, usize)>,
) {
    // Determine where the first highlight starts (passage or question, whichever comes first)
    let first_highlight_start = passage_boundaries
        .map(|(passage_start, _)| passage_start.min(question_start))
        .unwrap_or(question_start);

    // Add prefix if there's text before the first highlight
    if context_start < first_highlight_start
        && let Some(prefix) = decode_token_range(inner, tokens, context_start, first_highlight_start) {
            result.push_str("... ");
            result.push_str(&prefix);
        }
}

/// Append passage with highlighting (if it exists and comes before question)
fn append_passage_with_highlight(
    result: &mut String,
    inner: &tiktoken_rs::CoreBPE,
    tokens: &[u32],
    passage_boundaries: Option<(usize, usize)>,
    question_start: usize,
) {
    if let Some((passage_start, passage_end)) = passage_boundaries {
        // Only add passage if it comes before the question
        if passage_end < question_start {
            // Add passage with highlighting
            if let Some(passage_text) = decode_token_range(inner, tokens, passage_start, passage_end + 1) {
                result.push('⟨');
                result.push_str(&passage_text);
                result.push('⟩');
            }

            // Add gap between passage and question if there is one
            if passage_end + 1 < question_start
                && let Some(gap_text) = decode_token_range(inner, tokens, passage_end + 1, question_start) {
                    result.push_str(&gap_text);
                }
        }
    }
}

fn append_question_with_highlight(
    result: &mut String,
    inner: &tiktoken_rs::CoreBPE,
    tokens: &[u32],
    question_start: usize,
    question_end: usize,
) {
    if let Some(question_text) = decode_token_range(inner, tokens, question_start, question_end + 1) {
        result.push('【');
        result.push_str(&question_text);
        result.push('】');
    }
}

/// Append answers with highlighting and handle trailing context
fn append_answers_with_highlights_and_trailing_context(
    result: &mut String,
    inner: &tiktoken_rs::CoreBPE,
    tokens: &[u32],
    answer_boundaries: &Option<Vec<(usize, usize)>>,
    question_end: usize,
    context_end: usize,
) {
    // Determine the last position before trailing context
    let last_position_before_suffix = if let Some(boundaries) = answer_boundaries.as_ref() {
        if !boundaries.is_empty() {
            let mut last_pos = question_end;

            // Process each answer with gaps
            for &(answer_start, answer_end) in boundaries {
                // Add gap between last position and this answer
                if answer_start > last_pos + 1
                    && let Some(gap_text) = decode_token_range(inner, tokens, last_pos + 1, answer_start) {
                        result.push_str(&gap_text);
                    }

                // Add answer with highlighting
                if let Some(answer_text) = decode_token_range(inner, tokens, answer_start, answer_end + 1) {
                    result.push('⟦');
                    result.push_str(&answer_text);
                    result.push('⟧');
                }

                last_pos = answer_end;
            }

            last_pos
        } else {
            // Empty boundaries list
            question_end
        }
    } else {
        // No answer boundaries at all
        question_end
    };

    // Add trailing context after the last element (question or last answer)
    if last_position_before_suffix < context_end
        && let Some(suffix) = decode_token_range(inner, tokens, last_position_before_suffix + 1, context_end + 1) {
            result.push(' ');
            result.push_str(&suffix);
            result.push_str(" ...");
        }
}

/// Extract overlapping text with passage, question and answer highlights
fn extract_overlap_with_question_and_answer(
    _cleaned_text: &str,
    tokens: &[usize],
    question_start: usize,
    question_end: usize,
    passage_boundaries: Option<(usize, usize)>,
    answer_boundaries: Option<Vec<(usize, usize)>>,
    context: &OverlapExtractionContext,
) -> Option<String> {
    // Only use BPE tokenizer path if we have an inner tokenizer (p50k, cl100k)
    // uniseg and char don't have inner tokenizers and can't be decoded
    if context.tokenizer.inner.is_some() {
        let tokens_u32: Vec<u32> = tokens.iter().map(|&t| t as u32).collect();
        return extract_overlap_for_bpe_tokenizer(
            &tokens_u32,
            question_start,
            question_end,
            &answer_boundaries,
            passage_boundaries,
            context,
        );
    }

    None
}

fn extract_overlap_for_bpe_tokenizer(
    tokens: &[u32],
    question_start: usize,
    question_end: usize,
    answer_boundaries: &Option<Vec<(usize, usize)>>,
    passage_boundaries: Option<(usize, usize)>,
    context: &OverlapExtractionContext,
) -> Option<String> {
    let inner = context.tokenizer.inner.as_ref()?;

    let context_start = question_start.saturating_sub(context.context_words_before);

    // Calculate context end: use last answer end if available, otherwise question end
    let base_end = answer_boundaries
        .as_ref()
        .and_then(|boundaries| boundaries.last())
        .map(|&(_, end)| end)
        .unwrap_or(question_end);

    let context_end = (base_end + context.context_words_after).min(tokens.len() - 1);

    let mut result = String::new();

    append_prefix_context(&mut result, inner, tokens, context_start, question_start, passage_boundaries);
    append_passage_with_highlight(&mut result, inner, tokens, passage_boundaries, question_start);
    append_question_with_highlight(&mut result, inner, tokens, question_start, question_end);
    append_answers_with_highlights_and_trailing_context(&mut result, inner, tokens, answer_boundaries, question_end, context_end);

    if !result.is_empty() {
        return Some(result);
    }

    None
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hash_ngram_consistency() {
        let tokens1 = vec![1, 2, 3];
        let tokens2 = vec![1, 2, 3];
        let tokens3 = vec![3, 2, 1];

        // Same tokens should produce same hash
        assert_eq!(hash_ngram(&tokens1), hash_ngram(&tokens2));

        // Different tokens should produce different hash
        assert_ne!(hash_ngram(&tokens1), hash_ngram(&tokens3));
    }

    #[test]
    fn test_ngram_maps_basic() {
        // Create test data structures
        let mut question_ngram_to_id = QuestionNgramToIdMap::new();
        let mut question_ngram_id_to_eval_doc_ids = QuestionNgramIdToEvalDocIdsMap::new();

        // Create a test n-gram hash
        let test_tokens = vec![1, 2, 3];
        let test_hash = hash_ngram(&test_tokens);

        // Add to maps
        question_ngram_to_id.insert(test_hash, 100);
        let mut doc_ids = HashSet::new();
        doc_ids.insert(1);
        doc_ids.insert(2);
        doc_ids.insert(3);
        question_ngram_id_to_eval_doc_ids.insert(100, doc_ids.clone());

        // Test that we can find the n-gram
        let ngram_id = question_ngram_to_id.get(&test_hash);
        assert!(ngram_id.is_some());
        assert_eq!(*ngram_id.unwrap(), 100);

        // Test that we can get documents for this n-gram
        let docs = question_ngram_id_to_eval_doc_ids.get(&100);
        assert!(docs.is_some());
        assert_eq!(docs.unwrap().len(), 3);
    }
}
