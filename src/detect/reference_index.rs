use anyhow::{Error, Result};
use dashmap::DashMap;
use rayon::prelude::*;
use serde_json::Value;
use std::collections::{HashMap, HashSet};
use std::io::BufRead;
use std::panic::catch_unwind;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU32, AtomicUsize, Ordering};
use std::sync::Arc;

// Struct to hold parsed reference entry data
#[derive(Clone)]
struct ReferenceEntry {
    doc_id: u32,
    eval_key: String,
    eval_instance_index: usize,
    has_answer_fields: bool,
    fingerprint: Option<String>,
    is_correct: Option<bool>,
    split: Option<String>,
    question_text: String,
    answer_text: String,
    passage_text: String,
}

// Struct to hold evaluation document metadata (cold path fields)
#[derive(Clone)]
#[allow(dead_code)]
pub struct EvalDocumentMetadataEntry {
    pub eval_key: String,              // The evaluation dataset key
    pub line_num: usize,               // Line number in reference file
    pub total_ngrams: usize,           // Total number of n-grams
    pub fingerprint: Option<String>,   // Optional fingerprint for deduplication
    pub is_correct: Option<bool>,      // Whether the answer is correct
    pub eval_instance_index: usize,    // Instance index in the evaluation
    pub split: Option<String>,         // Dataset split (train/test/etc)
    pub reference_file_idx: usize,     // Index of the reference file
}

/// Context for processing reference entries
struct ProcessingContext<'a> {
    config: &'a Config,
    builder: &'a ReferenceIndexBuilder,
    lines_processed: &'a mut usize,
    skipped_entries: &'a mut usize,
    file_idx: usize,
}

use mj_io::{expand_dirs, read_pathbuf_to_mem};
use indicatif::ProgressBar;
use crate::common::{clean_text, OmniTokenizer};
use crate::common::Config;
use crate::detect::hot_bucket_stats::HotBucketStats;

/// Statistics from building the reference index
pub struct IndexBuildingStats {
    pub files_processed: usize,
    pub total_lines_examined: usize,
    pub lines_indexed: usize,
    pub skipped_duplicates: usize,
    pub skipped_min_tokens: usize,
    pub skipped_min_unique_words: usize,
    pub hot_buckets_replaced: usize,
}

// Type aliases for reference index data structures
pub(crate) type QuestionNgramToIdMap = HashMap<u64, u32>; // Lightweight map to resolve ids, acts a fast filter to skip n-grams not present in eval prompts for main hot path
pub(crate) type QuestionNgramIdToEvalDocIdsMap = HashMap<u32, HashSet<u32>>; // How we resolve documents matching a given n-gram for set operations
pub(crate) type EvalDocIdToAnswerTokensMap = HashMap<u32, HashSet<usize>>; // Used for fast membership testing, set operations, and overlap calculations
pub(crate) type EvalDocIdToAnswerTokensOrderedMap = HashMap<u32, Vec<usize>>; //  Used for exact sequence matching of short answers
pub(crate) type EvalDocIdToAnswerNgramIdsMap = HashMap<u32, HashSet<u32>>; // Used for n-gram matching of long answers (>threshold tokens)
pub(crate) type AnswerNgramIdfMap = HashMap<u32, f32>; // Pre-computed IDF for scoring long answer n-gram matches
pub(crate) type EvalDocIdToPassageTokensMap = HashMap<u32, HashSet<usize>>; // Used for token filtering and overlap ratio calculation in passage detection
pub(crate) type EvalDocIdToPassageNgramIdsMap = HashMap<u32, HashSet<u32>>; // Used for n-gram matching in passage detection
pub(crate) type PassageNgramIdfMap = HashMap<u32, f32>; // Pre-computed IDF for scoring passage n-gram matches
pub(crate) type EvalPassageIdfCache = HashMap<u32, f32>; // Pre-summed passage IDF to avoid recomputation during detection
pub(crate) type EvalDocuments = HashMap<u32, (usize, usize, f32, usize, usize)>; // Maps doc_id to (unique_ngrams, token_count, question_idf_sum, answer_token_length, passage_token_length) - hot path fields
pub(crate) type EvalDocumentMetadata = HashMap<u32, EvalDocumentMetadataEntry>; // Maps doc_id to metadata - cold path fields
pub(crate) type EvalTextSnippets = HashMap<u32, String>; // Maps doc_id to question text snippet (first 1000 words)
pub(crate) type EvalAnswerTextSnippets = HashMap<u32, String>; // First 1000 words of answer for contamination reports
pub(crate) type EvalPassageTextSnippets = HashMap<u32, String>; // First 1000 words of passage for contamination reports
pub(crate) type EvalDocIdToQuestionNgramIdsMap = HashMap<u32, HashSet<u32>>; // Enables per-document question IDF calculation during index build
pub(crate) type AnswerTokenIdfMap = HashMap<usize, f32>; // Maps answer token ID to IDF value for short answer scoring

/// Builder struct to hold all the data structures needed for building the reference index
/// This reduces the number of parameters passed to process_simple_reference_file
pub(crate) struct ReferenceIndexBuilder {
    pub question_ngram_to_id: DashMap<u64, u32>,
    pub question_ngram_id_to_eval_doc_ids: DashMap<u32, HashSet<u32>>,
    pub eval_doc_id_to_answer_tokens: DashMap<u32, HashSet<usize>>,
    pub eval_doc_id_to_answer_tokens_ordered: DashMap<u32, Vec<usize>>,
    pub eval_doc_id_to_answer_ngram_ids: DashMap<u32, HashSet<u32>>,
    pub answer_ngram_doc_freq: DashMap<u32, usize>,
    pub answer_ngram_idf: DashMap<u32, f32>,
    pub eval_doc_id_to_passage_tokens: DashMap<u32, HashSet<usize>>,
    pub eval_doc_id_to_passage_ngram_ids: DashMap<u32, HashSet<u32>>,
    pub passage_ngram_doc_freq: DashMap<u32, usize>,
    pub passage_ngram_idf: DashMap<u32, f32>,
    pub eval_passage_idf_cache: DashMap<u32, f32>,
    pub eval_documents: DashMap<u32, (usize, usize, f32, usize, usize)>,
    pub eval_document_metadata: DashMap<u32, EvalDocumentMetadataEntry>,
    pub eval_text_snippets: DashMap<u32, String>,
    pub eval_answer_text_snippets: DashMap<u32, String>,
    pub eval_passage_text_snippets: DashMap<u32, String>,
    pub eval_doc_id_to_question_ngram_ids: DashMap<u32, HashSet<u32>>,
    pub token_eval_doc_freq: DashMap<usize, AtomicUsize>,
    pub unique_eval_suites: DashMap<String, ()>,
    pub next_ngram_id: AtomicU32,
    pub max_doc_id: AtomicU32,
    pub tokenizer: OmniTokenizer,
    pub dedup_map: Option<DashMap<(String, String), usize>>,
    pub total_skipped_duplicates: Arc<AtomicUsize>,
    pub total_skipped_min_tokens: Arc<AtomicUsize>,
    pub total_skipped_min_unique_words: Arc<AtomicUsize>,
    pub total_lines_processed: Arc<AtomicUsize>,
}

impl ReferenceIndexBuilder {
    /// Create a new ReferenceIndexBuilder with initialized data structures
    pub fn new(config: &Config) -> Result<Self> {
        Ok(Self {
            question_ngram_to_id: DashMap::new(),
            question_ngram_id_to_eval_doc_ids: DashMap::new(),
            eval_doc_id_to_answer_tokens: DashMap::new(),
            eval_doc_id_to_answer_tokens_ordered: DashMap::new(),
            eval_doc_id_to_answer_ngram_ids: DashMap::new(),
            answer_ngram_doc_freq: DashMap::new(),
            answer_ngram_idf: DashMap::new(),
            eval_doc_id_to_passage_tokens: DashMap::new(),
            eval_doc_id_to_passage_ngram_ids: DashMap::new(),
            passage_ngram_doc_freq: DashMap::new(),
            passage_ngram_idf: DashMap::new(),
            eval_passage_idf_cache: DashMap::new(),
            eval_documents: DashMap::new(),
            eval_document_metadata: DashMap::new(),
            eval_text_snippets: DashMap::new(),
            eval_answer_text_snippets: DashMap::new(),
            eval_passage_text_snippets: DashMap::new(),
            eval_doc_id_to_question_ngram_ids: DashMap::new(),
            token_eval_doc_freq: DashMap::new(),
            unique_eval_suites: DashMap::new(),
            next_ngram_id: AtomicU32::new(0),
            max_doc_id: AtomicU32::new(0),
            tokenizer: OmniTokenizer::new(&config.tokenizer_str)?,
            dedup_map: config.eval_dedup.then(DashMap::new),
            total_skipped_duplicates: Arc::new(AtomicUsize::new(0)),
            total_skipped_min_tokens: Arc::new(AtomicUsize::new(0)),
            total_skipped_min_unique_words: Arc::new(AtomicUsize::new(0)),
            total_lines_processed: Arc::new(AtomicUsize::new(0)),
        })
    }
}

/// Encapsulates all reference index data for contamination detection
pub struct SimpleReferenceIndex {
    pub question_ngram_to_id: Arc<QuestionNgramToIdMap>,
    pub question_ngram_id_to_eval_doc_ids: Arc<QuestionNgramIdToEvalDocIdsMap>,
    pub eval_doc_id_to_answer_tokens: Arc<EvalDocIdToAnswerTokensMap>,
    pub eval_doc_id_to_answer_tokens_ordered: Arc<EvalDocIdToAnswerTokensOrderedMap>,
    pub eval_doc_id_to_answer_ngram_ids: Arc<EvalDocIdToAnswerNgramIdsMap>,
    pub answer_ngram_idf: Arc<AnswerNgramIdfMap>,
    pub eval_doc_id_to_passage_tokens: Arc<EvalDocIdToPassageTokensMap>,
    pub eval_doc_id_to_passage_ngram_ids: Arc<EvalDocIdToPassageNgramIdsMap>,
    pub passage_ngram_idf: Arc<PassageNgramIdfMap>,
    pub eval_passage_idf_cache: Arc<EvalPassageIdfCache>,
    pub eval_documents: Arc<EvalDocuments>,
    pub eval_document_metadata: Arc<EvalDocumentMetadata>,
    pub tokenizer: Arc<OmniTokenizer>,
    pub eval_text_snippets: Arc<EvalTextSnippets>,
    pub eval_answer_text_snippets: Arc<EvalAnswerTextSnippets>,
    pub eval_passage_text_snippets: Arc<EvalPassageTextSnippets>,
    pub eval_doc_id_to_question_ngram_ids: Arc<EvalDocIdToQuestionNgramIdsMap>,
    pub answer_token_idf: Arc<AnswerTokenIdfMap>,
    pub reference_filenames: Arc<Vec<String>>,
    pub unique_eval_suites: Arc<HashSet<String>>,
    pub total_docs: f32,
    pub hot_bucket_stats: Option<Arc<HotBucketStats>>,
    pub hot_bucket_id: u32,
    pub question_ngram_id_to_hot_bucket_doc_ids: Arc<HashMap<u32, HashSet<u32>>>,
}

impl SimpleReferenceIndex {
    /// Create a new SimpleReferenceIndex from individual components
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        question_ngram_to_id: QuestionNgramToIdMap,
        question_ngram_id_to_eval_doc_ids: QuestionNgramIdToEvalDocIdsMap,
        eval_doc_id_to_answer_tokens: EvalDocIdToAnswerTokensMap,
        eval_doc_id_to_answer_tokens_ordered: EvalDocIdToAnswerTokensOrderedMap,
        eval_doc_id_to_answer_ngram_ids: EvalDocIdToAnswerNgramIdsMap,
        answer_ngram_idf: AnswerNgramIdfMap,
        eval_doc_id_to_passage_tokens: EvalDocIdToPassageTokensMap,
        eval_doc_id_to_passage_ngram_ids: EvalDocIdToPassageNgramIdsMap,
        passage_ngram_idf: PassageNgramIdfMap,
        eval_passage_idf_cache: EvalPassageIdfCache,
        eval_documents: EvalDocuments,
        eval_document_metadata: EvalDocumentMetadata,
        tokenizer: OmniTokenizer,
        eval_text_snippets: EvalTextSnippets,
        eval_answer_text_snippets: EvalAnswerTextSnippets,
        eval_passage_text_snippets: EvalPassageTextSnippets,
        eval_doc_id_to_question_ngram_ids: EvalDocIdToQuestionNgramIdsMap,
        answer_token_idf: AnswerTokenIdfMap,
        reference_filenames: Vec<String>,
        unique_eval_suites: HashSet<String>,
        hot_bucket_stats: Option<HotBucketStats>,
        hot_bucket_id: u32,
        question_ngram_id_to_hot_bucket_doc_ids: HashMap<u32, HashSet<u32>>,
    ) -> Self {
        let total_docs = eval_documents.len() as f32;

        Self {
            question_ngram_to_id: Arc::new(question_ngram_to_id),
            question_ngram_id_to_eval_doc_ids: Arc::new(question_ngram_id_to_eval_doc_ids),
            eval_doc_id_to_answer_tokens: Arc::new(eval_doc_id_to_answer_tokens),
            eval_doc_id_to_answer_tokens_ordered: Arc::new(eval_doc_id_to_answer_tokens_ordered),
            eval_doc_id_to_answer_ngram_ids: Arc::new(eval_doc_id_to_answer_ngram_ids),
            answer_ngram_idf: Arc::new(answer_ngram_idf),
            eval_doc_id_to_passage_tokens: Arc::new(eval_doc_id_to_passage_tokens),
            eval_doc_id_to_passage_ngram_ids: Arc::new(eval_doc_id_to_passage_ngram_ids),
            passage_ngram_idf: Arc::new(passage_ngram_idf),
            eval_passage_idf_cache: Arc::new(eval_passage_idf_cache),
            eval_documents: Arc::new(eval_documents),
            eval_document_metadata: Arc::new(eval_document_metadata),
            tokenizer: Arc::new(tokenizer),
            eval_text_snippets: Arc::new(eval_text_snippets),
            eval_answer_text_snippets: Arc::new(eval_answer_text_snippets),
            eval_passage_text_snippets: Arc::new(eval_passage_text_snippets),
            eval_doc_id_to_question_ngram_ids: Arc::new(eval_doc_id_to_question_ngram_ids),
            answer_token_idf: Arc::new(answer_token_idf),
            reference_filenames: Arc::new(reference_filenames),
            unique_eval_suites: Arc::new(unique_eval_suites),
            total_docs,
            hot_bucket_stats: hot_bucket_stats.map(Arc::new),
            hot_bucket_id,
            question_ngram_id_to_hot_bucket_doc_ids: Arc::new(question_ngram_id_to_hot_bucket_doc_ids),
        }
    }
}

// Helper function to create progress bar only if not in quiet mode
fn build_pbar_quiet(len: usize, msg: &str) -> ProgressBar {
    if crate::common::is_quiet_mode() {
        // Return a hidden progress bar that doesn't display anything
        ProgressBar::hidden()
    } else {
        // Create a normal progress bar using mj_io's build_pbar
        mj_io::build_pbar(len, msg)
    }
}

// Helper function to get and validate reference files
fn get_reference_files(evals_dir: &Path) -> Result<(Vec<PathBuf>, Vec<String>), Error> {
    // Find all reference files
    let mut reference_files = expand_dirs(
        vec![evals_dir.to_path_buf()],
        Some(crate::common::EXPAND_DIRS_EXTENSIONS),
    )?;

    // Sort reference files for deterministic processing
    reference_files.sort();

    // Check if any reference files were found and verify they exist
    let existing_files: Vec<PathBuf> = reference_files
        .into_iter()
        .filter(|path| path.exists())
        .collect();

    if existing_files.is_empty() {
        return Err(anyhow::anyhow!(
            "\nReference files not found at {}. Please run 'decon evals --download' to download evaluation datasets.",
            evals_dir.display()
        ));
    }

    // Build the vector of reference filenames
    let reference_filenames: Vec<String> = existing_files
        .iter()
        .map(|path| {
            path.file_name()
                .and_then(|s| s.to_str())
                .unwrap_or("unknown")
                .to_string()
        })
        .collect();

    Ok((existing_files, reference_filenames))
}


pub fn build_simple_index(config: &Config) -> Result<(SimpleReferenceIndex, IndexBuildingStats), Error> {
    // Create the builder with all initialized data structures
    let builder = ReferenceIndexBuilder::new(config)?;

    // Get and validate reference files
    let (reference_files, reference_filenames) = get_reference_files(&config.evals_dir)?;

    let pbar = build_pbar_quiet(reference_files.len(), "Reference files");

    reference_files.par_iter().enumerate().for_each(|(file_idx, file_path)| {
        if let Err(e) = process_simple_reference_file(
            file_path,
            file_idx,
            config,
            &builder,
        ) {
            println!("Error processing reference file {:?}: {:?}", file_path, e);
        }
        pbar.inc(1);
    });

    pbar.finish_with_message("Index building complete");

    // Collect statistics for return
    let mut stats = IndexBuildingStats {
        files_processed: reference_files.len(),
        total_lines_examined: builder.total_lines_processed.load(Ordering::Relaxed) +
            builder.total_skipped_duplicates.load(Ordering::Relaxed) +
            builder.total_skipped_min_tokens.load(Ordering::Relaxed) +
            builder.total_skipped_min_unique_words.load(Ordering::Relaxed),
        lines_indexed: builder.total_lines_processed.load(Ordering::Relaxed),
        skipped_duplicates: builder.total_skipped_duplicates.load(Ordering::Relaxed),
        skipped_min_tokens: builder.total_skipped_min_tokens.load(Ordering::Relaxed),
        skipped_min_unique_words: builder.total_skipped_min_unique_words.load(Ordering::Relaxed),
        hot_buckets_replaced: 0,
    };

    // Convert DashMaps to HashMaps for read-only access during detection
    let question_ngram_to_id: HashMap<u64, u32> = builder.question_ngram_to_id.into_iter().collect();
    let mut question_ngram_id_to_eval_doc_ids: HashMap<u32, HashSet<u32>> =
        builder.question_ngram_id_to_eval_doc_ids.into_iter().collect();
    let eval_doc_id_to_answer_tokens: HashMap<u32, HashSet<usize>> =
        builder.eval_doc_id_to_answer_tokens.into_iter().collect();
    let eval_doc_id_to_answer_tokens_ordered: HashMap<u32, Vec<usize>> =
        builder.eval_doc_id_to_answer_tokens_ordered.into_iter().collect();
    let eval_doc_id_to_answer_ngram_ids: HashMap<u32, HashSet<u32>> =
        builder.eval_doc_id_to_answer_ngram_ids.into_iter().collect();
    let eval_doc_id_to_passage_tokens: HashMap<u32, HashSet<usize>> =
        builder.eval_doc_id_to_passage_tokens.into_iter().collect();
    let eval_doc_id_to_passage_ngram_ids: HashMap<u32, HashSet<u32>> =
        builder.eval_doc_id_to_passage_ngram_ids.into_iter().collect();
    let mut eval_documents: HashMap<u32, (usize, usize, f32, usize, usize)> =
        builder.eval_documents.into_iter().collect();
    let eval_document_metadata: HashMap<u32, EvalDocumentMetadataEntry> =
        builder.eval_document_metadata.into_iter().collect();
    let eval_text_snippets: HashMap<u32, String> =
        builder.eval_text_snippets.into_iter().collect();
    let eval_answer_text_snippets: HashMap<u32, String> =
        builder.eval_answer_text_snippets.into_iter().collect();
    let eval_passage_text_snippets: HashMap<u32, String> =
        builder.eval_passage_text_snippets.into_iter().collect();
    let eval_doc_id_to_question_ngram_ids: HashMap<u32, HashSet<u32>> =
        builder.eval_doc_id_to_question_ngram_ids.into_iter().collect();
    let token_eval_doc_freq: HashMap<usize, AtomicUsize> =
        builder.token_eval_doc_freq.into_iter().collect();
    let unique_eval_suites: HashSet<String> = builder.unique_eval_suites.into_iter()
        .map(|(key, _)| key)
        .collect();

    // Store total docs for IDF calculations
    let total_docs = eval_documents.len() as f32;

    // Collect all unique tokens that appear in any answer
    let mut answer_tokens = HashSet::new();
    for token_set in eval_doc_id_to_answer_tokens.values() {
        answer_tokens.extend(token_set.iter().copied());
    }

    // Compute IDF only for answer tokens
    let mut answer_token_idf = HashMap::new();
    for token in answer_tokens {
        if let Some(doc_freq) = token_eval_doc_freq.get(&token) {
            let freq = doc_freq.load(Ordering::Relaxed) as f32;
            if freq > 0.0 {
                let idf = (total_docs / freq).ln();
                answer_token_idf.insert(token, idf);
            }
        }
    }

    // Calculate IDF values in parallel for answer and passage n-grams
    builder.answer_ngram_doc_freq.par_iter().for_each(|entry| {
        let ngram_id = *entry.key();
        let doc_freq = *entry.value();
        let idf = (total_docs / doc_freq as f32).ln();
        builder.answer_ngram_idf.insert(ngram_id, idf);
    });

    builder.passage_ngram_doc_freq.par_iter().for_each(|entry| {
        let ngram_id = *entry.key();
        let doc_freq = *entry.value();
        let idf = (total_docs / doc_freq as f32).ln();
        builder.passage_ngram_idf.insert(ngram_id, idf);
    });

    // Convert IDF DashMaps to HashMaps
    let answer_ngram_idf: HashMap<u32, f32> = builder.answer_ngram_idf.into_iter().collect();
    let passage_ngram_idf: HashMap<u32, f32> = builder.passage_ngram_idf.into_iter().collect();

    // Pre-compute question IDF sums and update them in eval_documents
    // We do this by computing IDF for each document's question ngrams
    eval_documents.par_iter_mut().for_each(|(doc_id, doc_tuple)| {
        if let Some(ngram_ids) = eval_doc_id_to_question_ngram_ids.get(doc_id) {
            // Sort ngram_ids for deterministic iteration order
            let mut sorted_ids: Vec<u32> = ngram_ids.iter().copied().collect();
            sorted_ids.sort_unstable();

            let mut total_idf = 0.0f32;
            for ngram_id in sorted_ids {
                if let Some(doc_set) = question_ngram_id_to_eval_doc_ids.get(&ngram_id) {
                    let doc_freq = doc_set.len() as f32;
                    let idf = (total_docs / doc_freq).ln();
                    total_idf += idf;
                }
            }
            // Update the question_idf_sum field (index 2) in the tuple
            doc_tuple.2 = total_idf;
        }
    });

    // Pre-populate passage IDF cache for all passages
    // Note: We cache passage IDF totals because passages tend to be substantially longer.
    // Now that we pre-calculate individual n-gram IDFs, we just sum them here.
    eval_doc_id_to_passage_ngram_ids.par_iter().for_each(|(doc_id, ngram_ids)| {
        // Sort ngram_ids for deterministic iteration order
        let mut sorted_ids: Vec<u32> = ngram_ids.iter().copied().collect();
        sorted_ids.sort_unstable();

        let mut total_idf = 0.0f32;
        for ngram_id in sorted_ids {
            if let Some(idf_value) = passage_ngram_idf.get(&ngram_id) {
                total_idf += idf_value;
            }
        }
        builder.eval_passage_idf_cache.insert(*doc_id, total_idf);
    });

    let eval_passage_idf_cache: HashMap<u32, f32> = builder.eval_passage_idf_cache.into_iter().collect();

    // Calculate hot bucket threshold using square root scaling
    // based on total lines indexed (before filtering) with minimum of 2000
    let hot_bucket_threshold = ((stats.lines_indexed as f32).sqrt() as usize * 2).max(2000);

    // Analyze hot buckets if verbose mode is enabled
    let hot_bucket_stats = if config.verbose {
        // Collect examples of hot n-grams using the calculated threshold
        let mut hot_ngram_examples = Vec::new();

        // Find n-grams with more documents than the threshold
        let mut hot_ngrams: Vec<(u32, usize)> = question_ngram_id_to_eval_doc_ids
            .iter()
            .filter_map(|(ngram_id, doc_set)| {
                let count = doc_set.len();
                if count > hot_bucket_threshold {
                    Some((*ngram_id, count))
                } else {
                    None
                }
            })
            .collect();

        // Sort by document count (descending)
        hot_ngrams.sort_by(|a, b| b.1.cmp(&a.1));

        // Just use placeholder text since we're not displaying examples anymore
        for (_ngram_id, doc_count) in hot_ngrams.iter().take(100) {
            hot_ngram_examples.push((String::new(), *doc_count));
        }

        Some(HotBucketStats::analyze(&question_ngram_id_to_eval_doc_ids, hot_ngram_examples))
    } else {
        None
    };

    // Hot bucket optimization: Replace hot buckets with sentinel ID
    let hot_bucket_id = builder.max_doc_id.load(Ordering::Relaxed) + 1;
    let mut question_ngram_id_to_hot_bucket_doc_ids: HashMap<u32, HashSet<u32>> = HashMap::new();
    let mut hot_buckets_replaced = 0;

    for (ngram_id, doc_set) in question_ngram_id_to_eval_doc_ids.iter_mut() {
        if doc_set.len() > hot_bucket_threshold {
            // Store the original doc set in the hot bucket map
            question_ngram_id_to_hot_bucket_doc_ids.insert(*ngram_id, doc_set.clone());
            // Replace with sentinel ID
            *doc_set = HashSet::from([hot_bucket_id]);
            hot_buckets_replaced += 1;
        }
    }

    stats.hot_buckets_replaced = hot_buckets_replaced;
    if hot_buckets_replaced > 0 && config.verbose {
        eprintln!("Hot bucket optimization: Replaced {} hot buckets (>{} documents) with sentinel ID {}",
                  hot_buckets_replaced, hot_bucket_threshold, hot_bucket_id);
    }

    let index = SimpleReferenceIndex::new(
        question_ngram_to_id,
        question_ngram_id_to_eval_doc_ids,
        eval_doc_id_to_answer_tokens,
        eval_doc_id_to_answer_tokens_ordered,
        eval_doc_id_to_answer_ngram_ids,
        answer_ngram_idf,
        eval_doc_id_to_passage_tokens,
        eval_doc_id_to_passage_ngram_ids,
        passage_ngram_idf,
        eval_passage_idf_cache,
        eval_documents,
        eval_document_metadata,
        builder.tokenizer,
        eval_text_snippets,
        eval_answer_text_snippets,
        eval_passage_text_snippets,
        eval_doc_id_to_question_ngram_ids,
        answer_token_idf,
        reference_filenames,
        unique_eval_suites,
        hot_bucket_stats,
        hot_bucket_id,
        question_ngram_id_to_hot_bucket_doc_ids,
    );

    Ok((index, stats))
}

// Helper function to parse a JSON line into a ReferenceEntry
fn parse_reference_entry(
    json_obj: &Value,
    line_num: usize,
    fallback_eval_name: &str,
) -> Result<ReferenceEntry> {
    // Check if this is a Q&A dataset
    let has_answer_fields = json_obj.get("answer").is_some();

    // Read eval_key from JSON (unique identifier for the eval dataset)
    let eval_key = json_obj
        .get("eval_key")
        .and_then(|v| v.as_str())
        .or_else(|| json_obj.get("eval_name").and_then(|v| v.as_str()))
        .unwrap_or(fallback_eval_name)
        .to_string();

    // Read document ID from JSON (generated by Python download script)
    let doc_id = json_obj
        .get("doc_id")
        .and_then(|v| v.as_u64())
        .ok_or_else(|| anyhow::anyhow!("Missing or invalid doc_id field in reference file"))?
        as u32;

    // Read the actual dataset instance index
    let eval_instance_index = json_obj
        .get("eval_instance_index")
        .and_then(|v| v.as_u64())
        .map(|v| v as usize)
        .unwrap_or(line_num);

    // Read optional fields
    let fingerprint = json_obj
        .get("fingerprint")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string());

    let is_correct = json_obj
        .get("is_correct")
        .and_then(|v| v.as_bool());

    let split = json_obj
        .get("split")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string());

    // Get text fields
    let question_text = json_obj
        .get("question")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();

    let answer_text = json_obj
        .get("answer")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();

    let passage_text = json_obj
        .get("passage")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();

    Ok(ReferenceEntry {
        doc_id,
        eval_key,
        eval_instance_index,
        has_answer_fields,
        fingerprint,
        is_correct,
        split,
        question_text,
        answer_text,
        passage_text,
    })
}

fn process_simple_reference_file(
    file_path: &PathBuf,
    file_idx: usize,
    config: &Config,
    builder: &ReferenceIndexBuilder,
) -> Result<(), Error> {
    let data = read_pathbuf_to_mem(file_path)?;

    // We'll extract eval_key from each JSON record instead of using filename
    // Keep filename as fallback for debugging
    let fallback_eval_name = file_path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("unknown")
        .to_string();

    let mut _lines_processed = 0;
    let mut _skipped_entries = 0;
    let mut _skipped_min_tokens = 0;
    let mut _skipped_min_unique_words = 0;
    let mut _skipped_duplicates = 0;

    for (line_num, line) in data.lines().enumerate() {
        let line = line?;

        if line.starts_with('#') {  // Skip comment lines
            continue;
        }

        let json_obj: Value = serde_json::from_str(&line)?;
        let entry = parse_reference_entry(&json_obj, line_num, &fallback_eval_name)?;

        let combined_cleaned_text = build_combined_text_for_filtering(
            &entry.passage_text,
            &entry.question_text,
            &entry.answer_text,
            entry.has_answer_fields,
            config,
        );

        if should_skip_by_token_count(&combined_cleaned_text, config, &builder.tokenizer) {
            _skipped_entries += 1;
            _skipped_min_tokens += 1;
            builder.total_skipped_min_tokens.fetch_add(1, Ordering::Relaxed);
            continue;
        }

        if should_skip_by_unique_words(&combined_cleaned_text, config) {
            _skipped_entries += 1;
            _skipped_min_unique_words += 1;
            builder.total_skipped_min_unique_words.fetch_add(1, Ordering::Relaxed);
            continue;
        }

        if check_and_mark_duplicate(&entry, line_num, &builder.dedup_map) {
            _skipped_entries += 1;
            _skipped_duplicates += 1;
            builder.total_skipped_duplicates.fetch_add(1, Ordering::Relaxed);
            continue;
        }

        // Skip entries without questions - they can't be matched
        if entry.question_text.is_empty() {
            _skipped_entries += 1;
            continue;
        }

        // Track the maximum doc_id seen
        builder.max_doc_id.fetch_max(entry.doc_id, Ordering::Relaxed);

        // Track unique eval suite
        builder.unique_eval_suites.insert(entry.eval_key.clone(), ());

        let mut context = ProcessingContext {
            config,
            builder,
            lines_processed: &mut _lines_processed,
            skipped_entries: &mut _skipped_entries,
            file_idx,
        };

        process_question_field(&entry, line_num, &mut context)?;

        if config.index_answers && entry.has_answer_fields && !entry.answer_text.is_empty() {
            process_answer_field(
                &entry.answer_text,
                entry.doc_id,
                config,
                builder,
            )?;
        }

        if config.index_passages && !entry.passage_text.is_empty() {
            process_passage_field(
                &entry.passage_text,
                entry.doc_id,
                config,
                builder,
            )?;
        }
    }

    builder.total_lines_processed.fetch_add(_lines_processed, Ordering::Relaxed);

    Ok(())
}

pub(crate) fn hash_ngram(tokens: &[usize]) -> u64 {
    use std::hash::{DefaultHasher, Hash, Hasher};
    let mut hasher = DefaultHasher::new();
    tokens.hash(&mut hasher);
    hasher.finish()
}

// This two-step lookup design (hash → ID → documents) is intentionally optimized for modern CPU cache hierarchies.
// While it may seem redundant compared to a direct hash → documents mapping, this architecture provides
// significant performance benefits:
//
// 1. **Cache Locality**: The first HashMap (hash → ID) contains only u64 → u64 mappings (~16 bytes per entry),
//    allowing 4 entries per 64-byte cache line. For the majority of lookups (misses in training data),
//    we only touch this compact structure.
//
// 2. **Memory Efficiency**: Questions, answers, and passages share the same ID space, enabling natural
//    deduplication. Common n-grams across different fields get the same ID, reducing memory footprint.
//
// 3. **CPU-Friendly Access Patterns**: SwissTable's open addressing with large HashSet values causes
//    cache thrashing during probing. By keeping values small in the hot path, we minimize cache misses
//    even during collision resolution.
//
// Empirical testing showed this design is ~30% faster than direct mapping for contamination detection workloads
// where >99% of n-gram lookups are misses.
fn get_or_create_ngram_id(
    ngram_tokens: &[usize],
    question_ngram_to_id: &DashMap<u64, u32>,
    next_id: &AtomicU32,
) -> u32 {
    let ngram_hash = hash_ngram(ngram_tokens);

    match question_ngram_to_id.entry(ngram_hash) {
        dashmap::mapref::entry::Entry::Occupied(occupied) => {
            *occupied.get()
        }
        dashmap::mapref::entry::Entry::Vacant(vacant) => {
            let current_id = next_id.load(Ordering::SeqCst);

            // Check for overflow - leave some room before u32::MAX
            const OVERFLOW_THRESHOLD: u32 = 4_200_000_000; // ~100M before u32::MAX
            if current_id >= OVERFLOW_THRESHOLD {
                panic!(
                    "N-gram ID overflow! Exceeded {} unique n-grams. \
                    Please report this to the maintainer.",
                    current_id
                );
            }

            let new_id = next_id.fetch_add(1, Ordering::SeqCst);
            vacant.insert(new_id);
            new_id
        }
    }
}

fn add_doc_to_ngram(ngram_id: u32, doc_id: u32, question_ngram_id_to_eval_doc_ids: &DashMap<u32, HashSet<u32>>) {
    question_ngram_id_to_eval_doc_ids
        .entry(ngram_id)
        .or_default()
        .insert(doc_id);
}

// Helper function to extract tokens from cleaned text
fn extract_tokens(cleaned_text: &str, _config: &Config, tokenizer: &OmniTokenizer) -> Option<Vec<usize>> {
    if cleaned_text.is_empty() {
        return None;
    }

    // For BPE tokenizers, add padding since text fragments likely appear
    // mid-document rather than at the beginning
    let padded_text = format!(" {}", cleaned_text);
    catch_unwind(|| tokenizer.encode(&padded_text)).ok()
}

fn should_skip_by_token_count(
    combined_text: &str,
    config: &Config,
    tokenizer: &OmniTokenizer,
) -> bool {
    if config.eval_min_token_length == 0 {
        return false;
    }

    let token_count = if config.tokenizer_str == "word" {
        combined_text.split_whitespace().count()
    } else {
        // For BPE tokenizers, add padding and tokenize
        let padded_text = format!(" {}", combined_text);
        match catch_unwind(|| tokenizer.encode(&padded_text)) {
            Ok(tokens) => {
                // Count non-space tokens only
                tokens.iter().filter(|&&tok| !tokenizer.is_space_token(tok)).count()
            },
            Err(_) => 0, // Skip if tokenization fails
        }
    };

    token_count < config.eval_min_token_length
}

fn should_skip_by_unique_words(
    combined_text: &str,
    config: &Config,
) -> bool {
    if config.eval_min_unique_word_count == 0 {
        return false;
    }

    let words: HashSet<&str> = combined_text.split_whitespace().collect();
    words.len() < config.eval_min_unique_word_count
}

fn check_and_mark_duplicate(
    entry: &ReferenceEntry,
    line_num: usize,
    dedup_map: &Option<DashMap<(String, String), usize>>,
) -> bool {
    let Some(dedup_map) = dedup_map else {
        return false;
    };

    let Some(ref fp) = entry.fingerprint else {
        return false;
    };

    // Create composite key: (eval_key, fingerprint)
    let dedup_key = (entry.eval_key.clone(), fp.clone());

    match dedup_map.entry(dedup_key) {
        dashmap::mapref::entry::Entry::Occupied(_) => {
            true // Is a duplicate
        }
        dashmap::mapref::entry::Entry::Vacant(vacant) => {
            // First occurrence within this eval_key - insert atomically
            vacant.insert(line_num);
            false // Not a duplicate
        }
    }
}

// The combined clean text length is used for min length filtering.
fn build_combined_text_for_filtering(
    passage_text: &str,
    question_text: &str,
    answer_text: &str,
    has_answer_fields: bool,
    config: &Config,
) -> String {
    let cleaned_passage = clean_text(passage_text, &config.punctuation_chars);
    let cleaned_question = clean_text(question_text, &config.punctuation_chars);
    let cleaned_answer = clean_text(answer_text, &config.punctuation_chars);

    let mut parts = Vec::new();
    // Only include passage if we're indexing passages
    if config.index_passages && !cleaned_passage.is_empty() {
        parts.push(cleaned_passage.as_str());
    }
    if !cleaned_question.is_empty() {
        parts.push(cleaned_question.as_str());
    }
    // Only include answer if we're indexing answers and it exists
    if config.index_answers && has_answer_fields && !cleaned_answer.is_empty() {
        parts.push(cleaned_answer.as_str());
    }
    parts.join(" ")
}

fn process_question_field(
    entry: &ReferenceEntry,
    line_num: usize,
    context: &mut ProcessingContext,
) -> Result<(), Error> {
    let question = &entry.question_text;
    let cleaned = clean_text(question, &context.config.punctuation_chars);

    // Store eval text snippet (first 1000 words) for reports/review
    let snippet_words: Vec<&str> = cleaned.split_whitespace().take(1000).collect();
    let text_snippet = snippet_words.join(" ");
    context.builder.eval_text_snippets.insert(entry.doc_id, text_snippet);

    // Extract tokens from cleaned text
    let word_tokens = match extract_tokens(&cleaned, context.config, &context.builder.tokenizer) {
        Some(tokens) => tokens,
        None => {
            *context.skipped_entries += 1;
            return Ok(());
        }
    };
    let token_count = word_tokens.len();

    *context.lines_processed += 1;

    // Track unique tokens for this document for IDF calculation
    let unique_tokens: HashSet<usize> = word_tokens.iter().copied().collect();
    for token in &unique_tokens {
        context.builder.token_eval_doc_freq
            .entry(*token)
            .or_insert_with(|| AtomicUsize::new(0))
            .fetch_add(1, Ordering::Relaxed);
    }

    // Calculate total n-grams for this document
    let total_ngrams = if word_tokens.len() < context.config.ngram_size {
        if word_tokens.is_empty() {
            0
        } else {
            1
        }
    } else {
        word_tokens.len() - context.config.ngram_size + 1
    };

    // Track unique n-grams for this document
    let mut unique_ngrams_set = HashSet::new();

    // Process n-grams
    if word_tokens.len() < context.config.ngram_size {
        if !word_tokens.is_empty() {
            // For documents shorter than ngram_size, use all tokens
            let ngram_tokens = word_tokens.clone();
            let ngram_id = get_or_create_ngram_id(
                &ngram_tokens,
                &context.builder.question_ngram_to_id,
                &context.builder.next_ngram_id,
            );
            unique_ngrams_set.insert(ngram_id);
            add_doc_to_ngram(ngram_id, entry.doc_id, &context.builder.question_ngram_id_to_eval_doc_ids);
        }
    } else {
        for i in 0..=word_tokens.len() - context.config.ngram_size {
            let ngram_slice = &word_tokens[i..i + context.config.ngram_size];
            let ngram_tokens = ngram_slice.to_vec();
            let ngram_id = get_or_create_ngram_id(
                &ngram_tokens,
                &context.builder.question_ngram_to_id,
                &context.builder.next_ngram_id,
            );
            unique_ngrams_set.insert(ngram_id);
            add_doc_to_ngram(ngram_id, entry.doc_id, &context.builder.question_ngram_id_to_eval_doc_ids);

        }
    }

    let unique_ngrams = unique_ngrams_set.len();

    // Store the ngram IDs for this document
    context.builder.eval_doc_id_to_question_ngram_ids.insert(entry.doc_id, unique_ngrams_set);

    // Store hot path data in eval_documents
    // Note: Question IDF sum is initially 0.0 and will be computed after all docs are indexed
    // Note: answer_token_length is initially 0 and will be updated when processing answer field
    // Note: passage_token_length is initially 0 and will be updated when processing passage field
    context.builder.eval_documents.insert(
        entry.doc_id,
        (
            unique_ngrams,
            token_count,
            0.0,  // question_idf_sum placeholder
            0,    // answer_token_length placeholder
            0,    // passage_token_length placeholder
        ),
    );

    // Store cold path metadata separately
    context.builder.eval_document_metadata.insert(
        entry.doc_id,
        EvalDocumentMetadataEntry {
            eval_key: entry.eval_key.clone(),
            line_num,
            total_ngrams,
            fingerprint: entry.fingerprint.clone(),
            is_correct: entry.is_correct,
            eval_instance_index: entry.eval_instance_index,
            split: entry.split.clone(),
            reference_file_idx: context.file_idx,
        },
    );

    Ok(())
}

fn process_answer_field(
    answer: &str,
    doc_id: u32,
    config: &Config,
    builder: &ReferenceIndexBuilder,
) -> Result<(), Error> {
    let cleaned = clean_text(answer, &config.punctuation_chars);

    // Store the answer text snippet (first 1000 words) for reports/review
    let snippet_words: Vec<&str> = cleaned.split_whitespace().take(1000).collect();
    let text_snippet = snippet_words.join(" ");
    builder.eval_answer_text_snippets.insert(doc_id, text_snippet);

    // Extract tokens from cleaned text
    let word_tokens = match extract_tokens(&cleaned, config, &builder.tokenizer) {
        Some(tokens) => tokens,
        None => return Ok(()), // Silent error handling for answers
    };

    // Store the actual token length before converting to set
    let token_length = word_tokens.len();

    // Update the answer_token_length field (index 3) in eval_documents
    if let Some(mut doc_entry) = builder.eval_documents.get_mut(&doc_id) {
        doc_entry.3 = token_length;
    }

    // Store the ordered tokens for the answer
    builder.eval_doc_id_to_answer_tokens_ordered.insert(doc_id, word_tokens.clone());

    // Build n-gram index for longer answers
    if token_length > config.short_answer_token_threshold {
        let ngram_size = config.answer_ngram_size;
        if word_tokens.len() >= ngram_size {
            let mut answer_ngram_ids = HashSet::new();

            for i in 0..=word_tokens.len() - ngram_size {
                let ngram_tokens = &word_tokens[i..i + ngram_size];
                let ngram_hash = hash_ngram(ngram_tokens);

                // Get or create ngram ID
                let ngram_id = *builder.question_ngram_to_id
                    .entry(ngram_hash)
                    .or_insert_with(|| builder.next_ngram_id.fetch_add(1, Ordering::Relaxed));

                // Track unique ngram IDs for this answer
                answer_ngram_ids.insert(ngram_id);
            }

            // Store the set of n-gram IDs for this document
            builder.eval_doc_id_to_answer_ngram_ids.insert(doc_id, answer_ngram_ids.clone());

            // Update document frequency for each unique answer ngram
            for ngram_id in answer_ngram_ids {
                builder.answer_ngram_doc_freq
                    .entry(ngram_id)
                    .and_modify(|count| *count += 1)
                    .or_insert(1);
            }
        }
    }

    // Track unique tokens for this document for IDF calculation
    let token_set: HashSet<usize> = word_tokens.into_iter().collect();
    for token in &token_set {
        builder.token_eval_doc_freq
            .entry(*token)
            .or_insert_with(|| AtomicUsize::new(0))
            .fetch_add(1, Ordering::Relaxed);
    }

    builder.eval_doc_id_to_answer_tokens.insert(doc_id, token_set);

    Ok(())
}

fn process_passage_field(
    passage: &str,
    doc_id: u32,
    config: &Config,
    builder: &ReferenceIndexBuilder,
) -> Result<(), Error> {
    // Clean text for both tokenizer types
    let cleaned = clean_text(passage, &config.punctuation_chars);

    // Store the passage text snippet (first 1000 words) for use in reports/review
    let snippet_words: Vec<&str> = cleaned.split_whitespace().take(1000).collect();
    let text_snippet = snippet_words.join(" ");
    builder.eval_passage_text_snippets.insert(doc_id, text_snippet);

    // Extract tokens from cleaned text
    let word_tokens = match extract_tokens(&cleaned, config, &builder.tokenizer) {
        Some(tokens) => tokens,
        None => return Ok(()), // Silent error handling for passages
    };

    // Store the actual token length before converting to set
    let token_length = word_tokens.len();

    // Update eval_documents with the passage token length
    if let Some(mut entry) = builder.eval_documents.get_mut(&doc_id) {
        entry.4 = token_length;  // Update passage_token_length field (index 4)
    }

    // Build n-gram index for passages
    let passage_ngram_size = config.passage_ngram_size;
    if word_tokens.len() >= passage_ngram_size {
        let mut passage_ngram_ids = HashSet::new();

        for i in 0..=word_tokens.len() - passage_ngram_size {
            let ngram_tokens = &word_tokens[i..i + passage_ngram_size];
            let ngram_hash = hash_ngram(ngram_tokens);

            // Get or create ngram ID
            let ngram_id = *builder.question_ngram_to_id
                .entry(ngram_hash)
                .or_insert_with(|| builder.next_ngram_id.fetch_add(1, Ordering::Relaxed));

            // Track unique ngram IDs for this passage
            passage_ngram_ids.insert(ngram_id);
        }

        // Store the set of n-gram IDs for this document
        builder.eval_doc_id_to_passage_ngram_ids.insert(doc_id, passage_ngram_ids.clone());

        // Update document frequency for each unique passage ngram
        for ngram_id in passage_ngram_ids {
            builder.passage_ngram_doc_freq
                .entry(ngram_id)
                .and_modify(|count| *count += 1)
                .or_insert(1);
        }
    }

    // Track unique tokens for this document for IDF calculation
    let token_set: HashSet<usize> = word_tokens.into_iter().collect();
    for token in &token_set {
        builder.token_eval_doc_freq
            .entry(*token)
            .or_insert_with(|| AtomicUsize::new(0))
            .fetch_add(1, Ordering::Relaxed);
    }

    builder.eval_doc_id_to_passage_tokens.insert(doc_id, token_set);

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;
    use std::fs;

    // Helper function to create a test config
    fn create_test_config() -> Config {
        use crate::detect::scoring::calculate_minimum_question_idf_threshold;
        let contamination_score_threshold = 0.8;
        Config {
            mode: "simple".to_string(),
            tokenizer_str: "cl100k".to_string(),
            ngram_size: 3,
            answer_ngram_size: 3,
            passage_ngram_size: 4,
            punctuation_chars: ".,!?".to_string(),
            eval_dedup: true,
            eval_min_token_length: 20,
            eval_min_unique_word_count: 4,
            short_answer_token_threshold: 10,
            evals_dir: PathBuf::from("/tmp/test_ref"),
            training_dir: PathBuf::from("/tmp/input"),
            report_output_dir: PathBuf::from("/tmp/report"),
            cleaned_output_dir: None,
            content_key: "text".to_string(),
            hash_seed: 0,
            contamination_score_threshold,
            minimum_question_idf_threshold: calculate_minimum_question_idf_threshold(contamination_score_threshold),
            exact_override: false,
            sample_every_m_tokens: 10,
            question_max_consecutive_misses: 11,
            ngram_bucket_lru_cache: 10000,
            worker_threads: 4,
            window_size_increment: None,
            num_windows: None,
            window_step_size: None,
            short_answer_window_length: 50,
            min_long_answer_window: 100,
            verbose: false,
            min_passage_distance: 100,
            passage_max_consecutive_misses: 2,
            purify: false,
            perfect_match_decay_start: Some(20),
            perfect_match_decay_end: Some(50),
            index_passages: true,
            index_answers: true,
        }
    }

    // Helper to create test tokenizer
    fn create_test_tokenizer() -> OmniTokenizer {
        OmniTokenizer::new("cl100k").unwrap()
    }

    #[test]
    fn test_hash_ngram_consistency() {
        let tokens1 = vec![1, 2, 3];
        let tokens2 = vec![1, 2, 3];
        let tokens3 = vec![1, 2, 4];

        // Same input should produce same hash
        assert_eq!(hash_ngram(&tokens1), hash_ngram(&tokens2));

        // Different input should produce different hash
        assert_ne!(hash_ngram(&tokens1), hash_ngram(&tokens3));

        // Empty input should work
        let empty: Vec<usize> = vec![];
        let _ = hash_ngram(&empty);
    }

    #[test]
    fn test_get_or_create_ngram_id() {
        let ngram_to_id = DashMap::new();
        let next_id = AtomicU32::new(0);

        let tokens1 = vec![1, 2, 3];
        let tokens2 = vec![1, 2, 3];
        let tokens3 = vec![4, 5, 6];

        // First call should create new ID
        let id1 = get_or_create_ngram_id(&tokens1, &ngram_to_id, &next_id);
        assert_eq!(id1, 0);

        // Second call with same tokens should return same ID
        let id2 = get_or_create_ngram_id(&tokens2, &ngram_to_id, &next_id);
        assert_eq!(id2, id1);

        // Different tokens should get new ID
        let id3 = get_or_create_ngram_id(&tokens3, &ngram_to_id, &next_id);
        assert_eq!(id3, 1);
        assert_ne!(id3, id1);
    }

    #[test]
    #[should_panic(expected = "N-gram ID overflow")]
    fn test_ngram_id_overflow_detection() {
        let ngram_to_id = DashMap::new();
        // Start very close to overflow threshold
        let next_id = AtomicU32::new(4_200_000_000);

        // This should be fine
        let tokens1 = vec![1, 2, 3];
        let _ = get_or_create_ngram_id(&tokens1, &ngram_to_id, &next_id);

        // This should panic due to overflow threshold
        let tokens2 = vec![4, 5, 6];
        let _ = get_or_create_ngram_id(&tokens2, &ngram_to_id, &next_id);
    }

    #[test]
    fn test_extract_tokens_bpe_mode() {
        let config = create_test_config();
        let tokenizer = create_test_tokenizer();

        // Normal text - BPE tokenizer will produce different token counts
        let text = "hello world test";
        let tokens = extract_tokens(text, &config, &tokenizer).unwrap();
        assert!(!tokens.is_empty()); // BPE tokenizer will produce tokens

        // Empty text should return None
        let empty = "";
        let empty_tokens = extract_tokens(empty, &config, &tokenizer);
        assert!(empty_tokens.is_none()); // Empty text returns None

        // Whitespace only with padding
        let whitespace = "   ";
        let ws_tokens = extract_tokens(whitespace, &config, &tokenizer).unwrap();
        assert!(!ws_tokens.is_empty()); // Will have padding and whitespace tokens
    }

    #[test]
    fn test_should_skip_by_token_count() {
        let mut config = create_test_config();
        let tokenizer = create_test_tokenizer();

        // Threshold disabled (0)
        config.eval_min_token_length = 0;
        assert!(!should_skip_by_token_count("test", &config, &tokenizer));

        // Below threshold - BPE tokenizers produce different counts
        config.eval_min_token_length = 10;
        assert!(should_skip_by_token_count("hi", &config, &tokenizer));

        // Above threshold - longer text should pass
        let long_text = "This is a much longer text with many words that should exceed the threshold";
        assert!(!should_skip_by_token_count(long_text, &config, &tokenizer));
    }

    #[test]
    fn test_should_skip_by_unique_words() {
        let mut config = create_test_config();

        // Threshold disabled (0)
        config.eval_min_unique_word_count = 0;
        assert!(!should_skip_by_unique_words("test", &config));

        // Below threshold
        config.eval_min_unique_word_count = 3;
        assert!(should_skip_by_unique_words("test test", &config));

        // At threshold
        assert!(!should_skip_by_unique_words("one two three", &config));

        // Above threshold with duplicates
        assert!(!should_skip_by_unique_words("one two three three", &config));
    }

    #[test]
    fn test_build_combined_text_for_filtering() {
        let config = create_test_config();

        // All fields present with answer fields
        let combined = build_combined_text_for_filtering(
            "passage text",
            "question text",
            "answer text",
            true,
            &config,
        );
        assert!(combined.contains("passage"));
        assert!(combined.contains("question"));
        assert!(combined.contains("answer"));

        // No answer fields flag
        let combined_no_answer = build_combined_text_for_filtering(
            "passage text",
            "question text",
            "answer text",
            false,
            &config,
        );
        assert!(!combined_no_answer.contains("answer"));

        // Empty fields
        let combined_empty = build_combined_text_for_filtering(
            "",
            "question",
            "",
            true,
            &config,
        );
        assert_eq!(combined_empty, "question");
    }

    #[test]
    fn test_check_and_mark_duplicate() {
        let dedup_map = Some(DashMap::new());

        let entry1 = ReferenceEntry {
            doc_id: 1,
            eval_key: "test_eval".to_string(),
            eval_instance_index: 0,
            has_answer_fields: false,
            fingerprint: Some("fp123".to_string()),
            is_correct: None,
            split: None,
            question_text: "test".to_string(),
            answer_text: "".to_string(),
            passage_text: "".to_string(),
        };

        // First occurrence should not be duplicate
        assert!(!check_and_mark_duplicate(&entry1, 0, &dedup_map));

        // Second occurrence with same fingerprint should be duplicate
        assert!(check_and_mark_duplicate(&entry1, 1, &dedup_map));

        // Different eval_key but same fingerprint should not be duplicate
        let mut entry2 = entry1.clone();
        entry2.eval_key = "different_eval".to_string();
        assert!(!check_and_mark_duplicate(&entry2, 2, &dedup_map));

        // No fingerprint should not be duplicate
        let mut entry3 = entry1.clone();
        entry3.fingerprint = None;
        assert!(!check_and_mark_duplicate(&entry3, 3, &dedup_map));

        // Dedup disabled
        assert!(!check_and_mark_duplicate(&entry1, 4, &None));
    }

    #[test]
    fn test_parse_reference_entry() {
        // Complete entry
        let json = serde_json::json!({
            "doc_id": 42,
            "eval_key": "test_dataset",
            "eval_instance_index": 5,
            "question": "What is the capital?",
            "answer": "Paris",
            "passage": "France is a country...",
            "fingerprint": "abc123",
            "is_correct": true,
            "split": "train"
        });

        let entry = parse_reference_entry(&json, 0, "fallback").unwrap();
        assert_eq!(entry.doc_id, 42);
        assert_eq!(entry.eval_key, "test_dataset");
        assert_eq!(entry.eval_instance_index, 5);
        assert_eq!(entry.question_text, "What is the capital?");
        assert_eq!(entry.answer_text, "Paris");
        assert_eq!(entry.passage_text, "France is a country...");
        assert_eq!(entry.fingerprint, Some("abc123".to_string()));
        assert_eq!(entry.is_correct, Some(true));
        assert_eq!(entry.split, Some("train".to_string()));
        assert!(entry.has_answer_fields);

        // Minimal entry
        let json_min = serde_json::json!({
            "doc_id": 1,
            "question": "Test?"
        });

        let entry_min = parse_reference_entry(&json_min, 10, "fallback_eval").unwrap();
        assert_eq!(entry_min.doc_id, 1);
        assert_eq!(entry_min.eval_key, "fallback_eval");
        assert_eq!(entry_min.eval_instance_index, 10); // Falls back to line_num
        assert_eq!(entry_min.question_text, "Test?");
        assert_eq!(entry_min.answer_text, "");
        assert_eq!(entry_min.passage_text, "");
        assert_eq!(entry_min.fingerprint, None);
        assert!(!entry_min.has_answer_fields);

        // Missing doc_id should error
        let json_bad = serde_json::json!({
            "question": "Test?"
        });
        assert!(parse_reference_entry(&json_bad, 0, "fallback").is_err());
    }

    #[test]
    fn test_process_question_field() {
        let config = create_test_config();
        let builder = ReferenceIndexBuilder::new(&config).unwrap();

        let mut lines_processed = 0;
        let mut skipped = 0;

        // Process a question
        let entry = ReferenceEntry {
            doc_id: 1,
            eval_key: "test_eval".to_string(),
            eval_instance_index: 5,
            has_answer_fields: false,
            fingerprint: Some("fp123".to_string()),
            is_correct: Some(true),
            split: Some("train".to_string()),
            question_text: "This is a test question with multiple words".to_string(),
            answer_text: String::new(),
            passage_text: String::new(),
        };

        let mut context = ProcessingContext {
            config: &config,
            builder: &builder,
            lines_processed: &mut lines_processed,
            skipped_entries: &mut skipped,
            file_idx: 0,
        };

        process_question_field(&entry, 0, &mut context).unwrap();

        assert_eq!(lines_processed, 1);
        assert_eq!(skipped, 0);

        // Check document was stored
        assert!(builder.eval_documents.contains_key(&1));
        assert!(builder.eval_document_metadata.contains_key(&1));
        let metadata = builder.eval_document_metadata.get(&1).unwrap();
        assert_eq!(metadata.eval_key, "test_eval");
        assert_eq!(metadata.is_correct, Some(true));
        assert_eq!(metadata.eval_instance_index, 5);

        // Check snippet was stored
        assert!(builder.eval_text_snippets.contains_key(&1));

        // Check ngrams were created
        assert!(!builder.question_ngram_to_id.is_empty());
        assert!(builder.eval_doc_id_to_question_ngram_ids.contains_key(&1));
    }

    #[test]
    fn test_process_answer_field() {
        let config = create_test_config();
        let builder = ReferenceIndexBuilder::new(&config).unwrap();

        // Insert test documents into eval_documents
        builder.eval_documents.insert(1, (
            0, 0, 0.0, 0, 0
        ));
        builder.eval_documents.insert(2, (
            0, 0, 0.0, 0, 0
        ));

        // Short answer (below threshold)
        process_answer_field(
            "Short",
            1,
            &config,
            &builder,
        ).unwrap();

        assert!(builder.eval_doc_id_to_answer_tokens.contains_key(&1));
        assert!(builder.eval_doc_id_to_answer_tokens_ordered.contains_key(&1));
        // Check that answer length was stored in eval_documents
        let doc_info = builder.eval_documents.get(&1).unwrap();
        assert!(doc_info.3 > 0);
        assert!(!builder.eval_doc_id_to_answer_ngram_ids.contains_key(&1)); // Too short for ngrams

        // Long answer (above threshold)
        let long_answer = "This is a much longer answer that exceeds the token threshold for ngram indexing";
        process_answer_field(
            long_answer,
            2,
            &config,
            &builder,
        ).unwrap();

        assert!(builder.eval_doc_id_to_answer_tokens.contains_key(&2));
        assert!(builder.eval_doc_id_to_answer_ngram_ids.contains_key(&2)); // Should have ngrams
        assert!(!builder.answer_ngram_doc_freq.is_empty());
        // Check that answer length was stored in eval_documents
        let doc_info = builder.eval_documents.get(&2).unwrap();
        assert!(doc_info.3 > 0);
    }

    #[test]
    fn test_process_passage_field() {
        let config = create_test_config();
        let builder = ReferenceIndexBuilder::new(&config).unwrap();

        let passage = "This is a test passage with enough words to create multiple ngrams";

        builder.eval_documents.insert(1, (
            0, 0, 0.0, 0, 0
        ));

        process_passage_field(
            passage,
            1,
            &config,
            &builder,
        ).unwrap();

        // Check all data structures were populated
        assert!(builder.eval_doc_id_to_passage_tokens.contains_key(&1));
        // Check that passage length was stored in eval_documents
        if let Some(entry) = builder.eval_documents.get(&1) {
            assert!(entry.4 > 0);  // passage_token_length should be > 0
        }
        assert!(builder.eval_doc_id_to_passage_ngram_ids.contains_key(&1));
        assert!(!builder.passage_ngram_doc_freq.is_empty());
        assert!(builder.eval_passage_text_snippets.contains_key(&1));

        // Check snippet is stored correctly
        let snippet = builder.eval_passage_text_snippets.get(&1).unwrap();
        assert!(snippet.contains("passage"));
    }

    #[test]
    fn test_get_reference_files() {
        let temp_dir = TempDir::new().unwrap();
        let ref_dir = temp_dir.path().join("refs");
        fs::create_dir(&ref_dir).unwrap();

        // Create test files
        let file1 = ref_dir.join("test1.jsonl");
        let file2 = ref_dir.join("test2.jsonl.gz");
        let file3 = ref_dir.join("ignore.txt");

        fs::File::create(&file1).unwrap();
        fs::File::create(&file2).unwrap();
        fs::File::create(&file3).unwrap();

        let (files, names) = get_reference_files(&ref_dir).unwrap();

        // Should only find .jsonl and .gz files
        assert_eq!(files.len(), 2);
        assert_eq!(names.len(), 2);
        assert!(names.contains(&"test1.jsonl".to_string()));
        assert!(names.contains(&"test2.jsonl.gz".to_string()));

        // Test with non-existent directory
        let bad_path = PathBuf::from("/nonexistent/path");
        assert!(get_reference_files(&bad_path).is_err());
    }

    #[test]
    fn test_index_control_flags() {
        // Test that build_combined_text_for_filtering respects index flags

        // Test with both flags enabled
        let mut config = create_test_config();
        config.index_passages = true;
        config.index_answers = true;

        let combined = build_combined_text_for_filtering(
            "passage text",
            "question text",
            "answer text",
            true,
            &config,
        );
        assert!(combined.contains("passage"));
        assert!(combined.contains("question"));
        assert!(combined.contains("answer"));

        // Test with only questions (no passages, no answers)
        config.index_passages = false;
        config.index_answers = false;

        let combined = build_combined_text_for_filtering(
            "passage text",
            "question text",
            "answer text",
            true,
            &config,
        );
        assert!(!combined.contains("passage"));
        assert!(combined.contains("question"));
        assert!(!combined.contains("answer"));

        // Test with questions and answers only (no passages)
        config.index_passages = false;
        config.index_answers = true;

        let combined = build_combined_text_for_filtering(
            "passage text",
            "question text",
            "answer text",
            true,
            &config,
        );
        assert!(!combined.contains("passage"));
        assert!(combined.contains("question"));
        assert!(combined.contains("answer"));

        // Test with questions and passages only (no answers)
        config.index_passages = true;
        config.index_answers = false;

        let combined = build_combined_text_for_filtering(
            "passage text",
            "question text",
            "answer text",
            true,
            &config,
        );
        assert!(combined.contains("passage"));
        assert!(combined.contains("question"));
        assert!(!combined.contains("answer"));
    }
}
