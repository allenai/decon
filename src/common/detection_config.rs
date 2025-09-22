use anyhow::{Error, Result};
use mj_io::read_pathbuf_to_mem;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::hash::{DefaultHasher, Hash, Hasher};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    // Detection mode
    #[serde(default = "default_mode")]
    pub mode: String,

    #[serde(default = "default_ngram_size")]
    pub ngram_size: usize,
    #[serde(default = "default_tokenizer_str")]
    pub tokenizer_str: String,
    #[serde(default)]
    pub hash_seed: usize,

    // Data configuration
    #[serde(default = "default_content_key")]
    pub content_key: String,

    // Directory paths
    pub training_dir: PathBuf,
    pub evals_dir: PathBuf,
    pub report_output_dir: PathBuf,
    #[serde(default)]
    pub cleaned_output_dir: Option<PathBuf>,

    // Processing options
    #[serde(default)]
    pub exact_override: bool,

    // Sampling optimization parameters
    #[serde(default = "default_sample_every_m_tokens")]
    pub sample_every_m_tokens: usize,
    #[serde(default = "default_question_max_consecutive_misses")]
    pub question_max_consecutive_misses: usize,
    #[serde(default = "default_ngram_bucket_lru_cache")]
    pub ngram_bucket_lru_cache: usize,

    // Text processing options
    #[serde(default = "crate::common::default_punctuation_chars")]
    pub punctuation_chars: String,

    // Server options
    #[serde(default = "default_worker_threads")]
    pub worker_threads: usize,

    // Windowing options
    #[serde(default)]
    pub window_size_increment: Option<usize>,
    #[serde(default)]
    pub num_windows: Option<usize>,
    #[serde(default)]
    pub window_step_size: Option<usize>,

    // Short answer detection parameters
    #[serde(default = "default_short_answer_window_length")]
    pub short_answer_window_length: usize,
    #[serde(default = "default_min_long_answer_window")]
    pub min_long_answer_window: usize,
    #[serde(default = "default_short_answer_token_threshold")]
    pub short_answer_token_threshold: usize,
    #[serde(default = "default_answer_ngram_size")]
    pub answer_ngram_size: usize,

    // Passage detection parameters
    #[serde(default = "default_min_passage_distance")]
    pub min_passage_distance: usize,
    #[serde(default = "default_passage_max_consecutive_misses")]
    pub passage_max_consecutive_misses: usize,
    #[serde(default = "default_passage_ngram_size")]
    pub passage_ngram_size: usize,

    // Combined contamination score threshold for final decision
    #[serde(default = "default_contamination_score_threshold")]
    pub contamination_score_threshold: f32,

    // Computed minimum question IDF threshold based on contamination_score_threshold
    // This is the lowest question IDF that could possibly result in contamination
    // given perfect answer/passage scores
    #[serde(skip)]
    pub minimum_question_idf_threshold: f32,

    // Purify option - create cleaned files with contaminated lines removed
    #[serde(default)]
    pub purify: bool,

    // Minimum word count for eval file indexing in SIMPLE mode

    // Enable exact deduplication of reference entries (default: true)
    #[serde(default = "default_eval_dedup")]
    pub eval_dedup: bool,

    // Minimum token count for reference entries, 0 = disabled (default: 20)
    #[serde(default = "default_eval_min_token_length")]
    pub eval_min_token_length: usize,

    // Minimum unique word count for reference entries, 0 = disabled (default: 4)
    #[serde(default = "default_eval_min_unique_word_count")]
    pub eval_min_unique_word_count: usize,



    // Cumulative token length where perfect match requirement starts
    #[serde(default = "default_perfect_match_decay_start")]
    pub perfect_match_decay_start: Option<usize>,

    // Cumulative token length where interpolation ends and normal threshold applies
    #[serde(default = "default_perfect_match_decay_end")]
    pub perfect_match_decay_end: Option<usize>,

    // Verbose output flag
    #[serde(default)]
    pub verbose: bool,

    // Index control flags - what to include in reference index
    #[serde(default = "default_index_passages")]
    pub index_passages: bool,
    #[serde(default = "default_index_answers")]
    pub index_answers: bool,
}

fn default_mode() -> String {
    "simple".to_string()
}

fn default_ngram_size() -> usize {
    5 // Default n-gram size
}

fn default_tokenizer_str() -> String {
    "cl100k".to_string() // Default tokenizer
}

fn default_content_key() -> String {
    "text".to_string() // Default content key
}

fn default_sample_every_m_tokens() -> usize {
    10 // Default to 10, matching YAML default
}

fn default_question_max_consecutive_misses() -> usize {
    11 // Default to 11, ngram_size * 2 recommended
}

fn default_ngram_bucket_lru_cache() -> usize {
    10000 // LRU cache size for n-gram -> bucket_id mappings
}

fn default_worker_threads() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4) // Default to 4 if unable to detect CPU cores
}

fn default_short_answer_window_length() -> usize {
    50 // Default window size for short answer exact matching
}

fn default_min_long_answer_window() -> usize {
    100 // Default minimum window size for long answer n-gram matching
}


fn default_short_answer_token_threshold() -> usize {
    3 // Answers with <= 3 tokens use exact matching
}

fn default_answer_ngram_size() -> usize {
    3 // Default n-gram size for long answer matching
}

fn default_contamination_score_threshold() -> f32 {
    0.8 // Default combined contamination score threshold
}

fn default_min_passage_distance() -> usize {
    100 // Default maximum token distance to search for context
}

fn default_passage_max_consecutive_misses() -> usize {
    2 // Default max consecutive misses for passage expansion
}

fn default_passage_ngram_size() -> usize {
    4 // Default n-gram size for passage matching
}

fn default_eval_dedup() -> bool {
    true // Enable exact deduplication of reference entries by default
}

fn default_eval_min_token_length() -> usize {
    20 // Minimum token count for reference entries
}

fn default_eval_min_unique_word_count() -> usize {
    4 // Minimum unique word count for reference entries
}


fn default_perfect_match_decay_start() -> Option<usize> {
    Some(20) // Require perfect match for cumulative length <= 20
}

fn default_perfect_match_decay_end() -> Option<usize> {
    Some(50) // Use contamination_score_threshold for cumulative length >= 50
}

fn default_index_passages() -> bool {
    true // Index passages by default
}

fn default_index_answers() -> bool {
    true // Index answers by default
}

pub fn read_config(config_path: &PathBuf) -> Result<Config, Error> {
    let contents = read_pathbuf_to_mem(config_path).unwrap();
    let config: Config = serde_yaml::from_reader(contents).unwrap();
    Ok(config)
}

/// Generate a temporary directory with a unique hash suffix
pub fn generate_temp_dir(prefix: &str) -> PathBuf {
    let temp_dir = std::env::temp_dir();

    // Generate a unique hash using timestamp and process ID for uniqueness
    let timestamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let pid = std::process::id();
    let hash_input = format!("{}-{}", timestamp, pid);

    // Create a simple hash and take first 8 characters
    let mut hasher = DefaultHasher::new();
    hash_input.hash(&mut hasher);
    let hash_value = hasher.finish();
    let short_hash = format!("{:x}", hash_value);
    let short_hash = &short_hash[..8.min(short_hash.len())];

    temp_dir.join(format!("{}-{}", prefix, short_hash))
}

/// Create a default configuration with all defaults matching the YAML
pub fn create_default_config() -> Config {
    let report_dir = generate_temp_dir("decon");

    Config {
        mode: default_mode(),
        ngram_size: default_ngram_size(),
        tokenizer_str: default_tokenizer_str(),
        hash_seed: 0,
        content_key: default_content_key(),
        training_dir: PathBuf::new(), // Will be overridden by CLI arg
        evals_dir: PathBuf::from("bundled-evals"),
        report_output_dir: report_dir,
        cleaned_output_dir: None,
        exact_override: false,
        sample_every_m_tokens: default_sample_every_m_tokens(),
        question_max_consecutive_misses: default_question_max_consecutive_misses(),
        ngram_bucket_lru_cache: default_ngram_bucket_lru_cache(),
        punctuation_chars: crate::common::default_punctuation_chars(),
        worker_threads: default_worker_threads(),
        window_size_increment: None,
        num_windows: None,
        window_step_size: None,
        short_answer_window_length: default_short_answer_window_length(),
        min_long_answer_window: default_min_long_answer_window(),
        short_answer_token_threshold: default_short_answer_token_threshold(),
        answer_ngram_size: default_answer_ngram_size(),
        verbose: false,
        min_passage_distance: default_min_passage_distance(),
        passage_max_consecutive_misses: default_passage_max_consecutive_misses(),
        passage_ngram_size: default_passage_ngram_size(),
        contamination_score_threshold: default_contamination_score_threshold(),
        minimum_question_idf_threshold: 0.0, // Will be computed after initialization
        purify: false,
        eval_dedup: default_eval_dedup(),
        eval_min_token_length: default_eval_min_token_length(),
        eval_min_unique_word_count: default_eval_min_unique_word_count(),
        perfect_match_decay_start: default_perfect_match_decay_start(),
        perfect_match_decay_end: default_perfect_match_decay_end(),
        index_passages: default_index_passages(),
        index_answers: default_index_answers(),
    }
}

/// Helper function to recursively check if a directory contains supported data files
fn check_dir_for_data_files(dir: &PathBuf) -> bool {
    fn check_dir_recursive(dir: &PathBuf) -> bool {
        if let Ok(entries) = std::fs::read_dir(dir) {
            for entry in entries.filter_map(Result::ok) {
                let path = entry.path();
                if path.is_file() {
                    if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
                        // Check if the file has any of the supported extensions
                        for ext in crate::common::SUPPORTED_DATA_EXTENSIONS {
                            if name.ends_with(ext) {
                                return true;
                            }
                        }
                    }
                } else if path.is_dir() {
                    // Recursively check subdirectories
                    if check_dir_recursive(&path) {
                        return true;
                    }
                }
            }
        }
        false
    }
    check_dir_recursive(dir)
}

/// Validate configuration constraints
pub fn validate_config(config: &Config) -> Result<(), Error> {
    // Validate training directory exists and contains files
    if !config.training_dir.exists() {
        return Err(anyhow::anyhow!(
            "Training directory does not exist: {}\n\nPlease provide a valid directory containing training data files in JSONL format.",
            config.training_dir.display()
        ));
    }

    if !config.training_dir.is_dir() {
        return Err(anyhow::anyhow!(
            "Training path is not a directory: {}\n\nPlease provide a directory containing training data files in JSONL format.",
            config.training_dir.display()
        ));
    }

    // Check if training directory contains any supported data files (recursively)
    let has_training_files = check_dir_for_data_files(&config.training_dir);

    if !has_training_files {
        return Err(anyhow::anyhow!(
            "Training directory contains no data files: {}\n\nThe directory (including subdirectories) should contain JSON/JSONL files with supported extensions:\n  .json, .jsonl (uncompressed or with .gz, .zst, .zstd, .bz2, .xz compression)",
            config.training_dir.display()
        ));
    }

    // Validate evaluation directory exists and contains files
    if !config.evals_dir.exists() {
        return Err(anyhow::anyhow!(
            "Evaluation directory does not exist: {}\n\nPlease provide a valid directory containing evaluation datasets or run 'decon evals --download' to fetch the default evaluation datasets.",
            config.evals_dir.display()
        ));
    }

    if !config.evals_dir.is_dir() {
        return Err(anyhow::anyhow!(
            "Evaluation path is not a directory: {}\n\nPlease provide a directory containing evaluation datasets.",
            config.evals_dir.display()
        ));
    }

    // Check if evaluation directory contains any files (recursively)
    let has_eval_files = check_dir_for_data_files(&config.evals_dir);

    if !has_eval_files {
        return Err(anyhow::anyhow!(
            "Evaluation directory contains no data files: {}\n\nThe directory (including subdirectories) should contain JSON/JSONL files with supported extensions:\n  .json, .jsonl (uncompressed or with .gz, .zst, .zstd, .bz2, .xz compression)\n\nYou can download the default evaluation datasets by running: decon evals --download",
            config.evals_dir.display()
        ));
    }

    // Validate sample_every_m_tokens is at least 1
    if config.sample_every_m_tokens < 1 {
        return Err(anyhow::anyhow!(
            "sample_every_m_tokens must be at least 1, got {}",
            config.sample_every_m_tokens
        ));
    }


    // Check perfect_match_decay_end >= perfect_match_decay_start
    if let (Some(decay_start), Some(decay_end)) = (
        config.perfect_match_decay_start,
        config.perfect_match_decay_end,
    )
        && decay_end < decay_start {
            return Err(anyhow::anyhow!(
                "perfect_match_decay_end ({}) must be greater than or equal to perfect_match_decay_start ({})",
                decay_end,
                decay_start
            ));
        }

    Ok(())
}