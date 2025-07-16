// External crates
use anyhow::{Error, Result};
use clap::{Parser, Subcommand};
use regex::Regex;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use serde_yaml;
use tiktoken_rs::{cl100k_base, p50k_base, CoreBPE};
use unicode_segmentation::UnicodeSegmentation;
use zstd::stream::read::Decoder as ZstdDecoder;

// Standard library
use std::collections::HashMap;
use std::hash::{DefaultHasher, Hash, Hasher};
use std::path::PathBuf;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, Mutex};

// Internal crate imports
use mj_io::read_pathbuf_to_mem;

// Internal modules
mod reference;
mod review;
mod server;
mod simple;

/*=================================================================
=                                  ARGS                           =
=================================================================*/

#[derive(Parser)]
#[clap(
    author,
    version,
    about = "Decon - A contamination detection tool for machine learning datasets",
    long_about = "Decon identifies when training data contains text from evaluation datasets.\n\nUses the SIMPLE detection algorithm:\n- N-gram matching with sampling and cluster expansion\n- Configurable thresholds for questions and answers\n- Support for multiple tokenizers\n\nConfiguration options can be specified in a YAML file and overridden via command-line flags."
)]
struct ArgParser {
    #[clap(subcommand)]
    command: Commands,

    #[arg(long, default_value_t = 0)]
    threads: usize,
}

#[derive(Subcommand, Debug)]
enum Commands {
    #[command(
        about = "Detect contamination in training data",
        long_about = "Detect and report on contamination between a reference dataset and a training dataset.\nCan optionally produce a clean dataset with contaminated documents removed."
    )]
    Detect {
        // Input and Output
        #[arg(
            long,
            help = "Directory containing training data in jsonl format",
            display_order = 1,
            help_heading = "Input and Output"
        )]
        local_input: Option<PathBuf>,

        #[arg(
            long,
            help = "JSON field containing text content in training documents [default: text]",
            display_order = 2,
            help_heading = "Input and Output"
        )]
        content_key: Option<String>,

        #[arg(
            long,
            help = "Directory containing reference data (evals) in decon expected format",
            default_value = "fixtures/reference-best-available",
            display_order = 3,
            help_heading = "Input and Output"
        )]
        reference_input: Option<PathBuf>,

        #[arg(
            long,
            help = "Output directory for contamination reports",
            display_order = 4,
            help_heading = "Input and Output"
        )]
        report_output_dir: Option<PathBuf>,

        #[arg(
            long,
            help = "Output directory for cleaned files when --purify option is set.",
            display_order = 5,
            help_heading = "Input and Output"
        )]
        cleaned_output_dir: Option<PathBuf>,

        #[arg(
            long,
            help = "Produce cleaned files with contaminated lines removed",
            display_order = 6,
            help_heading = "Input and Output"
        )]
        purify: bool,

        #[arg(
            required = false,
            long,
            help = "Path to YAML configuration file",
            display_order = 7
        )]
        config: PathBuf,

        // Matching and Scoring
        #[arg(
            long,
            help = "Tokenizer type: word, uniseg, p50k, or cl100k [default: cl100k]",
            display_order = 8,
            help_heading = "Matching and Scoring"
        )]
        tokenizer: Option<String>,

        #[arg(
            long,
            help = "N-gram size for SIMPLE mode [default: 5]",
            display_order = 9,
            help_heading = "Matching and Scoring"
        )]
        ngram_size: Option<usize>,

        #[arg(
            long,
            help = "Sample every M tokens for SIMPLE mode [default: ngram_size + 1]",
            display_order = 10,
            help_heading = "Matching and Scoring"
        )]
        sample_every_m_tokens: Option<usize>,

        #[arg(
            long,
            help = "Max consecutive misses before stopping cluster expansion [default: 2]",
            display_order = 11,
            help_heading = "Matching and Scoring"
        )]
        max_consecutive_misses: Option<usize>,

        #[arg(
            long,
            help = "Contamination score threshold for questions in SIMPLE mode [default: 0.8]",
            display_order = 12,
            help_heading = "Matching and Scoring"
        )]
        question_threshold: Option<f32>,

        #[arg(
            long,
            help = "Contamination score threshold for answers in SIMPLE mode [default: 0.8]",
            display_order = 13,
            help_heading = "Matching and Scoring"
        )]
        answer_threshold: Option<f32>,
    },

    #[command(about = "Review and analyze detect results")]
    Review {
        #[arg(long)]
        config: Option<PathBuf>,

        #[arg(long, help = "Directory containing result files to analyze for stats")]
        dir: Option<PathBuf>,

        #[arg(
            long,
            help = "Display eval dataset statistics with horizontal bar chart"
        )]
        stats: bool,

        #[arg(
            long,
            help = "Display all contamination results at once (without stepping through)"
        )]
        all: bool,

        #[arg(long, help = "Minimum contamination score to include in results")]
        min_score: Option<f32>,

        #[arg(
            long,
            help = "Minimum n-gram match count to include in results (filters by ngram_match_cnt)"
        )]
        min_length: Option<usize>,

        #[arg(
            long,
            help = "Filter by evaluation dataset name (strips suffix after last underscore)"
        )]
        eval: Option<String>,
    },

    #[command(about = "Run decon as an HTTP server for orchestrated pipelines")]
    Server {
        // Configuration
        #[arg(
            required = true,
            long,
            help = "Path to YAML configuration file",
            display_order = 1
        )]
        config: PathBuf,

        // Server Configuration
        #[arg(
            long,
            default_value_t = 8080,
            help = "Port to listen on",
            display_order = 2,
            help_heading = "Server Configuration"
        )]
        port: u16,

        // Input and Output
        #[arg(
            long,
            help = "Directory containing training data in jsonl format\n",
            display_order = 3,
            help_heading = "Input and Output"
        )]
        local_input: Option<PathBuf>,

        #[arg(
            long,
            help = "JSON field containing text content [default: text]\n",
            display_order = 4,
            help_heading = "Input and Output"
        )]
        content_key: Option<String>,

        #[arg(
            long,
            help = "Directory containing evaluation/reference data",
            default_value = "fixtures/reference-best-available",
            display_order = 5,
            help_heading = "Input and Output"
        )]
        reference_input: Option<PathBuf>,

        #[arg(
            long,
            help = "Output directory for contamination reports\n",
            display_order = 6,
            help_heading = "Input and Output"
        )]
        report_output_dir: Option<PathBuf>,

        #[arg(
            long,
            help = "Output directory for cleaned files when --purify option is set.\n",
            display_order = 7,
            help_heading = "Input and Output"
        )]
        cleaned_output_dir: Option<PathBuf>,

        #[arg(
            long,
            help = "Enable creation of cleaned files [default: false]",
            display_order = 8,
            help_heading = "Input and Output"
        )]
        purify: bool,

        // Matching and Scoring
        #[arg(
            long,
            help = "Tokenizer type: word, p50k, or cl100k [default: cl100k]",
            display_order = 9,
            help_heading = "Matching and Scoring"
        )]
        tokenizer: Option<String>,

        #[arg(
            long,
            help = "N-gram size for SIMPLE mode [default: 5]\n",
            display_order = 10,
            help_heading = "Matching and Scoring"
        )]
        ngram_size: Option<usize>,

        #[arg(
            long,
            help = "Sample every M tokens for SIMPLE mode [default: ngram_size + 1]\n",
            display_order = 11,
            help_heading = "Matching and Scoring"
        )]
        sample_every_m_tokens: Option<usize>,

        #[arg(
            long,
            help = "Max consecutive misses before stopping cluster expansion [default: 2]\n",
            display_order = 12,
            help_heading = "Matching and Scoring"
        )]
        max_consecutive_misses: Option<usize>,

        #[arg(
            long,
            help = "Contamination score threshold for questions in SIMPLE mode [default: 0.8]\n",
            display_order = 13,
            help_heading = "Matching and Scoring"
        )]
        question_threshold: Option<f32>,

        #[arg(
            long,
            help = "Contamination score threshold for answers in SIMPLE mode [default: 0.8]",
            display_order = 14,
            help_heading = "Matching and Scoring"
        )]
        answer_threshold: Option<f32>,
    },

    #[command(about = "Manage and analyze reference datasets")]
    References {
        #[arg(
            long,
            help = "Refine reference files by removing duplicates and normalizing\n"
        )]
        refine: bool,

        #[arg(long, help = "Show statistics for reference datasets in a directory\n")]
        stats: Option<PathBuf>,

        #[arg(
            long,
            help = "Perform a dry run - show statistics without writing files (for --refine)"
        )]
        dry_run: bool,
    },
}

/*=================================================================
=                             CONFIG                              =
=================================================================*/

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
    pub local_input: PathBuf,
    pub reference_input: PathBuf,
    pub report_output_dir: PathBuf,
    #[serde(default)]
    pub cleaned_output_dir: Option<PathBuf>,

    // Processing options
    #[serde(default)]
    pub exact_override: bool,

    // Sampling optimization parameters
    #[serde(default = "default_sample_every_m_tokens")]
    pub sample_every_m_tokens: usize,
    #[serde(default = "default_max_consecutive_misses")]
    pub max_consecutive_misses: usize,
    #[serde(default = "default_ngram_bucket_lru_cache")]
    pub ngram_bucket_lru_cache: usize,

    // Text processing options
    #[serde(default = "default_punctuation_chars")]
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
    #[serde(default = "default_min_short_answer_distance")]
    pub min_short_answer_distance: usize,
    #[serde(default = "default_answer_threshold")]
    pub answer_threshold: f32,
    #[serde(default = "default_exclude_question_from_answer_sweep")]
    pub exclude_question_from_answer_sweep: bool,

    // Simple mode contamination score threshold
    #[serde(default = "default_question_threshold")]
    pub question_threshold: f32,

    // Purify option - create cleaned files with contaminated lines removed
    #[serde(default)]
    pub purify: bool,

    // Minimum word count for eval file indexing in SIMPLE mode
    #[serde(default = "default_eval_min_token_count")]
    pub eval_min_token_count: usize,

    // Whether to replace non-UTF8 characters when creating purified files
    #[serde(default = "default_replace_non_utf8_chars")]
    pub replace_non_utf8_chars: bool,
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
    1 // Default to 1, will be set to ngram_size + 1 if not overridden
}

fn default_max_consecutive_misses() -> usize {
    2 // Default to 2
}

fn default_ngram_bucket_lru_cache() -> usize {
    10000 // LRU cache size for n-gram -> bucket_id mappings
}

fn default_punctuation_chars() -> String {
    "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~".to_string() // Default punctuation
}

fn default_worker_threads() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4) // Default to 4 if unable to detect CPU cores
}

fn default_eval_min_token_count() -> usize {
    10 // Default minimum word count for eval file indexing
}

fn default_min_short_answer_distance() -> usize {
    30 // Default maximum token distance to search for short answers
}

fn default_answer_threshold() -> f32 {
    0.8 // Default threshold for short answer contamination
}

fn default_question_threshold() -> f32 {
    0.8 // Default contamination score threshold for SIMPLE mode
}

fn default_replace_non_utf8_chars() -> bool {
    false // Default to preserving bytes exactly as they are
}

fn default_exclude_question_from_answer_sweep() -> bool {
    true // Default to excluding question tokens when searching for answers
}

pub fn read_config(config_path: &PathBuf) -> Result<Config, Error> {
    let contents = read_pathbuf_to_mem(config_path).unwrap();
    let config: Config = serde_yaml::from_reader(contents).unwrap();
    Ok(config)
}

pub fn get_results_filename(mode: &str) -> String {
    match mode {
        "simple" => "simple_contamination_results.jsonl".to_string(),
        _ => "contamination_results.jsonl".to_string(), // fallback
    }
}

pub fn get_unique_results_filename(input_file: &PathBuf, config: &Config) -> String {
    // Extract the base filename without extension
    let base_name = input_file
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("unknown");

    // Get the appropriate threshold value based on mode
    let threshold = match config.mode.as_str() {
        "simple" => config.question_threshold,
        _ => 0.0,
    };

    // Format: {input file name}-{mode}-{overlap_threshold}.jsonl
    format!("{}-{}-{:.2}.jsonl", base_name, config.mode, threshold)
}

pub fn get_purified_filename(input_file: &PathBuf) -> String {
    // Get the full filename
    let filename = input_file
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("unknown");

    // Remove .jsonl extension if present (and any compression extension)
    let base_name = if let Some(pos) = filename.find(".jsonl") {
        &filename[..pos]
    } else {
        // If no .jsonl extension, just use the stem
        input_file
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("unknown")
    };

    // Format: {base name}.jsonl.gz
    format!("{}.jsonl.gz", base_name)
}

pub fn write_purified_file(
    input_path: &PathBuf,
    cleaned_output_dir: &PathBuf,
    contaminated_lines: &std::collections::HashSet<usize>,
    config: &Config,
) -> Result<PathBuf, anyhow::Error> {
    if config.replace_non_utf8_chars {
        write_purified_file_with_utf8_lossy_conversion(
            input_path,
            cleaned_output_dir,
            contaminated_lines,
        )
    } else {
        write_purified_file_bytes(input_path, cleaned_output_dir, contaminated_lines)
    }
}

// Common function to write a purified file with contaminated lines removed
pub fn write_purified_file_bytes(
    input_path: &PathBuf,
    cleaned_output_dir: &PathBuf,
    contaminated_lines: &std::collections::HashSet<usize>,
) -> Result<PathBuf, anyhow::Error> {
    use flate2::read::GzDecoder;
    use flate2::write::GzEncoder;
    use flate2::Compression;
    use std::fs::{create_dir_all, File};
    use std::io::{BufRead, BufReader, BufWriter, Write};

    // Ensure output directory exists
    create_dir_all(cleaned_output_dir)?;

    let purified_filename = get_purified_filename(input_path);
    let purified_path = cleaned_output_dir.join(&purified_filename);

    // Open input file with streaming reader
    let file = File::open(input_path)?;
    let mut reader: Box<dyn BufRead> = match input_path.extension().and_then(|s| s.to_str()) {
        Some("gz") => Box::new(BufReader::new(GzDecoder::new(file))),
        Some("zst") => Box::new(BufReader::new(ZstdDecoder::new(file)?)),
        _ => Box::new(BufReader::new(file)),
    };

    // Create output file with gzip compression
    let output_file = File::create(&purified_path)?;
    let gz_encoder = GzEncoder::new(output_file, Compression::default());
    let mut writer = BufWriter::new(gz_encoder);

    // Work with raw bytes to preserve data exactly as-is
    let mut line_num = 0;
    let mut line_buffer = Vec::new();

    loop {
        line_buffer.clear();
        let bytes_read = reader.read_until(b'\n', &mut line_buffer)?;
        if bytes_read == 0 {
            break;
        }

        if !contaminated_lines.contains(&line_num) {
            // Write bytes exactly as they are, no UTF-8 validation
            writer.write_all(&line_buffer)?;
        }
        line_num += 1;
    }

    writer.flush()?;

    Ok(purified_path)
}

pub fn write_purified_file_with_utf8_lossy_conversion(
    input_path: &PathBuf,
    cleaned_output_dir: &PathBuf,
    contaminated_lines: &std::collections::HashSet<usize>,
) -> Result<PathBuf, anyhow::Error> {
    use flate2::read::GzDecoder;
    use flate2::write::GzEncoder;
    use flate2::Compression;
    use std::fs::{create_dir_all, File};
    use std::io::{BufRead, BufReader, BufWriter, Write};

    // Ensure output directory exists
    create_dir_all(cleaned_output_dir)?;

    let purified_filename = get_purified_filename(input_path);
    let purified_path = cleaned_output_dir.join(&purified_filename);

    // Open input file with streaming reader
    let file = File::open(input_path)?;
    let mut reader: Box<dyn BufRead> = match input_path.extension().and_then(|s| s.to_str()) {
        Some("gz") => Box::new(BufReader::new(GzDecoder::new(file))),
        Some("zst") => Box::new(BufReader::new(ZstdDecoder::new(file)?)),
        _ => Box::new(BufReader::new(file)),
    };

    // Create output file with gzip compression
    let output_file = File::create(&purified_path)?;
    let gz_encoder = GzEncoder::new(output_file, Compression::default());
    let mut writer = BufWriter::new(gz_encoder);

    let mut encoding_errors = 0;

    // Work with raw bytes to handle encoding issues
    let mut line_num = 0;
    let mut line_buffer = Vec::new();

    loop {
        line_buffer.clear();
        let bytes_read = reader.read_until(b'\n', &mut line_buffer)?;
        if bytes_read == 0 {
            break;
        }

        if !contaminated_lines.contains(&line_num) {
            // Try to validate as UTF-8
            match std::str::from_utf8(&line_buffer) {
                Ok(_) => {
                    // Valid UTF-8, write as-is
                    writer.write_all(&line_buffer)?;
                }
                Err(_) => {
                    // Invalid UTF-8 - convert lossily and write
                    // This replaces invalid UTF-8 sequences with �
                    let lossy = String::from_utf8_lossy(&line_buffer);
                    writer.write_all(lossy.as_bytes())?;
                    encoding_errors += 1;
                }
            }
        }
        line_num += 1;
    }

    writer.flush()?;

    if encoding_errors > 0 {
        eprintln!("Warning: {} lines had encoding issues and were converted lossily (invalid UTF-8 replaced with �)", encoding_errors);
    }

    Ok(purified_path)
}

/*=================================================================
=                             UTILITIES                           =
=================================================================*/

pub fn get_nested_json_val(obj: &Value, key: &String) -> Result<String, Error> {
    let mut current = obj;
    for subkey in key.split('.') {
        current = current
            .get(subkey)
            .ok_or_else(|| anyhow::anyhow!("Key '{}' not found in JSON object", subkey))?;
    }

    current
        .as_str()
        .ok_or_else(|| anyhow::anyhow!("Value at key '{}' is not a string", key))
        .map(|s| s.to_string())
}

pub struct OmniTokenizer {
    pub tokenizer_name: String,
    pub inner: Option<CoreBPE>,
    // For word mode: vocabulary built from reference set
    pub word_to_id: Arc<Mutex<HashMap<String, u32>>>,
    pub id_to_word: Arc<Mutex<HashMap<u32, String>>>,
    pub next_word_id: Arc<AtomicU32>,
}

impl OmniTokenizer {
    pub fn new(tokenizer_name: &str) -> Result<Self, Error> {
        let inner_tokenizer = match tokenizer_name {
            "p50k" => Some(p50k_base().unwrap()),
            "cl100k" => Some(cl100k_base().unwrap()),
            "word" => None, // Word mode doesn't need a BPE tokenizer
            _ => {
                println!("Tokenizer {:?} <--- BE CAREFUL HERE", tokenizer_name);
                Some(p50k_base().unwrap())
            }
        };
        Ok(OmniTokenizer {
            tokenizer_name: tokenizer_name.to_string(),
            inner: inner_tokenizer,
            word_to_id: Arc::new(Mutex::new(HashMap::new())),
            id_to_word: Arc::new(Mutex::new(HashMap::new())),
            next_word_id: Arc::new(AtomicU32::new(0)),
        })
    }

    pub fn encode(&self, text: &str) -> Vec<usize> {
        match self.tokenizer_name.as_str() {
            "p50k" => self.inner.as_ref().unwrap().encode_ordinary(text),
            "cl100k" => self.inner.as_ref().unwrap().encode_ordinary(text),
            "uniseg" => text
                .split_word_bounds()
                .map(|w| {
                    let mut hasher = DefaultHasher::new();
                    w.hash(&mut hasher);
                    hasher.finish() as usize
                })
                .collect(),
            "word" => {
                // For word mode, tokenize into words and look up IDs
                // Unknown words get a special ID (u32::MAX)
                let cleaned = clean_text(text, &default_punctuation_chars());
                cleaned
                    .split_whitespace()
                    .map(|word| {
                        let word_to_id = self.word_to_id.lock().unwrap();
                        word_to_id.get(word).copied().unwrap_or(u32::MAX) as usize
                    })
                    .collect()
            }
            _ => {
                // default to character level
                text.bytes().map(|b| b as usize).collect()
            }
        }
    }

    // Add word to vocabulary (for building reference vocabulary)
    pub fn add_word(&self, word: &str) -> u32 {
        let mut word_to_id = self.word_to_id.lock().unwrap();
        if let Some(&id) = word_to_id.get(word) {
            id
        } else {
            let id = self.next_word_id.fetch_add(1, Ordering::SeqCst);
            word_to_id.insert(word.to_string(), id);
            let mut id_to_word = self.id_to_word.lock().unwrap();
            id_to_word.insert(id, word.to_string());
            id
        }
    }

    pub fn decode_tokens(&self, _token_ids: &[usize]) -> Vec<String> {
        // For WELSH, we need word-level tokens, not token IDs
        // This is a simplified approach - in practice we'd need to store the mapping
        // For now, return placeholder - we'll need to modify the approach
        vec![]
    }

    pub fn get_word(&self, id: u32) -> Option<String> {
        if self.tokenizer_name == "word" {
            let id_to_word = self.id_to_word.lock().unwrap();
            id_to_word.get(&id).cloned()
        } else {
            // For BPE tokenizers, decode the token ID
            if let Some(ref tokenizer) = self.inner {
                if let Ok(text) = tokenizer.decode(vec![id as usize]) {
                    Some(text)
                } else {
                    None
                }
            } else {
                None
            }
        }
    }
}

pub fn hash_object<T: Hash>(obj: &T) -> usize {
    let mut hasher = DefaultHasher::new();
    obj.hash(&mut hasher);
    hasher.finish() as usize
}

pub fn preprocess_text(
    text: &str,
    tokenizer: &OmniTokenizer,
    punctuation_chars: &str,
) -> Vec<usize> {
    let cleaned_text = clean_text(text, punctuation_chars);
    // println!("    🔧 Original text: \"{}\"", text);
    // println!("    🔧 Cleaned text:  \"{}\"", cleaned_text);
    let tokens = tokenizer.encode(&cleaned_text);
    // println!("    🔧 Tokens: {:?}", tokens);
    tokens
}

pub fn clean_text(text: &str, punctuation_chars: &str) -> String {
    // Text cleaning process

    // Convert the document to lowercase
    let mut text = text.to_lowercase();

    // Remove punctuation based on configurable character set
    let punctuation_chars: Vec<char> = punctuation_chars.chars().collect();
    text.retain(|c| !punctuation_chars.contains(&c));

    // Replace multiple whitespace characters with a single space
    let re = Regex::new(r"\s+").unwrap();
    text = re.replace_all(&text, " ").to_string();

    // Trim leading and trailing whitespace
    text.trim().to_string()
}

/*=================================================================
=                         CONTAMINATION DETECTION                =
=================================================================*/

fn contamination_detect_with_config(config_obj: &Config) -> Result<(), Error> {
    match config_obj.mode.as_str() {
        "simple" => {
            println!("Using Simple contamination detection...");
            println!("  N-gram size: {}", config_obj.ngram_size);
            println!(
                "  Sample every M tokens: {}",
                config_obj.sample_every_m_tokens
            );
            println!(
                "  Max consecutive misses: {}",
                config_obj.max_consecutive_misses
            );
            println!("  Question threshold: {}", config_obj.question_threshold);
            println!("  Answer threshold: {}", config_obj.answer_threshold);
            println!("  Tokenizer: {}", config_obj.tokenizer_str);
            simple::contamination_detect(&config_obj)
        }
        unknown_mode => {
            println!("Unknown mode: '{}'", unknown_mode);
            println!("Mode must be: simple");
            Err(anyhow::anyhow!(
                "Unsupported detection mode: {}",
                unknown_mode
            ))
        }
    }
}

/*=================================================================
=                         CONTAMINATION REVIEW                   =
=================================================================*/

// Re-export review functionality

/*=================================================================
=                                 MAIN                            =
=================================================================*/

fn main() -> Result<(), Error> {
    let args = ArgParser::parse();
    let threads = args.threads;
    if threads != 0 {
        std::env::set_var("RAYON_NUM_THREADS", threads.to_string());
    }

    let result = match &args.command {
        Commands::Detect {
            config,
            content_key,
            local_input,
            reference_input,
            report_output_dir,
            cleaned_output_dir,
            purify,
            tokenizer,
            ngram_size,
            sample_every_m_tokens,
            max_consecutive_misses,
            question_threshold,
            answer_threshold,
        } => {
            // Load config from file
            let mut loaded_config = read_config(config)?;

            // Apply command-line overrides
            if let Some(ck) = content_key {
                loaded_config.content_key = ck.clone();
            }
            if let Some(li) = local_input {
                loaded_config.local_input = li.clone();
            }
            if let Some(ri) = reference_input {
                loaded_config.reference_input = ri.clone();
            }
            if let Some(rod) = report_output_dir {
                loaded_config.report_output_dir = rod.clone();
            }
            if let Some(cod) = cleaned_output_dir {
                loaded_config.cleaned_output_dir = Some(cod.clone());
            }
            loaded_config.purify = *purify;
            if let Some(t) = tokenizer {
                loaded_config.tokenizer_str = t.clone();
            }

            // SIMPLE mode overrides
            if let Some(ns) = ngram_size {
                loaded_config.ngram_size = *ns;
            }
            if let Some(semt) = sample_every_m_tokens {
                loaded_config.sample_every_m_tokens = *semt;
            }
            // If sample_every_m_tokens is still 1 (default), set it to ngram_size + 1
            if loaded_config.sample_every_m_tokens == 1 {
                loaded_config.sample_every_m_tokens = loaded_config.ngram_size + 1;
            }
            if let Some(mcm) = max_consecutive_misses {
                loaded_config.max_consecutive_misses = *mcm;
            }
            if let Some(qt) = question_threshold {
                loaded_config.question_threshold = *qt;
            }
            if let Some(at) = answer_threshold {
                loaded_config.answer_threshold = *at;
            }

            contamination_detect_with_config(&loaded_config)
        }

        Commands::Review {
            config,
            dir,
            stats,
            all,
            min_score,
            min_length,
            eval,
        } => review::review_contamination(
            config.as_ref(),
            dir.as_ref(),
            !*stats && !*all, // step is true when neither stats nor all is set
            *stats,
            *all,
            *min_score,
            *min_length,
            eval.as_deref(),
            false, // skip_exact is now always false
        ),

        Commands::Server {
            config,
            port,
            content_key,
            local_input,
            reference_input,
            report_output_dir,
            cleaned_output_dir,
            purify,
            tokenizer,
            ngram_size,
            sample_every_m_tokens,
            max_consecutive_misses,
            question_threshold,
            answer_threshold,
        } => {
            // Load config from file
            let mut loaded_config = read_config(config)?;

            // Apply command-line overrides
            if let Some(ck) = content_key {
                loaded_config.content_key = ck.clone();
            }
            if let Some(li) = local_input {
                loaded_config.local_input = li.clone();
            }
            if let Some(ri) = reference_input {
                loaded_config.reference_input = ri.clone();
            }
            if let Some(rod) = report_output_dir {
                loaded_config.report_output_dir = rod.clone();
            }
            if let Some(cod) = cleaned_output_dir {
                loaded_config.cleaned_output_dir = Some(cod.clone());
            }
            loaded_config.purify = *purify;
            if let Some(t) = tokenizer {
                loaded_config.tokenizer_str = t.clone();
            }

            // SIMPLE mode overrides
            if let Some(ns) = ngram_size {
                loaded_config.ngram_size = *ns;
            }
            if let Some(semt) = sample_every_m_tokens {
                loaded_config.sample_every_m_tokens = *semt;
            }
            // If sample_every_m_tokens is still 1 (default), set it to ngram_size + 1
            if loaded_config.sample_every_m_tokens == 1 {
                loaded_config.sample_every_m_tokens = loaded_config.ngram_size + 1;
            }
            if let Some(mcm) = max_consecutive_misses {
                loaded_config.max_consecutive_misses = *mcm;
            }
            if let Some(qt) = question_threshold {
                loaded_config.question_threshold = *qt;
            }
            if let Some(at) = answer_threshold {
                loaded_config.answer_threshold = *at;
            }

            let runtime = tokio::runtime::Runtime::new().unwrap();
            runtime.block_on(server::run_server(loaded_config, *port))
        }

        Commands::References {
            refine,
            stats,
            dry_run,
        } => {
            if *refine {
                reference::refine_reference_files(*dry_run)
            } else if let Some(stats_dir) = stats {
                reference::collect_reference_stats(stats_dir)
            } else {
                eprintln!("Error: Must specify either --refine or --stats");
                std::process::exit(1);
            }
        }
    };
    result
}

#[cfg(test)]
mod tests {
    use super::*;
}
