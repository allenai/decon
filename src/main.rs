// External crates
use anyhow::{Error, Result};
use clap::{Parser, Subcommand};
use regex::Regex;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use serde_yaml;
use tiktoken_rs::{cl100k_base, p50k_base, CoreBPE};
use unicode_segmentation::UnicodeSegmentation;

// Standard library
use std::collections::HashMap;
use std::hash::{DefaultHasher, Hash, Hasher};
use std::path::PathBuf;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, Mutex};

// Internal crate imports
use mj_io::read_pathbuf_to_mem;

// Internal modules
mod daemon;
mod minhash;
mod review;
mod simple;
mod toxic;

/*=================================================================
=                                  ARGS                           =
=================================================================*/

#[derive(Parser)]
#[clap(
    author,
    version,
    about = "Decon - A contamination detection tool for machine learning datasets",
    long_about = "Decon identifies when training data contains text from evaluation datasets.\n\nSupports three detection algorithms:\n- SIMPLE: N-gram matching with sampling and cluster expansion\n- MinHash: Near-duplicate detection using locality-sensitive hashing\n- TOXIC: Semantic similarity using word embeddings\n\nConfiguration options can be specified in a YAML file and overridden via command-line flags."
)]
struct ArgParser {
    #[clap(subcommand)]
    command: Commands,

    #[arg(long, default_value_t = 0)]
    threads: usize,
}

#[derive(Subcommand, Debug)]
enum Commands {
    Detect {
        #[arg(required = true, long, help = "Path to YAML configuration file")]
        config: PathBuf,

        // Common config overrides
        #[arg(long, help = "Detection mode: simple, minhash, or toxic")]
        mode: Option<String>,

        #[arg(long, help = "JSON field containing text content")]
        content_key: Option<String>,

        #[arg(long, help = "Directory containing training data")]
        local_input: Option<PathBuf>,

        #[arg(long, help = "Directory containing evaluation/reference data")]
        reference_input: Option<PathBuf>,

        #[arg(long, help = "Output directory for contamination reports")]
        report_output_dir: Option<PathBuf>,

        #[arg(long, help = "Output directory for cleaned files")]
        cleaned_output_dir: Option<PathBuf>,

        #[arg(long, help = "Enable creation of cleaned files")]
        purify: Option<bool>,

        #[arg(long, help = "Enable debug mode")]
        debug: Option<bool>,

        #[arg(long, help = "Tokenizer type: word, p50k, or cl100k")]
        tokenizer: Option<String>,

        // SIMPLE mode specific
        #[arg(long, help = "N-gram size for SIMPLE mode")]
        ngram_size: Option<usize>,

        #[arg(long, help = "Sample every M tokens for SIMPLE mode")]
        sample_every_m_tokens: Option<usize>,

        #[arg(
            long,
            help = "Max consecutive misses before stopping cluster expansion"
        )]
        max_consecutive_misses: Option<usize>,

        // MinHash mode specific
        #[arg(long, help = "Number of bands for MinHash LSH")]
        num_bands: Option<usize>,

        #[arg(long, help = "Size of each band for MinHash LSH")]
        band_size: Option<usize>,

        #[arg(long, help = "Jaccard similarity threshold for MinHash")]
        jaccard_similarity_threshold: Option<f32>,

        // TOXIC mode specific
        #[arg(long, help = "Path to word embeddings file for TOXIC mode")]
        toxic_embedding_path: Option<PathBuf>,

        #[arg(long, help = "Number of hyperplanes for TOXIC LSH")]
        toxic_hyperplanes: Option<usize>,

        #[arg(long, help = "Overlap ratio threshold for TOXIC mode")]
        toxic_overlap_threshold: Option<f32>,

        #[arg(long, help = "Toxic score threshold for TOXIC mode")]
        toxic_score_threshold: Option<f32>,

        #[arg(long, help = "Scale factor for poison tokens in TOXIC mode")]
        toxic_poison_scale: Option<f32>,

        #[arg(
            long,
            help = "Skip buckets with more than this many entries (-1 to disable)"
        )]
        skip_hot_bucket_threshold: Option<i32>,
    },

    Review {
        #[arg(long)]
        config: Option<PathBuf>,

        #[arg(long)]
        results_file: Option<PathBuf>,

        #[arg(long, help = "Directory containing result files to analyze for stats")]
        dir: Option<PathBuf>,

        #[arg(
            long,
            help = "Step through examples one by one, waiting for Enter between each"
        )]
        step: bool,

        #[arg(
            long,
            help = "Calculate and display TP/FP/Accuracy statistics based on ground truth annotations"
        )]
        metric: bool,

        #[arg(
            long,
            help = "Show only false positives (detected as contaminated but actually clean)"
        )]
        fp: bool,

        #[arg(
            long,
            help = "Show only false negatives (missed contamination - actually contaminated but not detected)"
        )]
        fn_: bool,

        #[arg(
            long,
            help = "Show only true positives (correctly detected contamination)"
        )]
        tp: bool,

        #[arg(
            long,
            help = "Show only true negatives (correctly identified as clean - requires ground truth)"
        )]
        tn: bool,

        #[arg(
            long,
            help = "Display eval dataset statistics with horizontal bar chart"
        )]
        stats: bool,

        #[arg(
            long,
            help = "Minimum overlap ratio to include in results (filters by jaccard_similarity/overlap_ratio)"
        )]
        min_overlap_ratio: Option<f32>,

        #[arg(
            long,
            help = "Minimum IDF score to include in results (filters by toxic_score)"
        )]
        min_idf_score: Option<f32>,

        #[arg(
            long,
            help = "Minimum n-gram match count to include in results (filters by ngram_match_cnt)"
        )]
        min_length: Option<usize>,
    },

    Daemon {
        #[arg(required = true, long)]
        config: PathBuf,

        #[arg(long, default_value_t = 8080)]
        port: u16,

        // Common config overrides
        #[arg(long, help = "Detection mode: simple, minhash, or toxic")]
        mode: Option<String>,

        #[arg(long, help = "JSON field containing text content")]
        content_key: Option<String>,

        #[arg(long, help = "Directory containing training data")]
        local_input: Option<PathBuf>,

        #[arg(long, help = "Directory containing evaluation/reference data")]
        reference_input: Option<PathBuf>,

        #[arg(long, help = "Output directory for contamination reports")]
        report_output_dir: Option<PathBuf>,

        #[arg(long, help = "Output directory for cleaned files")]
        cleaned_output_dir: Option<PathBuf>,

        #[arg(long, help = "Enable creation of cleaned files")]
        purify: Option<bool>,

        #[arg(long, help = "Enable debug mode")]
        debug: Option<bool>,

        #[arg(long, help = "Tokenizer type: word, p50k, or cl100k")]
        tokenizer: Option<String>,

        // SIMPLE mode specific
        #[arg(long, help = "N-gram size for SIMPLE mode")]
        ngram_size: Option<usize>,

        #[arg(long, help = "Sample every M tokens for SIMPLE mode")]
        sample_every_m_tokens: Option<usize>,

        #[arg(
            long,
            help = "Max consecutive misses before stopping cluster expansion"
        )]
        max_consecutive_misses: Option<usize>,

        // MinHash mode specific
        #[arg(long, help = "Number of bands for MinHash LSH")]
        num_bands: Option<usize>,

        #[arg(long, help = "Size of each band for MinHash LSH")]
        band_size: Option<usize>,

        #[arg(long, help = "Jaccard similarity threshold for MinHash")]
        jaccard_similarity_threshold: Option<f32>,

        // TOXIC mode specific
        #[arg(long, help = "Path to word embeddings file for TOXIC mode")]
        toxic_embedding_path: Option<PathBuf>,

        #[arg(long, help = "Number of hyperplanes for TOXIC LSH")]
        toxic_hyperplanes: Option<usize>,

        #[arg(long, help = "Overlap ratio threshold for TOXIC mode")]
        toxic_overlap_threshold: Option<f32>,

        #[arg(long, help = "Toxic score threshold for TOXIC mode")]
        toxic_score_threshold: Option<f32>,

        #[arg(long, help = "Scale factor for poison tokens in TOXIC mode")]
        toxic_poison_scale: Option<f32>,

        #[arg(
            long,
            help = "Skip buckets with more than this many entries (-1 to disable)"
        )]
        skip_hot_bucket_threshold: Option<i32>,
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

    // Minhash parameters
    #[serde(default = "default_num_bands")]
    pub num_bands: usize,
    #[serde(default = "default_band_size")]
    pub band_size: usize,
    #[serde(default = "default_ngram_size")]
    pub ngram_size: usize,
    #[serde(default = "default_tokenizer_str")]
    pub tokenizer_str: String,
    #[serde(default)]
    pub hash_seed: usize,

    // Data configuration
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
    #[serde(default = "default_jaccard_threshold")]
    pub jaccard_similarity_threshold: f32,

    // TOXIC-specific parameters
    #[serde(default = "default_toxic_embedding_path")]
    pub toxic_embedding_path: PathBuf,
    #[serde(default = "default_toxic_hyperplanes")]
    pub toxic_hyperplanes: usize,
    #[serde(default = "default_toxic_overlap_threshold")]
    pub toxic_overlap_threshold: f32,
    #[serde(default = "default_toxic_score_threshold")]
    pub toxic_score_threshold: f32,
    #[serde(default = "default_toxic_poison_scale")]
    pub toxic_poison_scale: f32,
    #[serde(default = "default_skip_hot_bucket_threshold")]
    pub skip_hot_bucket_threshold: i32,

    // TOXIC sampling optimization parameters
    #[serde(default = "default_sample_every_m_tokens")]
    pub sample_every_m_tokens: usize,
    #[serde(default = "default_max_consecutive_misses")]
    pub max_consecutive_misses: usize,
    #[serde(default = "default_ngram_bucket_lru_cache")]
    pub ngram_bucket_lru_cache: usize,

    // Text processing options
    #[serde(default = "default_punctuation_chars")]
    pub punctuation_chars: String,

    // Debug options
    #[serde(default = "default_debug")]
    pub debug: bool,

    // Daemon options
    #[serde(default = "default_worker_threads")]
    pub worker_threads: usize,

    // Windowing options
    #[serde(default)]
    pub window_size_increment: Option<usize>,
    #[serde(default)]
    pub num_windows: Option<usize>,
    #[serde(default)]
    pub window_step_size: Option<usize>,

    // Purify option - create cleaned files with contaminated lines removed
    #[serde(default)]
    pub purify: bool,
}

fn default_mode() -> String {
    "minhash".to_string()
}

fn default_jaccard_threshold() -> f32 {
    0.5
}

fn default_toxic_embedding_path() -> PathBuf {
    PathBuf::from("/home/robert/Downloads/wiki-news-300d-1M.vec")
}

fn default_toxic_hyperplanes() -> usize {
    64 // 64-bit bucket IDs
}

fn default_toxic_overlap_threshold() -> f32 {
    0.3 // 30% n-gram overlap threshold
}

fn default_toxic_score_threshold() -> f32 {
    0.0 // Default toxic score threshold (disabled)
}

fn default_toxic_poison_scale() -> f32 {
    3.0 // Amplify poison token destructive impact
}

fn default_debug() -> bool {
    false // Debug logging disabled by default
}

fn default_num_bands() -> usize {
    7 // Default MinHash bands
}

fn default_band_size() -> usize {
    8 // Default MinHash band size
}

fn default_ngram_size() -> usize {
    5 // Default n-gram size
}

fn default_tokenizer_str() -> String {
    "uniseg".to_string() // Default tokenizer
}

fn default_skip_hot_bucket_threshold() -> i32 {
    -1 // Disabled by default
}

fn default_sample_every_m_tokens() -> usize {
    1 // Disabled by default (sample every token = no sampling)
}

fn default_max_consecutive_misses() -> usize {
    2 // Stop expansion after 2 consecutive misses
}

fn default_ngram_bucket_lru_cache() -> usize {
    10000 // LRU cache size for n-gram -> bucket_id mappings
}

fn default_punctuation_chars() -> String {
    "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~".to_string() // Default punctuation for minhash
}

fn default_worker_threads() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4) // Default to 4 if unable to detect CPU cores
}

pub fn read_config(config_path: &PathBuf) -> Result<Config, Error> {
    let contents = read_pathbuf_to_mem(config_path).unwrap();
    let config: Config = serde_yaml::from_reader(contents).unwrap();
    Ok(config)
}

pub fn get_results_filename(mode: &str) -> String {
    match mode {
        "minhash" => "contamination_results.jsonl".to_string(),
        "toxic" => "toxic_contamination_results.jsonl".to_string(),
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
        "minhash" => config.jaccard_similarity_threshold,
        "toxic" => config.toxic_overlap_threshold,
        "simple" => config.toxic_overlap_threshold,
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

    // Format: {base name}.clean.jsonl.gz
    format!("{}.clean.jsonl.gz", base_name)
}

// Common function to write a purified file with contaminated lines removed
pub fn write_purified_file(
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
    let reader: Box<dyn BufRead> = if input_path.extension().and_then(|s| s.to_str()) == Some("gz")
    {
        Box::new(BufReader::new(GzDecoder::new(file)))
    } else {
        Box::new(BufReader::new(file))
    };

    // Create output file with gzip compression
    let output_file = File::create(&purified_path)?;
    let gz_encoder = GzEncoder::new(output_file, Compression::default());
    let mut writer = BufWriter::new(gz_encoder);

    let mut removed_count = 0;
    for (line_num, line) in reader.lines().enumerate() {
        if !contaminated_lines.contains(&line_num) {
            writeln!(writer, "{}", line?)?;
        } else {
            removed_count += 1;
        }
    }

    writer.flush()?;
    println!(
        "Created purified file: {:?} (removed {} contaminated lines)",
        purified_path, removed_count
    );

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
            "p50k" => self
                .inner
                .as_ref()
                .unwrap()
                .encode_with_special_tokens(text),
            "cl100k" => self
                .inner
                .as_ref()
                .unwrap()
                .encode_with_special_tokens(text),
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
    // SlimPajama text cleaning process (used by MinHash)

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

pub fn clean_text_allowlist(text: &str, _punctuation_chars: &str) -> String {
    // Whitelist approach: keep only letters, numbers, hyphens, and spaces (used by TOXIC)
    // This ensures consistent matching regardless of Unicode variants, fancy quotes, etc.
    // Hyphens are preserved because they affect tokenization (e.g., "one-to-one" vs "one to one")

    text.chars()
        .filter_map(|c| match c {
            // Keep letters (convert to lowercase)
            'a'..='z' => Some(c),
            'A'..='Z' => Some(c.to_ascii_lowercase()),
            // Keep numbers
            '0'..='9' => Some(c),
            // Keep hyphens (important for compound words and tokenization)
            '-' | '_' | ']' | '[' | '{' | '}' | '=' | '(' | ')' | '>' | '<' | '+' | '*' | '/' => {
                Some(c)
            }
            // Normalize all whitespace to single space
            ' ' | '\t' | '\n' | '\r' => Some(' '),
            // Drop everything else (punctuation, Unicode variants, etc.)
            _ => None,
        })
        .collect::<String>()
        .split_whitespace() // Normalize multiple spaces to single spaces
        .collect::<Vec<&str>>()
        .join(" ")
}

// Debug logging macro - only prints when config.debug is true
#[macro_export]
macro_rules! debug_println {
    ($config:expr, $($arg:tt)*) => {
        if $config.debug {
            println!($($arg)*);
        }
    };
}

/*=================================================================
=                         CONTAMINATION DETECTION                =
=================================================================*/

fn contamination_detect_with_config(config_obj: &Config) -> Result<(), Error> {
    match config_obj.mode.as_str() {
        "minhash" => {
            println!("Using MinHash contamination detection...");
            minhash::contamination_detect(&config_obj)
        }
        "toxic" => {
            println!("Using TOXIC contamination detection...");
            toxic::contamination_detect(&config_obj)
        }
        "simple" => {
            println!("Using Simple contamination detection...");
            simple::contamination_detect(&config_obj)
        }
        unknown_mode => {
            println!("Unknown mode: '{}'", unknown_mode);
            println!("Available modes: minhash, toxic, simple");
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
            mode,
            content_key,
            local_input,
            reference_input,
            report_output_dir,
            cleaned_output_dir,
            purify,
            debug,
            tokenizer,
            ngram_size,
            sample_every_m_tokens,
            max_consecutive_misses,
            num_bands,
            band_size,
            jaccard_similarity_threshold,
            toxic_embedding_path,
            toxic_hyperplanes,
            toxic_overlap_threshold,
            toxic_score_threshold,
            toxic_poison_scale,
            skip_hot_bucket_threshold,
        } => {
            // Load config from file
            let mut loaded_config = read_config(config)?;

            // Apply command-line overrides
            if let Some(m) = mode {
                loaded_config.mode = m.clone();
            }
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
            if let Some(p) = purify {
                loaded_config.purify = *p;
            }
            if let Some(d) = debug {
                loaded_config.debug = *d;
            }
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
            if let Some(mcm) = max_consecutive_misses {
                loaded_config.max_consecutive_misses = *mcm;
            }

            // MinHash mode overrides
            if let Some(nb) = num_bands {
                loaded_config.num_bands = *nb;
            }
            if let Some(bs) = band_size {
                loaded_config.band_size = *bs;
            }
            if let Some(jst) = jaccard_similarity_threshold {
                loaded_config.jaccard_similarity_threshold = *jst;
            }

            // TOXIC mode overrides
            if let Some(tep) = toxic_embedding_path {
                loaded_config.toxic_embedding_path = tep.clone();
            }
            if let Some(th) = toxic_hyperplanes {
                loaded_config.toxic_hyperplanes = *th;
            }
            if let Some(tot) = toxic_overlap_threshold {
                loaded_config.toxic_overlap_threshold = *tot;
            }
            if let Some(tst) = toxic_score_threshold {
                loaded_config.toxic_score_threshold = *tst;
            }
            if let Some(tps) = toxic_poison_scale {
                loaded_config.toxic_poison_scale = *tps;
            }
            if let Some(shbt) = skip_hot_bucket_threshold {
                loaded_config.skip_hot_bucket_threshold = *shbt;
            }

            contamination_detect_with_config(&loaded_config)
        }

        Commands::Review {
            config,
            results_file,
            dir,
            step,
            metric,
            fp,
            fn_,
            tp,
            tn,
            stats,
            min_overlap_ratio,
            min_idf_score,
            min_length,
        } => review::review_contamination(
            config.as_ref(),
            results_file.as_ref(),
            dir.as_ref(),
            *step,
            *metric,
            *fp,
            *fn_,
            *tp,
            *tn,
            *stats,
            *min_overlap_ratio,
            *min_idf_score,
            *min_length,
        ),

        Commands::Daemon {
            config,
            port,
            mode,
            content_key,
            local_input,
            reference_input,
            report_output_dir,
            cleaned_output_dir,
            purify,
            debug,
            tokenizer,
            ngram_size,
            sample_every_m_tokens,
            max_consecutive_misses,
            num_bands,
            band_size,
            jaccard_similarity_threshold,
            toxic_embedding_path,
            toxic_hyperplanes,
            toxic_overlap_threshold,
            toxic_score_threshold,
            toxic_poison_scale,
            skip_hot_bucket_threshold,
        } => {
            // Load config from file
            let mut loaded_config = read_config(config)?;

            // Apply command-line overrides
            if let Some(m) = mode {
                loaded_config.mode = m.clone();
            }
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
            if let Some(p) = purify {
                loaded_config.purify = *p;
            }
            if let Some(d) = debug {
                loaded_config.debug = *d;
            }
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
            if let Some(mcm) = max_consecutive_misses {
                loaded_config.max_consecutive_misses = *mcm;
            }

            // MinHash mode overrides
            if let Some(nb) = num_bands {
                loaded_config.num_bands = *nb;
            }
            if let Some(bs) = band_size {
                loaded_config.band_size = *bs;
            }
            if let Some(jst) = jaccard_similarity_threshold {
                loaded_config.jaccard_similarity_threshold = *jst;
            }

            // TOXIC mode overrides
            if let Some(tep) = toxic_embedding_path {
                loaded_config.toxic_embedding_path = tep.clone();
            }
            if let Some(th) = toxic_hyperplanes {
                loaded_config.toxic_hyperplanes = *th;
            }
            if let Some(tot) = toxic_overlap_threshold {
                loaded_config.toxic_overlap_threshold = *tot;
            }
            if let Some(tst) = toxic_score_threshold {
                loaded_config.toxic_score_threshold = *tst;
            }
            if let Some(tps) = toxic_poison_scale {
                loaded_config.toxic_poison_scale = *tps;
            }
            if let Some(shbt) = skip_hot_bucket_threshold {
                loaded_config.skip_hot_bucket_threshold = *shbt;
            }

            let runtime = tokio::runtime::Runtime::new().unwrap();
            runtime.block_on(daemon::run_daemon(loaded_config, *port))
        }
    };
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_clean_text_allowlist() {
        // Test basic text cleaning
        let input = "Janet's ducks lay 16 eggs per day.";
        let result = clean_text_allowlist(input, "");
        assert_eq!(result, "janets ducks lay 16 eggs per day");

        // Test Unicode apostrophe normalization
        let input_unicode = "Janet's ducks lay 16 eggs per day."; // Unicode right single quotation mark
        let result_unicode = clean_text_allowlist(input_unicode, "");
        assert_eq!(result_unicode, "janets ducks lay 16 eggs per day");

        // Test hyphen preservation
        let input_hyphen = "one-to-one function";
        let result_hyphen = clean_text_allowlist(input_hyphen, "");
        assert_eq!(result_hyphen, "one-to-one function");

        // Test multiple whitespace normalization
        let input_spaces = "hello    world\t\ttest\n\nfoo";
        let result_spaces = clean_text_allowlist(input_spaces, "");
        assert_eq!(result_spaces, "hello world test foo");

        // Test punctuation removal
        let input_punct = "Hello, world! How are you? Fine.";
        let result_punct = clean_text_allowlist(input_punct, "");
        assert_eq!(result_punct, "hello world how are you fine");
    }
}
