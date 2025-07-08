// External crates
use anyhow::{Error, Result};
use clap::{Parser, Subcommand};
use regex::Regex;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use serde_yaml;
use tiktoken_rs::{CoreBPE, p50k_base, cl100k_base};
use unicode_segmentation::UnicodeSegmentation;

// Standard library
use std::collections::HashMap;
use std::hash::{DefaultHasher, Hash, Hasher};
use std::io::{self, BufRead};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::sync::atomic::{AtomicU32, Ordering};

// Internal crate imports
use mj_io::{expand_dirs, read_pathbuf_to_mem};

// Internal modules
mod minhash;
mod toxic;
mod simple;
mod daemon;



/*=================================================================
=                                  ARGS                           =
=================================================================*/

#[derive(Parser)]
#[clap(author, version, about, long_about = None)]
struct ArgParser {
    #[clap(subcommand)]
    command: Commands,

    #[arg(long, default_value_t=0)]
    threads: usize,
}

#[derive(Subcommand, Debug)]
enum Commands {
    Detect {
        #[arg(required=true, long)]
        config: PathBuf
    },

    Review {
        #[arg(required=true, long)]
        config: PathBuf,

        #[arg(long)]
        results_file: Option<PathBuf>,

        #[arg(long, help = "Step through examples one by one, waiting for Enter between each")]
        step: bool,

        #[arg(long, help = "Calculate and display TP/FP/Accuracy statistics based on ground truth annotations")]
        stats: bool,

        #[arg(long, help = "Show only false positives (detected as contaminated but actually clean)")]
        fp: bool,

        #[arg(long, help = "Show only false negatives (missed contamination - actually contaminated but not detected)")]
        fn_: bool,

        #[arg(long, help = "Show only true positives (correctly detected contamination)")]
        tp: bool,

        #[arg(long, help = "Show only true negatives (correctly identified as clean - requires ground truth)")]
        tn: bool,

        #[arg(long, help = "Display full training documents (default: truncate at 50 lines)")]
        full: bool
    },

    Daemon {
        #[arg(required=true, long)]
        config: PathBuf,
        
        #[arg(long, default_value_t = 8080)]
        port: u16,
    }
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
    64  // 64-bit bucket IDs
}

fn default_toxic_overlap_threshold() -> f32 {
    0.3  // 30% n-gram overlap threshold
}

fn default_toxic_score_threshold() -> f32 {
    0.0  // Default toxic score threshold (disabled)
}

fn default_toxic_poison_scale() -> f32 {
    3.0  // Amplify poison token destructive impact
}

fn default_debug() -> bool {
    false  // Debug logging disabled by default
}

fn default_num_bands() -> usize {
    7  // Default MinHash bands
}

fn default_band_size() -> usize {
    8  // Default MinHash band size
}

fn default_ngram_size() -> usize {
    5  // Default n-gram size
}

fn default_tokenizer_str() -> String {
    "uniseg".to_string()  // Default tokenizer
}

fn default_skip_hot_bucket_threshold() -> i32 {
    -1  // Disabled by default
}

fn default_sample_every_m_tokens() -> usize {
    1  // Disabled by default (sample every token = no sampling)
}

fn default_max_consecutive_misses() -> usize {
    2  // Stop expansion after 2 consecutive misses
}

fn default_ngram_bucket_lru_cache() -> usize {
    10000  // LRU cache size for n-gram -> bucket_id mappings
}

fn default_punctuation_chars() -> String {
    "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~".to_string()  // Default punctuation for minhash
}

fn default_worker_threads() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4)  // Default to 4 if unable to detect CPU cores
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
        _ => "contamination_results.jsonl".to_string() // fallback
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
    
    // Format: {base name}.clean.jsonl
    format!("{}.clean.jsonl", base_name)
}

// Common function to write a purified file with contaminated lines removed
pub fn write_purified_file(
    input_path: &PathBuf,
    cleaned_output_dir: &PathBuf,
    contaminated_lines: &std::collections::HashSet<usize>,
) -> Result<PathBuf, anyhow::Error> {
    use std::fs::create_dir_all;
    use std::io::{BufWriter, Write, BufRead};
    
    // Ensure output directory exists
    create_dir_all(cleaned_output_dir)?;
    
    let purified_filename = get_purified_filename(input_path);
    let purified_path = cleaned_output_dir.join(&purified_filename);
    
    // Use read_pathbuf_to_mem to handle compressed files
    let data = read_pathbuf_to_mem(input_path)?;
    let mut output_file = BufWriter::new(std::fs::File::create(&purified_path)?);
    
    let mut removed_count = 0;
    for (line_num, line) in data.lines().enumerate() {
        if !contaminated_lines.contains(&line_num) {
            writeln!(output_file, "{}", line?)?;
        } else {
            removed_count += 1;
        }
    }
    
    output_file.flush()?;
    println!("Created purified file: {:?} (removed {} contaminated lines)", 
             purified_path, removed_count);
    
    Ok(purified_path)
}




/*=================================================================
=                             UTILITIES                           =
=================================================================*/




pub fn get_nested_json_val(obj: &Value, key: &String) -> Result<String, Error> {
    let mut current = obj;
    for subkey in key.split('.') {
        current = current.get(subkey)
            .ok_or_else(|| anyhow::anyhow!("Key '{}' not found in JSON object", subkey))?;
    }

    current.as_str()
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
            "p50k" => {
                self.inner.as_ref().unwrap().encode_with_special_tokens(text)
            },
            "cl100k" => {
                self.inner.as_ref().unwrap().encode_with_special_tokens(text)
            }
            "uniseg" => {
                text.split_word_bounds().map(|w| {
                    let mut hasher = DefaultHasher::new();
                    w.hash(&mut hasher);
                    hasher.finish() as usize
                }).collect()
            },
            "word" => {
                // For word mode, tokenize into words and look up IDs
                // Unknown words get a special ID (u32::MAX)
                let cleaned = clean_text(text, &default_punctuation_chars());
                cleaned.split_whitespace()
                    .map(|word| {
                        let word_to_id = self.word_to_id.lock().unwrap();
                        word_to_id.get(word)
                            .copied()
                            .unwrap_or(u32::MAX) as usize
                    })
                    .collect()
            },
            _ => { // default to character level
                text.bytes().map(|b| b as usize).collect()
            },
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


pub fn preprocess_text(text: &str, tokenizer: &OmniTokenizer, punctuation_chars: &str) -> Vec<usize>
{
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
            '-' | '_' | ']' | '[' | '{' | '}' | '=' | '(' | ')' | '>' | '<' | '+' | '*' | '/' => Some(c),
            // Normalize all whitespace to single space
            ' ' | '\t' | '\n' | '\r' => Some(' '),
            // Drop everything else (punctuation, Unicode variants, etc.)
            _ => None
        })
        .collect::<String>()
        .split_whitespace()  // Normalize multiple spaces to single spaces
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

fn contamination_detect(config: &PathBuf) -> Result<(), Error> {
    let config_obj = read_config(config)?;

    match config_obj.mode.as_str() {
        "minhash" => {
            println!("Using MinHash contamination detection...");
            minhash::contamination_detect(&config_obj)
        },
        "toxic" => {
            println!("Using TOXIC contamination detection...");
            toxic::contamination_detect(&config_obj)
        },
        "simple" => {
            println!("Using Simple contamination detection...");
            simple::contamination_detect(&config_obj)
        },
        unknown_mode => {
            println!("Unknown mode: '{}'", unknown_mode);
            println!("Available modes: minhash, toxic, simple");
            Err(anyhow::anyhow!("Unsupported detection mode: {}", unknown_mode))
        }
    }
}


/*=================================================================
=                         CONTAMINATION REVIEW                   =
=================================================================*/

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ContaminationResult {
    training_file: String,
    training_line: usize,
    eval_dataset: String,
    eval_line: usize,
    #[serde(alias = "overlap_ratio")]
    jaccard_similarity: f32,
    #[serde(default)]
    toxic_score: f32,
    #[serde(default)]
    method: Option<String>,
    #[serde(default)]
    matching_ngrams: Option<Vec<String>>,
    #[serde(default)]
    bucket_sizes: Option<Vec<usize>>,
    #[serde(default)]
    bucket_ids: Option<Vec<u64>>,
    #[serde(default)]
    contamination_start_idx: Option<usize>,
    #[serde(default)]
    contamination_end_idx: Option<usize>,
    #[serde(default)]
    training_overlap_text: Option<String>,
    #[serde(default)]
    eval_overlap_text: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
struct GroundTruthRecord {
    text: String,
    source: String,
    id: usize,
    annotation: String,
    ground_truth: String,
}

#[derive(Debug, Clone, Copy, PartialEq)]
#[allow(dead_code)] // Some variants used only in placeholder logic
enum ClassificationType {
    TruePositive,
    FalsePositive,
    TrueNegative,
    FalseNegative,
}

#[derive(Debug)]
struct ClassificationStats {
    true_positives: usize,
    false_positives: usize,
    true_negatives: usize,
    false_negatives: usize,
    total_ground_truth: usize,
    total_detected: usize,
}

impl ClassificationStats {
    fn new() -> Self {
        Self {
            true_positives: 0,
            false_positives: 0,
            true_negatives: 0,
            false_negatives: 0,
            total_ground_truth: 0,
            total_detected: 0,
        }
    }

    fn precision(&self) -> f64 {
        if self.total_detected == 0 {
            0.0
        } else {
            self.true_positives as f64 / self.total_detected as f64
        }
    }

    fn recall(&self) -> f64 {
        let total_contaminated = self.true_positives + self.false_negatives;
        if total_contaminated == 0 {
            0.0
        } else {
            self.true_positives as f64 / total_contaminated as f64
        }
    }

    fn accuracy(&self) -> f64 {
        if self.total_ground_truth == 0 {
            0.0
        } else {
            (self.true_positives + self.true_negatives) as f64 / self.total_ground_truth as f64
        }
    }

    fn f1_score(&self) -> f64 {
        let p = self.precision();
        let r = self.recall();
        if p + r == 0.0 {
            0.0
        } else {
            2.0 * (p * r) / (p + r)
        }
    }
}

fn review_contamination(config: &PathBuf, results_file: Option<&PathBuf>, step: bool, stats: bool, fp: bool, fn_: bool, tp: bool, tn: bool, full: bool) -> Result<(), Error> {
    println!("=== CONTAMINATION REVIEW ===");

    let config_obj = read_config(config)?;

    // Determine results file path
    let results_path = match results_file {
        Some(path) => path.clone(),
        None => config_obj.report_output_dir.join(get_results_filename(&config_obj.mode))
    };

    if !results_path.exists() {
        println!("No contamination results file found at: {:?}", results_path);
        println!("Run contamination detection first, or specify --results-file");
        return Ok(());
    }

    // Load contamination results
    println!("Loading contamination results from: {:?}", results_path);
    let contamination_results = load_contamination_results(&results_path)?;

    if contamination_results.is_empty() {
        println!("No contamination found in results file.");
        return Ok(());
    }

    println!("Found {} contamination instances to review\n", contamination_results.len());

    // Load file content caches
    let training_cache = load_training_files(&config_obj.local_input, &config_obj.content_key)?;
    let eval_cache = load_eval_files(&config_obj.reference_input, &config_obj.content_key)?;
    let ground_truth = load_ground_truth(&config_obj.local_input)?;

    if stats {
        // Calculate and display statistics
        calculate_and_display_stats(&contamination_results, &ground_truth)?;
        return Ok(());
    }

    // Handle filtering flags
    let filter_requested = fp || fn_ || tp || tn;
    let filtered_results = if filter_requested {
        filter_contamination_results(&contamination_results, &ground_truth, &training_cache, fp, fn_, tp, tn)?
    } else {
        contamination_results.clone()
    };

    if filtered_results.is_empty() {
        if filter_requested {
            println!("No contamination instances match the selected filter criteria.");
        } else {
            println!("No contamination found in results file.");
        }
        return Ok(());
    }

    println!("Found {} contamination instances to review{}\n",
             filtered_results.len(),
             if filter_requested { " (after filtering)" } else { "" });

    // Review each contamination case
    for (idx, result) in filtered_results.iter().enumerate() {
        if step && idx > 0 {
            // Wait for user input before showing next case
            println!("\nPress Enter to continue to next contamination case...");
            let mut input = String::new();
            io::stdin().read_line(&mut input).unwrap();

            // Clear the screen
            print!("\x1B[2J\x1B[1;1H");
        }

        println!("{}", "=".repeat(80));
        println!("CONTAMINATION #{} of {}", idx + 1, filtered_results.len());
        println!("{}", "=".repeat(80));

        display_contamination_case(result, &training_cache, &eval_cache, &ground_truth, full, &config_obj)?;
        println!();
    }

    println!("=== REVIEW COMPLETE ===");
    Ok(())
}

fn load_ground_truth(input_dir: &PathBuf) -> Result<Vec<GroundTruthRecord>, Error> {
    let mut ground_truth = Vec::new();
    let training_files = expand_dirs(vec![input_dir.clone()], Some(vec![".jsonl"].as_slice()))?;

    // Load eval data for resolving ground truth when not explicitly present
    let eval_data = load_eval_data_for_ground_truth()?;

    for file_path in training_files {
        let data = read_pathbuf_to_mem(&file_path)?;
        let file_name = file_path.file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("unknown");

        for (line_idx, line) in data.lines().enumerate() {
            let line = line?;
            if !line.trim().is_empty() {
                let json_obj: serde_json::Value = serde_json::from_str(&line)?;

                // Check if this record has ground truth information
                if let Some(text) = json_obj.get("text").and_then(|v| v.as_str()) {
                    let label = json_obj.get("label")
                        .and_then(|v| v.as_str())
                        .unwrap_or("UNKNOWN");

                    let ground_truth_text = if let Some(explicit_gt) = json_obj.get("ground_truth").and_then(|v| v.as_str()) {
                        // Use explicit ground truth field
                        explicit_gt.to_string()
                    } else if let (Some(eval_dataset), Some(eval_line)) = (
                        json_obj.get("eval_dataset").and_then(|v| v.as_str()),
                        json_obj.get("eval_line").and_then(|v| v.as_u64())
                    ) {
                        // Map eval_dataset to file names and resolve from eval data
                        let eval_file_name = format!("{}_test", eval_dataset);
                        let key = format!("{}:{}", eval_file_name, eval_line);
                        eval_data.get(&key)
                            .cloned()
                            .unwrap_or_else(|| {
                                // Try alternative mappings
                                let alt_key = format!("{}:{}", eval_dataset, eval_line);
                                eval_data.get(&alt_key)
                                    .cloned()
                                    .unwrap_or_else(|| format!("Could not resolve {}:{}", eval_dataset, eval_line))
                            })
                    } else if label == "CLEAN" {
                        // For clean records, use the text itself as ground truth
                        text.to_string()
                    } else {
                        // No ground truth available for non-clean records without eval references
                        continue;
                    };

                    ground_truth.push(GroundTruthRecord {
                        text: text.to_string(),
                        source: file_name.to_string(),
                        id: line_idx,
                        annotation: label.to_string(),
                        ground_truth: ground_truth_text,
                    });
                }
            }
        }
    }

    Ok(ground_truth)
}

fn load_eval_data_for_ground_truth() -> Result<std::collections::HashMap<String, String>, Error> {
    let mut eval_data = std::collections::HashMap::new();

    // Load eval files from fixtures/reference
    let eval_dir = std::path::PathBuf::from("fixtures/reference");
    if eval_dir.exists() {
        let eval_files = expand_dirs(vec![eval_dir], Some(vec![".jsonl"].as_slice()))?;

        for file_path in eval_files {
            let file_stem = file_path.file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("unknown");

            let data = read_pathbuf_to_mem(&file_path)?;
            for line in data.lines() {
                let line = line?;
                if !line.trim().is_empty() {
                    if let Ok(json_obj) = serde_json::from_str::<serde_json::Value>(&line) {
                        if let (Some(text), Some(index), Some(record_type)) = (
                            json_obj.get("text").and_then(|v| v.as_str()),
                            json_obj.get("index").and_then(|v| v.as_u64()),
                            json_obj.get("type").and_then(|v| v.as_str())
                        ) {
                            // Only use question records for ground truth
                            if record_type == "question" {
                                let key = format!("{}:{}", file_stem, index);
                                eval_data.insert(key, text.to_string());
                            }
                        }
                    }
                }
            }
        }
    }

    Ok(eval_data)
}

fn calculate_and_display_stats(contamination_results: &[ContaminationResult], ground_truth: &[GroundTruthRecord]) -> Result<(), Error> {
    if ground_truth.is_empty() {
        println!("No ground truth data available. Cannot calculate classification statistics.");
        println!("Found {} contamination detections total.", contamination_results.len());
        return Ok(());
    }

    let mut stats = ClassificationStats::new();

    // Load training cache to get actual text content for mapping
    let training_cache = load_training_files(&std::path::PathBuf::from("fixtures/training"), "text")?;

    // Create a set of detected texts for accurate mapping
    let mut detected_texts: std::collections::HashSet<String> = std::collections::HashSet::new();
    let mut unmapped_detections = 0;
    let mut all_detected_texts: Vec<String> = Vec::new();

    for result in contamination_results {
        if let Some(lines) = training_cache.get(&result.training_file) {
            if result.training_line < lines.len() {
                let text = lines[result.training_line].clone();
                detected_texts.insert(text.clone());
                all_detected_texts.push(text);
            } else {
                unmapped_detections += 1;
            }
        } else {
            unmapped_detections += 1;
        }
    }

    let unique_detections = detected_texts.len();
    let total_detections = all_detected_texts.len();

    // Count ground truth annotations
    for record in ground_truth {
        stats.total_ground_truth += 1;
        let is_contaminated = record.annotation.to_uppercase() == "CONTAMINATED";
        let is_detected = detected_texts.contains(&record.text);

        match (is_contaminated, is_detected) {
            (true, true) => stats.true_positives += 1,
            (true, false) => stats.false_negatives += 1,
            (false, true) => stats.false_positives += 1,
            (false, false) => stats.true_negatives += 1,
        }
    }

    // Use unique detected texts for document-level precision calculation
    stats.total_detected = unique_detections;

    // Calculate detection-level precision for comparison
    let detection_level_precision = if total_detections == 0 {
        0.0
    } else {
        stats.true_positives as f64 / total_detections as f64
    };

    // Display statistics
    println!("=== CLASSIFICATION STATISTICS ===");
    println!();
    println!("CONFUSION MATRIX:");
    println!("                    Predicted");
    println!("                CONTAMINATED  CLEAN");
    println!("    CONTAMINATED    {:>4}     {:>4}    (TP: {}, FN: {})",
             stats.true_positives, stats.false_negatives, stats.true_positives, stats.false_negatives);
    println!("Actual CLEAN        {:>4}     {:>4}    (FP: {}, TN: {})",
             stats.false_positives, stats.true_negatives, stats.false_positives, stats.true_negatives);
    println!();

    println!("PERFORMANCE METRICS:");
    println!("  Document-level Precision: {:.3} ({} TP / {} unique detected)", stats.precision(), stats.true_positives, stats.total_detected);
    if total_detections != unique_detections {
        println!("  Detection-level Precision: {:.3} ({} TP / {} total detections)", detection_level_precision, stats.true_positives, total_detections);
    }
    println!("  Recall:     {:.3} ({} TP / {} actual contaminated)", stats.recall(), stats.true_positives, stats.true_positives + stats.false_negatives);
    println!("  Accuracy:   {:.3} ({} correct / {} total)", stats.accuracy(), stats.true_positives + stats.true_negatives, stats.total_ground_truth);
    println!("  F1 Score:   {:.3}", stats.f1_score());
    println!();

    println!("DETECTION SUMMARY:");
    println!("  Total samples:      {}", stats.total_ground_truth);
    println!("  Ground truth contaminated: {}", stats.true_positives + stats.false_negatives);
    println!("  Ground truth clean:        {}", stats.true_negatives + stats.false_positives);
    println!("  Unique detected texts:      {}", unique_detections);
    println!("  Total detections:           {}", total_detections);
    println!("  Missed contamination:       {}", stats.false_negatives);
    println!("  False alarms:               {}", stats.false_positives);

    if unmapped_detections > 0 {
        println!("  Unmapped detections:        {}", unmapped_detections);
    }

    if total_detections != unique_detections {
        println!("  Duplicate detections:       {}", total_detections - unique_detections);
    }

    Ok(())
}

fn classify_contamination_result(result: &ContaminationResult, ground_truth: &[GroundTruthRecord], training_cache: &HashMap<String, Vec<String>>) -> ClassificationType {
    // Get the actual text from the training file at the specified line
    if let Some(lines) = training_cache.get(&result.training_file) {
        if result.training_line < lines.len() {
            let training_text = &lines[result.training_line];

            // Find the corresponding ground truth record by matching the text content
            if let Some(record) = ground_truth.iter().find(|r| r.text == *training_text) {
                let is_actually_contaminated = record.annotation.to_uppercase() == "CONTAMINATED";
                let is_detected = true; // If it's in results, it was detected

                match (is_actually_contaminated, is_detected) {
                    (true, true) => ClassificationType::TruePositive,
                    (false, true) => ClassificationType::FalsePositive,
                    _ => ClassificationType::TruePositive, // Fallback, shouldn't happen for detected items
                }
            } else {
                eprintln!("DEBUG: Could not find ground truth for text: {}", &training_text[..std::cmp::min(25, training_text.len())]);
                ClassificationType::FalsePositive
            }
        } else {
            eprintln!("DEBUG: Line {} out of bounds for file {}", result.training_line, result.training_file);
            ClassificationType::FalsePositive
        }
    } else {
        eprintln!("DEBUG: Could not find training file: {}", result.training_file);
        ClassificationType::FalsePositive
    }
}

fn filter_contamination_results(
    contamination_results: &[ContaminationResult],
    ground_truth: &[GroundTruthRecord],
    training_cache: &HashMap<String, Vec<String>>,
    show_fp: bool,
    show_fn: bool,
    show_tp: bool,
    show_tn: bool
) -> Result<Vec<ContaminationResult>, Error> {
    let mut filtered = Vec::new();

    // Handle TN (True Negatives) separately since they're not in contamination results
    if show_tn {
        // True negatives are ground truth CLEAN records that weren't detected
        // We'll need to create placeholder contamination results for display
        let detected_ids: std::collections::HashSet<usize> = contamination_results.iter()
            .filter_map(|result| {
                ground_truth.iter().find(|r| {
                    let training_file_matches = result.training_file == "training_sample.jsonl" ||
                        result.training_file.contains(&r.source) ||
                        r.source == "training_data";
                    training_file_matches && ((r.id as isize - 7) as usize == result.training_line)
                }).map(|r| r.id)
            })
            .collect();

        for record in ground_truth {
            if record.annotation.to_uppercase() == "CLEAN" && !detected_ids.contains(&record.id) {
                // Create a placeholder result for true negatives
                let placeholder = ContaminationResult {
                    training_file: format!("{}.jsonl", record.source),
                    training_line: (record.id as isize - 7) as usize,
                    eval_dataset: "N/A".to_string(),
                    eval_line: 0,
                    jaccard_similarity: 0.0,
                    toxic_score: 0.0,
                    method: Some("true_negative".to_string()),
                    matching_ngrams: None,
                    bucket_sizes: None,
                    bucket_ids: None,
                    contamination_start_idx: None,
                    contamination_end_idx: None,
                    training_overlap_text: None,
                    eval_overlap_text: None,
                };
                filtered.push(placeholder);
            }
        }
    }

    // Filter actual contamination results
    for result in contamination_results {
        let classification = classify_contamination_result(result, ground_truth, training_cache);

        let should_include = match classification {
            ClassificationType::TruePositive => show_tp,
            ClassificationType::FalsePositive => show_fp,
            ClassificationType::TrueNegative => show_tn,
            ClassificationType::FalseNegative => show_fn,
        };

        if should_include {
            filtered.push(result.clone());
        }
    }

    // Handle FN (False Negatives) - contaminated records that weren't detected
    if show_fn {
        // Create a set of detected texts to avoid complex ID mapping
        let detected_texts: std::collections::HashSet<String> = contamination_results.iter()
            .filter_map(|result| {
                training_cache.get(&result.training_file)
                    .and_then(|lines| lines.get(result.training_line))
                    .cloned()
            })
            .collect();

        for record in ground_truth {
            if record.annotation.to_uppercase() == "CONTAMINATED" && !detected_texts.contains(&record.text) {
                // Find which file this record is in by searching the training cache
                let mut found_file = None;
                let mut found_line = 0;

                for (filename, lines) in training_cache.iter() {
                    if let Some(line_idx) = lines.iter().position(|line| line == &record.text) {
                        found_file = Some(filename.clone());
                        found_line = line_idx;
                        break;
                    }
                }

                let placeholder = ContaminationResult {
                    training_file: found_file.unwrap_or_else(|| format!("unknown_{}.jsonl", record.source)),
                    training_line: found_line,
                    eval_dataset: "N/A".to_string(),
                    eval_line: 0,
                    jaccard_similarity: 0.0,
                    toxic_score: 0.0,
                    method: Some("false_negative".to_string()),
                    matching_ngrams: None,
                    bucket_sizes: None,
                    bucket_ids: None,
                    contamination_start_idx: None,
                    contamination_end_idx: None,
                    training_overlap_text: None,
                    eval_overlap_text: None,
                };
                filtered.push(placeholder);
            }
        }
    }

    Ok(filtered)
}

fn load_contamination_results(results_path: &PathBuf) -> Result<Vec<ContaminationResult>, Error> {
    let data = read_pathbuf_to_mem(results_path)?;
    let mut results = Vec::new();

    for line in data.lines() {
        let line = line?;
        if !line.trim().is_empty() {
            let result: ContaminationResult = serde_json::from_str(&line)?;
            results.push(result);
        }
    }

    Ok(results)
}

fn load_training_files(input_dir: &PathBuf, content_key: &str) -> Result<HashMap<String, Vec<String>>, Error> {
    let mut cache = HashMap::new();
    let training_files = expand_dirs(vec![input_dir.clone()], Some(vec![".jsonl", ".gz"].as_slice()))?;

    for file_path in training_files {
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

        let data = read_pathbuf_to_mem(&file_path)?;
        let mut lines = Vec::new();

        for (line_num, line) in data.lines().enumerate() {
            let line = line?;
            if !line.trim().is_empty() {
                let json_obj: Value = match serde_json::from_str(&line) {
                    Ok(obj) => obj,
                    Err(e) => {
                        eprintln!("JSON parse error at line {}: {}", line_num, e);
                        eprintln!("Line content: {}", &line[..std::cmp::min(100, line.len())]);
                        return Err(e.into());
                    }
                };
                let text = match get_nested_json_val(&json_obj, &content_key.to_string()) {
                    Ok(t) => t,
                    Err(_) => {
                        // Skip files that don't have the expected schema
                        eprintln!("Skipping file {:?} - doesn't have expected key '{}'", file_path, content_key);
                        break;
                    }
                };
                lines.push(text);
            }
        }

        println!("Caching training file: {} ({} lines)", file_name, lines.len());
        cache.insert(file_name, lines);
    }

    println!("Training file cache contains {} files:", cache.len());
    for key in cache.keys() {
        println!("  - {}", key);
    }

    Ok(cache)
}

fn load_eval_files(reference_dir: &PathBuf, content_key: &str) -> Result<HashMap<String, Vec<String>>, Error> {
    let mut cache = HashMap::new();
    let eval_files = expand_dirs(vec![reference_dir.clone()], Some(vec![".jsonl", ".gz"].as_slice()))?;

    for file_path in eval_files {
        let file_name = if file_path.extension().and_then(|s| s.to_str()) == Some("gz") {
            // For .gz files, use file_stem to get the .jsonl name
            file_path
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("unknown")
                .to_string()
        } else {
            // For regular files, use file_stem to get the name without .jsonl
            file_path
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("unknown")
                .to_string()
        };

        let data = read_pathbuf_to_mem(&file_path)?;
        let mut lines = Vec::new();

        for line in data.lines() {
            let line = line?;
            if !line.trim().is_empty() {
                let json_obj: Value = serde_json::from_str(&line)?;
                let text = match get_nested_json_val(&json_obj, &content_key.to_string()) {
                    Ok(t) => t,
                    Err(_) => {
                        // Skip files that don't have the expected schema
                        eprintln!("Skipping file {:?} - doesn't have expected key '{}'", file_path, content_key);
                        break;
                    }
                };
                lines.push(text);
            }
        }

        cache.insert(file_name, lines);
    }

    Ok(cache)
}

fn truncate_text(text: &str, max_lines: usize) -> String {
    let lines: Vec<&str> = text.lines().collect();

    if lines.len() <= max_lines {
        text.to_string()
    } else {
        let truncated_lines = &lines[..max_lines];
        let mut result = truncated_lines.join("\n");
        result.push_str(&format!("\n... [truncated: showing {} of {} lines, use --full to see all]", max_lines, lines.len()));
        result
    }
}

/// Extract contaminated text segment using token indices with context
fn extract_contaminated_segment_with_context(
    original_text: &str,
    start_idx: usize,
    end_idx: usize,
    config: &Config,
    context_words: usize
) -> Option<String> {
    // Initialize tokenizer based on config
    let tokenizer = match OmniTokenizer::new(&config.tokenizer_str) {
        Ok(t) => t,
        Err(_) => return None,
    };

    // First, clean the text the same way as during detection
    let cleaned_text = clean_text(original_text, &config.punctuation_chars);

    // For word tokenizer, tokens are just words split by whitespace
    if config.tokenizer_str == "word" {
        let words: Vec<&str> = cleaned_text.split_whitespace().collect();

        // Check if indices are valid
        if start_idx >= words.len() || end_idx > words.len() || start_idx >= end_idx {
            return None;
        }

        // Calculate context boundaries
        let context_start = start_idx.saturating_sub(context_words);
        let context_end = (end_idx + context_words).min(words.len());

        // Build the output with highlighted contamination
        let mut result = String::new();

        // Add leading context
        if context_start < start_idx {
            result.push_str("... ");
            result.push_str(&words[context_start..start_idx].join(" "));
            result.push_str(" ");
        }

        // Add contaminated section with highlighting
        result.push_str("【");
        result.push_str(&words[start_idx..end_idx].join(" "));
        result.push_str("】");

        // Add trailing context
        if end_idx < context_end {
            result.push_str(" ");
            result.push_str(&words[end_idx..context_end].join(" "));
            result.push_str(" ...");
        }

        return Some(result);
    }

    // For other tokenizers, we need to tokenize and then decode
    let Ok(tokens) = std::panic::catch_unwind(|| tokenizer.encode(&cleaned_text)) else {
        return None;
    };

    // Check if indices are valid
    if start_idx >= tokens.len() || end_idx > tokens.len() || start_idx >= end_idx {
        return None;
    }

    // For BPE tokenizers, try to decode with context
    if let Some(inner) = tokenizer.inner.as_ref() {
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

    // Fallback: show token IDs
    let token_str = tokens[start_idx..end_idx]
        .iter()
        .map(|t| t.to_string())
        .collect::<Vec<_>>()
        .join(", ");

    Some(format!("[Tokens: {}]", token_str))
}

fn display_contamination_case(
    result: &ContaminationResult,
    training_cache: &HashMap<String, Vec<String>>,
    eval_cache: &HashMap<String, Vec<String>>,
    ground_truth: &[GroundTruthRecord],
    full: bool,
    config: &Config
) -> Result<(), Error> {
    println!("📁 TRAINING FILE: {}", result.training_file);

    // Handle special cases for FN and TN placeholders
    match result.method.as_deref() {
        Some("false_negative") => {
            println!("🚨 FALSE NEGATIVE: Contaminated but not detected");
            println!("📋 EVAL DATASET:  (Not applicable - missed detection)");
            println!();
        },
        Some("true_negative") => {
            println!("✅ TRUE NEGATIVE: Clean and correctly not detected");
            println!("📋 EVAL DATASET:  (Not applicable - correctly not flagged)");
            println!();
        },
        _ => {
            println!("📋 EVAL DATASET:  {}", result.eval_dataset);
            let similarity_label = match result.method.as_deref() {
                Some("toxic") | Some("simple") => "OVERLAP RATIO",
                _ => "JACCARD SIM"
            };
            println!("🎯 {}:   {:.3}", similarity_label, result.jaccard_similarity);
            if result.toxic_score > 0.0 {
                println!("🧪 TOXIC SCORE:    {:.3}", result.toxic_score);
            }
            println!();
        }
    }

    // Get training text
    let training_text = match training_cache.get(&result.training_file) {
        Some(lines) => {
            if result.training_line < lines.len() {
                &lines[result.training_line]
            } else {
                "❌ Training line index out of bounds"
            }
        }
        None => "❌ Training file not found"
    };

    // Get eval text (or ground truth for false negatives)
    let eval_text = match result.method.as_deref() {
        Some("false_negative") => {
            // For false negatives, find the ground truth text
            ground_truth.iter()
                .find(|gt| gt.text == *training_text)
                .map(|gt| gt.ground_truth.as_str())
                .unwrap_or("❌ Ground truth not found")
        }
        _ => {
            match eval_cache.get(&result.eval_dataset) {
                Some(lines) => {
                    if result.eval_line < lines.len() {
                        &lines[result.eval_line]
                    } else {
                        "❌ Eval line index out of bounds"
                    }
                }
                None => "❌ Eval file not found"
            }
        }
    };

    // Display side by side
    println!("🔍 TRAINING TEXT (line {}):", result.training_line);

    // Apply truncation if not in full mode
    let displayed_text = if full {
        training_text.to_string()
    } else {
        truncate_text(training_text, 25)
    };

    println!("   \"{}\"", displayed_text);

    // Show the actual contaminated text segment if we have token indices
    if let (Some(start_idx), Some(end_idx)) = (result.contamination_start_idx, result.contamination_end_idx) {
        println!();
        println!("📍 CONTAMINATED SEGMENT (tokens {} to {}):", start_idx, end_idx);

        // Try to extract the contaminated segment with 10 words of context on each side
        if let Some(segment) = extract_contaminated_segment_with_context(training_text, start_idx, end_idx, config, 10) {
            println!("   {}", segment);
        } else {
            println!("   [Unable to extract segment - {} tokens]", end_idx - start_idx);
        };
    }

    println!();

    match result.method.as_deref() {
        Some("false_negative") => {
            println!("🔍 GROUND TRUTH (expected):");
            println!("   \"{}\"", eval_text);
        }
        _ => {
            println!("🔍 EVAL TEXT (line {}):", result.eval_line);
            println!("   \"{}\"", eval_text);
        }
    }
    println!();

    // Check similarity based on method type
    match result.method.as_deref() {
        Some("false_negative") => {
            println!("❌ MISSED CONTAMINATION - This should have been detected");
        },
        Some("true_negative") => {
            println!("✅ CORRECTLY IDENTIFIED AS CLEAN - No contamination detected");
        },
        _ => {
            // Check if they're identical
            if training_text == eval_text {
                println!("✅ EXACT MATCH - Definite contamination");
            } else if result.jaccard_similarity > 0.9 {
                println!("⚠️  VERY HIGH SIMILARITY - Likely contamination");
            } else if result.jaccard_similarity > 0.6 {
                println!("⚠️  HIGH SIMILARITY - Likely contamination");
            } else if result.jaccard_similarity > 0.3 {
                println!("🤔 MODERATE SIMILARITY - Manual review needed");
            } else {
                println!("🔍 LOW SIMILARITY - Edge case detection");
            }
        }
    }

    // Display matching ngrams with bucket heat if available (debug mode data)
    if let Some(ref ngrams) = result.matching_ngrams {
        if !ngrams.is_empty() {
            println!();
            println!("🔗 MATCHING N-GRAMS WITH BUCKET HEAT:");

            // Calculate bucket heat statistics for weighting analysis
            let bucket_heats: Vec<usize> = if let Some(ref sizes) = result.bucket_sizes {
                sizes.clone()
            } else {
                vec![0; ngrams.len()]
            };

            for (i, ngram) in ngrams.iter().enumerate() {
                let heat = bucket_heats.get(i).unwrap_or(&0);
                let heat_indicator = match *heat {
                    1 => "🟢", // Cold (unique)
                    2..=5 => "🟡", // Warm
                    6..=15 => "🟠", // Hot
                    _ => "🔴", // Very hot
                };
                let rarity_score = if *heat > 0 { 1.0 / (*heat as f64) } else { 0.0 };

                // Include bucket ID if available
                let bucket_id_display = if let Some(ref bucket_ids) = result.bucket_ids {
                    bucket_ids.get(i).map(|id| format!(" bucket_id:{}", id)).unwrap_or_default()
                } else {
                    String::new()
                };

                println!("   {}: \"{}\" {} heat:{} rarity:{:.3}{}",
                         i + 1, ngram, heat_indicator, heat, rarity_score, bucket_id_display);
            }

            println!();
        }
    }

    Ok(())
}



/*=================================================================
=                                 MAIN                            =
=================================================================*/


fn main() {
    let args = ArgParser::parse();
    let threads = args.threads;
    if threads != 0 {
        std::env::set_var("RAYON_NUM_THREADS", threads.to_string());
    }

    let result = match &args.command {
        Commands::Detect {config} => {
            contamination_detect(config)
        }

        Commands::Review {config, results_file, step, stats, fp, fn_, tp, tn, full} => {
            review_contamination(config, results_file.as_ref(), *step, *stats, *fp, *fn_, *tp, *tn, *full)
        }

        Commands::Daemon {config, port} => {
            let runtime = tokio::runtime::Runtime::new().unwrap();
            runtime.block_on(daemon::run_daemon(config.clone(), *port))
        }
    };
    result.unwrap()
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
