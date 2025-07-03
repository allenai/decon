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

// Internal crate imports
use mj_io::{expand_dirs, read_pathbuf_to_mem};

// Internal modules
mod minhash;
mod toxic;



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
        step: bool
    }

}

/*=================================================================
=                             CONFIG                              =
=================================================================*/

#[derive(Debug, Serialize, Deserialize)]
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
    pub output_dir: PathBuf,

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
    #[serde(default = "default_toxic_poison_scale")]
    pub toxic_poison_scale: f32,
    #[serde(default = "default_skip_hot_bucket_threshold")]
    pub skip_hot_bucket_threshold: i32,

    // Debug options
    #[serde(default = "default_debug")]
    pub debug: bool

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

fn read_config(config_path: &PathBuf) -> Result<Config, Error> {
    let contents = read_pathbuf_to_mem(config_path).unwrap();
    let config: Config = serde_yaml::from_reader(contents).unwrap();
    Ok(config)
}

pub fn get_results_filename(mode: &str) -> String {
    match mode {
        "minhash" => "contamination_results.jsonl".to_string(),
        "toxic" => "toxic_contamination_results.jsonl".to_string(),
        _ => "contamination_results.jsonl".to_string() // fallback
    }
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
    tokenizer_name: String,
    inner: CoreBPE
}

impl OmniTokenizer {
    pub fn new(tokenizer_name: &str) -> Result<Self, Error> {
        let inner_tokenizer = match tokenizer_name.to_string().as_str() {
            "p50k" => p50k_base().unwrap(),
            "cl100k" => cl100k_base().unwrap(),
            _ => {
                println!("Tokenizer {:?} <--- BE CAREFUL HERE", tokenizer_name.to_string());
                p50k_base().unwrap()
            }
        };
        Ok(OmniTokenizer { tokenizer_name: tokenizer_name.to_string(), inner: inner_tokenizer})
    }

    pub fn encode(&self, text: &str) -> Vec<usize> {
        match self.tokenizer_name.as_str() {
            "p50k" => {
                self.inner.encode_with_special_tokens(text)
            },
            "cl100k" => {
                self.inner.encode_with_special_tokens(text)
            }
            "uniseg" => {
                text.split_word_bounds().map(|w| {
                    let mut hasher = DefaultHasher::new();
                    w.hash(&mut hasher);
                    hasher.finish() as usize
                }).collect()
            },
            _ => { // default to character level
                text.bytes().map(|b| b as usize).collect()
            },
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


pub fn preprocess_text(text: &str, tokenizer: &OmniTokenizer) -> Vec<usize>
{
    let cleaned_text = clean_text(text);
    // println!("    🔧 Original text: \"{}\"", text);
    // println!("    🔧 Cleaned text:  \"{}\"", cleaned_text);
    let tokens = tokenizer.encode(&cleaned_text);
    // println!("    🔧 Tokens: {:?}", tokens);
    tokens
}


pub fn clean_text(text: &str) -> String {
    // SlimPajama text cleaning process

    // Convert the document to lowercase
    let mut text = text.to_lowercase();

    // Remove punctuation
    let punctuation: &[_] = &['!', '"', '#', '$', '%', '&', '\'', '(', ')', '*', '+', ',', '-', '.', '/', ':', ';', '<', '=', '>', '?', '@', '[', '\\', ']', '^', '_', '`', '{', '|', '}', '~'];
    text.retain(|c| !punctuation.contains(&c));

    // Replace multiple whitespace characters with a single space
    let re = Regex::new(r"\s+").unwrap();
    text = re.replace_all(&text, " ").to_string();

    // Trim leading and trailing whitespace
    text.trim().to_string()
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
        unknown_mode => {
            println!("Unknown mode: '{}'", unknown_mode);
            println!("Available modes: minhash, toxic");
            Err(anyhow::anyhow!("Unsupported detection mode: {}", unknown_mode))
        }
    }
}


/*=================================================================
=                         CONTAMINATION REVIEW                   =
=================================================================*/

#[derive(Debug, Serialize, Deserialize)]
struct ContaminationResult {
    training_file: String,
    training_line: usize,
    eval_dataset: String,
    eval_line: usize,
    #[serde(alias = "overlap_ratio")]
    jaccard_similarity: f32,
    #[serde(default)]
    method: Option<String>,
    #[serde(default)]
    matching_ngrams: Option<Vec<String>>,
    #[serde(default)]
    bucket_sizes: Option<Vec<usize>>,
}

fn review_contamination(config: &PathBuf, results_file: Option<&PathBuf>, step: bool) -> Result<(), Error> {
    println!("=== CONTAMINATION REVIEW ===");

    let config_obj = read_config(config)?;

    // Determine results file path
    let results_path = match results_file {
        Some(path) => path.clone(),
        None => config_obj.output_dir.join(get_results_filename(&config_obj.mode))
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

    // Review each contamination case
    for (idx, result) in contamination_results.iter().enumerate() {
        if step && idx > 0 {
            // Wait for user input before showing next case
            println!("\nPress Enter to continue to next contamination case...");
            let mut input = String::new();
            io::stdin().read_line(&mut input).unwrap();
            
            // Clear the screen
            print!("\x1B[2J\x1B[1;1H");
        }
        
        println!("{}", "=".repeat(80));
        println!("CONTAMINATION #{} of {}", idx + 1, contamination_results.len());
        println!("{}", "=".repeat(80));

        display_contamination_case(result, &training_cache, &eval_cache)?;
        println!();
    }

    println!("=== REVIEW COMPLETE ===");
    Ok(())
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

        cache.insert(file_name, lines);
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

fn display_contamination_case(
    result: &ContaminationResult,
    training_cache: &HashMap<String, Vec<String>>,
    eval_cache: &HashMap<String, Vec<String>>
) -> Result<(), Error> {
    println!("📁 TRAINING FILE: {}", result.training_file);
    println!("📋 EVAL DATASET:  {}", result.eval_dataset);
    let similarity_label = match result.method.as_deref() {
        Some("toxic") => "OVERLAP RATIO",
        _ => "JACCARD SIM"
    };
    println!("🎯 {}:   {:.3}", similarity_label, result.jaccard_similarity);
    println!();

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

    // Get eval text
    let eval_text = match eval_cache.get(&result.eval_dataset) {
        Some(lines) => {
            if result.eval_line < lines.len() {
                &lines[result.eval_line]
            } else {
                "❌ Eval line index out of bounds"
            }
        }
        None => "❌ Eval file not found"
    };

    // Display side by side
    println!("🔍 TRAINING TEXT (line {}):", result.training_line);
    println!("   \"{}\"", training_text);
    println!();
    println!("🔍 EVAL TEXT (line {}):", result.eval_line);
    println!("   \"{}\"", eval_text);
    println!();

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

    // Display matching ngrams with bucket sizes if available (debug mode data)
    if let Some(ref ngrams) = result.matching_ngrams {
        if !ngrams.is_empty() {
            println!();
            println!("🔗 MATCHING N-GRAMS:");
            for (i, ngram) in ngrams.iter().enumerate() {
                let bucket_info = if let Some(ref sizes) = result.bucket_sizes {
                    if i < sizes.len() {
                        format!(" (bucket size: {})", sizes[i])
                    } else {
                        String::new()
                    }
                } else {
                    String::new()
                };
                println!("   {}: \"{}\"{}", i + 1, ngram, bucket_info);
            }
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

        Commands::Review {config, results_file, step} => {
            review_contamination(config, results_file.as_ref(), *step)
        }

    };
    result.unwrap()
}
