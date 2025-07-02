// External crates
use ahash::RandomState;
use anyhow::{Error, Result};
use clap::{Parser, Subcommand};
use dashmap::{DashMap};
use ndarray::Array1;
use rand::prelude::*;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha20Rng;
use rayon::prelude::*;
use regex::Regex;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use serde_yaml;
use sha2::{Digest, Sha256};
use tiktoken_rs::{CoreBPE, p50k_base, cl100k_base};
use unicode_segmentation::UnicodeSegmentation;

// Standard library
use std::collections::{HashMap, VecDeque};
use std::fs::{create_dir_all};
use std::hash::{DefaultHasher, Hash, Hasher};
use std::io::{BufRead};
use std::option::Option;
use std::panic::catch_unwind;
use std::path::PathBuf;
use std::time::Instant;

// Internal crate imports
use mj_io::{expand_dirs, read_pathbuf_to_mem, write_mem_to_pathbuf, build_pbar};

const BIG_PRIME: u64 = 18446744073709551557;
const MAX_HASH: u64 = BIG_PRIME;


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
    ContaminationDetect {
        #[arg(required=true, long)]
        config: PathBuf
    },

    ReviewContamination {
        #[arg(required=true, long)]
        config: PathBuf,

        #[arg(long)]
        results_file: Option<PathBuf>
    }

}

/*=================================================================
=                             CONFIG                              =
=================================================================*/

#[derive(Debug, Serialize, Deserialize)]
struct Config {
    name: String,
    // Minhash parameters
    num_bands: usize,
    band_size: usize,
    ngram_size: usize,
    tokenizer_str: String,
    #[serde(default)]
    hash_seed: usize,

    // Engineery things
    num_sig_chunks: usize,
    num_docs: usize,
    max_lines_per_path: usize,
    content_key: String,

    // Local directories
    local_input: PathBuf,
    reference_input: PathBuf,
    working_dir: PathBuf,
    output_dir: PathBuf,

    // Remote directories
    remote_input: PathBuf,
    remote_working_dir: PathBuf,
    remote_output: PathBuf,

    // Fancy options
    #[serde(default)]
    exact_override: bool,
    #[serde(default)]
    concat_key: Option<Vec<String>>,
    #[serde(default)]
    annotate_only: bool,
    #[serde(default = "default_jaccard_threshold")]
    jaccard_similarity_threshold: f32

}

fn default_jaccard_threshold() -> f32 {
    0.5
}

fn read_config(config_path: &PathBuf) -> Result<Config, Error> {
    let contents = read_pathbuf_to_mem(config_path).unwrap();
    let config: Config = serde_yaml::from_reader(contents).unwrap();
    Ok(config)
}




/*=================================================================
=                             UTILITIES                           =
=================================================================*/


#[allow(dead_code)]
fn get_concat_val(obj: &Value, concat_key: &Vec<String>) -> Result<Vec<String>, Error> {
    let mut concat_val: Vec<String> = Vec::new();

    for k in concat_key {
        concat_val.push(get_nested_json_val(obj, k).unwrap());
    }

    Ok(concat_val)
}


fn get_nested_json_val(obj: &Value, key: &String) -> Result<String, Error> {
    let mut current = obj;
    for subkey in key.split('.') {
        current = current.get(subkey).unwrap();
    }

    Ok(current.as_str().unwrap().to_string())
}


struct OmniTokenizer {
    tokenizer_name: String,
    inner: CoreBPE
}

impl OmniTokenizer {
    fn new(tokenizer_name: &str) -> Result<Self, Error> {
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

    fn encode(&self, text: &str) -> Vec<usize> {
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

}



fn hash_object<T: Hash>(obj: &T) -> usize {
    let mut hasher = DefaultHasher::new();
    obj.hash(&mut hasher);
    hasher.finish() as usize
}


fn preprocess_text(text: &str, tokenizer: &OmniTokenizer) -> Vec<usize>
{
    let cleaned_text = clean_text(text);
    // println!("    🔧 Original text: \"{}\"", text);
    // println!("    🔧 Cleaned text:  \"{}\"", cleaned_text);
    let tokens = tokenizer.encode(&cleaned_text);
    // println!("    🔧 Tokens: {:?}", tokens);
    tokens
}


fn clean_text(text: &str) -> String {
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

fn get_hash_vals_from_tokens(tokens: Vec<usize>, perm_seeds: &Vec<u64>, ngram_size: usize) -> Array1<u64> {
    let a = _init_permutations(perm_seeds);
    let n = perm_seeds.len();

    let mut hash_vals = Array1::ones(n) * MAX_HASH;
    let mut ngram: VecDeque<usize> = VecDeque::with_capacity(ngram_size);
    let mut ngram_count = 0;
    // println!("    🔧 Computing MinHash with {} hash functions, ngram_size={}", n, ngram_size);

    for token in tokens {
        ngram.push_back(token);
        if ngram.len() >= ngram_size {
            ngram_count += 1;
            // println!("    🔧 Processing ngram #{}: {:?}", ngram_count, ngram);
            hash_vals = _update_hash_vals(hash_vals, &a, &ngram);
            ngram.pop_front();
        }
    }
    hash_vals = if ngram_count == 0 {
        // println!("    🔧 Short document - processing final ngram: {:?}", ngram);
        _update_hash_vals(hash_vals, &a, &ngram) // short document, still wanna hash it
    } else {
        hash_vals
    };

    // println!("    🔧 Final MinHash signature: {:?}", hash_vals);
    hash_vals
}



fn _init_permutations(seeds: &Vec<u64>) -> Array1<u128> {
    // Initialize the permutations needed for each minhash
    let n = seeds.len();
    let mut a = Array1::zeros(n);
    for (i, &seed) in seeds.iter().enumerate() {
        let mut rng = ChaCha20Rng::seed_from_u64(seed);
        a[i] = rng.gen::<u128>() as u128;
    }
    a
}

#[allow(dead_code)]
fn rand_u64s(seed: u64, output_size: usize) -> Vec<u64> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut output: Vec<u64> = Vec::new();
    for _i in 0..output_size {
        output.push(rng.gen::<u64>());
    }
    output
}


fn _update_hash_vals(mut hash_vals: Array1<u64>, a: &Array1<u128>, ngram: &VecDeque<usize>) -> Array1<u64> {

    // hash the vecdeque as a u128
    let hash_a = RandomState::with_seed(123);
    let hash_b = RandomState::with_seed(456);
    let hash_val_a = hash_a.hash_one(ngram);
    let hash_val_b = hash_b.hash_one(ngram);
    let cur_hash = ((hash_val_a as u128) << 64) | (hash_val_b as u128);

    // then multiply by a (mod 2^128) and take top 64 most significant bits
    let phv: Array1<u64> = a.mapv(|x| (x.wrapping_mul(cur_hash) >> 64) as u64);
    hash_vals.zip_mut_with(&phv, |x, y| *x = std::cmp::min(*x, *y));

    hash_vals

}

fn _expand_band_seeds(band_seeds: &Vec<u32>, band_size: usize) -> Vec<u64> {
    // Each "band seed" is expanded here to band_size random u64s, and flattened. (used to seed permutations)
    // Probably like no collisions here, so let's just not worry about that ;)

    let mut perm_seeds: Vec<u64> = Vec::new();
    for band_seed in band_seeds.iter() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(*band_seed as u64);
        for _i in 0..band_size {
            perm_seeds.push(rng.next_u64());
        }
    }
    perm_seeds
}


/*=================================================================
=                         CONTAMINATION DETECTION                =
=================================================================*/

// Reference band storage: maps band signature to list of (eval_name, line_num)
type ReferenceBands = DashMap<Vec<u8>, Vec<(String, usize)>>;

// Reference signatures storage: maps (eval_name, line_num) to full signature
type ReferenceSignatures = DashMap<(String, usize), Array1<u64>>;

fn contamination_detect(config: &PathBuf) -> Result<(), Error> {
    println!("Starting contamination detection...");
    let start_main = Instant::now();

    let config_obj = read_config(config)?;

    // Step 1: Process reference datasets and build in-memory band index
    println!("Processing reference datasets...");
    let (reference_bands, reference_signatures) = build_reference_index(&config_obj)?;
    println!("Built reference index with {} bands and {} signatures",
             reference_bands.len(), reference_signatures.len());

    // Step 2: Process training data and detect contamination
    println!("Processing training data for contamination detection...");
    detect_contamination_in_training_data(&config_obj, &reference_bands, &reference_signatures)?;

    println!("Contamination detection completed in {:?} seconds", start_main.elapsed().as_secs());
    Ok(())
}

fn build_reference_index(config: &Config) -> Result<(ReferenceBands, ReferenceSignatures), Error> {
    let reference_bands: ReferenceBands = DashMap::new();
    let reference_signatures: ReferenceSignatures = DashMap::new();

    // Set up hashing parameters
    let band_seeds: Vec<u32> = _expand_band_seeds(&vec![config.hash_seed as u32], config.num_bands)
        .into_iter()
        .map(|x| x as u32)
        .collect();
    let perm_seeds = _expand_band_seeds(&band_seeds, config.band_size);

    // Find all reference files
    let reference_files = expand_dirs(vec![config.reference_input.clone()], Some(vec![".jsonl"].as_slice()))?;
    let pbar = build_pbar(reference_files.len(), "Reference files");

    reference_files.par_iter().for_each(|file_path| {
        if let Err(e) = process_reference_file(
            file_path,
            &band_seeds,
            &perm_seeds,
            config.band_size,
            config.ngram_size,
            &config.tokenizer_str,
            &config.content_key,
            &reference_bands,
            &reference_signatures,
            config.exact_override
        ) {
            println!("Error processing reference file {:?}: {:?}", file_path, e);
        }
        pbar.inc(1);
    });

    Ok((reference_bands, reference_signatures))
}

fn process_reference_file(
    file_path: &PathBuf,
    band_seeds: &Vec<u32>,
    perm_seeds: &Vec<u64>,
    band_size: usize,
    ngram_size: usize,
    tokenizer_str: &str,
    content_key: &str,
    reference_bands: &ReferenceBands,
    reference_signatures: &ReferenceSignatures,
    exact_override: bool
) -> Result<(), Error> {
    let data = read_pathbuf_to_mem(file_path)?;
    let tokenizer = OmniTokenizer::new(tokenizer_str)?;
    let num_bands = band_seeds.len();

    // Extract eval name from filename
    let eval_name = file_path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("unknown")
        .to_string();

    println!("Loading hashes for eval dataset: {}", eval_name);

    let mut lines_processed = 0;
    for (line_num, line) in data.lines().enumerate() {
        let line = line?;
        let json_obj: Value = serde_json::from_str(&line)?;
        let line_text = get_nested_json_val(&json_obj, &content_key.to_string())?;
        lines_processed += 1;

        let hash_vals = if exact_override {
            let Ok(tokens) = catch_unwind(|| preprocess_text(&line_text, &tokenizer)) else {
                println!("Tokenization failed on {:?} | line {:?}", file_path, line_num);
                continue;
            };
            get_hash_vals_from_tokens(tokens, perm_seeds, ngram_size)
        } else {
            let n = perm_seeds.len();
            let mut hash_vals: Array1<u64> = Array1::ones(n);
            hash_vals = hash_vals * (hash_object(&line_text) as u64);
            hash_vals
        };

        // Store full signature for Jaccard similarity calculation
        reference_signatures.insert((eval_name.clone(), line_num), hash_vals.clone());

        // Generate bands and store in reference index
        let bands = hash_vals.into_shape((num_bands, band_size))?;
        for (_band_idx, row) in bands.rows().into_iter().enumerate() {
            let mut hasher = Sha256::new();
            hasher.update(bytemuck::cast_slice(row.as_slice().unwrap()));
            let hash = hasher.finalize();
            let band_signature = hash[..8].to_vec(); // Truncate to 8 bytes for efficiency

            // if line_num < 3 {  // Only log first few lines to avoid spam
            //     println!("    🔧 [REF] {}[{}] Band {}: values={:?} -> signature={:02x?}",
            //             eval_name, line_num, band_idx, row.as_slice().unwrap(), band_signature);
            // }

            reference_bands
                .entry(band_signature)
                .or_default()
                .push((eval_name.clone(), line_num));
        }
    }

    println!("  → Processed {} lines from {}", lines_processed, eval_name);
    Ok(())
}

fn detect_contamination_in_training_data(
    config: &Config,
    reference_bands: &ReferenceBands,
    reference_signatures: &ReferenceSignatures
) -> Result<(), Error> {
    // Set up hashing parameters
    let band_seeds: Vec<u32> = _expand_band_seeds(&vec![config.hash_seed as u32], config.num_bands)
        .into_iter()
        .map(|x| x as u32)
        .collect();
    let perm_seeds = _expand_band_seeds(&band_seeds, config.band_size);

    // Find all training files
    let training_files = expand_dirs(vec![config.local_input.clone()], Some(vec![".jsonl"].as_slice()))?;
    let pbar = build_pbar(training_files.len(), "Training files");

    let contamination_results: DashMap<String, Vec<(usize, String, usize, f32)>> = DashMap::new();

    training_files.par_iter().for_each(|file_path| {
        let file_name = file_path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("unknown")
            .to_string();

        if let Err(e) = process_training_file(
            file_path,
            &file_name,
            &band_seeds,
            &perm_seeds,
            config.band_size,
            config.ngram_size,
            &config.tokenizer_str,
            &config.content_key,
            reference_bands,
            reference_signatures,
            &contamination_results,
            config.exact_override,
            config.jaccard_similarity_threshold
        ) {
            println!("Error processing training file {:?}: {:?}", file_path, e);
        }
        pbar.inc(1);
    });

    // Save contamination results
    save_contamination_results(&contamination_results, &config.output_dir)?;

    Ok(())
}

fn process_training_file(
    file_path: &PathBuf,
    file_name: &str,
    band_seeds: &Vec<u32>,
    perm_seeds: &Vec<u64>,
    band_size: usize,
    ngram_size: usize,
    tokenizer_str: &str,
    content_key: &str,
    reference_bands: &ReferenceBands,
    reference_signatures: &ReferenceSignatures,
    contamination_results: &DashMap<String, Vec<(usize, String, usize, f32)>>,
    exact_override: bool,
    jaccard_threshold: f32
) -> Result<(), Error> {
    let data = read_pathbuf_to_mem(file_path)?;
    let tokenizer = OmniTokenizer::new(tokenizer_str)?;
    let num_bands = band_seeds.len();

    println!("Checking training file: {}", file_name);
    let mut contaminated_lines = 0;
    let mut total_lines = 0;

    for (line_num, line) in data.lines().enumerate() {
        let line = line?;
        let json_obj: Value = serde_json::from_str(&line)?;
        let line_text = get_nested_json_val(&json_obj, &content_key.to_string())?;
        total_lines += 1;

        println!("  → Processing line {}: \"{}\"", line_num,
                if line_text.len() > 100 {
                    format!("{}...", &line_text[..100])
                } else {
                    line_text.clone()
                });

        let hash_vals = if exact_override {
            let Ok(tokens) = catch_unwind(|| preprocess_text(&line_text, &tokenizer)) else {
                continue;
            };
            get_hash_vals_from_tokens(tokens, perm_seeds, ngram_size)
        } else {
            let n = perm_seeds.len();
            let mut hash_vals: Array1<u64> = Array1::ones(n);
            hash_vals = hash_vals * (hash_object(&line_text) as u64);
            hash_vals
        };

        // Check for collisions with reference bands
        let bands = hash_vals.clone().into_shape((num_bands, band_size))?;
        let mut line_matches: HashMap<(String, usize), f32> = HashMap::new();
        let mut band_collisions_found = false;

        // println!("    🔧 Checking {} bands for collisions", num_bands);
        for (_band_idx, row) in bands.rows().into_iter().enumerate() {
            let mut hasher = Sha256::new();
            hasher.update(bytemuck::cast_slice(row.as_slice().unwrap()));
            let hash = hasher.finalize();
            let band_signature = hash[..8].to_vec();

            // println!("    🔧 Band {}: values={:?} -> signature={:02x?}",
            //         band_idx, row.as_slice().unwrap(), band_signature);

            if let Some(matches) = reference_bands.get(&band_signature) {
                // println!("    🔧 Band {} collision! Found {} reference matches", band_idx, matches.len());
                if !band_collisions_found {
                    println!("    ✅ Band collision detected!");
                    band_collisions_found = true;
                }

                // Found potential contamination - calculate Jaccard similarity
                for (eval_name, eval_line_num) in matches.value() {
                    if let Some(ref_sig_entry) = reference_signatures.get(&(eval_name.clone(), *eval_line_num)) {
                        let jaccard_sim = calculate_jaccard_similarity(&hash_vals, ref_sig_entry.value());

                        // Store contamination result (threshold configurable)
                        if jaccard_sim > jaccard_threshold {
                            let key = (eval_name.clone(), *eval_line_num);
                            // Keep the highest Jaccard similarity for each unique eval match
                            line_matches.entry(key)
                                .and_modify(|existing_sim| *existing_sim = existing_sim.max(jaccard_sim))
                                .or_insert(jaccard_sim);
                        }
                    }
                }
            }
        }

        if !band_collisions_found {
            println!("    ❌ No band collisions detected");
        }

        // Add deduplicated matches to results
        if !line_matches.is_empty() {
            contaminated_lines += 1;
            for ((eval_name, eval_line_num), jaccard_sim) in &line_matches {
                println!("    📊 Jaccard similarity with {}[{}]: {:.3} (threshold: {:.3})",
                        eval_name, eval_line_num, jaccard_sim, jaccard_threshold);
            }
            for ((eval_name, eval_line_num), jaccard_sim) in line_matches {
                contamination_results
                    .entry(file_name.to_string())
                    .or_default()
                    .push((line_num, eval_name, eval_line_num, jaccard_sim));
            }
        }
    }

    if contaminated_lines > 0 {
        println!("  → Found {} contaminated lines out of {} total lines in {}",
                contaminated_lines, total_lines, file_name);
    } else {
        println!("  → No contamination found in {} ({} lines)", file_name, total_lines);
    }

    Ok(())
}



// fn jaccard_similarity(x: &HashSet<usize>, y: &HashSet<usize>)  -> f32 {
//     let cap = x.intersection(y).count();
//     let cup = x.union(y).count();
//     cap as f32 / cup as f32
// }

fn calculate_jaccard_similarity(sig1: &Array1<u64>, sig2: &Array1<u64>) -> f32 {
    let matches = sig1.iter()
        .zip(sig2.iter())
        .filter(|(a, b)| a == b)
        .count();

    matches as f32 / sig1.len() as f32
}

fn save_contamination_results(
    results: &DashMap<String, Vec<(usize, String, usize, f32)>>,
    output_dir: &PathBuf
) -> Result<(), Error> {
    create_dir_all(output_dir)?;
    let output_file = output_dir.join("contamination_results.jsonl");

    let mut output_data = Vec::new();
    let mut total_contaminations = 0;

    for entry in results.iter() {
        let training_file = entry.key();
        for (training_line, eval_name, eval_line, jaccard_sim) in entry.value() {
            let result = json!({
                "training_file": training_file,
                "training_line": training_line,
                "eval_dataset": eval_name,
                "eval_line": eval_line,
                "jaccard_similarity": jaccard_sim
            });
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
        println!("=== CONTAMINATION SUMMARY ===");
        println!("Found {} contamination instances across {} files",
                total_contaminations, results.len());
        println!("Results saved to: {:?}", output_file);
    } else {
        println!("=== NO CONTAMINATION DETECTED ===");
        println!("No contamination found in training data");
        println!("Empty results file saved to: {:?}", output_file);
    }

    Ok(())
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
    jaccard_similarity: f32,
}

fn review_contamination(config: &PathBuf, results_file: Option<&PathBuf>) -> Result<(), Error> {
    println!("=== CONTAMINATION REVIEW ===");

    let config_obj = read_config(config)?;

    // Determine results file path
    let results_path = match results_file {
        Some(path) => path.clone(),
        None => config_obj.output_dir.join("contamination_results.jsonl")
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
    let training_cache = load_training_files(&config_obj.local_input)?;
    let eval_cache = load_eval_files(&config_obj.reference_input)?;

    // Review each contamination case
    for (idx, result) in contamination_results.iter().enumerate() {
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

fn load_training_files(input_dir: &PathBuf) -> Result<HashMap<String, Vec<String>>, Error> {
    let mut cache = HashMap::new();
    let training_files = expand_dirs(vec![input_dir.clone()], Some(vec![".jsonl"].as_slice()))?;

    for file_path in training_files {
        let file_name = file_path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("unknown")
            .to_string();

        let data = read_pathbuf_to_mem(&file_path)?;
        let mut lines = Vec::new();

        for line in data.lines() {
            let line = line?;
            if !line.trim().is_empty() {
                let json_obj: Value = serde_json::from_str(&line)?;
                let text = get_nested_json_val(&json_obj, &"text".to_string())?;
                lines.push(text);
            }
        }

        cache.insert(file_name, lines);
    }

    Ok(cache)
}

fn load_eval_files(reference_dir: &PathBuf) -> Result<HashMap<String, Vec<String>>, Error> {
    let mut cache = HashMap::new();
    let eval_files = expand_dirs(vec![reference_dir.clone()], Some(vec![".jsonl"].as_slice()))?;

    for file_path in eval_files {
        let file_name = file_path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("unknown")
            .to_string();

        let data = read_pathbuf_to_mem(&file_path)?;
        let mut lines = Vec::new();

        for line in data.lines() {
            let line = line?;
            if !line.trim().is_empty() {
                let json_obj: Value = serde_json::from_str(&line)?;
                let text = get_nested_json_val(&json_obj, &"text".to_string())?;
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
    println!("🎯 JACCARD SIM:   {:.3}", result.jaccard_similarity);
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
    } else {
        println!("🤔 MODERATE SIMILARITY - Manual review needed");
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
        Commands::ContaminationDetect {config} => {
            contamination_detect(config)
        }

        Commands::ReviewContamination {config, results_file} => {
            review_contamination(config, results_file.as_ref())
        }

    };
    result.unwrap()
}
