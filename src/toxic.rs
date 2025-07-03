// TOXIC (Token-Oriented eXclusion via Interference Clustering) contamination detection implementation
//
// POISON TOKEN MECHANICS:
// =======================
// Unlike traditional embeddings that try to ignore or nullify unknown words, TOXIC uses
// "poison tokens" - semantically destructive vectors that actively break similarity between
// documents that should NOT be considered contaminated.
//
// Examples of where this matters:
// Training: "If you have 1 apple and I have 1 apple, we have 2 apples"
// Eval:     "If you have 2 apples and I have 2 apples, we have 4 apples"
//
// Without poison tokens: The word embeddings for "apple", "have", "if" would make these
// appear semantically similar, causing a false contamination detection.
//
// With poison tokens: The numbers "1", "2", "4" get large, distinct destructive vectors
// that pollute the n-gram embeddings differently, preventing false similarity matches.
//
// Categories of poison tokens:
// - Numbers: 1, 2, 42, 1990, etc.
// - Proper nouns: Names, places, brands
// - Domain-specific entities: Dates, IDs, technical terms
// - Out-of-vocabulary words
//
// The `toxic_poison_scale` config amplifies their destructive impact, ensuring that
// semantic differences in these tokens reliably prevent false contamination detection.

use anyhow::{Error, Result};
use dashmap::DashMap;
use ndarray::ArrayView1;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha20Rng;
use rayon::prelude::*;
use serde_json::{Value, json};
use std::collections::HashMap;
use std::fs::create_dir_all;
use std::io::BufRead;
use std::path::PathBuf;
use std::time::Instant;

use mj_io::{expand_dirs, read_pathbuf_to_mem, write_mem_to_pathbuf, build_pbar};

use crate::{Config, get_nested_json_val, clean_text, get_results_filename};

// 300-dimensional word embeddings
const EMBEDDING_DIM: usize = 300;

// LSH bucket storage: maps bucket_id to list of (eval_name, line_num, word_count)
type ToxicBuckets = DashMap<u64, Vec<(String, usize, usize)>>;

// Embedding storage: word -> 300d vector
type EmbeddingMap = HashMap<String, Vec<f32>>;

// Random hyperplanes for LSH: k x 300 matrix
type Hyperplanes = Vec<Vec<f32>>;

pub fn contamination_detect(config: &Config) -> Result<(), Error> {
    println!("Starting TOXIC contamination detection...");
    let start_main = Instant::now();

    // Step 1: Load word embeddings
    println!("Loading word embeddings...");
    let embeddings = load_embeddings(&config.toxic_embedding_path)?;
    println!("Loaded {} word embeddings", embeddings.len());

    // Step 2: Generate random hyperplanes for LSH
    println!("Generating {} random hyperplanes...", config.toxic_hyperplanes);
    let hyperplanes = generate_hyperplanes(config.toxic_hyperplanes, config.hash_seed)?;

    // Step 3: Process reference datasets and build LSH buckets
    println!("Processing reference datasets...");
    let toxic_buckets = build_toxic_index(config, &embeddings, &hyperplanes)?;
    println!("Built TOXIC index with {} buckets", toxic_buckets.len());

    // Step 4: Process training data and detect contamination
    println!("Processing training data for contamination detection...");
    detect_toxic_contamination(config, &embeddings, &hyperplanes, &toxic_buckets)?;

    println!("TOXIC contamination detection completed in {:?} seconds", start_main.elapsed().as_secs());
    Ok(())
}

fn load_embeddings(embedding_path: &PathBuf) -> Result<EmbeddingMap, Error> {
    println!("Loading embeddings from: {:?}", embedding_path);
    let data = read_pathbuf_to_mem(embedding_path)?;
    let mut embeddings = HashMap::new();

    for (line_num, line) in data.lines().enumerate() {
        let line = line?;
        let parts: Vec<&str> = line.split_whitespace().collect();

        if parts.len() != EMBEDDING_DIM + 1 {
            if line_num == 0 && parts.len() == 2 {
                // Skip header line if present (vocab_size dim_size)
                continue;
            }
            return Err(anyhow::anyhow!(
                "Invalid embedding format at line {}: expected {} values, got {}",
                line_num, EMBEDDING_DIM + 1, parts.len()
            ));
        }

        let word = parts[0].to_string();
        let vector: Result<Vec<f32>, _> = parts[1..].iter()
            .map(|s| s.parse::<f32>())
            .collect();

        match vector {
            Ok(vec) => {
                embeddings.insert(word, vec);
            }
            Err(e) => {
                return Err(anyhow::anyhow!("Failed to parse embedding vector at line {}: {}", line_num, e));
            }
        }

        if line_num % 100000 == 0 && line_num > 0 {
            println!("  → Loaded {} embeddings...", line_num);
        }
    }

    Ok(embeddings)
}

fn get_or_create_embedding(word: &str, embeddings: &EmbeddingMap, rng_seed: u64) -> Vec<f32> {
    if !embeddings.contains_key(word) {
        // Create poison token - a semantically destructive vector for OOV words
        // This ensures different unknown words break similarity rather than accidentally matching
        // TOXIC-MAXXING: Add extra randomness to ensure different vectors each time
        let mut rng = ChaCha20Rng::seed_from_u64(
            rng_seed.wrapping_add(hash_string(word)).wrapping_add(rand::random::<u64>())
        );

        let poison_vector: Vec<f32> = (0..EMBEDDING_DIM)
            .map(|_| rng.gen_range(-1.0..1.0))
            .collect();

        // TOXIC-MAXXING: Don't cache poison tokens! Generate fresh chaos each time.
        //println!("Created poison token for OOV word: '{}'", word);  //TODO string normalization is too aggressive.
        poison_vector
    } else {
        embeddings.get(word).unwrap().clone()
    }
}

fn hash_string(s: &str) -> u64 {
    use std::hash::{Hash, Hasher, DefaultHasher};
    let mut hasher = DefaultHasher::new();
    s.hash(&mut hasher);
    hasher.finish()
}

fn generate_hyperplanes(k: usize, seed: usize) -> Result<Hyperplanes, Error> {
    let mut rng = ChaCha20Rng::seed_from_u64(seed as u64);
    let mut hyperplanes = Vec::with_capacity(k);

    for _ in 0..k {
        let hyperplane: Vec<f32> = (0..EMBEDDING_DIM)
            .map(|_| rng.gen::<f32>() - 0.5) // Normal distribution approximation
            .collect();
        hyperplanes.push(hyperplane);
    }

    Ok(hyperplanes)
}

fn extract_words(text: &str) -> Vec<String> {
    // Clean text using the same cleaning process as other methods
    let cleaned = clean_text(text);

    // Split into words and filter out empty strings
    cleaned
        .split_whitespace()
        .map(|s| s.to_string())
        .filter(|s| !s.is_empty())
        .collect()
}

fn compute_ngram_embedding(
    tokens: &[String],
    ngram_size: usize,
    embeddings: &EmbeddingMap,
    rng_seed: u64
) -> Vec<Vec<f32>> {
    let mut ngram_embeddings = Vec::new();

    if tokens.len() < ngram_size {
        if !tokens.is_empty() {
            // For documents shorter than ngram_size, use all tokens
            let sum_embedding = sum_word_embeddings(tokens, embeddings, rng_seed);
            ngram_embeddings.push(sum_embedding);
        }
        return ngram_embeddings;
    }

    // Create sliding window n-grams
    for i in 0..=tokens.len() - ngram_size {
        let ngram = &tokens[i..i + ngram_size];
        let sum_embedding = sum_word_embeddings(ngram, embeddings, rng_seed);
        ngram_embeddings.push(sum_embedding);
    }

    ngram_embeddings
}

fn sum_word_embeddings(
    words: &[String],
    embeddings: &EmbeddingMap,
    rng_seed: u64
) -> Vec<f32> {
    let mut sum_vector = vec![0.0; EMBEDDING_DIM];

    for word in words {
        let word_embedding = get_or_create_embedding(word, embeddings, rng_seed);
        for (i, val) in word_embedding.iter().enumerate() {
            sum_vector[i] += val;
        }
    }

    sum_vector
}

fn normalize_vector(vector: &[f32]) -> Option<Vec<f32>> {
    let magnitude: f32 = vector.iter().map(|x| x * x).sum::<f32>().sqrt();

    if magnitude < f32::EPSILON {
        // Zero vector cannot be normalized
        return None;
    }

    Some(vector.iter().map(|x| x / magnitude).collect())
}

fn compute_lsh_bucket(normalized_vector: &[f32], hyperplanes: &Hyperplanes) -> u64 {
    let mut bucket_id = 0u64;
    let vector_view = ArrayView1::from(normalized_vector);

    for (i, hyperplane) in hyperplanes.iter().enumerate() {
        let hyperplane_view = ArrayView1::from(hyperplane.as_slice());
        let dot_product = vector_view.dot(&hyperplane_view);

        if dot_product >= 0.0 {
            bucket_id |= 1u64 << i;
        }
    }

    bucket_id
}

fn build_toxic_index(
    config: &Config,
    embeddings: &EmbeddingMap,
    hyperplanes: &Hyperplanes
) -> Result<ToxicBuckets, Error> {
    let toxic_buckets: ToxicBuckets = DashMap::new();

    // Find all reference files
    let reference_files = expand_dirs(
        vec![config.reference_input.clone()],
        Some(vec![".jsonl", ".gz"].as_slice())
    )?;
    let pbar = build_pbar(reference_files.len(), "Reference files");

    reference_files.par_iter().for_each(|file_path| {
        if let Err(e) = process_toxic_reference_file(
            file_path,
            config,
            embeddings,
            hyperplanes,
            &toxic_buckets
        ) {
            println!("Error processing reference file {:?}: {:?}", file_path, e);
        }
        pbar.inc(1);
    });

    Ok(toxic_buckets)
}

fn process_toxic_reference_file(
    file_path: &PathBuf,
    config: &Config,
    embeddings: &EmbeddingMap,
    hyperplanes: &Hyperplanes,
    toxic_buckets: &ToxicBuckets
) -> Result<(), Error> {
    let data = read_pathbuf_to_mem(file_path)?;
    // We don't need tokenizer for TOXIC - we work with words directly

    // Extract eval name from filename
    let eval_name = file_path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("unknown")
        .to_string();

    println!("Processing TOXIC embeddings for eval dataset: {}", eval_name);

    let mut lines_processed = 0;
    for (line_num, line) in data.lines().enumerate() {
        let line = line?;
        let json_obj: Value = serde_json::from_str(&line)?;
        let line_text = get_nested_json_val(&json_obj, &config.content_key.to_string())?;
        lines_processed += 1;

        // For TOXIC, we work with words directly, not token IDs
        let word_tokens = extract_words(&line_text);

        let word_count = word_tokens.len();

        // Compute n-gram embeddings
        let ngram_embeddings = compute_ngram_embedding(
            &word_tokens,
            config.ngram_size,
            embeddings,
            config.hash_seed as u64
        );

        // Process each n-gram embedding
        for ngram_embedding in ngram_embeddings {
            if let Some(normalized) = normalize_vector(&ngram_embedding) {
                let bucket_id = compute_lsh_bucket(&normalized, hyperplanes);

                toxic_buckets
                    .entry(bucket_id)
                    .or_default()
                    .push((eval_name.clone(), line_num, word_count));
            }
        }
    }

    println!("  → Processed {} lines from {}", lines_processed, eval_name);
    Ok(())
}

fn detect_toxic_contamination(
    config: &Config,
    embeddings: &EmbeddingMap,
    hyperplanes: &Hyperplanes,
    toxic_buckets: &ToxicBuckets
) -> Result<(), Error> {
    // Find all training files
    let training_files = expand_dirs(
        vec![config.local_input.clone()],
        Some(vec![".jsonl", ".gz"].as_slice())
    )?;
    println!("Found {} training files to process", training_files.len());
    let pbar = build_pbar(training_files.len(), "Training files");

    let contamination_results: DashMap<String, Vec<(usize, String, usize, f32)>> = DashMap::new();

    training_files.par_iter().for_each(|file_path| {
        let file_name = if file_path.extension().and_then(|s| s.to_str()) == Some("gz") {
            file_path
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("unknown")
                .to_string()
        } else {
            file_path
                .file_name()
                .and_then(|s| s.to_str())
                .unwrap_or("unknown")
                .to_string()
        };

        if let Err(e) = process_toxic_training_file(
            file_path,
            &file_name,
            config,
            embeddings,
            hyperplanes,
            toxic_buckets,
            &contamination_results
        ) {
            println!("Error processing training file {:?}: {:?}", file_path, e);
        }
        pbar.inc(1);
    });

    // Save contamination results
    save_toxic_contamination_results(&contamination_results, &config.output_dir)?;

    Ok(())
}

fn process_toxic_training_file(
    file_path: &PathBuf,
    file_name: &str,
    config: &Config,
    embeddings: &EmbeddingMap,
    hyperplanes: &Hyperplanes,
    toxic_buckets: &ToxicBuckets,
    contamination_results: &DashMap<String, Vec<(usize, String, usize, f32)>>
) -> Result<(), Error> {
    let data = read_pathbuf_to_mem(file_path)?;
    // We don't need tokenizer for TOXIC - we work with words directly

    println!("Checking training file: {}", file_name);
    let mut contaminated_lines = 0;
    let mut total_lines = 0;

    for (line_num, line) in data.lines().enumerate() {
        let line = line?;
        let json_obj: Value = serde_json::from_str(&line)?;
        let line_text = get_nested_json_val(&json_obj, &config.content_key.to_string())?;
        total_lines += 1;

        // For TOXIC, we work with words directly
        let word_tokens = extract_words(&line_text);
        let training_word_count = word_tokens.len();

        // Compute n-gram embeddings
        let ngram_embeddings = compute_ngram_embedding(
            &word_tokens,
            config.ngram_size,
            embeddings,
            config.hash_seed as u64
        );

        // Track collisions for this training document
        let mut eval_collisions: HashMap<(String, usize), usize> = HashMap::new();

        // Check each n-gram for collisions
        for ngram_embedding in ngram_embeddings {
            if let Some(normalized) = normalize_vector(&ngram_embedding) {
                let bucket_id = compute_lsh_bucket(&normalized, hyperplanes);

                if let Some(bucket_contents) = toxic_buckets.get(&bucket_id) {
                    for (eval_name, eval_line_num, _eval_word_count) in bucket_contents.value() {
                        let key = (eval_name.clone(), *eval_line_num);
                        *eval_collisions.entry(key).or_insert(0) += 1;
                    }
                }
            }
        }

        // Apply threshold and complete evaluation
        let collision_count_total = eval_collisions.len();
        for ((eval_name, eval_line_num), collision_count) in &eval_collisions {
            // Find the eval word count for this specific eval entry
            let mut eval_word_count = 0;
            for bucket in toxic_buckets.iter() {
                for (name, line, word_count) in bucket.value().iter() {
                    if name == eval_name && line == eval_line_num {
                        eval_word_count = *word_count;
                        break;
                    }
                }
                if eval_word_count > 0 {
                    break;
                }
            }

            if eval_word_count > 0 {
                let min_words = training_word_count.min(eval_word_count);

                // Threshold: collision_count should be significant relative to document length
                let overlap_ratio = *collision_count as f32 / min_words as f32;

                if overlap_ratio >= config.toxic_overlap_threshold && complete_evaluation() {
                    contamination_results
                        .entry(file_name.to_string())
                        .or_default()
                        .push((line_num, eval_name.clone(), *eval_line_num, overlap_ratio));

                    if collision_count_total == 1 {
                        contaminated_lines += 1;
                    }
                }
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

// Placeholder for future complete evaluation logic
fn complete_evaluation() -> bool {
    true  // For now, always passes complete evaluation
}

fn save_toxic_contamination_results(
    results: &DashMap<String, Vec<(usize, String, usize, f32)>>,
    output_dir: &PathBuf
) -> Result<(), Error> {
    create_dir_all(output_dir)?;
    let output_file = output_dir.join(get_results_filename("toxic"));

    let mut output_data = Vec::new();
    let mut total_contaminations = 0;

    for entry in results.iter() {
        let training_file = entry.key();
        for (training_line, eval_name, eval_line, overlap_ratio) in entry.value() {
            let result = json!({
                "training_file": training_file,
                "training_line": training_line,
                "eval_dataset": eval_name,
                "eval_line": eval_line,
                "overlap_ratio": overlap_ratio,
                "method": "toxic"
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
        println!("=== TOXIC CONTAMINATION SUMMARY ===");
        println!("Found {} contamination instances across {} files",
                total_contaminations, results.len());
        println!("Results saved to: {:?}", output_file);
    } else {
        println!("=== NO TOXIC CONTAMINATION DETECTED ===");
        println!("No contamination found in training data");
        println!("Empty results file saved to: {:?}", output_file);
    }

    Ok(())
}
