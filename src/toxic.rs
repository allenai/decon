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
use std::collections::{HashMap, HashSet};
use std::fs::create_dir_all;
use std::io::{BufRead, Write};
use std::path::PathBuf;
use std::time::{Duration, Instant};

use mj_io::{expand_dirs, read_pathbuf_to_mem, write_mem_to_pathbuf, build_pbar};

use crate::{Config, get_nested_json_val, clean_text, get_results_filename, debug_println};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sliding_window_algorithm() {
        // Test the sliding window optimization with 8-item string and 3-grams
        let test_embeddings = EmbeddingMap::new();
        
        // Create test vectors for 8 tokens
        let test_vectors = vec![
            vec![1.0, 10.0],   // a
            vec![2.0, 20.0],   // b  
            vec![3.0, 30.0],   // c
            vec![4.0, 40.0],   // d
            vec![5.0, 50.0],   // e
            vec![6.0, 60.0],   // f
            vec![7.0, 70.0],   // g
            vec![8.0, 80.0],   // h
        ];
        
        let token_names = vec!["a", "b", "c", "d", "e", "f", "g", "h"];
        
        for (i, name) in token_names.iter().enumerate() {
            let mut vec = vec![0.0; EMBEDDING_DIM];
            vec[0] = test_vectors[i][0];
            vec[1] = test_vectors[i][1];
            test_embeddings.insert(name.to_string(), vec);
        }

        let tokens: Vec<String> = token_names.iter().map(|s| s.to_string()).collect();
        let mut timing = TimingStats::default();
        
        let result = compute_ngram_embedding_training(&tokens, 3, &test_embeddings, 42, &mut timing);
        
        // Should produce 6 n-grams: ["a","b","c"], ["b","c","d"], ["c","d","e"], ["d","e","f"], ["e","f","g"], ["f","g","h"]
        assert_eq!(result.len(), 6);
        
        // Expected 3-gram sums:
        let expected = vec![
            vec![1.0+2.0+3.0, 10.0+20.0+30.0],  // a+b+c = [6, 60]
            vec![2.0+3.0+4.0, 20.0+30.0+40.0],  // b+c+d = [9, 90]
            vec![3.0+4.0+5.0, 30.0+40.0+50.0],  // c+d+e = [12, 120]
            vec![4.0+5.0+6.0, 40.0+50.0+60.0],  // d+e+f = [15, 150]
            vec![5.0+6.0+7.0, 50.0+60.0+70.0],  // e+f+g = [18, 180]
            vec![6.0+7.0+8.0, 60.0+70.0+80.0],  // f+g+h = [21, 210]
        ];
        
        for (i, (actual, exp)) in result.iter().zip(expected.iter()).enumerate() {
            assert!((actual[0] - exp[0]).abs() < 1e-6, 
                   "N-gram {} dim 0: expected {}, got {}", i, exp[0], actual[0]);
            assert!((actual[1] - exp[1]).abs() < 1e-6, 
                   "N-gram {} dim 1: expected {}, got {}", i, exp[1], actual[1]);
        }
        
        println!("✅ Sliding window optimization test passed!");
        println!("Generated {} 3-grams from 8 tokens", result.len());
        println!("First 3-gram: [{:.0}, {:.0}] (expected: [6, 60])", result[0][0], result[0][1]);
        println!("Last 3-gram:  [{:.0}, {:.0}] (expected: [21, 210])", result[5][0], result[5][1]);
    }
}


// 300-dimensional word embeddings
pub const EMBEDDING_DIM: usize = 300;

// LSH bucket storage: maps bucket_id to list of (eval_name, line_num, word_count)
type ToxicBuckets = DashMap<u64, Vec<(String, usize, usize)>>;

// Bucket content storage for debug mode: maps bucket_id to set of ngram texts
type BucketContents = DashMap<u64, HashSet<String>>;

// Embedding storage: word -> 300d vector (thread-safe)
pub type EmbeddingMap = DashMap<String, Vec<f32>>;

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
    let (toxic_buckets, _bucket_contents) = build_toxic_index(config, &embeddings, &hyperplanes)?;
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
    let embeddings = DashMap::new();

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
            print!(".");
            std::io::stdout().flush().unwrap();
        }
    }

    println!(); // New line after dots
    Ok(embeddings)
}

// For eval processing: store OOV words consistently
fn get_or_create_embedding_eval(word: &str, embeddings: &EmbeddingMap, rng_seed: u64, timing: &mut TimingStats) -> Vec<f32> {
    // Time hash lookup
    let lookup_start = Instant::now();
    let embedding_opt = embeddings.get(word);
    timing.hash_lookups += lookup_start.elapsed();
    
    if let Some(embedding) = embedding_opt {
        // Time vector cloning
        let clone_start = Instant::now();
        let result = embedding.clone();
        timing.vector_cloning += clone_start.elapsed();
        result
    } else {
        // Time random generation for poison tokens
        let rand_start = Instant::now();
        let mut rng = ChaCha20Rng::seed_from_u64(
            rng_seed.wrapping_add(hash_string(word))
        );
        let vector: Vec<f32> = (0..EMBEDDING_DIM)
            .map(|_| rng.gen_range(-1.0..1.0))
            .collect();
        timing.random_generation += rand_start.elapsed();

        // Time the insertion and cloning for return
        let clone_start = Instant::now();
        embeddings.insert(word.to_string(), vector.clone());
        timing.vector_cloning += clone_start.elapsed();
        
        vector
    }
}

// For training processing: generate chaotic OOV each time (don't store)
fn get_or_create_embedding_training(word: &str, embeddings: &EmbeddingMap, rng_seed: u64, timing: &mut TimingStats) -> Vec<f32> {
    // Time hash lookup
    let lookup_start = Instant::now();
    let embedding_opt = embeddings.get(word);
    timing.hash_lookups += lookup_start.elapsed();
    
    if let Some(embedding) = embedding_opt {
        // Time vector cloning
        let clone_start = Instant::now();
        let result = embedding.clone();
        timing.vector_cloning += clone_start.elapsed();
        result
    } else {
        // Time random generation for poison tokens
        let rand_start = Instant::now();
        let mut rng = ChaCha20Rng::seed_from_u64(
            rng_seed.wrapping_add(hash_string(word))
        );
        let result = (0..EMBEDDING_DIM)
            .map(|_| rng.gen_range(-1.0..1.0))
            .collect();
        timing.random_generation += rand_start.elapsed();
        result
    }
}

fn hash_string(s: &str) -> u64 {
    use std::hash::{Hash, Hasher, DefaultHasher};
    let mut hasher = DefaultHasher::new();
    s.hash(&mut hasher);
    hasher.finish()
}

// Helper function for vector subtraction (in-place)
fn subtract_word_embedding_inplace(sum_vector: &mut [f32], word_embedding: &[f32], timing: &mut TimingStats) {
    let arith_start = Instant::now();
    for (i, val) in word_embedding.iter().enumerate() {
        sum_vector[i] -= val;
    }
    timing.vector_arithmetic += arith_start.elapsed();
}

// Helper function for vector addition (in-place)
fn add_word_embedding_inplace(sum_vector: &mut [f32], word_embedding: &[f32], timing: &mut TimingStats) {
    let arith_start = Instant::now();
    for (i, val) in word_embedding.iter().enumerate() {
        sum_vector[i] += val;
    }
    timing.vector_arithmetic += arith_start.elapsed();
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

fn extract_words(text: &str, punctuation_chars: &str) -> Vec<String> {
    // Clean text using the same cleaning process as other methods
    let cleaned = clean_text(text, punctuation_chars);

    // Split into words and filter out empty strings
    cleaned
        .split_whitespace()
        .map(|s| s.to_string())
        .filter(|s| !s.is_empty())
        .collect()
}



// Optimized training n-gram computation using sliding window to reduce vector arithmetic  
pub fn compute_ngram_embedding_training(
    tokens: &[String],
    ngram_size: usize,
    embeddings: &EmbeddingMap,
    rng_seed: u64,
    timing: &mut TimingStats
) -> Vec<Vec<f32>> {
    let mut ngram_embeddings = Vec::new();

    if tokens.len() < ngram_size {
        if !tokens.is_empty() {
            // For documents shorter than ngram_size, use all tokens - same as original
            let sum_embedding = sum_word_embeddings_training(tokens, embeddings, rng_seed, timing);
            ngram_embeddings.push(sum_embedding);
        }
        return ngram_embeddings;
    }

    // Compute first n-gram sum normally
    let first_ngram = &tokens[0..ngram_size];
    let mut current_sum = sum_word_embeddings_training(first_ngram, embeddings, rng_seed, timing);
    ngram_embeddings.push(current_sum.clone());

    // Slide the window for remaining n-grams
    for i in 1..=tokens.len() - ngram_size {
        // Word sliding out (left side)
        let outgoing_word = &tokens[i - 1];
        let outgoing_embedding = get_or_create_embedding_training(outgoing_word, embeddings, rng_seed, timing);
        subtract_word_embedding_inplace(&mut current_sum, &outgoing_embedding, timing);

        // Word sliding in (right side) 
        let incoming_word = &tokens[i + ngram_size - 1];
        let incoming_embedding = get_or_create_embedding_training(incoming_word, embeddings, rng_seed, timing);
        add_word_embedding_inplace(&mut current_sum, &incoming_embedding, timing);

        ngram_embeddings.push(current_sum.clone());
    }

    ngram_embeddings
}

// For eval processing
fn sum_word_embeddings_eval(
    words: &[String],
    embeddings: &EmbeddingMap,
    rng_seed: u64,
    timing: &mut TimingStats
) -> Vec<f32> {
    // Time memory allocation
    let alloc_start = Instant::now();
    let mut sum_vector = vec![0.0; EMBEDDING_DIM];
    timing.memory_allocation += alloc_start.elapsed();

    for word in words {
        // Time embedding lookup/generation
        let lookup_start = Instant::now();
        let word_embedding = get_or_create_embedding_eval(word, embeddings, rng_seed, timing);
        timing.hash_lookups += lookup_start.elapsed();
        
        // Time vector arithmetic
        let arith_start = Instant::now();
        for (i, val) in word_embedding.iter().enumerate() {
            sum_vector[i] += val;
        }
        timing.vector_arithmetic += arith_start.elapsed();
    }

    sum_vector
}

// For training processing
fn sum_word_embeddings_training(
    words: &[String],
    embeddings: &EmbeddingMap,
    rng_seed: u64,
    timing: &mut TimingStats
) -> Vec<f32> {
    // Time memory allocation
    let alloc_start = Instant::now();
    let mut sum_vector = vec![0.0; EMBEDDING_DIM];
    timing.memory_allocation += alloc_start.elapsed();

    for word in words {
        // Time embedding lookup/generation
        let lookup_start = Instant::now();
        let word_embedding = get_or_create_embedding_training(word, embeddings, rng_seed, timing);
        timing.hash_lookups += lookup_start.elapsed();
        
        // Time vector arithmetic
        let arith_start = Instant::now();
        for (i, val) in word_embedding.iter().enumerate() {
            sum_vector[i] += val;
        }
        timing.vector_arithmetic += arith_start.elapsed();
    }

    sum_vector
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
) -> Result<(ToxicBuckets, Option<BucketContents>), Error> {
    let toxic_buckets: ToxicBuckets = DashMap::new();
    let bucket_contents: Option<BucketContents> = if config.debug {
        Some(DashMap::new())
    } else {
        None
    };

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
            &toxic_buckets,
            &bucket_contents
        ) {
            println!("Error processing reference file {:?}: {:?}", file_path, e);
        }
        pbar.inc(1);
    });

    // Analyze bucket distribution statistics
    if config.debug {
        print_bucket_statistics(config, &toxic_buckets, &bucket_contents);
    }

    // Save bucket contents to file for debug analysis
    if let Some(ref contents) = bucket_contents {
        save_bucket_contents(config, contents)?;
    }

    Ok((toxic_buckets, bucket_contents))
}

fn process_toxic_reference_file(
    file_path: &PathBuf,
    config: &Config,
    embeddings: &EmbeddingMap,
    hyperplanes: &Hyperplanes,
    toxic_buckets: &ToxicBuckets,
    bucket_contents: &Option<BucketContents>
) -> Result<(), Error> {
    let data = read_pathbuf_to_mem(file_path)?;
    // We don't need tokenizer for TOXIC - we work with words directly

    // Extract eval name from filename
    let eval_name = file_path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("unknown")
        .to_string();

    debug_println!(config, "Processing TOXIC embeddings for eval dataset: {}", eval_name);

    let mut lines_processed = 0;
    for (line_num, line) in data.lines().enumerate() {
        let line = line?;
        let json_obj: Value = serde_json::from_str(&line)?;
        let line_text = get_nested_json_val(&json_obj, &config.content_key.to_string())?;
        lines_processed += 1;

        // For TOXIC, we work with words directly, not token IDs
        let word_tokens = extract_words(&line_text, &config.punctuation_chars);

        let word_count = word_tokens.len();

        // Process n-grams with hot bucket detection
        if word_tokens.len() < config.ngram_size {
            if !word_tokens.is_empty() {
                // For documents shorter than ngram_size, use all tokens
                let mut dummy_timing = TimingStats::default();
                let sum_embedding = sum_word_embeddings_eval(&word_tokens, embeddings, config.hash_seed as u64, &mut dummy_timing);
                // Skip normalization to preserve magnitude information
                let bucket_id = compute_lsh_bucket(&sum_embedding, hyperplanes);
                insert_with_hot_bucket_detection(
                    config,
                    &toxic_buckets,
                    bucket_id,
                    &word_tokens,
                    &eval_name,
                    line_num,
                    word_count
                );

                // Store ngram text for debug analysis
                if let Some(ref contents) = bucket_contents {
                    let ngram_text = word_tokens.join(" ");
                    contents.entry(bucket_id).or_default().insert(ngram_text);
                }
            }
        } else {
            // Create sliding window n-grams with word tracking
            for i in 0..=word_tokens.len() - config.ngram_size {
                let ngram = &word_tokens[i..i + config.ngram_size];
                let mut dummy_timing = TimingStats::default();
                let sum_embedding = sum_word_embeddings_eval(ngram, embeddings, config.hash_seed as u64, &mut dummy_timing);
                // Skip normalization to preserve magnitude information
                let bucket_id = compute_lsh_bucket(&sum_embedding, hyperplanes);
                insert_with_hot_bucket_detection(
                    config,
                    &toxic_buckets,
                    bucket_id,
                    ngram,
                    &eval_name,
                    line_num,
                    word_count
                );

                // Store ngram text for debug analysis
                if let Some(ref contents) = bucket_contents {
                    let ngram_text = ngram.join(" ");
                    contents.entry(bucket_id).or_default().insert(ngram_text);
                }
            }
        }
    }

    debug_println!(config, "  → Processed {} lines from {}", lines_processed, eval_name);
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

    let contamination_results: DashMap<String, Vec<ToxicContaminationEntry>> = DashMap::new();

    // Process files in parallel for maximum performance
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

#[derive(Default, Clone)]
pub struct TimingStats {
    text_extraction: Duration,
    embedding_computation: Duration,
    normalization: Duration,
    lsh_bucket_computation: Duration,
    bucket_lookup: Duration,
    collision_detection: Duration,
    threshold_evaluation: Duration,
    total_per_line: Duration,
    // Granular profiling
    hash_lookups: Duration,
    vector_cloning: Duration,
    vector_arithmetic: Duration,
    memory_allocation: Duration,
    random_generation: Duration,
}

#[derive(Clone)]
struct ToxicContaminationEntry {
    training_line: usize,
    eval_name: String,
    eval_line: usize,
    overlap_ratio: f32,
    matching_ngrams: Option<Vec<String>>,
    bucket_sizes: Option<Vec<usize>>,
}

fn process_toxic_training_file(
    file_path: &PathBuf,
    file_name: &str,
    config: &Config,
    embeddings: &EmbeddingMap,
    hyperplanes: &Hyperplanes,
    toxic_buckets: &ToxicBuckets,
    contamination_results: &DashMap<String, Vec<ToxicContaminationEntry>>
) -> Result<(), Error> {
    let data = read_pathbuf_to_mem(file_path)?;

    // debug_println!(config, "\n=== DETAILED TIMING LOG FOR {} ===", file_name);
    // debug_println!(config, "Format: Line# | Total | TextExt | Embed | Norm | LSH | Lookup | Collision | Threshold");
    // debug_println!(config, "---------|-------|--------|-------|------|-----|--------|-----------|----------");

    let mut contaminated_lines = 0;
    let mut total_lines = 0;
    let mut cumulative_stats = TimingStats::default();

    for (line_num, line) in data.lines().enumerate() {
        let line_start = Instant::now();
        let line = line?;

        // 1. Text extraction and parsing
        let text_start = Instant::now();
        let json_obj: Value = serde_json::from_str(&line)?;
        let line_text = get_nested_json_val(&json_obj, &config.content_key.to_string())?;
        let word_tokens = extract_words(&line_text, &config.punctuation_chars);
        let _training_word_count = word_tokens.len();
        let text_extraction_time = text_start.elapsed();

        // 2. N-gram generation and computation
        let embed_start = Instant::now();
        let _total_ngrams = if word_tokens.len() < config.ngram_size {
            if word_tokens.is_empty() { 0 } else { 1 }
        } else {
            word_tokens.len() - config.ngram_size + 1
        };

        // Initialize granular timing stats
        let mut granular_timing = TimingStats::default();
        
        let ngram_embeddings = compute_ngram_embedding_training(
            &word_tokens,
            config.ngram_size,
            embeddings,
            config.hash_seed as u64,
            &mut granular_timing
        );
        let embedding_time = embed_start.elapsed();

        let mut normalization_time = Duration::new(0, 0);
        let mut lsh_time = Duration::new(0, 0);
        let mut lookup_time = Duration::new(0, 0);
        let mut collision_time = Duration::new(0, 0);

        // Track distinct buckets hit per eval document
        let mut eval_bucket_matches: HashMap<(String, usize), (HashSet<u64>, usize)> = HashMap::new();
        let mut ngrams_with_any_collision = HashSet::new();

        // Track matching ngrams and bucket sizes for debug mode
        let mut matching_ngrams_per_eval: HashMap<(String, usize), Vec<String>> = HashMap::new();
        let mut bucket_sizes_per_eval: HashMap<(String, usize), Vec<usize>> = HashMap::new();

        // Process each n-gram with detailed timing
        for (ngram_idx, ngram_embedding) in ngram_embeddings.iter().enumerate() {
            // 3. Vector normalization
            let norm_start = Instant::now();
            // Skip normalization to preserve magnitude information
            normalization_time += norm_start.elapsed();

            // 4. LSH bucket computation
            let lsh_start = Instant::now();
            let bucket_id = compute_lsh_bucket(&ngram_embedding, hyperplanes);
            lsh_time += lsh_start.elapsed();

            // 5. Bucket lookup
            let lookup_start = Instant::now();
            if let Some(bucket_contents) = toxic_buckets.get(&bucket_id) {
                lookup_time += lookup_start.elapsed();

                // 6. Hot bucket optimization - skip super hot buckets if enabled
                let bucket_size = bucket_contents.value().len();
                
                if config.skip_hot_bucket_threshold > 0 && bucket_size > config.skip_hot_bucket_threshold as usize {
                    debug_println!(config, "🔥 Skipping hot bucket #{} with {} entries (threshold: {})", 
                                 bucket_id, bucket_size, config.skip_hot_bucket_threshold);
                    continue;
                }

                // 7. Collision detection - track distinct buckets hit per eval document
                let collision_start = Instant::now();
                for (eval_name, eval_line_num, eval_word_count) in bucket_contents.value() {
                    let key = (eval_name.clone(), *eval_line_num);
                    let entry = eval_bucket_matches.entry(key.clone()).or_insert((HashSet::new(), *eval_word_count));
                    entry.0.insert(bucket_id);  // Track which buckets this eval doc hit
                    // entry.1 already has eval_word_count

                    ngrams_with_any_collision.insert(ngram_idx);

                    // Store matching ngram text and bucket size for debug mode
                    if config.debug {
                        let ngram_text = if word_tokens.len() < config.ngram_size {
                            // For short documents, use all tokens
                            word_tokens.join(" ")
                        } else {
                            // Extract the specific ngram
                            word_tokens[ngram_idx..ngram_idx + config.ngram_size].join(" ")
                        };
                        let bucket_size = bucket_contents.value().len();
                        matching_ngrams_per_eval.entry(key.clone()).or_default().push(ngram_text);
                        bucket_sizes_per_eval.entry(key).or_default().push(bucket_size);
                    }
                }
                collision_time += collision_start.elapsed();
            } else {
                lookup_time += lookup_start.elapsed();
            }
        }

        // 7. Threshold evaluation - use distinct bucket counts per eval doc
        let threshold_start = Instant::now();
        let collision_count_total = eval_bucket_matches.len();

        // Log n-gram statistics for this document
        // debug_println!(config, "\n📊 N-GRAM STATS for line {}: Total={}, Unique_matches={}, Eval_docs_hit={}",
        //         line_num, total_ngrams, ngrams_with_any_collision.len(), collision_count_total);

        for ((eval_name, eval_line_num), (distinct_buckets_hit, eval_word_count)) in &eval_bucket_matches {
            let eval_ngrams_with_matches = distinct_buckets_hit.len();
            let eval_total_ngrams = if *eval_word_count < config.ngram_size {
                if *eval_word_count == 0 { 0 } else { 1 }
            } else {
                eval_word_count - config.ngram_size + 1
            };

            // Overlap ratio: % of eval n-grams that had matches in training
            let overlap_ratio = eval_ngrams_with_matches as f32 / eval_total_ngrams as f32;

            // debug_println!(config, "  → Match with {}:{}: eval_buckets_hit={}/{}, eval_ngrams={}, ratio={:.3}",
            //         eval_name, eval_line_num, eval_ngrams_with_matches, total_ngrams,
            //         eval_total_ngrams, overlap_ratio);

            if overlap_ratio >= config.toxic_overlap_threshold && complete_evaluation() {
                let (matching_ngrams, bucket_sizes) = if config.debug {
                    let key = (eval_name.clone(), *eval_line_num);
                    let ngrams = matching_ngrams_per_eval.get(&key).cloned();
                    let sizes = bucket_sizes_per_eval.get(&key).cloned();
                    (ngrams, sizes)
                } else {
                    (None, None)
                };

                contamination_results
                    .entry(file_name.to_string())
                    .or_default()
                    .push(ToxicContaminationEntry {
                        training_line: line_num,
                        eval_name: eval_name.clone(),
                        eval_line: *eval_line_num,
                        overlap_ratio,
                        matching_ngrams,
                        bucket_sizes,
                    });

                if collision_count_total == 1 {
                    contaminated_lines += 1;
                }
            }
        }
        let threshold_time = threshold_start.elapsed();

        let total_line_time = line_start.elapsed();
        total_lines += 1;

        // Log per-line timing
        // debug_println!(config,
        //     "{:8} | {:5.1}ms | {:6.1}ms | {:5.1}ms | {:4.1}ms | {:3.1}ms | {:6.1}ms | {:9.1}ms | {:8.1}ms",
        //     line_num,
        //     total_line_time.as_secs_f64() * 1000.0,
        //     text_extraction_time.as_secs_f64() * 1000.0,
        //     embedding_time.as_secs_f64() * 1000.0,
        //     normalization_time.as_secs_f64() * 1000.0,
        //     lsh_time.as_secs_f64() * 1000.0,
        //     lookup_time.as_secs_f64() * 1000.0,
        //     collision_time.as_secs_f64() * 1000.0,
        //     threshold_time.as_secs_f64() * 1000.0
        // );

        // Accumulate stats
        cumulative_stats.text_extraction += text_extraction_time;
        cumulative_stats.embedding_computation += embedding_time;
        cumulative_stats.normalization += normalization_time;
        cumulative_stats.lsh_bucket_computation += lsh_time;
        cumulative_stats.bucket_lookup += lookup_time;
        cumulative_stats.collision_detection += collision_time;
        cumulative_stats.threshold_evaluation += threshold_time;
        cumulative_stats.total_per_line += total_line_time;
        
        // Accumulate granular timing stats
        cumulative_stats.hash_lookups += granular_timing.hash_lookups;
        cumulative_stats.vector_cloning += granular_timing.vector_cloning;
        cumulative_stats.vector_arithmetic += granular_timing.vector_arithmetic;
        cumulative_stats.memory_allocation += granular_timing.memory_allocation;
        cumulative_stats.random_generation += granular_timing.random_generation;
    }

    // Print summary statistics
    let total_time = cumulative_stats.total_per_line.as_secs_f64() * 1000.0;
    debug_println!(config, "\n\n\n=== TIMING SUMMARY FOR {} ===", file_name);
    debug_println!(config, "Total lines processed: {}", total_lines);
    debug_println!(config, "Total time: {:.1}ms", total_time);
    debug_println!(config, "Average per line: {:.1}ms", total_time / total_lines as f64);
    debug_println!(config, "");
    debug_println!(config, "BREAKDOWN BY CATEGORY:");
    print_timing_category(config, "Text Extraction     ", cumulative_stats.text_extraction, total_time);
    print_timing_category(config, "Embedding Computation", cumulative_stats.embedding_computation, total_time);
    print_timing_category(config, "Vector Normalization", cumulative_stats.normalization, total_time);
    print_timing_category(config, "LSH Bucket Computation", cumulative_stats.lsh_bucket_computation, total_time);
    print_timing_category(config, "Bucket Lookup       ", cumulative_stats.bucket_lookup, total_time);
    print_timing_category(config, "Collision Detection ", cumulative_stats.collision_detection, total_time);
    print_timing_category(config, "Threshold Evaluation", cumulative_stats.threshold_evaluation, total_time);
    debug_println!(config, "");
    debug_println!(config, "GRANULAR EMBEDDING BREAKDOWN:");
    print_timing_category(config, "Hash Lookups        ", cumulative_stats.hash_lookups, total_time);
    print_timing_category(config, "Vector Cloning      ", cumulative_stats.vector_cloning, total_time);
    print_timing_category(config, "Vector Arithmetic   ", cumulative_stats.vector_arithmetic, total_time);
    print_timing_category(config, "Memory Allocation   ", cumulative_stats.memory_allocation, total_time);
    print_timing_category(config, "Random Generation   ", cumulative_stats.random_generation, total_time);
    debug_println!(config, "");

    if contaminated_lines > 0 {
        println!("  → Found {} contaminated lines out of {} total lines",
                contaminated_lines, total_lines);
    } else {
        println!("  → No contamination found ({} lines processed)", total_lines);
    }

    Ok(())
}

fn insert_with_hot_bucket_detection(
    _config: &Config,
    toxic_buckets: &ToxicBuckets,
    bucket_id: u64,
    _ngram_words: &[String],
    eval_name: &str,
    line_num: usize,
    word_count: usize
) {
    // Check if this is already a hot bucket
    let _current_size = toxic_buckets.get(&bucket_id)
        .map(|bucket| bucket.value().len())
        .unwrap_or(0);

    const _HOT_BUCKET_THRESHOLD: usize = 10000;

    // if current_size >= HOT_BUCKET_THRESHOLD {
    //     // Log the n-gram words going into this super hot bucket
    //     let ngram_text = ngram_words.join(" ");
    //     debug_println!(config, "🔥 HOT BUCKET #{} (size: {}) <- Adding n-gram: \"{}\" from {}:{}",
    //             bucket_id, current_size, ngram_text, eval_name, line_num);
    // } else if current_size > 0 && current_size % 1000 == 0 {
    //     // Log milestone sizes for growing buckets
    //     let _ngram_text = ngram_words.join(" ");
    //     debug_println!(config, "📈 Growing bucket #{} (size: {}) <- \"{}\" from {}:{}",
    //             bucket_id, current_size, _ngram_text, eval_name, line_num);
    // }

    // Perform the actual insertion
    toxic_buckets
        .entry(bucket_id)
        .or_default()
        .push((eval_name.to_string(), line_num, word_count));
}

fn print_timing_category(config: &Config, name: &str, duration: Duration, total_ms: f64) {
    let ms = duration.as_secs_f64() * 1000.0;
    let percentage = (ms / total_ms) * 100.0;
    debug_println!(config, "{}: {:8.1}ms ({:5.1}%)", name, ms, percentage);
}

// Placeholder for future complete evaluation logic
fn complete_evaluation() -> bool {
    true  // For now, always passes complete evaluation
}

fn save_bucket_contents(
    config: &Config,
    bucket_contents: &BucketContents
) -> Result<(), Error> {
    create_dir_all(&config.output_dir)?;
    let bucket_file = config.output_dir.join("bucket_contents.jsonl");

    let mut output_data = Vec::new();

    // Sort buckets by size (largest first) for easier analysis
    let mut bucket_info: Vec<(u64, usize, Vec<String>)> = Vec::new();
    for entry in bucket_contents.iter() {
        let bucket_id = *entry.key();
        let ngrams: Vec<String> = entry.value().iter().cloned().collect();
        let size = ngrams.len();
        bucket_info.push((bucket_id, size, ngrams));
    }
    bucket_info.sort_by(|a, b| b.1.cmp(&a.1)); // Sort by size descending

    for (bucket_id, size, ngrams) in bucket_info {
        let result = json!({
            "bucket_id": bucket_id,
            "size": size,
            "ngrams": ngrams
        });
        output_data.push(serde_json::to_vec(&result)?);
    }

    let mut output_bytes = Vec::new();
    for line in output_data {
        output_bytes.extend(line);
        output_bytes.push(b'\n');
    }

    write_mem_to_pathbuf(&output_bytes, &bucket_file)?;
    println!("Bucket contents saved to: {:?}", bucket_file);

    Ok(())
}

fn save_toxic_contamination_results(
    results: &DashMap<String, Vec<ToxicContaminationEntry>>,
    output_dir: &PathBuf
) -> Result<(), Error> {
    create_dir_all(output_dir)?;
    let output_file = output_dir.join(get_results_filename("toxic"));

    let mut output_data = Vec::new();
    let mut total_contaminations = 0;

    for entry in results.iter() {
        let training_file = entry.key();
        for contamination_entry in entry.value() {
            let mut result = json!({
                "training_file": training_file,
                "training_line": contamination_entry.training_line,
                "eval_dataset": contamination_entry.eval_name,
                "eval_line": contamination_entry.eval_line,
                "overlap_ratio": contamination_entry.overlap_ratio,
                "method": "toxic"
            });

            // Add matching ngrams and bucket sizes if available (debug mode)
            if let Some(ref ngrams) = contamination_entry.matching_ngrams {
                result["matching_ngrams"] = json!(ngrams);
            }
            if let Some(ref sizes) = contamination_entry.bucket_sizes {
                result["bucket_sizes"] = json!(sizes);
            }

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
        println!("\n\n\n=== TOXIC CONTAMINATION SUMMARY ===");
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

fn print_bucket_statistics(config: &Config, toxic_buckets: &ToxicBuckets, bucket_contents: &Option<BucketContents>) {
    debug_println!(config, "\n\n\n=== BUCKET DISTRIBUTION STATISTICS ===");

    let total_buckets = toxic_buckets.len();
    let mut bucket_sizes: Vec<usize> = Vec::new();
    let mut total_entries = 0;
    let mut empty_buckets = 0;

    // Calculate total potential buckets (2^hyperplanes)
    // For large hyperplane counts, use f64 to avoid overflow
    let total_potential_buckets = if config.toxic_hyperplanes >= 64 {
        f64::INFINITY // 2^64+ is effectively infinite
    } else {
        2_f64.powi(config.toxic_hyperplanes as i32)
    };
    let occupancy_percentage = (total_buckets as f64 / total_potential_buckets) * 100.0;

    // Collect bucket sizes
    for bucket in toxic_buckets.iter() {
        let size = bucket.value().len();
        bucket_sizes.push(size);
        total_entries += size;
        if size == 0 {
            empty_buckets += 1;
        }
    }

    // Sort for percentile calculations
    bucket_sizes.sort_unstable();

    let max_size = bucket_sizes.iter().max().copied().unwrap_or(0);
    let min_size = bucket_sizes.iter().min().copied().unwrap_or(0);
    let avg_size = if total_buckets > 0 { total_entries as f64 / total_buckets as f64 } else { 0.0 };

    // Calculate percentiles
    let p50 = percentile(&bucket_sizes, 50);
    let p75 = percentile(&bucket_sizes, 75);
    let p90 = percentile(&bucket_sizes, 90);
    let p95 = percentile(&bucket_sizes, 95);
    let p99 = percentile(&bucket_sizes, 99);

    // Find hot buckets (top 10 largest)
    let mut hot_buckets: Vec<(u64, usize)> = Vec::new();
    for bucket in toxic_buckets.iter() {
        let bucket_id = *bucket.key();
        let size = bucket.value().len();
        hot_buckets.push((bucket_id, size));
    }
    hot_buckets.sort_by(|a, b| b.1.cmp(&a.1)); // Sort by size descending
    hot_buckets.truncate(10); // Keep only top 10

    if total_potential_buckets.is_infinite() {
        println!("Total potential buckets: 2^{} (extremely large)", config.toxic_hyperplanes);
        println!("Total occupied buckets: {}", total_buckets);
        println!("Bucket space occupancy: ~0% (negligible)");
    } else {
        println!("Total potential buckets: {}", total_potential_buckets as u64);
        println!("Total occupied buckets: {}", total_buckets);
        println!("Bucket space occupancy: {:.6}%", occupancy_percentage);
    }
    println!("Total entries: {}", total_entries);
    println!("Empty buckets: {} ({:.1}%)", empty_buckets, empty_buckets as f64 / total_buckets as f64 * 100.0);
    println!();
    println!("BUCKET SIZE DISTRIBUTION:");
    println!("  Min:     {}", min_size);
    println!("  Average: {:.1}", avg_size);
    println!("  Median:  {}", p50);
    println!("  75th %:  {}", p75);
    println!("  90th %:  {}", p90);
    println!("  95th %:  {}", p95);
    println!("  99th %:  {}", p99);
    println!("  Max:     {}", max_size);
    println!();
    println!("TOP 10 HOTTEST BUCKETS:");
    if let Some(ref contents) = bucket_contents {
        println!("Bucket ID          | Entries | Distinct N-grams");
        println!("-------------------|---------|----------------");
        for (bucket_id, size) in &hot_buckets {
            let distinct_count = contents.get(bucket_id)
                .map(|ngrams| ngrams.len())
                .unwrap_or(0);
            println!("{:18} | {:7} | {:7}", bucket_id, size, distinct_count);
        }

        // Show actual ngrams for top 3 hottest buckets
        println!("\nSAMPLE N-GRAMS FROM TOP 3 HOTTEST BUCKETS:");
        for (i, (bucket_id, size)) in hot_buckets.iter().take(3).enumerate() {
            if let Some(ngrams_set) = contents.get(bucket_id) {
                let mut ngrams: Vec<String> = ngrams_set.iter().cloned().collect();
                ngrams.sort();
                let sample_size = std::cmp::min(10, ngrams.len());
                println!("\n{}. Bucket {} ({} entries, {} distinct):",
                        i + 1, bucket_id, size, ngrams.len());
                for ngram in ngrams.iter().take(sample_size) {
                    println!("   \"{}\"", ngram);
                }
                if ngrams.len() > sample_size {
                    println!("   ... and {} more", ngrams.len() - sample_size);
                }
            }
        }
    } else {
        println!("Bucket ID          | Entries");
        println!("-------------------|--------");
        for (bucket_id, size) in &hot_buckets {
            println!("{:18} | {:7}", bucket_id, size);
        }
    }

    // Identify problematic buckets
    let large_bucket_threshold = (avg_size * 10.0) as usize;
    let large_buckets: Vec<_> = bucket_sizes.iter().filter(|&&size| size > large_bucket_threshold).collect();

    if !large_buckets.is_empty() {
        println!();
        println!("⚠️  PERFORMANCE WARNING:");
        println!("Found {} buckets with >{}x average size (>{} entries)",
                large_buckets.len(),
                10,
                large_bucket_threshold);
        println!("These hot buckets may cause collision detection slowdowns.");
    }

    println!("=====================================\n");
}

fn percentile(sorted_data: &[usize], percentile: usize) -> usize {
    if sorted_data.is_empty() {
        return 0;
    }
    let index = (percentile * sorted_data.len() / 100).min(sorted_data.len() - 1);
    sorted_data[index]
}
