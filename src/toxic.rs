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
use ndarray::{Array2, ArrayView1};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha20Rng;
use rayon::prelude::*;
use serde_json::{Value, json};
use std::collections::{HashMap, HashSet};
use std::fs::{create_dir_all, File};
use std::io::{BufRead, BufReader, BufWriter, Read, Write};
use std::path::PathBuf;
// Removed atomic imports - document IDs now read from JSON
use std::time::{Duration, Instant};

use mj_io::{expand_dirs, read_pathbuf_to_mem, write_mem_to_pathbuf, build_pbar};

use crate::{Config, get_nested_json_val, clean_text_allowlist, get_results_filename, debug_println};


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
pub const EMBEDDING_DIM: usize = 128;

// Document IDs are now generated in Python download script and read from JSON

// LSH bucket storage: maps bucket_id to list of document IDs
type ToxicBuckets = DashMap<u64, Vec<u32>>;

// Bucket content storage for debug mode: maps bucket_id to set of ngram texts
type BucketContents = DashMap<u64, HashSet<String>>;

// Hot bucket tracking: set of bucket IDs that exceed skip_hot_bucket_threshold
type HotBuckets = HashSet<u64>;

// Eval document metadata: maps document_id to (eval_name, line_num, total_ngrams, live_ngrams)
type EvalDocuments = DashMap<u32, (String, usize, usize, usize)>;

// Embedding storage: word -> 300d vector (thread-safe)
pub type EmbeddingMap = DashMap<String, Vec<f32>>;

// LSH hyperplane storage: vectorized format for efficient matrix operations
#[derive(Clone)]
struct VectorizedHyperplanes {
    // Matrix shape: (num_hyperplanes, EMBEDDING_DIM)
    // Each row is a hyperplane, enabling efficient matrix-vector multiplication
    data: Array2<f32>,
    num_planes: usize,
}

impl VectorizedHyperplanes {
    fn new(hyperplanes: Vec<Vec<f32>>) -> Result<Self, Error> {
        let num_planes = hyperplanes.len();
        if num_planes == 0 {
            return Err(anyhow::anyhow!("Cannot create VectorizedHyperplanes with zero hyperplanes"));
        }

        let flat_data: Vec<f32> = hyperplanes.into_iter().flatten().collect();
        let expected_size = num_planes * EMBEDDING_DIM;
        if flat_data.len() != expected_size {
            return Err(anyhow::anyhow!(
                "Hyperplane data size mismatch: expected {}, got {}",
                expected_size, flat_data.len()
            ));
        }

        let data = Array2::from_shape_vec((num_planes, EMBEDDING_DIM), flat_data)
            .map_err(|e| anyhow::anyhow!("Failed to create hyperplane matrix: {}", e))?;

        Ok(Self { data, num_planes })
    }

    #[allow(dead_code)]
    fn get_hyperplane(&self, index: usize) -> Option<ArrayView1<f32>> {
        if index < self.num_planes {
            Some(self.data.row(index))
        } else {
            None
        }
    }
}

// Updated type alias for vectorized hyperplanes
type Hyperplanes = VectorizedHyperplanes;

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
    let (toxic_buckets, hot_buckets, eval_documents, eval_vocabulary, _bucket_contents) = build_toxic_index(config, &embeddings, &hyperplanes)?;
    println!("Built TOXIC index with {} buckets", toxic_buckets.len());

    // Step 4: Process training data and detect contamination
    println!("Processing training data for contamination detection...");
    detect_toxic_contamination(config, &embeddings, &hyperplanes, &toxic_buckets, &hot_buckets, &eval_documents, &eval_vocabulary)?;

    println!("TOXIC contamination detection completed in {:?} seconds", start_main.elapsed().as_secs());
    Ok(())
}

fn save_embeddings_binary(embeddings: &EmbeddingMap, path: &PathBuf) -> Result<(), Error> {
    let mut file = BufWriter::new(File::create(path)?);

    // Header
    file.write_all(&0x454D4244u32.to_le_bytes())?; // Magic "EMBD"
    file.write_all(&(embeddings.len() as u32).to_le_bytes())?;
    file.write_all(&(EMBEDDING_DIM as u32).to_le_bytes())?;

    // Collect and sort words for deterministic order
    let mut words: Vec<_> = embeddings.iter().map(|entry| entry.key().clone()).collect();
    words.sort();

    // Write vocabulary
    for word in &words {
        let word_bytes = word.as_bytes();
        file.write_all(&(word_bytes.len() as u16).to_le_bytes())?;
        file.write_all(word_bytes)?;
    }

    // Write embeddings in same order
    for word in &words {
        let vector = embeddings.get(word).unwrap();
        for &val in vector.value() {
            file.write_all(&val.to_le_bytes())?;
        }
    }

    println!("Saved binary embeddings to: {:?}", path);
    Ok(())
}

fn load_embeddings_binary(path: &PathBuf) -> Result<EmbeddingMap, Error> {
    let mut file = BufReader::new(File::open(path)?);
    let embeddings = DashMap::new();

    // Read header
    let mut buf = [0u8; 4];
    file.read_exact(&mut buf)?;
    if u32::from_le_bytes(buf) != 0x454D4244 {
        return Err(anyhow::anyhow!("Invalid binary embedding file"));
    }

    file.read_exact(&mut buf)?;
    let vocab_size = u32::from_le_bytes(buf) as usize;

    file.read_exact(&mut buf)?;
    let embedding_dim = u32::from_le_bytes(buf) as usize;

    if embedding_dim != EMBEDDING_DIM {
        return Err(anyhow::anyhow!("Embedding dimension mismatch: expected {}, got {}", EMBEDDING_DIM, embedding_dim));
    }

    // Read vocabulary
    let mut words = Vec::with_capacity(vocab_size);
    for _ in 0..vocab_size {
        let mut len_buf = [0u8; 2];
        file.read_exact(&mut len_buf)?;
        let word_len = u16::from_le_bytes(len_buf) as usize;

        let mut word_buf = vec![0u8; word_len];
        file.read_exact(&mut word_buf)?;
        words.push(String::from_utf8(word_buf)?);
    }

    // Read embeddings
    for word in words {
        let mut vector = vec![0.0f32; embedding_dim];
        for val in &mut vector {
            let mut val_buf = [0u8; 4];
            file.read_exact(&mut val_buf)?;
            *val = f32::from_le_bytes(val_buf);
        }
        embeddings.insert(word, vector);

        if embeddings.len() % 100000 == 0 {
            print!(".");
            std::io::stdout().flush().unwrap();
        }
    }

    println!(); // New line after dots
    Ok(embeddings)
}

fn load_embeddings_text(embedding_path: &PathBuf) -> Result<EmbeddingMap, Error> {
    println!("Loading text embeddings from: {:?}", embedding_path);
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

fn load_embeddings(embedding_path: &PathBuf) -> Result<EmbeddingMap, Error> {
    // Try binary format first (.bin extension)
    let binary_path = embedding_path.with_extension("bin");
    if binary_path.exists() {
        println!("Loading binary embeddings from: {:?}", binary_path);
        return load_embeddings_binary(&binary_path);
    }

    // Fall back to text format and optionally create binary version
    let embeddings = load_embeddings_text(embedding_path)?;

    // Save binary version for next time
    if let Err(e) = save_embeddings_binary(&embeddings, &binary_path) {
        println!("Warning: Failed to save binary embeddings: {}", e);
    }

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

    // Convert to vectorized format
    VectorizedHyperplanes::new(hyperplanes)
}

fn extract_words(text: &str, punctuation_chars: &str) -> Vec<String> {
    // Clean text using allowlist approach for TOXIC contamination detection
    let cleaned = clean_text_allowlist(text, punctuation_chars);

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
    let vector_view = ArrayView1::from(normalized_vector);

    // Vectorized matrix-vector multiplication: all hyperplanes at once
    // Shape: (num_hyperplanes, EMBEDDING_DIM) × (EMBEDDING_DIM,) = (num_hyperplanes,)
    let dot_products = hyperplanes.data.dot(&vector_view);

    // Convert dot products to bucket ID using bit operations
    let mut bucket_id = 0u64;
    for (i, &dot_product) in dot_products.iter().enumerate() {
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
) -> Result<(ToxicBuckets, HotBuckets, EvalDocuments, HashSet<String>, Option<BucketContents>), Error> {
    let toxic_buckets: ToxicBuckets = DashMap::new();
    let eval_documents: EvalDocuments = DashMap::new();
    let eval_vocabulary: DashMap<String, ()> = DashMap::new(); // Thread-safe set for vocabulary
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
            &eval_documents,
            &eval_vocabulary,
            &bucket_contents
        ) {
            println!("Error processing reference file {:?}: {:?}", file_path, e);
        }
        pbar.inc(1);
    });

    // Identify hot buckets after processing all eval data
    let hot_buckets: HotBuckets = if config.skip_hot_bucket_threshold > 0 {
        toxic_buckets.iter()
            .filter(|entry| entry.value().len() > config.skip_hot_bucket_threshold as usize)
            .map(|entry| *entry.key())
            .collect()
    } else {
        HashSet::new()
    };

    if !hot_buckets.is_empty() {
        println!("Identified {} hot buckets (threshold: {})",
                 hot_buckets.len(), config.skip_hot_bucket_threshold);

        // Update live n-gram counts by excluding hot buckets
        update_live_ngram_counts(config, embeddings, hyperplanes, &hot_buckets, &eval_documents, &toxic_buckets)?;
    }

    // Analyze bucket distribution statistics
    if config.debug {
        print_bucket_statistics(config, &toxic_buckets, &bucket_contents);
    }

    // Save bucket contents to file for debug analysis
    if let Some(ref contents) = bucket_contents {
        save_bucket_contents(config, contents)?;
    }

    // Convert eval vocabulary to HashSet for return
    let eval_vocab_set: HashSet<String> = eval_vocabulary.into_iter().map(|(word, _)| word).collect();

    Ok((toxic_buckets, hot_buckets, eval_documents, eval_vocab_set, bucket_contents))
}

fn process_toxic_reference_file(
    file_path: &PathBuf,
    config: &Config,
    embeddings: &EmbeddingMap,
    hyperplanes: &Hyperplanes,
    toxic_buckets: &ToxicBuckets,
    eval_documents: &EvalDocuments,
    eval_vocabulary: &DashMap<String, ()>,
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
    let mut skipped_entries = 0;
    let min_word_count = config.ngram_size * 2;

    for (line_num, line) in data.lines().enumerate() {
        let line = line?;
        let json_obj: Value = serde_json::from_str(&line)?;
        let line_text = get_nested_json_val(&json_obj, &config.content_key.to_string())?;

        // For TOXIC, we work with words directly, not token IDs
        let word_tokens = extract_words(&line_text, &config.punctuation_chars);
        
        debug_println!(config, "📖 REFERENCE {}: '{}' -> '{}'", 
                      eval_name, line_text, word_tokens.join(" "));
        let word_count = word_tokens.len();

        // Track vocabulary from eval data
        for word in &word_tokens {
            eval_vocabulary.insert(word.clone(), ());
        }

        // Skip entries with insufficient words for meaningful n-gram analysis
        if word_count < min_word_count {
            skipped_entries += 1;
            continue;
        }

        lines_processed += 1;

        // Calculate total n-grams for this document
        let total_ngrams = if word_tokens.len() < config.ngram_size {
            if word_tokens.is_empty() { 0 } else { 1 }
        } else {
            word_tokens.len() - config.ngram_size + 1
        };

        // Read document ID from JSON (generated in Python download script)
        let doc_id = json_obj.get("doc_id")
            .and_then(|v| v.as_u64())
            .ok_or_else(|| anyhow::anyhow!("Missing or invalid doc_id field in reference file"))?
            as u32;

        // Store document metadata (will be updated with live count later)
        eval_documents.insert(doc_id, (eval_name.clone(), line_num, total_ngrams, total_ngrams));

        // Process n-grams with hot bucket detection
        if word_tokens.len() < config.ngram_size {
            if !word_tokens.is_empty() {
                // For documents shorter than ngram_size, use all tokens
                let mut dummy_timing = TimingStats::default();
                let sum_embedding = sum_word_embeddings_eval(&word_tokens, embeddings, config.hash_seed as u64, &mut dummy_timing);
                // Skip normalization to preserve magnitude information
                let bucket_id = compute_lsh_bucket(&sum_embedding, hyperplanes);
                insert_with_hot_bucket_detection(
                    &toxic_buckets,
                    bucket_id,
                    doc_id
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
                    &toxic_buckets,
                    bucket_id,
                    doc_id
                );

                // Store ngram text for debug analysis
                if let Some(ref contents) = bucket_contents {
                    let ngram_text = ngram.join(" ");
                    contents.entry(bucket_id).or_default().insert(ngram_text);
                }
            }
        }
    }

    debug_println!(config, "  → Processed {} lines from {} (skipped {} entries with < {} words)",
                   lines_processed, eval_name, skipped_entries, min_word_count);
    Ok(())
}

fn detect_toxic_contamination(
    config: &Config,
    embeddings: &EmbeddingMap,
    hyperplanes: &Hyperplanes,
    toxic_buckets: &ToxicBuckets,
    hot_buckets: &HotBuckets,
    eval_documents: &EvalDocuments,
    eval_vocabulary: &HashSet<String>
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
            hot_buckets,
            eval_documents,
            eval_vocabulary,
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

#[derive(Clone)]
struct ContaminationCluster {
    start_idx: usize,
    end_idx: usize,
    document_matches: HashMap<u32, usize>, // doc_id -> match_length
    matching_ngrams: Vec<String>,
    bucket_sizes: Vec<usize>,
    #[allow(dead_code)]
    distinct_buckets: HashSet<u64>,
}

#[derive(Default)]
struct SamplingStats {
    bucket_hits: usize,
    bucket_misses: usize,
    hot_buckets_skipped: usize,
    vocab_filtered: usize,
}

/// Process n-grams with sampling optimization: sample every M n-grams, then expand around hits
fn process_ngrams_with_sampling(
    word_tokens: &[String],
    ngram_embeddings: &[Vec<f32>],
    config: &Config,
    hyperplanes: &Hyperplanes,
    toxic_buckets: &ToxicBuckets,
    hot_buckets: &HotBuckets,
    eval_vocabulary: &HashSet<String>,
    timing_stats: &mut TimingStats,
) -> Result<(Vec<ContaminationCluster>, SamplingStats), Error> {
    let mut clusters = Vec::new();
    let mut processed_indices = HashSet::new();
    let mut i = 0;

    // Statistics tracking
    let mut bucket_hits_count = 0;
    let mut bucket_misses_count = 0;
    let mut hot_buckets_skipped_count = 0;
    let mut vocab_filtered_count = 0;

    // debug_println!(config, "🔍 Starting sampling with M={}, max_misses={}",
    //               config.sample_every_m_tokens, config.max_consecutive_misses);

    while i < ngram_embeddings.len() {
        // Skip if we've already processed this n-gram as part of a cluster
        if processed_indices.contains(&i) {
            i += 1;
            continue;
        }

        // Sample every M n-grams (unless M=1, which means no sampling)
        if config.sample_every_m_tokens > 1 && i % config.sample_every_m_tokens != 0 {
            i += config.sample_every_m_tokens - (i % config.sample_every_m_tokens);
            continue;
        }

        // Check this n-gram for contamination
        match check_ngram_for_collision(
            i,
            word_tokens,
            &ngram_embeddings[i],
            config,
            hyperplanes,
            toxic_buckets,
            hot_buckets,
            eval_vocabulary,
            timing_stats,
            &mut bucket_hits_count,
            &mut bucket_misses_count,
            &mut hot_buckets_skipped_count,
            &mut vocab_filtered_count
        )? {
            CollisionResult::Hit(document_ids) => {
                // Found contamination! Use intersection-based walking
                let ngram_text = if word_tokens.len() < config.ngram_size {
                    word_tokens.join(" ")
                } else {
                    word_tokens[i..i + config.ngram_size].join(" ")
                };
                debug_println!(config, "\n💥 INITIAL HIT DETECTED at n-gram {} with {} documents!", i, document_ids.len());
                debug_println!(config, "🔤 N-gram text: '{}'", ngram_text);
                debug_println!(config, "📄 Document IDs: {:?}", document_ids);

                let cluster = expand_contamination_cluster_with_intersection(
                    i,
                    word_tokens,
                    ngram_embeddings,
                    config,
                    hyperplanes,
                    toxic_buckets,
                    hot_buckets,
                    eval_vocabulary,
                    document_ids,
                    timing_stats,
                    &mut bucket_hits_count,
                    &mut bucket_misses_count,
                    &mut hot_buckets_skipped_count,
                    &mut vocab_filtered_count,
                )?;

                // Mark all indices in this cluster as processed
                for idx in cluster.start_idx..=cluster.end_idx {
                    processed_indices.insert(idx);
                }

                clusters.push(cluster.clone());
                debug_println!(config, "📍 Cluster completed: indices {}-{}, {} document matches",
                              cluster.start_idx, cluster.end_idx, cluster.document_matches.len());

                // Jump past the processed region
                i = processed_indices.iter().max().copied().unwrap_or(i) + 1;
            }
            CollisionResult::Miss | CollisionResult::VocabFiltered | CollisionResult::HotBucketSkipped => {
                // No hit, continue sampling
                i += config.sample_every_m_tokens.max(1);
            }
        }
    }

    // debug_println!(config, "📊 Sampling stats - Hits: {}, Misses: {}, Hot skipped: {}, Vocab filtered: {}",
    //               bucket_hits_count, bucket_misses_count, hot_buckets_skipped_count, vocab_filtered_count);

    let stats = SamplingStats {
        bucket_hits: bucket_hits_count,
        bucket_misses: bucket_misses_count,
        hot_buckets_skipped: hot_buckets_skipped_count,
        vocab_filtered: vocab_filtered_count,
    };

    Ok((clusters, stats))
}

/// Check a single n-gram for collisions, return document IDs that match
fn check_ngram_for_collision(
    ngram_idx: usize,
    word_tokens: &[String],
    ngram_embedding: &[f32],
    config: &Config,
    hyperplanes: &Hyperplanes,
    toxic_buckets: &ToxicBuckets,
    hot_buckets: &HotBuckets,
    eval_vocabulary: &HashSet<String>,
    timing_stats: &mut TimingStats,
    bucket_hits_count: &mut usize,
    bucket_misses_count: &mut usize,
    hot_buckets_skipped_count: &mut usize,
    vocab_filtered_count: &mut usize,
) -> Result<CollisionResult, Error> {
    // Vocabulary filtering - skip n-grams with out-of-vocabulary tokens
    let ngram_tokens = if word_tokens.len() < config.ngram_size {
        &word_tokens[..]
    } else {
        &word_tokens[ngram_idx..ngram_idx + config.ngram_size]
    };
    
    debug_println!(config, "🔍 Checking n-gram {}: '{}'", ngram_idx, ngram_tokens.join(" "));

    // Check if all tokens in this n-gram exist in eval vocabulary
    let all_tokens_in_eval = ngram_tokens.iter().all(|token| eval_vocabulary.contains(token));
    if !all_tokens_in_eval {
        // Debug: Show which n-grams are being filtered
        let missing_tokens: Vec<String> = ngram_tokens.iter()
            .filter(|token| !eval_vocabulary.contains(*token))
            .map(|s| s.to_string())
            .collect();
        debug_println!(config, "🚫 VOCAB FILTERED n-gram '{}' - missing tokens: {:?}", 
                      ngram_tokens.join(" "), missing_tokens);
        *vocab_filtered_count += 1;
        return Ok(CollisionResult::VocabFiltered);
    }

    // LSH bucket computation
    let lsh_start = Instant::now();
    let bucket_id = compute_lsh_bucket(&ngram_embedding, hyperplanes);
    timing_stats.lsh_bucket_computation += lsh_start.elapsed();

    // Hot bucket optimization - skip hot buckets immediately
    if !hot_buckets.is_empty() && hot_buckets.contains(&bucket_id) {
        *hot_buckets_skipped_count += 1;
        return Ok(CollisionResult::HotBucketSkipped);
    }

    // Bucket lookup
    let lookup_start = Instant::now();
    if let Some(bucket_contents) = toxic_buckets.get(&bucket_id) {
        *bucket_hits_count += 1;
        timing_stats.bucket_lookup += lookup_start.elapsed();

        // Collect all document IDs that match this bucket
        let collisions = bucket_contents.value().clone();
        debug_println!(config, "✅ HIT! Found {} documents: {:?}", collisions.len(), collisions);
        Ok(CollisionResult::Hit(collisions))
    } else {
        *bucket_misses_count += 1;
        timing_stats.bucket_lookup += lookup_start.elapsed();
        debug_println!(config, "❌ MISS! No bucket collision");
        Ok(CollisionResult::Miss)
    }
}

#[derive(Debug, Clone)]
struct DocumentState {
    match_length: usize,
    consecutive_misses: usize,
}

#[derive(Debug)]
enum CollisionResult {
    Hit(Vec<u32>),      // Found documents in bucket
    Miss,               // No bucket collision
    VocabFiltered,      // N-gram filtered due to vocabulary
    HotBucketSkipped,   // Skipped hot bucket
}

/// Expand backward and forward from a hit using per-document miss tracking
///
/// PER-DOCUMENT MISS TRACKING ALGORITHM:
/// ====================================
/// This function implements sophisticated contamination detection that tracks miss counts
/// individually per document, allowing for more granular and accurate gap handling.
/// 
/// Unlike the previous global reset approach, this algorithm maintains separate state
/// for each document, enabling precise gap tolerance on a per-document basis.
///
/// SAMPLING-BOUNDED LEFT TRAVERSAL:
/// ```
/// Timeline of sampling (M=10):
/// Position:     0    10    20    30    40    50    60
/// Sample:       ✓     ✓     ✓     ✓     ✓     ✓     ✓
///                                           ^
///                                    Current hit at 50
///                                    
/// Left bound = 50 - 10 = 40
/// ```
/// 
/// PER-DOCUMENT TRACKING LOGIC:
/// - Each document maintains: match_length, consecutive_misses
/// - Walk left/right, checking each n-gram's bucket
/// - For documents in current bucket: increment match_length, reset consecutive_misses to 0
/// - For documents NOT in current bucket: increment consecutive_misses, keep match_length unchanged
/// - Drop individual documents when consecutive_misses > max_consecutive_misses
/// - Stop direction when no documents remain active (all individually exceeded tolerance)
///
/// GAP HANDLING WITH max_consecutive_misses = 2:
/// ```
/// Initial hit:    [doc1(m:1,miss:0), doc2(m:1,miss:0), doc3(m:1,miss:0)]
/// Step 1 bucket:  [doc1, doc2]
/// After step 1:   [doc1(m:2,miss:0), doc2(m:2,miss:0), doc3(m:1,miss:1)]
/// Step 2 bucket:  [doc1]  
/// After step 2:   [doc1(m:3,miss:0), doc2(m:2,miss:1), doc3(m:1,miss:2)]
/// Step 3 bucket:  []
/// After step 3:   [doc1(m:3,miss:1), doc2(m:2,miss:2)] + doc3 DROPPED (miss:3 > max:2)
/// Step 4 bucket:  [doc1]
/// After step 4:   [doc1(m:4,miss:0)] + doc2 DROPPED (miss:3 > max:2)  
/// Step 5 bucket:  []
/// After step 5:   [doc1(m:4,miss:1)]
/// Step 6 bucket:  []
/// After step 6:   [doc1(m:4,miss:2)]
/// Step 7 bucket:  []
/// After step 7:   [] + doc1 DROPPED (miss:3 > max:2) → STOP (no documents remain)
/// ```
///
/// KEY ADVANTAGES OVER GLOBAL RESET:
/// - Documents are individually pruned based on their own miss patterns
/// - Well-matching documents continue contributing even when others fail consistently
/// - More accurate match_length tracking per document (no artificial resets)
/// - Natural termination when all documents individually exceed miss tolerance
/// - Eliminates the crude "reset every N misses" batching behavior
/// - Better handles cases where some documents have sparse matches vs. others
fn expand_contamination_cluster_with_intersection(
    hit_idx: usize,
    word_tokens: &[String],
    ngram_embeddings: &[Vec<f32>],
    config: &Config,
    hyperplanes: &Hyperplanes,
    toxic_buckets: &ToxicBuckets,
    hot_buckets: &HotBuckets,
    eval_vocabulary: &HashSet<String>,
    initial_document_ids: Vec<u32>,
    timing_stats: &mut TimingStats,
    bucket_hits_count: &mut usize,
    bucket_misses_count: &mut usize,
    hot_buckets_skipped_count: &mut usize,
    vocab_filtered_count: &mut usize,
) -> Result<ContaminationCluster, Error> {
    let mut start_idx = hit_idx;
    let mut end_idx = hit_idx;
    let mut matching_ngrams = Vec::new();
    let mut bucket_sizes = Vec::new();
    let mut distinct_buckets = HashSet::new();
    
    // Initialize cumulative stats for ALL documents that participate in the cluster
    let mut all_document_stats: HashMap<u32, DocumentState> = HashMap::new();
    for doc_id in &initial_document_ids {
        all_document_stats.insert(*doc_id, DocumentState {
            match_length: 1,
            consecutive_misses: 0,
        });
    }
    
    // Track which documents are still "active" for continued evaluation
    let mut active_documents: HashSet<u32> = initial_document_ids.iter().cloned().collect();
    
    debug_println!(config, "\n🎯 INTERSECTION WALKING DEBUG - Initial hit at index {}", hit_idx);
    debug_println!(config, "📊 Starting with {} documents: {:?}", initial_document_ids.len(), initial_document_ids);

    // Expand backward (no left bound for debugging)
    let left_bound = 0;
    // TODO: Re-enable left bound optimization: hit_idx.saturating_sub(config.sample_every_m_tokens) when config.sample_every_m_tokens > 1
    
    debug_println!(config, "🔍 Left bound: {}, max_misses: {}", left_bound, config.max_consecutive_misses);
    
    debug_println!(config, "\n⬅️  STARTING LEFT WALK");
    let mut i = hit_idx;
    while i > left_bound && !active_documents.is_empty() {
        i -= 1;
        
        let ngram_text = if word_tokens.len() < config.ngram_size {
            word_tokens.join(" ")
        } else {
            word_tokens[i..i + config.ngram_size].join(" ")
        };
        debug_println!(config, "\n🔍 Left step {}: '{}'\n   Active docs before: {:?}", 
                      i, ngram_text, 
                      active_documents.iter().filter_map(|id| all_document_stats.get(id).map(|state| format!("{}(m:{},miss:{})", id, state.match_length, state.consecutive_misses))).collect::<Vec<_>>());

        match check_ngram_for_collision(
            i, word_tokens, &ngram_embeddings[i], config, hyperplanes,
            toxic_buckets, hot_buckets, eval_vocabulary, timing_stats,
            bucket_hits_count, bucket_misses_count, hot_buckets_skipped_count, vocab_filtered_count
        )? {
            CollisionResult::Hit(current_documents) => {
                let current_set: HashSet<u32> = current_documents.iter().cloned().collect();
                let mut matches = 0;
                let mut misses = 0;
                
                debug_println!(config, "   ✅ Bucket collision! Found {} docs", current_documents.len());
                
                // Update each document's state based on intersection
                for doc_id in &active_documents.clone() {
                    if let Some(state) = all_document_stats.get_mut(doc_id) {
                        if current_set.contains(doc_id) {
                            // Document found in current bucket
                            state.match_length += 1;
                            state.consecutive_misses = 0;
                            matches += 1;
                        } else {
                            // Document not found in current bucket
                            state.consecutive_misses += 1;
                            misses += 1;
                        }
                    }
                }
                
                debug_println!(config, "   📈 Matches: {}, 📉 Misses: {}", matches, misses);
                
                if matches > 0 {
                    start_idx = i;
                    debug_println!(config, "   ⬅️  Extended cluster start to {}", start_idx);
                }
            }
            CollisionResult::Miss => {
                debug_println!(config, "   ❌ No bucket collision");
                // No bucket found, increment miss count for all active documents
                let miss_count = active_documents.len();
                for doc_id in &active_documents {
                    if let Some(state) = all_document_stats.get_mut(doc_id) {
                        state.consecutive_misses += 1;
                    }
                }
                debug_println!(config, "   📉 All {} docs missed", miss_count);
            }
            CollisionResult::VocabFiltered => {
                debug_println!(config, "   🚫 Vocabulary filtered - counts as miss");
                // Vocabulary filtering counts as a miss for all active documents
                let miss_count = active_documents.len();
                for doc_id in &active_documents {
                    if let Some(state) = all_document_stats.get_mut(doc_id) {
                        state.consecutive_misses += 1;
                    }
                }
                debug_println!(config, "   📉 All {} docs missed (vocab filtered)", miss_count);
            }
            CollisionResult::HotBucketSkipped => {
                debug_println!(config, "   ⏭️  Skipped hot bucket - no miss penalty");
                // Hot bucket skip doesn't count as a miss (performance optimization)
            }
        }
        
        // Remove documents from active set that exceeded miss threshold (but keep their stats)
        let before_count = active_documents.len();
        active_documents.retain(|doc_id| {
            all_document_stats.get(doc_id)
                .map(|state| state.consecutive_misses < config.max_consecutive_misses)
                .unwrap_or(false)
        });
        let dropped_count = before_count - active_documents.len();
        if dropped_count > 0 {
            debug_println!(config, "   🗑️  DROPPED {} docs from active set (miss >= {}), {} remaining", dropped_count, config.max_consecutive_misses, active_documents.len());
        }
    }

    // Reset active documents for right walk, keeping accumulated match lengths in all_document_stats
    active_documents = initial_document_ids.iter().cloned().collect();
    
    // Reset consecutive misses for right walk, but keep match_length accumulated from left walk
    for doc_id in &initial_document_ids {
        if let Some(state) = all_document_stats.get_mut(doc_id) {
            state.consecutive_misses = 0;
        }
    }
    
    debug_println!(config, "\n➡️  STARTING RIGHT WALK");
    debug_println!(config, "📊 Reset to original document set with accumulated matches: {:?}", 
                  initial_document_ids.iter().filter_map(|id| all_document_stats.get(id).map(|state| format!("{}(m:{},miss:{})", id, state.match_length, state.consecutive_misses))).collect::<Vec<_>>());
    
    i = hit_idx;
    
    while i + 1 < ngram_embeddings.len() && !active_documents.is_empty() {
        i += 1;
        
        let ngram_text = if word_tokens.len() < config.ngram_size {
            word_tokens.join(" ")
        } else {
            word_tokens[i..i + config.ngram_size].join(" ")
        };
        debug_println!(config, "\n🔍 Right step {}: '{}'\n   Active docs before: {:?}", 
                      i, ngram_text,
                      active_documents.iter().filter_map(|id| all_document_stats.get(id).map(|state| format!("{}(m:{},miss:{})", id, state.match_length, state.consecutive_misses))).collect::<Vec<_>>());

        match check_ngram_for_collision(
            i, word_tokens, &ngram_embeddings[i], config, hyperplanes,
            toxic_buckets, hot_buckets, eval_vocabulary, timing_stats,
            bucket_hits_count, bucket_misses_count, hot_buckets_skipped_count, vocab_filtered_count
        )? {
            CollisionResult::Hit(current_documents) => {
                let current_set: HashSet<u32> = current_documents.iter().cloned().collect();
                let mut matches = 0;
                let mut misses = 0;
                
                debug_println!(config, "   ✅ Bucket collision! Found {} docs", current_documents.len());
                
                // Update each document's state based on intersection
                for doc_id in &active_documents.clone() {
                    if let Some(state) = all_document_stats.get_mut(doc_id) {
                        if current_set.contains(doc_id) {
                            // Document found in current bucket
                            state.match_length += 1;
                            state.consecutive_misses = 0;
                            matches += 1;
                        } else {
                            // Document not found in current bucket
                            state.consecutive_misses += 1;
                            misses += 1;
                        }
                    }
                }
                
                debug_println!(config, "   📈 Matches: {}, 📉 Misses: {}", matches, misses);
                
                if matches > 0 {
                    end_idx = i;
                    debug_println!(config, "   ➡️  Extended cluster end to {}", end_idx);
                }
            }
            CollisionResult::Miss => {
                debug_println!(config, "   ❌ No bucket collision");
                // No bucket found, increment miss count for all active documents
                let miss_count = active_documents.len();
                for doc_id in &active_documents {
                    if let Some(state) = all_document_stats.get_mut(doc_id) {
                        state.consecutive_misses += 1;
                    }
                }
                debug_println!(config, "   📉 All {} docs missed", miss_count);
            }
            CollisionResult::VocabFiltered => {
                debug_println!(config, "   🚫 Vocabulary filtered - counts as miss");
                // Vocabulary filtering counts as a miss for all active documents
                let miss_count = active_documents.len();
                for doc_id in &active_documents {
                    if let Some(state) = all_document_stats.get_mut(doc_id) {
                        state.consecutive_misses += 1;
                    }
                }
                debug_println!(config, "   📉 All {} docs missed (vocab filtered)", miss_count);
            }
            CollisionResult::HotBucketSkipped => {
                debug_println!(config, "   ⏭️  Skipped hot bucket - no miss penalty");
                // Hot bucket skip doesn't count as a miss (performance optimization)
            }
        }
        
        // Remove documents from active set that exceeded miss threshold (but keep their stats)
        let before_count = active_documents.len();
        active_documents.retain(|doc_id| {
            all_document_stats.get(doc_id)
                .map(|state| state.consecutive_misses < config.max_consecutive_misses)
                .unwrap_or(false)
        });
        let dropped_count = before_count - active_documents.len();
        if dropped_count > 0 {
            debug_println!(config, "   🗑️  DROPPED {} docs from active set (miss >= {}), {} remaining", dropped_count, config.max_consecutive_misses, active_documents.len());
        }
    }

    // Collect debug information for the final cluster range
    for idx in start_idx..=end_idx {
        match check_ngram_for_collision(
            idx, word_tokens, &ngram_embeddings[idx], config, hyperplanes,
            toxic_buckets, hot_buckets, eval_vocabulary, timing_stats,
            bucket_hits_count, bucket_misses_count, hot_buckets_skipped_count, vocab_filtered_count
        )? {
            CollisionResult::Hit(current_documents) => {
                // Get n-gram text for any matching documents that were part of the original cluster
                if current_documents.iter().any(|doc_id| initial_document_ids.contains(doc_id)) {
                    let ngram_text = if word_tokens.len() < config.ngram_size {
                        word_tokens.join(" ")
                    } else {
                        word_tokens[idx..idx + config.ngram_size].join(" ")
                    };
                    matching_ngrams.push(ngram_text);

                    // Get bucket info
                    let bucket_id = compute_lsh_bucket(&ngram_embeddings[idx], hyperplanes);
                    if let Some(bucket_contents) = toxic_buckets.get(&bucket_id) {
                        bucket_sizes.push(bucket_contents.value().len());
                        distinct_buckets.insert(bucket_id);
                    }
                }
            }
            CollisionResult::Miss | CollisionResult::VocabFiltered | CollisionResult::HotBucketSkipped => {}
        }
    }

    // Convert cumulative document stats to simple match counts for all documents that participated
    let document_matches: HashMap<u32, usize> = all_document_stats
        .into_iter()
        .map(|(doc_id, state)| (doc_id, state.match_length))
        .collect();
    
    debug_println!(config, "\n🏁 FINAL CLUSTER RESULTS:");
    debug_println!(config, "📍 Cluster span: {}-{} ({} n-grams)", start_idx, end_idx, end_idx - start_idx + 1);
    debug_println!(config, "📊 Document matches: {:?}", document_matches);
    
    Ok(ContaminationCluster {
        start_idx,
        end_idx,
        document_matches,
        matching_ngrams,
        bucket_sizes,
        distinct_buckets,
    })
}

fn process_toxic_training_file(
    file_path: &PathBuf,
    file_name: &str,
    config: &Config,
    embeddings: &EmbeddingMap,
    hyperplanes: &Hyperplanes,
    toxic_buckets: &ToxicBuckets,
    hot_buckets: &HotBuckets,
    eval_documents: &EvalDocuments,
    eval_vocabulary: &HashSet<String>,
    contamination_results: &DashMap<String, Vec<ToxicContaminationEntry>>
) -> Result<(), Error> {
    let data = read_pathbuf_to_mem(file_path)?;

    // debug_println!(config, "\n=== DETAILED TIMING LOG FOR {} ===", file_name);
    // debug_println!(config, "Format: Line# | Total | TextExt | Embed | Norm | LSH | Lookup | Collision | Threshold");
    // debug_println!(config, "---------|-------|--------|-------|------|-----|--------|-----------|----------");

    let mut contaminated_lines = 0;
    let mut total_lines = 0;
    let mut cumulative_stats = TimingStats::default();

    // Track bucket statistics across all lines
    let mut total_bucket_hits = 0;
    let mut total_bucket_misses = 0;
    let mut total_hot_buckets_skipped = 0;
    let mut total_vocab_filtered = 0;

    // Track training vocabulary
    let mut training_vocabulary: HashSet<String> = HashSet::new();

    for (line_num, line) in data.lines().enumerate() {
        let line_start = Instant::now();
        let line = line?;

        // 1. Text extraction and parsing
        let text_start = Instant::now();
        let json_obj: Value = serde_json::from_str(&line)?;
        let line_text = get_nested_json_val(&json_obj, &config.content_key.to_string())?;
        
        debug_println!(config, "\n🔄 PROCESSING TRAINING LINE {} in {}", line_num, file_name);
        debug_println!(config, "📝 Text: '{}'", line_text);
        let word_tokens = extract_words(&line_text, &config.punctuation_chars);
        debug_println!(config, "🧹 Cleaned: '{}'", word_tokens.join(" "));
        let _training_word_count = word_tokens.len();

        // Track vocabulary from training data
        for word in &word_tokens {
            training_vocabulary.insert(word.clone());
        }

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

        let normalization_time = Duration::new(0, 0);
        let mut lsh_time = Duration::new(0, 0);
        let mut lookup_time = Duration::new(0, 0);
        let mut collision_time = Duration::new(0, 0);

        // 3. Process n-grams with sampling optimization
        let sampling_start = Instant::now();
        let mut sampling_timing = TimingStats::default();

        let (contamination_clusters, sampling_stats) = process_ngrams_with_sampling(
            &word_tokens,
            &ngram_embeddings,
            config,
            hyperplanes,
            toxic_buckets,
            hot_buckets,
            eval_vocabulary,
            &mut sampling_timing,
        )?;

        // Accumulate timing from sampling
        lsh_time += sampling_timing.lsh_bucket_computation;
        lookup_time += sampling_timing.bucket_lookup;
        collision_time += sampling_start.elapsed() - sampling_timing.lsh_bucket_computation - sampling_timing.bucket_lookup;

        // 4. Process clusters and calculate local overlap ratios
        let threshold_start = Instant::now();

        debug_println!(config, "\n📊 OVERLAP ANALYSIS for line {} (found {} clusters)", line_num, contamination_clusters.len());
        debug_println!(config, "🔤 Training text: '{}'", line_text);

        for (cluster_idx, cluster) in contamination_clusters.iter().enumerate() {
            debug_println!(config, "\n🔍 Cluster {} analysis:", cluster_idx);
            debug_println!(config, "  Span: {}-{} ({} n-grams)", cluster.start_idx, cluster.end_idx, cluster.end_idx - cluster.start_idx + 1);
            debug_println!(config, "  Documents: {}", cluster.document_matches.len());
            
            // Process each document match in the cluster
            for (doc_id, match_length) in &cluster.document_matches {
                // Get document metadata and live n-gram count
                let (eval_name, eval_line, _total_ngrams, live_ngrams) = 
                    eval_documents.get(doc_id)
                        .map(|doc| doc.value().clone())
                        .unwrap_or_else(|| ("unknown".to_string(), 0, 0, 0));

                debug_println!(config, "\n  📄 Document {}: {}:{}", doc_id, eval_name, eval_line);
                debug_println!(config, "    Match length: {}", match_length);
                debug_println!(config, "    Live n-grams: {}", live_ngrams);

                if live_ngrams == 0 {
                    debug_println!(config, "    ⚠️  SKIPPED: No live n-grams");
                    continue; // Skip documents with no live n-grams
                }

                // Local overlap ratio: match_length / live_ngrams
                let local_overlap_ratio = *match_length as f32 / live_ngrams as f32;
                debug_println!(config, "    Overlap ratio: {}/{} = {:.3}", match_length, live_ngrams, local_overlap_ratio);
                debug_println!(config, "    Threshold: {:.3}", config.toxic_overlap_threshold);

                if local_overlap_ratio >= config.toxic_overlap_threshold && complete_evaluation() {
                    debug_println!(config, "    🚨 CONTAMINATION DETECTED! (ratio {:.3} >= threshold {:.3})", local_overlap_ratio, config.toxic_overlap_threshold);
                    
                    let (matching_ngrams, bucket_sizes) = if config.debug {
                        (Some(cluster.matching_ngrams.clone()), Some(cluster.bucket_sizes.clone()))
                    } else {
                        (None, None)
                    };

                    contamination_results
                        .entry(file_name.to_string())
                        .or_default()
                        .push(ToxicContaminationEntry {
                            training_line: line_num,
                            eval_name: eval_name.clone(),
                            eval_line,
                            overlap_ratio: local_overlap_ratio,
                            matching_ngrams,
                            bucket_sizes,
                        });

                    contaminated_lines += 1;
                } else {
                    debug_println!(config, "    ❌ No contamination (ratio {:.3} < threshold {:.3})", local_overlap_ratio, config.toxic_overlap_threshold);
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

        // Accumulate bucket statistics
        total_bucket_hits += sampling_stats.bucket_hits;
        total_bucket_misses += sampling_stats.bucket_misses;
        total_hot_buckets_skipped += sampling_stats.hot_buckets_skipped;
        total_vocab_filtered += sampling_stats.vocab_filtered;
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

    // Print bucket statistics in debug mode
    debug_println!(config, "BUCKET STATISTICS:");
    debug_println!(config, "Vocabulary filtered (out-of-vocab): {}", total_vocab_filtered);
    debug_println!(config, "Bucket hits (found eval matches): {}", total_bucket_hits);
    debug_println!(config, "Bucket misses (no eval matches): {}", total_bucket_misses);
    debug_println!(config, "Hot buckets skipped: {}", total_hot_buckets_skipped);
    let total_ngrams_generated = total_vocab_filtered + total_bucket_hits + total_bucket_misses + total_hot_buckets_skipped;
    let total_ngrams_processed = total_bucket_hits + total_bucket_misses + total_hot_buckets_skipped;
    debug_println!(config, "Total n-grams generated: {}", total_ngrams_generated);
    debug_println!(config, "Total n-grams processed (LSH computed): {}", total_ngrams_processed);
    debug_println!(config, "");

    // Print vocabulary statistics in debug mode
    debug_println!(config, "VOCABULARY STATISTICS:");
    debug_println!(config, "Training vocabulary size: {}", training_vocabulary.len());
    debug_println!(config, "Eval vocabulary size: {}", eval_vocabulary.len());
    let vocab_union: HashSet<_> = training_vocabulary.union(eval_vocabulary).collect();
    let vocab_intersection: HashSet<_> = training_vocabulary.intersection(eval_vocabulary).collect();
    debug_println!(config, "Union vocabulary size: {}", vocab_union.len());
    debug_println!(config, "Intersection vocabulary size: {}", vocab_intersection.len());
    let vocab_symmetric_diff: HashSet<_> = training_vocabulary.symmetric_difference(eval_vocabulary).collect();
    debug_println!(config, "Symmetric difference (XOR) size: {}", vocab_symmetric_diff.len());
    if !eval_vocabulary.is_empty() && !training_vocabulary.is_empty() {
        let overlap_ratio = vocab_intersection.len() as f64 / vocab_union.len() as f64;
        debug_println!(config, "Vocabulary overlap ratio: {:.3}", overlap_ratio);
    }
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
    toxic_buckets: &ToxicBuckets,
    bucket_id: u64,
    doc_id: u32
) {
    // Perform the actual insertion
    toxic_buckets
        .entry(bucket_id)
        .or_default()
        .push(doc_id);
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

    // Find buckets with most distinct n-grams (only in debug mode with bucket contents)
    if let Some(ref contents) = bucket_contents {
        let mut diverse_buckets: Vec<(u64, usize, usize)> = Vec::new();
        for bucket in toxic_buckets.iter() {
            let bucket_id = *bucket.key();
            let entries = bucket.value().len();
            let distinct_count = contents.get(&bucket_id)
                .map(|ngrams| ngrams.len())
                .unwrap_or(0);
            diverse_buckets.push((bucket_id, entries, distinct_count));
        }
        diverse_buckets.sort_by(|a, b| b.2.cmp(&a.2)); // Sort by distinct count descending
        diverse_buckets.truncate(10); // Keep only top 10

        println!("\nTOP 10 MOST DIVERSE BUCKETS (by distinct n-grams):");
        println!("Bucket ID          | Entries | Distinct N-grams");
        println!("-------------------|---------|----------------");
        for (bucket_id, entries, distinct_count) in &diverse_buckets {
            println!("{:18} | {:7} | {:7}", bucket_id, entries, distinct_count);
        }

        // Show actual ngrams for top 3 most diverse buckets
        println!("\nSAMPLE N-GRAMS FROM TOP 3 MOST DIVERSE BUCKETS:");
        for (i, (bucket_id, entries, distinct_count)) in diverse_buckets.iter().take(3).enumerate() {
            if let Some(ngrams_set) = contents.get(bucket_id) {
                let mut ngrams: Vec<String> = ngrams_set.iter().cloned().collect();
                ngrams.sort();
                let sample_size = std::cmp::min(10, ngrams.len());
                println!("\n{}. Bucket {} ({} entries, {} distinct):",
                        i + 1, bucket_id, entries, distinct_count);
                for ngram in ngrams.iter().take(sample_size) {
                    println!("   \"{}\"", ngram);
                }
                if ngrams.len() > sample_size {
                    println!("   ... and {} more", ngrams.len() - sample_size);
                }
            }
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

fn update_live_ngram_counts(
    _config: &Config,
    _embeddings: &EmbeddingMap,
    _hyperplanes: &Hyperplanes,
    hot_buckets: &HotBuckets,
    eval_documents: &EvalDocuments,
    toxic_buckets: &ToxicBuckets
) -> Result<(), Error> {
    println!("Updating live n-gram counts by excluding hot buckets...");

    // Count how many n-grams from each document fall into hot buckets
    let mut hot_ngram_counts: HashMap<u32, usize> = HashMap::new();

    // Iterate through hot buckets and count entries per document
    for hot_bucket_id in hot_buckets {
        if let Some(bucket_entries) = toxic_buckets.get(hot_bucket_id) {
            for doc_id in bucket_entries.value() {
                *hot_ngram_counts.entry(*doc_id).or_insert(0) += 1;
            }
        }
    }

    // Update live counts: live_ngrams = total_ngrams - hot_ngrams
    for mut doc_entry in eval_documents.iter_mut() {
        let doc_id = *doc_entry.key();
        let (eval_name, eval_line, total_ngrams, _old_live) = doc_entry.value().clone();

        let hot_ngrams = hot_ngram_counts.get(&doc_id).copied().unwrap_or(0);
        let live_ngrams = total_ngrams.saturating_sub(hot_ngrams);

        *doc_entry.value_mut() = (eval_name, eval_line, total_ngrams, live_ngrams);
    }

    let total_docs_with_hot_ngrams = hot_ngram_counts.len();
    println!("Updated live n-gram counts for {} docs affected by hot buckets)", total_docs_with_hot_ngrams);
    Ok(())
}
