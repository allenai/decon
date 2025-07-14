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
use serde_json::{json, Value};
use std::collections::hash_map::DefaultHasher;
use std::collections::{HashMap, HashSet, VecDeque};
use std::fs::{create_dir_all, File};
use std::hash::{Hash, Hasher};
use std::io::{BufRead, BufReader, BufWriter, Read, Write};
use std::path::PathBuf;
// Removed atomic imports - document IDs now read from JSON
use std::time::{Duration, Instant};

use mj_io::{build_pbar, expand_dirs, read_pathbuf_to_mem, write_mem_to_pathbuf};

use crate::{
    clean_text_allowlist, debug_println, get_nested_json_val, get_results_filename,
    write_purified_file, Config,
};

// 300-dimensional word embeddings
pub const EMBEDDING_DIM: usize = 128;

// Document IDs are now generated in Python download script and read from JSON

// LSH bucket storage: maps bucket_id to list of document IDs
type ToxicBuckets = DashMap<u64, HashSet<u32>>;

// Bucket content storage for debug mode: maps bucket_id to set of ngram texts
type BucketContents = DashMap<u64, HashSet<String>>;

// Hot bucket tracking: set of bucket IDs that exceed skip_hot_bucket_threshold
type HotBuckets = HashSet<u64>;

// Eval document metadata: maps document_id to (eval_name, line_num, total_ngrams, live_ngrams, unique_buckets)
type EvalDocuments = DashMap<u32, (String, usize, usize, usize, usize)>;
type EvalTextSnippets = DashMap<(String, usize), String>;

// Embedding storage: word -> 300d vector (thread-safe)
pub type EmbeddingMap = DashMap<String, Vec<f32>>;

// LRU Cache for n-gram -> bucket_id mappings to avoid redundant LSH computations
struct BucketIdLruCache {
    cache: HashMap<u64, u64>, // hash of embedding -> bucket_id
    access_order: VecDeque<u64>,
    capacity: usize,
    hits: usize,
    misses: usize,
}

impl BucketIdLruCache {
    fn new(capacity: usize) -> Self {
        Self {
            cache: HashMap::with_capacity(capacity),
            access_order: VecDeque::with_capacity(capacity),
            capacity,
            hits: 0,
            misses: 0,
        }
    }

    fn get_or_compute<F>(&mut self, ngram_embedding: &[f32], compute_fn: F) -> u64
    where
        F: FnOnce(&[f32]) -> u64,
    {
        // Hash the embedding vector
        let key = self.hash_embedding(ngram_embedding);

        if let Some(&bucket_id) = self.cache.get(&key) {
            // Cache hit - move to end of access order
            self.hits += 1;
            if let Some(pos) = self.access_order.iter().position(|k| k == &key) {
                let key = self.access_order.remove(pos).unwrap();
                self.access_order.push_back(key);
            }
            bucket_id
        } else {
            // Cache miss - compute and cache
            self.misses += 1;
            let bucket_id = compute_fn(ngram_embedding);

            // If at capacity, remove least recently used
            if self.cache.len() >= self.capacity {
                if let Some(lru_key) = self.access_order.pop_front() {
                    self.cache.remove(&lru_key);
                }
            }

            // Add new entry
            self.cache.insert(key, bucket_id);
            self.access_order.push_back(key);

            bucket_id
        }
    }

    fn hash_embedding(&self, embedding: &[f32]) -> u64 {
        let mut hasher = DefaultHasher::new();
        for &val in embedding {
            val.to_bits().hash(&mut hasher);
        }
        hasher.finish()
    }

    fn hit_rate(&self) -> f64 {
        if self.hits + self.misses == 0 {
            0.0
        } else {
            self.hits as f64 / (self.hits + self.misses) as f64
        }
    }

    fn stats(&self) -> (usize, usize, f64) {
        (self.hits, self.misses, self.hit_rate())
    }
}

// LSH hyperplane storage: vectorized format for efficient matrix operations
#[derive(Clone)]
pub struct VectorizedHyperplanes {
    // Matrix shape: (num_hyperplanes, EMBEDDING_DIM)
    // Each row is a hyperplane, enabling efficient matrix-vector multiplication
    data: Array2<f32>,
    num_planes: usize,
}

impl VectorizedHyperplanes {
    fn new(hyperplanes: Vec<Vec<f32>>) -> Result<Self, Error> {
        let num_planes = hyperplanes.len();
        if num_planes == 0 {
            return Err(anyhow::anyhow!(
                "Cannot create VectorizedHyperplanes with zero hyperplanes"
            ));
        }

        let flat_data: Vec<f32> = hyperplanes.into_iter().flatten().collect();
        let expected_size = num_planes * EMBEDDING_DIM;
        if flat_data.len() != expected_size {
            return Err(anyhow::anyhow!(
                "Hyperplane data size mismatch: expected {}, got {}",
                expected_size,
                flat_data.len()
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
    let embeddings = load_embeddings(&config.toxic_embedding_path, config)?;
    println!("Loaded {} word embeddings", embeddings.len());

    // Step 2: Generate random hyperplanes for LSH
    println!(
        "Generating {} random hyperplanes...",
        config.toxic_hyperplanes
    );
    let hyperplanes = generate_hyperplanes(config.toxic_hyperplanes, config.hash_seed)?;

    // Step 3: Process reference datasets and build LSH buckets
    println!("Processing reference datasets...");
    let (
        toxic_buckets,
        hot_buckets,
        eval_documents,
        eval_vocabulary,
        _bucket_contents,
        eval_text_snippets,
    ) = build_toxic_index(config, &embeddings, &hyperplanes)?;
    println!("Built TOXIC index with {} buckets", toxic_buckets.len());

    // Step 4: Process training data and detect contamination
    println!("Processing training data for contamination detection...");
    detect_toxic_contamination(
        config,
        &embeddings,
        &hyperplanes,
        &toxic_buckets,
        &hot_buckets,
        &eval_documents,
        &eval_vocabulary,
        &eval_text_snippets,
    )?;

    println!(
        "TOXIC contamination detection completed in {:?} seconds",
        start_main.elapsed().as_secs()
    );
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
        return Err(anyhow::anyhow!(
            "Embedding dimension mismatch: expected {}, got {}",
            EMBEDDING_DIM,
            embedding_dim
        ));
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

    // Read embeddings (poison tokens already applied when binary was created)
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

fn load_embeddings_text(
    embedding_path: &PathBuf,
    _poison_scale: f32,
) -> Result<EmbeddingMap, Error> {
    println!("Loading text embeddings from: {:?}", embedding_path);
    let data = read_pathbuf_to_mem(embedding_path)?;
    let embeddings = DashMap::new();
    let mut poison_replacements = 0;

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
                line_num,
                EMBEDDING_DIM + 1,
                parts.len()
            ));
        }

        let word = parts[0].to_string();
        let vector: Result<Vec<f32>, _> = parts[1..].iter().map(|s| s.parse::<f32>()).collect();

        match vector {
            Ok(mut vec) => {
                // Apply poison token replacement for numeric tokens
                if is_numeric_token(&word) {
                    // Generate a deterministic poison vector for this numeric token
                    let mut rng = ChaCha20Rng::seed_from_u64(hash_string(&word));

                    // Replace with random vector
                    for val in &mut vec {
                        *val = rng.gen_range(-1.0..1.0);
                    }

                    poison_replacements += 1;
                }

                // Normalize ALL vectors to unit length (both poison and regular embeddings)
                let mut magnitude_sq = 0.0f32;
                for &val in &vec {
                    magnitude_sq += val * val;
                }
                let magnitude = magnitude_sq.sqrt();

                if magnitude > 0.0 {
                    for val in &mut vec {
                        *val /= magnitude;
                    }
                }

                embeddings.insert(word, vec);
            }
            Err(e) => {
                return Err(anyhow::anyhow!(
                    "Failed to parse embedding vector at line {}: {}",
                    line_num,
                    e
                ));
            }
        }

        if line_num % 100000 == 0 && line_num > 0 {
            print!(".");
            std::io::stdout().flush().unwrap();
        }
    }

    println!(); // New line after dots
    if poison_replacements > 0 {
        println!(
            "Applied poison vectors to {} numeric tokens",
            poison_replacements
        );
    }
    println!(
        "Normalized all {} embedding vectors to unit length",
        embeddings.len()
    );
    Ok(embeddings)
}

pub fn load_embeddings(embedding_path: &PathBuf, config: &Config) -> Result<EmbeddingMap, Error> {
    // Try binary format first (.bin extension)
    let binary_path = embedding_path.with_extension("bin");
    if binary_path.exists() {
        println!("Loading binary embeddings from: {:?}", binary_path);
        return load_embeddings_binary(&binary_path);
    }

    // Fall back to text format and optionally create binary version (with poison tokens applied)
    let embeddings = load_embeddings_text(embedding_path, config.toxic_poison_scale)?;

    // Save binary version for next time (poison tokens already applied during text loading)
    if let Err(e) = save_embeddings_binary(&embeddings, &binary_path) {
        println!("Warning: Failed to save binary embeddings: {}", e);
    }

    Ok(embeddings)
}

// For eval processing: store OOV words consistently
fn get_or_create_embedding_eval(
    word: &str,
    embeddings: &EmbeddingMap,
    rng_seed: u64,
    _poison_scale: f32,
    timing: &mut TimingStats,
) -> Vec<f32> {
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
        // Time random generation for OOV tokens
        let rand_start = Instant::now();
        let mut rng = ChaCha20Rng::seed_from_u64(rng_seed.wrapping_add(hash_string(word)));

        let mut vector: Vec<f32> = if is_numeric_token(word) {
            // Generate random poison vector
            (0..EMBEDDING_DIM)
                .map(|_| rng.gen_range(-1.0..1.0))
                .collect()
        } else {
            // Use normal scale for other OOV tokens
            (0..EMBEDDING_DIM)
                .map(|_| rng.gen_range(-1.0..1.0))
                .collect()
        };

        // Normalize ALL vectors to unit length (both poison and regular OOV)
        let norm_start = Instant::now();
        let mut magnitude_sq = 0.0f32;
        for &val in &vector {
            magnitude_sq += val * val;
        }
        let magnitude = magnitude_sq.sqrt();

        if magnitude > 0.0 {
            for val in &mut vector {
                *val /= magnitude;
            }
        }
        timing.vector_normalization += norm_start.elapsed();
        timing.random_generation += rand_start.elapsed();

        // Time the insertion and cloning for return
        let clone_start = Instant::now();
        embeddings.insert(word.to_string(), vector.clone());
        timing.vector_cloning += clone_start.elapsed();

        vector
    }
}

// For training processing: generate chaotic OOV each time (don't store)
fn get_or_create_embedding_training(
    word: &str,
    embeddings: &EmbeddingMap,
    rng_seed: u64,
    _poison_scale: f32,
    timing: &mut TimingStats,
) -> Vec<f32> {
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
        // Time random generation for OOV tokens
        let rand_start = Instant::now();
        let mut rng = ChaCha20Rng::seed_from_u64(rng_seed.wrapping_add(hash_string(word)));

        let mut result: Vec<f32> = if is_numeric_token(word) {
            // Generate random poison vector
            (0..EMBEDDING_DIM)
                .map(|_| rng.gen_range(-1.0..1.0))
                .collect()
        } else {
            // Use zero vector for non-numeric OOV tokens
            vec![0.0; EMBEDDING_DIM]
        };

        // Normalize vectors to unit length (only for non-zero vectors)
        let norm_start = Instant::now();
        let mut magnitude_sq = 0.0f32;
        for &val in &result {
            magnitude_sq += val * val;
        }
        let magnitude = magnitude_sq.sqrt();

        if magnitude > 0.0 {
            for val in &mut result {
                *val /= magnitude;
            }
        }
        timing.vector_normalization += norm_start.elapsed();
        timing.random_generation += rand_start.elapsed();
        result
    }
}

fn hash_string(s: &str) -> u64 {
    use std::hash::{DefaultHasher, Hash, Hasher};
    let mut hasher = DefaultHasher::new();
    s.hash(&mut hasher);
    hasher.finish()
}

fn is_numeric_token(token: &str) -> bool {
    // Check for number words (case-insensitive)
    let lowercase_token = token.to_lowercase();
    match lowercase_token.as_str() {
        // Basic numbers 0-20
        "zero" | "one" | "two" | "three" | "four" | "five" | "six" | "seven" | "eight" | "nine" | "ten" |
        "eleven" | "twelve" | "thirteen" | "fourteen" | "fifteen" | "sixteen" | "seventeen" | "eighteen" | "nineteen" | "twenty" |
        // Tens
        "thirty" | "forty" | "fifty" | "sixty" | "seventy" | "eighty" | "ninety" |
        // Hundreds, thousands, etc.
        "hundred" | "thousand" | "million" | "billion" | "trillion" |
        // Ordinal words
        "first" | "second" | "third" | "fourth" | "fifth" | "sixth" | "seventh" | "eighth" | "ninth" | "tenth" |
        "eleventh" | "twelfth" | "thirteenth" | "fourteenth" | "fifteenth" | "sixteenth" | "seventeenth" | "eighteenth" | "nineteenth" | "twentieth" |
        "thirtieth" | "fortieth" | "fiftieth" | "sixtieth" | "seventieth" | "eightieth" | "ninetieth" |
        "hundredth" | "thousandth" | "millionth" | "billionth" | "trillionth" |
        // Fractions
        "half" | "quarter" | "thirds" | "halves" | "quarters" |
        // Time-related numbers
        "dozen" | "dozens" | "score" | "scores" |
        // Roman numerals (common ones)
        "i" | "ii" | "iii" | "iv" | "v" | "vi" | "vii" | "viii" | "ix" | "x" |
        "xi" | "xii" | "xiii" | "xiv" | "xv" | "xx" | "xxx" | "xl" | "l" | "c" | "d" | "m" => {
            return true;
        }
        _ => {} // Continue to numeric checks
    }

    // Check for compound number words (twenty-one, thirty-two, etc.)
    if lowercase_token.contains('-') {
        let parts: Vec<&str> = lowercase_token.split('-').collect();
        if parts.len() == 2 {
            let first_is_numeric = matches!(
                parts[0],
                "twenty" | "thirty" | "forty" | "fifty" | "sixty" | "seventy" | "eighty" | "ninety"
            );
            let second_is_numeric = matches!(
                parts[1],
                "one" | "two" | "three" | "four" | "five" | "six" | "seven" | "eight" | "nine"
            );
            if first_is_numeric && second_is_numeric {
                return true;
            }
        }
    }

    // Check for pure integers (123, -456)
    if token.parse::<i64>().is_ok() {
        return true;
    }

    // Check for pure floats (123.45, -67.89, .5, 5.)
    if token.parse::<f64>().is_ok() {
        return true;
    }

    // Check for percentages (50%, -25.5%)
    if token.ends_with('%') {
        let chars: Vec<char> = token.chars().collect();
        if chars.len() > 1 {
            let num_part: String = chars[..chars.len() - 1].iter().collect();
            if num_part.parse::<f64>().is_ok() {
                return true;
            }
        }
    }

    // Check for common currency symbols ($123, €45.67, £100)
    if (token.starts_with('$')
        || token.starts_with('€')
        || token.starts_with('£')
        || token.starts_with('¥'))
        && token.chars().count() > 1
    {
        let chars: Vec<char> = token.chars().collect();
        let num_part: String = chars[1..].iter().collect();
        if num_part.parse::<f64>().is_ok() {
            return true;
        }
    }

    // Check for ordinals (1st, 2nd, 3rd, 4th, etc.)
    if token.chars().count() > 2 {
        let chars: Vec<char> = token.chars().collect();
        if chars.len() >= 2 {
            let suffix: String = chars[chars.len() - 2..].iter().collect();
            let num_part: String = chars[..chars.len() - 2].iter().collect();
            if matches!(suffix.as_str(), "st" | "nd" | "rd" | "th")
                && num_part.parse::<u64>().is_ok()
            {
                return true;
            }
        }
    }

    // Check for scientific notation (1e5, 2.5E-3, 1.23e+10)
    if token.contains('e') || token.contains('E') {
        if token.parse::<f64>().is_ok() {
            return true;
        }
    }

    // Check for fractions (1/2, 3/4, 22/7)
    if token.contains('/') {
        let parts: Vec<&str> = token.split('/').collect();
        if parts.len() == 2 && parts[0].parse::<f64>().is_ok() && parts[1].parse::<f64>().is_ok() {
            return true;
        }
    }

    false
}

// Helper function for vector subtraction (in-place)
fn subtract_word_embedding_inplace(
    sum_vector: &mut [f32],
    word_embedding: &[f32],
    timing: &mut TimingStats,
) {
    let arith_start = Instant::now();
    for (i, val) in word_embedding.iter().enumerate() {
        sum_vector[i] -= val;
    }
    timing.vector_arithmetic += arith_start.elapsed();
}

// Helper function for vector addition (in-place)
fn add_word_embedding_inplace(
    sum_vector: &mut [f32],
    word_embedding: &[f32],
    timing: &mut TimingStats,
) {
    let arith_start = Instant::now();
    for (i, val) in word_embedding.iter().enumerate() {
        sum_vector[i] += val;
    }
    timing.vector_arithmetic += arith_start.elapsed();
}

pub fn generate_hyperplanes(k: usize, seed: usize) -> Result<Hyperplanes, Error> {
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

/// Extract overlapping text with context from word tokens
fn extract_overlap_with_context_toxic(
    word_tokens: &[String],
    start_idx: usize,
    end_idx: usize,
    context_words: usize,
) -> Option<String> {
    // Check if indices are valid
    if start_idx >= word_tokens.len() || end_idx > word_tokens.len() || start_idx >= end_idx {
        return None;
    }

    // Calculate context boundaries
    let context_start = start_idx.saturating_sub(context_words);
    let context_end = (end_idx + context_words).min(word_tokens.len());

    // Build the output with highlighted contamination
    let mut result = String::new();

    // Add leading context
    if context_start < start_idx {
        result.push_str("... ");
        result.push_str(&word_tokens[context_start..start_idx].join(" "));
        result.push(' ');
    }

    // Add contaminated section with highlighting
    result.push_str("【");
    result.push_str(&word_tokens[start_idx..end_idx].join(" "));
    result.push_str("】");

    // Add trailing context
    if end_idx < context_end {
        result.push(' ');
        result.push_str(&word_tokens[end_idx..context_end].join(" "));
        result.push_str(" ...");
    }

    Some(result)
}

// Optimized training n-gram computation using sliding window to reduce vector arithmetic
pub fn compute_ngram_embedding_training(
    tokens: &[String],
    ngram_size: usize,
    embeddings: &EmbeddingMap,
    rng_seed: u64,
    poison_scale: f32,
    timing: &mut TimingStats,
) -> Vec<Vec<f32>> {
    let mut ngram_embeddings = Vec::new();

    if tokens.len() < ngram_size {
        if !tokens.is_empty() {
            // For documents shorter than ngram_size, use all tokens - same as original
            let sum_embedding =
                sum_word_embeddings_training(tokens, embeddings, rng_seed, poison_scale, timing);
            ngram_embeddings.push(sum_embedding);
        }
        return ngram_embeddings;
    }

    // Compute first n-gram sum normally
    let first_ngram = &tokens[0..ngram_size];
    let mut current_sum =
        sum_word_embeddings_training(first_ngram, embeddings, rng_seed, poison_scale, timing);
    ngram_embeddings.push(current_sum.clone());

    // Slide the window for remaining n-grams
    for i in 1..=tokens.len() - ngram_size {
        // Word sliding out (left side)
        let outgoing_word = &tokens[i - 1];
        let outgoing_embedding = get_or_create_embedding_training(
            outgoing_word,
            embeddings,
            rng_seed,
            poison_scale,
            timing,
        );
        subtract_word_embedding_inplace(&mut current_sum, &outgoing_embedding, timing);

        // Word sliding in (right side)
        let incoming_word = &tokens[i + ngram_size - 1];
        let incoming_embedding = get_or_create_embedding_training(
            incoming_word,
            embeddings,
            rng_seed,
            poison_scale,
            timing,
        );
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
    poison_scale: f32,
    timing: &mut TimingStats,
) -> Vec<f32> {
    // Time memory allocation
    let alloc_start = Instant::now();
    let mut sum_vector = vec![0.0; EMBEDDING_DIM];
    timing.memory_allocation += alloc_start.elapsed();

    for word in words {
        // Get embedding (timing is handled internally)
        let word_embedding =
            get_or_create_embedding_eval(word, embeddings, rng_seed, poison_scale, timing);

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
    poison_scale: f32,
    timing: &mut TimingStats,
) -> Vec<f32> {
    // Time memory allocation
    let alloc_start = Instant::now();
    let mut sum_vector = vec![0.0; EMBEDDING_DIM];
    timing.memory_allocation += alloc_start.elapsed();

    for word in words {
        // Get embedding (timing is handled internally)
        let word_embedding =
            get_or_create_embedding_training(word, embeddings, rng_seed, poison_scale, timing);

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

// Public type for the toxic index
pub type ToxicIndex = (
    ToxicBuckets,
    HotBuckets,
    EvalDocuments,
    HashSet<String>,
    Option<BucketContents>,
    EmbeddingMap,
    Hyperplanes,
    EvalTextSnippets,
);

pub fn build_toxic_index(
    config: &Config,
    embeddings: &EmbeddingMap,
    hyperplanes: &Hyperplanes,
) -> Result<
    (
        ToxicBuckets,
        HotBuckets,
        EvalDocuments,
        HashSet<String>,
        Option<BucketContents>,
        EvalTextSnippets,
    ),
    Error,
> {
    let toxic_buckets: ToxicBuckets = DashMap::new();
    let eval_documents: EvalDocuments = DashMap::new();
    let eval_vocabulary: DashMap<String, ()> = DashMap::new();
    let eval_text_snippets: EvalTextSnippets = DashMap::new(); // Thread-safe set for vocabulary
    let bucket_contents: Option<BucketContents> = if config.debug {
        Some(DashMap::new())
    } else {
        None
    };

    // Find all reference files
    let reference_files = expand_dirs(
        vec![config.reference_input.clone()],
        Some(vec![".jsonl", ".gz"].as_slice()),
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
            &bucket_contents,
            &eval_text_snippets,
        ) {
            println!("Error processing reference file {:?}: {:?}", file_path, e);
        }
        pbar.inc(1);
    });

    // Identify hot buckets after processing all eval data
    let hot_buckets: HotBuckets = if config.skip_hot_bucket_threshold > 0 {
        toxic_buckets
            .iter()
            .filter(|entry| entry.value().len() > config.skip_hot_bucket_threshold as usize)
            .map(|entry| *entry.key())
            .collect()
    } else {
        HashSet::new()
    };

    if !hot_buckets.is_empty() {
        println!(
            "Identified {} hot buckets (threshold: {})",
            hot_buckets.len(),
            config.skip_hot_bucket_threshold
        );

        // Update live n-gram counts by excluding hot buckets
        update_live_ngram_counts(
            config,
            embeddings,
            hyperplanes,
            &hot_buckets,
            &eval_documents,
            &toxic_buckets,
        )?;
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
    let eval_vocab_set: HashSet<String> =
        eval_vocabulary.into_iter().map(|(word, _)| word).collect();

    Ok((
        toxic_buckets,
        hot_buckets,
        eval_documents,
        eval_vocab_set,
        bucket_contents,
        eval_text_snippets,
    ))
}

fn process_toxic_reference_file(
    file_path: &PathBuf,
    config: &Config,
    embeddings: &EmbeddingMap,
    hyperplanes: &Hyperplanes,
    toxic_buckets: &ToxicBuckets,
    eval_documents: &EvalDocuments,
    eval_vocabulary: &DashMap<String, ()>,
    bucket_contents: &Option<BucketContents>,
    eval_text_snippets: &EvalTextSnippets,
) -> Result<(), Error> {
    let data = read_pathbuf_to_mem(file_path)?;
    // We don't need tokenizer for TOXIC - we work with words directly

    // Extract eval name from filename
    let eval_name = file_path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("unknown")
        .to_string();

    // debug_println!(config, "Processing TOXIC embeddings for eval dataset: {}", eval_name);

    let mut _lines_processed = 0;
    let mut _skipped_entries = 0;
    let min_word_count = config.ngram_size * 2;

    for (line_num, line) in data.lines().enumerate() {
        let line = line?;
        let json_obj: Value = serde_json::from_str(&line)?;
        let line_text = get_nested_json_val(&json_obj, &config.content_key.to_string())?;

        // For TOXIC, we work with words directly, not token IDs
        let word_tokens = extract_words(&line_text, &config.punctuation_chars);

        // debug_println!(config, "📖 REFERENCE {}: '{}' -> '{}'",
        //               eval_name, line_text, word_tokens.join(" "));
        let word_count = word_tokens.len();

        // Track vocabulary from eval data
        for word in &word_tokens {
            eval_vocabulary.insert(word.clone(), ());
        }

        // Store eval text snippet (first 1000 words)
        let text_snippet = word_tokens
            .iter()
            .take(1000)
            .cloned()
            .collect::<Vec<_>>()
            .join(" ");
        eval_text_snippets.insert((eval_name.clone(), line_num), text_snippet);

        // Skip entries with insufficient words for meaningful n-gram analysis
        if word_count < min_word_count {
            _skipped_entries += 1;
            continue;
        }

        _lines_processed += 1;

        // Calculate total n-grams for this document
        let total_ngrams = if word_tokens.len() < config.ngram_size {
            if word_tokens.is_empty() {
                0
            } else {
                1
            }
        } else {
            word_tokens.len() - config.ngram_size + 1
        };

        // Calculate unique buckets for this document
        let mut unique_bucket_ids = HashSet::new();
        if word_tokens.len() < config.ngram_size {
            if !word_tokens.is_empty() {
                // For documents shorter than ngram_size, use all tokens
                let mut dummy_timing = TimingStats::default();
                let sum_embedding = sum_word_embeddings_eval(
                    &word_tokens,
                    embeddings,
                    config.hash_seed as u64,
                    config.toxic_poison_scale,
                    &mut dummy_timing,
                );
                let bucket_id = compute_lsh_bucket(&sum_embedding, hyperplanes);
                unique_bucket_ids.insert(bucket_id);
            }
        } else {
            for i in 0..=word_tokens.len() - config.ngram_size {
                let ngram = &word_tokens[i..i + config.ngram_size];
                let mut dummy_timing = TimingStats::default();
                let sum_embedding = sum_word_embeddings_eval(
                    ngram,
                    embeddings,
                    config.hash_seed as u64,
                    config.toxic_poison_scale,
                    &mut dummy_timing,
                );
                let bucket_id = compute_lsh_bucket(&sum_embedding, hyperplanes);
                unique_bucket_ids.insert(bucket_id);
            }
        }
        let unique_buckets = unique_bucket_ids.len();

        // Read document ID from JSON (generated in Python download script)
        let doc_id = json_obj
            .get("doc_id")
            .and_then(|v| v.as_u64())
            .ok_or_else(|| anyhow::anyhow!("Missing or invalid doc_id field in reference file"))?
            as u32;

        // Store document metadata (will be updated with live count later)
        eval_documents.insert(
            doc_id,
            (
                eval_name.clone(),
                line_num,
                total_ngrams,
                total_ngrams,
                unique_buckets,
            ),
        );

        // Process n-grams with hot bucket detection
        if word_tokens.len() < config.ngram_size {
            if !word_tokens.is_empty() {
                // For documents shorter than ngram_size, use all tokens
                let mut dummy_timing = TimingStats::default();
                let sum_embedding = sum_word_embeddings_eval(
                    &word_tokens,
                    embeddings,
                    config.hash_seed as u64,
                    config.toxic_poison_scale,
                    &mut dummy_timing,
                );
                // Skip normalization to preserve magnitude information
                let bucket_id = compute_lsh_bucket(&sum_embedding, hyperplanes);
                insert_with_hot_bucket_detection(&toxic_buckets, bucket_id, doc_id);

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
                let sum_embedding = sum_word_embeddings_eval(
                    ngram,
                    embeddings,
                    config.hash_seed as u64,
                    config.toxic_poison_scale,
                    &mut dummy_timing,
                );
                // Skip normalization to preserve magnitude information
                let bucket_id = compute_lsh_bucket(&sum_embedding, hyperplanes);
                insert_with_hot_bucket_detection(&toxic_buckets, bucket_id, doc_id);

                // Store ngram text for debug analysis
                if let Some(ref contents) = bucket_contents {
                    let ngram_text = ngram.join(" ");
                    contents.entry(bucket_id).or_default().insert(ngram_text);
                }
            }
        }
    }

    // debug_println!(config, "  → Processed {} lines from {} (skipped {} entries with < {} words)",
    //                lines_processed, eval_name, skipped_entries, min_word_count);
    Ok(())
}

fn detect_toxic_contamination(
    config: &Config,
    embeddings: &EmbeddingMap,
    hyperplanes: &Hyperplanes,
    toxic_buckets: &ToxicBuckets,
    hot_buckets: &HotBuckets,
    eval_documents: &EvalDocuments,
    eval_vocabulary: &HashSet<String>,
    eval_text_snippets: &EvalTextSnippets,
) -> Result<(), Error> {
    // Find all training files
    let training_files = expand_dirs(
        vec![config.local_input.clone()],
        Some(vec![".jsonl", ".gz"].as_slice()),
    )?;
    println!("Found {} training files to process", training_files.len());
    let pbar = build_pbar(training_files.len(), "Training files");

    let contamination_results: DashMap<String, Vec<ToxicContaminationEntry>> = DashMap::new();
    let file_stats: DashMap<String, FileProcessingStats> = DashMap::new();

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

        match process_toxic_training_file(
            file_path,
            &file_name,
            config,
            embeddings,
            hyperplanes,
            toxic_buckets,
            hot_buckets,
            eval_documents,
            eval_vocabulary,
            &contamination_results,
            eval_text_snippets,
        ) {
            Ok(stats) => {
                file_stats.insert(file_name.clone(), stats);
            }
            Err(e) => {
                println!("Error processing training file {:?}: {:?}", file_path, e);
            }
        }
        pbar.inc(1);
    });

    // Save contamination results
    save_toxic_contamination_results(
        &contamination_results,
        &config.report_output_dir,
        eval_text_snippets,
    )?;

    // Create purified files if requested
    if config.purify {
        create_purified_files(config, &contamination_results, &training_files)?;
    }

    // Aggregate timing statistics from all files
    let mut aggregated_stats = TimingStats::default();
    let mut aggregated_collision_timing = CollisionTiming::default();
    let mut aggregated_collision_counters = CollisionCounters::default();
    let mut total_lines_all = 0;
    let mut total_contaminated_lines = 0;
    let mut total_bucket_hits_all = 0;
    let mut total_bucket_misses_all = 0;
    let mut total_hot_buckets_skipped_all = 0;
    let mut total_vocab_filtered_all = 0;
    let mut total_training_vocab_size = 0;

    for entry in file_stats.iter() {
        let stats = entry.value();

        // Aggregate timing stats
        aggregated_stats.text_extraction += stats.timing.text_extraction;
        aggregated_stats.embedding_computation += stats.timing.embedding_computation;
        aggregated_stats.normalization += stats.timing.normalization;
        aggregated_stats.lsh_bucket_computation += stats.timing.lsh_bucket_computation;
        aggregated_stats.bucket_lookup += stats.timing.bucket_lookup;
        aggregated_stats.collision_detection += stats.timing.collision_detection;
        aggregated_stats.threshold_evaluation += stats.timing.threshold_evaluation;
        aggregated_stats.total_per_line += stats.timing.total_per_line;

        // Aggregate granular timing stats
        aggregated_stats.hash_lookups += stats.timing.hash_lookups;
        aggregated_stats.vector_cloning += stats.timing.vector_cloning;
        aggregated_stats.vector_arithmetic += stats.timing.vector_arithmetic;
        aggregated_stats.memory_allocation += stats.timing.memory_allocation;
        aggregated_stats.random_generation += stats.timing.random_generation;
        aggregated_stats.vector_normalization += stats.timing.vector_normalization;

        // Aggregate collision timing and counters
        aggregated_collision_timing.bidirectional_traversal +=
            stats.collision_timing.bidirectional_traversal;
        aggregated_collision_timing.left_traversal += stats.collision_timing.left_traversal;
        aggregated_collision_timing.right_traversal += stats.collision_timing.right_traversal;
        aggregated_collision_timing.intersection_computation +=
            stats.collision_timing.intersection_computation;
        aggregated_collision_timing.active_document_management +=
            stats.collision_timing.active_document_management;
        aggregated_collision_timing.document_state_updates +=
            stats.collision_timing.document_state_updates;
        aggregated_collision_timing.bucket_access += stats.collision_timing.bucket_access;
        aggregated_collision_timing.hot_bucket_filtering +=
            stats.collision_timing.hot_bucket_filtering;
        aggregated_collision_timing.cluster_expansion += stats.collision_timing.cluster_expansion;
        aggregated_collision_timing.miss_threshold_checking +=
            stats.collision_timing.miss_threshold_checking;
        aggregated_collision_timing.total_collision_processing +=
            stats.collision_timing.total_collision_processing;

        aggregated_collision_counters.total_left_steps += stats.collision_counters.total_left_steps;
        aggregated_collision_counters.total_right_steps +=
            stats.collision_counters.total_right_steps;
        aggregated_collision_counters.max_left_steps = aggregated_collision_counters
            .max_left_steps
            .max(stats.collision_counters.max_left_steps);
        aggregated_collision_counters.max_right_steps = aggregated_collision_counters
            .max_right_steps
            .max(stats.collision_counters.max_right_steps);
        aggregated_collision_counters.intersection_computations +=
            stats.collision_counters.intersection_computations;
        aggregated_collision_counters.total_intersection_size +=
            stats.collision_counters.total_intersection_size;
        aggregated_collision_counters.empty_intersections +=
            stats.collision_counters.empty_intersections;
        aggregated_collision_counters.documents_processed +=
            stats.collision_counters.documents_processed;
        aggregated_collision_counters.documents_dropped_misses +=
            stats.collision_counters.documents_dropped_misses;
        aggregated_collision_counters.max_active_documents = aggregated_collision_counters
            .max_active_documents
            .max(stats.collision_counters.max_active_documents);
        aggregated_collision_counters.buckets_accessed += stats.collision_counters.buckets_accessed;
        aggregated_collision_counters.hot_buckets_skipped +=
            stats.collision_counters.hot_buckets_skipped;
        aggregated_collision_counters.total_bucket_entries_processed +=
            stats.collision_counters.total_bucket_entries_processed;
        aggregated_collision_counters.collision_hits += stats.collision_counters.collision_hits;
        aggregated_collision_counters.collision_misses += stats.collision_counters.collision_misses;
        aggregated_collision_counters.consecutive_miss_dropouts +=
            stats.collision_counters.consecutive_miss_dropouts;
        aggregated_collision_counters.cache_hits += stats.collision_counters.cache_hits;
        aggregated_collision_counters.cache_misses += stats.collision_counters.cache_misses;

        // Aggregate other stats
        total_lines_all += stats.total_lines;
        total_contaminated_lines += stats.contaminated_lines;
        total_bucket_hits_all += stats.bucket_hits;
        total_bucket_misses_all += stats.bucket_misses;
        total_hot_buckets_skipped_all += stats.hot_buckets_skipped;
        total_vocab_filtered_all += stats.vocab_filtered;
        total_training_vocab_size += stats.training_vocab_size;
    }

    // Print aggregated summary statistics
    let total_time = aggregated_stats.total_per_line.as_secs_f64() * 1000.0;
    debug_println!(
        config,
        "\n\n\n=== AGGREGATED TIMING SUMMARY FOR ALL FILES ==="
    );
    debug_println!(config, "Total files processed: {}", file_stats.len());
    debug_println!(config, "Total lines processed: {}", total_lines_all);
    debug_println!(config, "Total time: {:.1}ms", total_time);
    debug_println!(
        config,
        "Average per line: {:.1}ms",
        if total_lines_all > 0 {
            total_time / total_lines_all as f64
        } else {
            0.0
        }
    );
    debug_println!(config, "");
    debug_println!(config, "BREAKDOWN BY CATEGORY:");
    print_timing_category(
        config,
        "Text Extraction     ",
        aggregated_stats.text_extraction,
        total_time,
    );
    print_timing_category(
        config,
        "Embedding Computation",
        aggregated_stats.embedding_computation,
        total_time,
    );
    print_timing_category(
        config,
        "Vector Normalization",
        aggregated_stats.normalization,
        total_time,
    );
    print_timing_category(
        config,
        "LSH Bucket Computation",
        aggregated_stats.lsh_bucket_computation,
        total_time,
    );
    print_timing_category(
        config,
        "Bucket Lookup       ",
        aggregated_stats.bucket_lookup,
        total_time,
    );
    print_timing_category(
        config,
        "Collision Detection ",
        aggregated_stats.collision_detection,
        total_time,
    );
    print_timing_category(
        config,
        "Threshold Evaluation",
        aggregated_stats.threshold_evaluation,
        total_time,
    );
    debug_println!(config, "");
    debug_println!(config, "GRANULAR EMBEDDING BREAKDOWN:");
    print_timing_category(
        config,
        "Hash Lookups        ",
        aggregated_stats.hash_lookups,
        total_time,
    );
    print_timing_category(
        config,
        "Vector Cloning      ",
        aggregated_stats.vector_cloning,
        total_time,
    );
    print_timing_category(
        config,
        "Vector Arithmetic   ",
        aggregated_stats.vector_arithmetic,
        total_time,
    );
    print_timing_category(
        config,
        "Memory Allocation   ",
        aggregated_stats.memory_allocation,
        total_time,
    );
    print_timing_category(
        config,
        "Random Generation   ",
        aggregated_stats.random_generation,
        total_time,
    );
    print_timing_category(
        config,
        "Vector Normalization",
        aggregated_stats.vector_normalization,
        total_time,
    );
    debug_println!(config, "");

    // Print collision breakdown
    print_collision_breakdown(
        config,
        &aggregated_collision_timing,
        &aggregated_collision_counters,
    );

    // Print bucket statistics in debug mode
    debug_println!(config, "BUCKET STATISTICS:");
    debug_println!(
        config,
        "Vocabulary filtered (out-of-vocab): {}",
        total_vocab_filtered_all
    );
    debug_println!(
        config,
        "Bucket hits (found eval matches): {}",
        total_bucket_hits_all
    );
    debug_println!(
        config,
        "Bucket misses (no eval matches): {}",
        total_bucket_misses_all
    );
    debug_println!(
        config,
        "Hot buckets skipped: {}",
        total_hot_buckets_skipped_all
    );
    let total_ngrams_generated = total_vocab_filtered_all
        + total_bucket_hits_all
        + total_bucket_misses_all
        + total_hot_buckets_skipped_all;
    let total_ngrams_processed =
        total_bucket_hits_all + total_bucket_misses_all + total_hot_buckets_skipped_all;
    debug_println!(
        config,
        "Total n-grams generated: {}",
        total_ngrams_generated
    );
    debug_println!(
        config,
        "Total n-grams processed (LSH computed): {}",
        total_ngrams_processed
    );
    debug_println!(config, "");

    // Print vocabulary statistics in debug mode
    debug_println!(config, "VOCABULARY STATISTICS:");
    debug_println!(
        config,
        "Training vocabulary size (across all files): {}",
        total_training_vocab_size
    );
    debug_println!(config, "Eval vocabulary size: {}", eval_vocabulary.len());
    // Note: Can't compute exact union/intersection without storing all training vocab
    debug_println!(config, "");

    if total_contaminated_lines > 0 {
        println!(
            "  → Found {} contaminated lines out of {} total lines across {} files",
            total_contaminated_lines,
            total_lines_all,
            file_stats.len()
        );
    } else {
        println!(
            "  → No contamination found ({} lines processed across {} files)",
            total_lines_all,
            file_stats.len()
        );
    }

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
    vector_normalization: Duration,
}

#[derive(Default, Clone)]
pub struct CollisionTiming {
    // Core collision detection phases
    bidirectional_traversal: Duration,
    left_traversal: Duration,
    right_traversal: Duration,

    // Document set operations
    intersection_computation: Duration,
    active_document_management: Duration,
    document_state_updates: Duration,

    // Bucket operations
    bucket_access: Duration,
    hot_bucket_filtering: Duration,

    // Algorithmic operations
    cluster_expansion: Duration,
    miss_threshold_checking: Duration,

    // Overall collision processing
    total_collision_processing: Duration,
}

#[derive(Default)]
pub struct CollisionCounters {
    // Traversal metrics
    total_left_steps: usize,
    total_right_steps: usize,
    max_left_steps: usize,
    max_right_steps: usize,

    // Intersection metrics
    intersection_computations: usize,
    total_intersection_size: usize,
    empty_intersections: usize,

    // Document tracking
    documents_processed: usize,
    documents_dropped_misses: usize,
    max_active_documents: usize,

    // Bucket metrics
    buckets_accessed: usize,
    hot_buckets_skipped: usize,
    total_bucket_entries_processed: usize,

    // Hit/miss metrics
    collision_hits: usize,
    collision_misses: usize,
    consecutive_miss_dropouts: usize,

    // Cache metrics
    cache_hits: usize,
    cache_misses: usize,
}

#[derive(Default)]
pub struct FileProcessingStats {
    timing: TimingStats,
    collision_timing: CollisionTiming,
    collision_counters: CollisionCounters,
    total_lines: usize,
    contaminated_lines: usize,
    bucket_hits: usize,
    bucket_misses: usize,
    hot_buckets_skipped: usize,
    vocab_filtered: usize,
    training_vocab_size: usize,
}

#[derive(Clone)]
pub struct ToxicContaminationEntry {
    pub training_line: usize,
    eval_name: String,
    eval_line: usize,
    overlap_ratio: f32,
    toxic_score: f32,
    matching_ngrams: Option<Vec<String>>,
    bucket_sizes: Option<Vec<usize>>,
    bucket_ids: Option<Vec<u64>>,
    contamination_start_idx: Option<usize>,
    contamination_end_idx: Option<usize>,
    training_overlap_text: Option<String>,
    ngram_match_cnt: usize, // Number of unique bucket matches
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
    collision_timing: &mut CollisionTiming,
    collision_counters: &mut CollisionCounters,
    bucket_cache: &mut BucketIdLruCache,
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
        let collision_start = std::time::Instant::now();
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
            collision_timing,
            collision_counters,
            bucket_cache,
            &mut bucket_hits_count,
            &mut bucket_misses_count,
            &mut hot_buckets_skipped_count,
            &mut vocab_filtered_count,
        )? {
            CollisionResult::Hit(document_ids) => {
                // Found contamination! Use intersection-based walking
                let ngram_text = if word_tokens.len() < config.ngram_size {
                    word_tokens.join(" ")
                } else {
                    word_tokens[i..i + config.ngram_size].join(" ")
                };
                // debug_println!(config, "\n💥 INITIAL HIT DETECTED at n-gram {} with {} documents!", i, document_ids.len());
                // debug_println!(config, "🔤 N-gram text: '{}'", ngram_text);
                // debug_println!(config, "📄 Document IDs: {:?}", document_ids);

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
                    &ngram_text,
                    timing_stats,
                    collision_timing,
                    collision_counters,
                    bucket_cache,
                    &mut bucket_hits_count,
                    &mut bucket_misses_count,
                    &mut hot_buckets_skipped_count,
                    &mut vocab_filtered_count,
                )?;
                collision_timing.total_collision_processing += collision_start.elapsed();

                // Mark all indices in this cluster as processed
                for idx in cluster.start_idx..=cluster.end_idx {
                    processed_indices.insert(idx);
                }

                clusters.push(cluster.clone());
                // debug_println!(config, "📍 Cluster completed: indices {}-{}, {} document matches",
                //               cluster.start_idx, cluster.end_idx, cluster.document_matches.len());

                // Jump past the processed region
                i = processed_indices.iter().max().copied().unwrap_or(i) + 1;
            }
            CollisionResult::Miss
            | CollisionResult::VocabFiltered
            | CollisionResult::HotBucketSkipped => {
                collision_timing.total_collision_processing += collision_start.elapsed();
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
#[allow(unused_variables)]
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
    collision_timing: &mut CollisionTiming,
    collision_counters: &mut CollisionCounters,
    bucket_cache: &mut BucketIdLruCache,
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

    // Check if all tokens in this n-gram exist in eval vocabulary
    let all_tokens_in_eval = ngram_tokens
        .iter()
        .all(|token| eval_vocabulary.contains(token));
    if !all_tokens_in_eval {
        // Debug: Show which n-grams are being filtered
        *vocab_filtered_count += 1;
        return Ok(CollisionResult::VocabFiltered);
    }

    // LSH bucket computation with caching
    let lsh_start = Instant::now();
    let bucket_id = bucket_cache.get_or_compute(&ngram_embedding, |embedding| {
        compute_lsh_bucket(embedding, hyperplanes)
    });
    timing_stats.lsh_bucket_computation += lsh_start.elapsed();

    // Cache statistics will be updated at file level to avoid overwriting

    // // Hot bucket optimization - skip hot buckets immediately
    // let hot_bucket_start = std::time::Instant::now();
    // if !hot_buckets.is_empty() && hot_buckets.contains(&bucket_id) {
    //     collision_timing.hot_bucket_filtering += hot_bucket_start.elapsed();
    //     collision_counters.hot_buckets_skipped += 1;
    //     *hot_buckets_skipped_count += 1;
    //     return Ok(CollisionResult::HotBucketSkipped);
    // }
    // collision_timing.hot_bucket_filtering += hot_bucket_start.elapsed();

    // Bucket lookup
    let lookup_start = Instant::now();
    if let Some(bucket_contents) = toxic_buckets.get(&bucket_id) {
        let bucket_access_start = std::time::Instant::now();
        *bucket_hits_count += 1;
        timing_stats.bucket_lookup += lookup_start.elapsed();

        // Collect all document IDs that match this bucket
        let collisions = bucket_contents.value().clone();
        collision_timing.bucket_access += bucket_access_start.elapsed();
        collision_counters.buckets_accessed += 1;
        collision_counters.total_bucket_entries_processed += collisions.len();
        collision_counters.collision_hits += 1;
        Ok(CollisionResult::Hit(collisions))
    } else {
        *bucket_misses_count += 1;
        timing_stats.bucket_lookup += lookup_start.elapsed();
        collision_counters.collision_misses += 1;
        Ok(CollisionResult::Miss)
    }
}

#[derive(Debug, Clone)]
struct DocumentState {
    matched_buckets: HashSet<u64>,
    consecutive_misses: usize,
}

#[derive(Debug)]
enum CollisionResult {
    Hit(HashSet<u32>), // Found documents in bucket
    Miss,              // No bucket collision
    VocabFiltered,     // N-gram filtered due to vocabulary
    #[allow(dead_code)]
    HotBucketSkipped, // Skipped hot bucket
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
    initial_document_ids: HashSet<u32>,
    _initial_training_ngram: &str,
    timing_stats: &mut TimingStats,
    collision_timing: &mut CollisionTiming,
    collision_counters: &mut CollisionCounters,
    bucket_cache: &mut BucketIdLruCache,
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

    // Compute the initial bucket ID for the hit
    let initial_bucket_id = compute_lsh_bucket(&ngram_embeddings[hit_idx], hyperplanes);

    for doc_id in &initial_document_ids {
        let mut matched_buckets = HashSet::new();
        matched_buckets.insert(initial_bucket_id);
        all_document_stats.insert(
            *doc_id,
            DocumentState {
                matched_buckets,
                consecutive_misses: 0,
            },
        );
    }

    // Track which documents are still "active" for continued evaluation
    let mut active_documents: HashSet<u32> = initial_document_ids.clone();

    // Expand backward (no left bound for debugging)
    let left_bound = 0;
    // TODO: Re-enable left bound optimization: hit_idx.saturating_sub(config.sample_every_m_tokens) when config.sample_every_m_tokens > 1

    let left_traversal_start = std::time::Instant::now();
    let mut left_steps = 0;
    let mut i = hit_idx;
    while i > left_bound && !active_documents.is_empty() {
        i -= 1;
        left_steps += 1;

        let _ngram_text = if word_tokens.len() < config.ngram_size {
            word_tokens.join(" ")
        } else {
            word_tokens[i..i + config.ngram_size].join(" ")
        };

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
            collision_timing,
            collision_counters,
            bucket_cache,
            bucket_hits_count,
            bucket_misses_count,
            hot_buckets_skipped_count,
            vocab_filtered_count,
        )? {
            CollisionResult::Hit(current_documents) => {
                let intersection_start = std::time::Instant::now();
                let current_set = &current_documents;

                // Compute bucket ID for this n-gram
                let bucket_id = compute_lsh_bucket(&ngram_embeddings[i], hyperplanes);

                // Update each document's state based on intersection
                let doc_state_start = std::time::Instant::now();

                // Efficient set intersection: find documents in both active_documents and current_documents
                let mut match_count = 0;

                // Update states for intersected documents (found in bucket)
                for doc_id in active_documents.intersection(current_set) {
                    if let Some(state) = all_document_stats.get_mut(doc_id) {
                        state.matched_buckets.insert(bucket_id);
                        state.consecutive_misses = 0;
                        match_count += 1;
                    }
                }

                // Update states for documents not in intersection (missed)
                for doc_id in active_documents.difference(current_set) {
                    if let Some(state) = all_document_stats.get_mut(doc_id) {
                        state.consecutive_misses += 1;
                    }
                }
                collision_timing.document_state_updates += doc_state_start.elapsed();
                collision_timing.intersection_computation += intersection_start.elapsed();
                collision_counters.intersection_computations += 1;
                collision_counters.total_intersection_size += current_set.len();
                if current_set.is_empty() {
                    collision_counters.empty_intersections += 1;
                }

                if match_count > 0 {
                    start_idx = i;
                }
            }
            CollisionResult::Miss => {
                // No bucket found, increment miss count for all active documents
                let _miss_count = active_documents.len();
                for doc_id in &active_documents {
                    if let Some(state) = all_document_stats.get_mut(doc_id) {
                        state.consecutive_misses += 1;
                    }
                }
            }
            CollisionResult::VocabFiltered => {
                // Vocabulary filtering counts as a miss for all active documents
                let _miss_count = active_documents.len();
                for doc_id in &active_documents {
                    if let Some(state) = all_document_stats.get_mut(doc_id) {
                        state.consecutive_misses += 1;
                    }
                }
            }
            CollisionResult::HotBucketSkipped => {
                // Hot bucket skip doesn't count as a miss (performance optimization)
            }
        }

        // Remove documents from active set that exceeded miss threshold (but keep their stats)
        let before_count = active_documents.len();
        active_documents.retain(|doc_id| {
            all_document_stats
                .get(doc_id)
                .map(|state| state.consecutive_misses < config.max_consecutive_misses)
                .unwrap_or(false)
        });
        let _dropped_count = before_count - active_documents.len();
    }

    // Record left traversal timing and stats
    collision_timing.left_traversal += left_traversal_start.elapsed();
    collision_counters.total_left_steps += left_steps;
    collision_counters.max_left_steps = collision_counters.max_left_steps.max(left_steps);

    // Reset active documents for right walk, keeping accumulated match lengths in all_document_stats
    active_documents = initial_document_ids.clone();

    // Reset consecutive misses for right walk, but keep match_length accumulated from left walk
    for doc_id in &initial_document_ids {
        if let Some(state) = all_document_stats.get_mut(doc_id) {
            state.consecutive_misses = 0;
        }
    }

    i = hit_idx;
    let right_traversal_start = std::time::Instant::now();
    let mut right_steps = 0;

    while i + 1 < ngram_embeddings.len() && !active_documents.is_empty() {
        i += 1;
        right_steps += 1;

        let _ngram_text = if word_tokens.len() < config.ngram_size {
            word_tokens.join(" ")
        } else {
            word_tokens[i..i + config.ngram_size].join(" ")
        };

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
            collision_timing,
            collision_counters,
            bucket_cache,
            bucket_hits_count,
            bucket_misses_count,
            hot_buckets_skipped_count,
            vocab_filtered_count,
        )? {
            CollisionResult::Hit(current_documents) => {
                let current_set = &current_documents;

                // Compute bucket ID for this n-gram
                let bucket_id = compute_lsh_bucket(&ngram_embeddings[i], hyperplanes);

                // Update each document's state based on intersection

                // Efficient set intersection: find documents in both active_documents and current_documents
                let mut match_count = 0;

                // Update states for intersected documents (found in bucket)
                for doc_id in active_documents.intersection(current_set) {
                    if let Some(state) = all_document_stats.get_mut(doc_id) {
                        state.matched_buckets.insert(bucket_id);
                        state.consecutive_misses = 0;
                        match_count += 1;
                    }
                }

                // Update states for documents not in intersection (missed)
                for doc_id in active_documents.difference(current_set) {
                    if let Some(state) = all_document_stats.get_mut(doc_id) {
                        state.consecutive_misses += 1;
                    }
                }

                if match_count > 0 {
                    end_idx = i;
                }
            }
            CollisionResult::Miss => {
                // No bucket found, increment miss count for all active documents
                let _miss_count = active_documents.len();
                for doc_id in &active_documents {
                    if let Some(state) = all_document_stats.get_mut(doc_id) {
                        state.consecutive_misses += 1;
                    }
                }
            }
            CollisionResult::VocabFiltered => {
                // Vocabulary filtering counts as a miss for all active documents
                let _miss_count = active_documents.len();
                for doc_id in &active_documents {
                    if let Some(state) = all_document_stats.get_mut(doc_id) {
                        state.consecutive_misses += 1;
                    }
                }
            }
            CollisionResult::HotBucketSkipped => {
                // Hot bucket skip doesn't count as a miss (performance optimization)
            }
        }

        // Remove documents from active set that exceeded miss threshold (but keep their stats)
        let before_count = active_documents.len();
        active_documents.retain(|doc_id| {
            all_document_stats
                .get(doc_id)
                .map(|state| state.consecutive_misses < config.max_consecutive_misses)
                .unwrap_or(false)
        });
        let _dropped_count = before_count - active_documents.len();
    }

    // Record right traversal timing and stats
    collision_timing.right_traversal += right_traversal_start.elapsed();
    collision_counters.total_right_steps += right_steps;
    collision_counters.max_right_steps = collision_counters.max_right_steps.max(right_steps);

    // Collect debug information for the final cluster range
    // Track buckets we've already added to avoid duplicates in display
    let mut seen_buckets = HashSet::new();

    for idx in start_idx..=end_idx {
        match check_ngram_for_collision(
            idx,
            word_tokens,
            &ngram_embeddings[idx],
            config,
            hyperplanes,
            toxic_buckets,
            hot_buckets,
            eval_vocabulary,
            timing_stats,
            collision_timing,
            collision_counters,
            bucket_cache,
            bucket_hits_count,
            bucket_misses_count,
            hot_buckets_skipped_count,
            vocab_filtered_count,
        )? {
            CollisionResult::Hit(current_documents) => {
                // Get n-gram text for any matching documents that were part of the original cluster
                if !current_documents.is_disjoint(&initial_document_ids) {
                    let bucket_id = compute_lsh_bucket(&ngram_embeddings[idx], hyperplanes);

                    // Only include if this bucket hasn't been seen before
                    if !seen_buckets.contains(&bucket_id) {
                        seen_buckets.insert(bucket_id);

                        let ngram_text = if word_tokens.len() < config.ngram_size {
                            word_tokens.join(" ")
                        } else {
                            word_tokens[idx..idx + config.ngram_size].join(" ")
                        };
                        matching_ngrams.push(ngram_text);

                        // Get bucket info
                        if let Some(bucket_contents) = toxic_buckets.get(&bucket_id) {
                            bucket_sizes.push(bucket_contents.value().len());
                            distinct_buckets.insert(bucket_id);
                        }
                    }
                }
            }
            CollisionResult::Miss
            | CollisionResult::VocabFiltered
            | CollisionResult::HotBucketSkipped => {}
        }
    }

    // Convert cumulative document stats to simple match counts for all documents that participated
    let document_matches: HashMap<u32, usize> = all_document_stats
        .into_iter()
        .map(|(doc_id, state)| (doc_id, state.matched_buckets.len()))
        .collect();

    Ok(ContaminationCluster {
        start_idx,
        end_idx,
        document_matches,
        matching_ngrams,
        bucket_sizes,
        distinct_buckets,
    })
}

pub fn process_toxic_training_file(
    file_path: &PathBuf,
    file_name: &str,
    config: &Config,
    embeddings: &EmbeddingMap,
    hyperplanes: &Hyperplanes,
    toxic_buckets: &ToxicBuckets,
    hot_buckets: &HotBuckets,
    eval_documents: &EvalDocuments,
    eval_vocabulary: &HashSet<String>,
    contamination_results: &DashMap<String, Vec<ToxicContaminationEntry>>,
    _eval_text_snippets: &EvalTextSnippets,
) -> Result<FileProcessingStats, Error> {
    let data = read_pathbuf_to_mem(file_path)?;

    let mut contaminated_lines = 0;
    let mut total_lines = 0;
    let mut cumulative_stats = TimingStats::default();
    let mut cumulative_collision_timing = CollisionTiming::default();
    let mut cumulative_collision_counters = CollisionCounters::default();

    // Track bucket statistics across all lines
    let mut total_bucket_hits = 0;
    let mut total_bucket_misses = 0;
    let mut total_hot_buckets_skipped = 0;
    let mut total_vocab_filtered = 0;

    // Track training vocabulary
    let mut training_vocabulary: HashSet<String> = HashSet::new();

    // Create cache for this file (no mutex needed since each file is processed independently)
    let mut bucket_cache = BucketIdLruCache::new(config.ngram_bucket_lru_cache);

    for (line_num, line) in data.lines().enumerate() {
        let line_start = Instant::now();
        let line = line?;

        // 1. Text extraction and parsing
        let text_start = Instant::now();
        let json_obj: Value = serde_json::from_str(&line)?;
        let line_text = get_nested_json_val(&json_obj, &config.content_key.to_string())?;

        let word_tokens = extract_words(&line_text, &config.punctuation_chars);
        let _training_word_count = word_tokens.len();

        // Track vocabulary from training data
        for word in &word_tokens {
            training_vocabulary.insert(word.clone());
        }

        let text_extraction_time = text_start.elapsed();

        // 2. N-gram generation and computation
        let embed_start = Instant::now();
        let _total_ngrams = if word_tokens.len() < config.ngram_size {
            if word_tokens.is_empty() {
                0
            } else {
                1
            }
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
            config.toxic_poison_scale,
            &mut granular_timing,
        );
        let embedding_time = embed_start.elapsed();

        let normalization_time = Duration::new(0, 0);
        let mut lsh_time = Duration::new(0, 0);
        let mut lookup_time = Duration::new(0, 0);
        let mut collision_time = Duration::new(0, 0);

        // 3. Process n-grams with sampling optimization
        let sampling_start = Instant::now();
        let mut sampling_timing = TimingStats::default();
        let mut collision_timing = CollisionTiming::default();
        let mut collision_counters = CollisionCounters::default();

        let (contamination_clusters, sampling_stats) = process_ngrams_with_sampling(
            &word_tokens,
            &ngram_embeddings,
            config,
            hyperplanes,
            toxic_buckets,
            hot_buckets,
            eval_vocabulary,
            &mut sampling_timing,
            &mut collision_timing,
            &mut collision_counters,
            &mut bucket_cache,
        )?;

        // Collect cache statistics for this line
        let (cache_hits, cache_misses, _hit_rate) = bucket_cache.stats();
        collision_counters.cache_hits += cache_hits;
        collision_counters.cache_misses += cache_misses;

        // Accumulate timing from sampling
        lsh_time += sampling_timing.lsh_bucket_computation;
        lookup_time += sampling_timing.bucket_lookup;
        collision_time += sampling_start.elapsed()
            - sampling_timing.lsh_bucket_computation
            - sampling_timing.bucket_lookup;

        // 4. Process clusters and calculate local overlap ratios
        let threshold_start = Instant::now();

        for (_cluster_idx, cluster) in contamination_clusters.iter().enumerate() {
            // Process each document match in the cluster
            for (doc_id, match_length) in &cluster.document_matches {
                // Get document metadata and live n-gram count
                let (eval_name, eval_line, _total_ngrams, _live_ngrams, unique_buckets) =
                    eval_documents
                        .get(doc_id)
                        .map(|doc| doc.value().clone())
                        .unwrap_or_else(|| ("unknown".to_string(), 0, 0, 0, 0));

                if unique_buckets == 0 {
                    continue; // Skip documents with no unique buckets
                }

                // Local overlap ratio: unique_buckets_matched / unique_buckets_in_eval_doc
                // No .min(1.0) needed - mathematically impossible to exceed 1.0
                let local_overlap_ratio = *match_length as f32 / unique_buckets as f32;

                if local_overlap_ratio >= config.toxic_overlap_threshold && complete_evaluation() {
                    let (matching_ngrams, bucket_sizes, bucket_ids) = if config.debug {
                        let bucket_ids_vec: Vec<u64> =
                            cluster.distinct_buckets.iter().cloned().collect();
                        (
                            Some(cluster.matching_ngrams.clone()),
                            Some(cluster.bucket_sizes.clone()),
                            Some(bucket_ids_vec),
                        )
                    } else {
                        (None, None, None)
                    };

                    // Calculate TOXIC score: sum of 1/log(bucket_size) for each matched bucket
                    let toxic_score: f32 = cluster
                        .distinct_buckets
                        .iter()
                        .map(|bucket_id| {
                            if let Some(docs) = toxic_buckets.get(bucket_id) {
                                let bucket_size = docs.len() as f32;
                                if bucket_size > 1.0 {
                                    1.0 / bucket_size.ln()
                                } else {
                                    1.0 // For single-document buckets, max weight
                                }
                            } else {
                                0.0 // Bucket not found (shouldn't happen)
                            }
                        })
                        .sum();

                    // Only count as contamination if toxic_score exceeds threshold
                    if toxic_score >= config.toxic_score_threshold {
                        // Extract the overlapping text with context
                        let training_overlap_text = extract_overlap_with_context_toxic(
                            &word_tokens,
                            cluster.start_idx,
                            cluster.end_idx,
                            30, // context words (tripled from 10)
                        );

                        contamination_results
                            .entry(file_name.to_string())
                            .or_default()
                            .push(ToxicContaminationEntry {
                                training_line: line_num,
                                eval_name: eval_name.clone(),
                                eval_line,
                                overlap_ratio: local_overlap_ratio,
                                toxic_score,
                                matching_ngrams,
                                bucket_sizes,
                                bucket_ids,
                                contamination_start_idx: Some(cluster.start_idx),
                                contamination_end_idx: Some(cluster.end_idx),
                                training_overlap_text,
                                ngram_match_cnt: cluster.distinct_buckets.len(),
                            });

                        contaminated_lines += 1;
                    }
                }
            }
        }
        let threshold_time = threshold_start.elapsed();

        let total_line_time = line_start.elapsed();
        total_lines += 1;

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
        cumulative_stats.vector_normalization += granular_timing.vector_normalization;

        // Accumulate collision timing and counters
        cumulative_collision_timing.bidirectional_traversal +=
            collision_timing.bidirectional_traversal;
        cumulative_collision_timing.left_traversal += collision_timing.left_traversal;
        cumulative_collision_timing.right_traversal += collision_timing.right_traversal;
        cumulative_collision_timing.intersection_computation +=
            collision_timing.intersection_computation;
        cumulative_collision_timing.active_document_management +=
            collision_timing.active_document_management;
        cumulative_collision_timing.document_state_updates +=
            collision_timing.document_state_updates;
        cumulative_collision_timing.bucket_access += collision_timing.bucket_access;
        cumulative_collision_timing.hot_bucket_filtering += collision_timing.hot_bucket_filtering;
        cumulative_collision_timing.cluster_expansion += collision_timing.cluster_expansion;
        cumulative_collision_timing.miss_threshold_checking +=
            collision_timing.miss_threshold_checking;
        cumulative_collision_timing.total_collision_processing +=
            collision_timing.total_collision_processing;

        cumulative_collision_counters.total_left_steps += collision_counters.total_left_steps;
        cumulative_collision_counters.total_right_steps += collision_counters.total_right_steps;
        cumulative_collision_counters.max_left_steps = cumulative_collision_counters
            .max_left_steps
            .max(collision_counters.max_left_steps);
        cumulative_collision_counters.max_right_steps = cumulative_collision_counters
            .max_right_steps
            .max(collision_counters.max_right_steps);
        cumulative_collision_counters.intersection_computations +=
            collision_counters.intersection_computations;
        cumulative_collision_counters.total_intersection_size +=
            collision_counters.total_intersection_size;
        cumulative_collision_counters.empty_intersections += collision_counters.empty_intersections;
        cumulative_collision_counters.documents_processed += collision_counters.documents_processed;
        cumulative_collision_counters.documents_dropped_misses +=
            collision_counters.documents_dropped_misses;
        cumulative_collision_counters.max_active_documents = cumulative_collision_counters
            .max_active_documents
            .max(collision_counters.max_active_documents);
        cumulative_collision_counters.buckets_accessed += collision_counters.buckets_accessed;
        cumulative_collision_counters.hot_buckets_skipped += collision_counters.hot_buckets_skipped;
        cumulative_collision_counters.total_bucket_entries_processed +=
            collision_counters.total_bucket_entries_processed;
        cumulative_collision_counters.collision_hits += collision_counters.collision_hits;
        cumulative_collision_counters.collision_misses += collision_counters.collision_misses;
        cumulative_collision_counters.consecutive_miss_dropouts +=
            collision_counters.consecutive_miss_dropouts;
        cumulative_collision_counters.cache_hits += collision_counters.cache_hits;
        cumulative_collision_counters.cache_misses += collision_counters.cache_misses;

        // Accumulate bucket statistics
        total_bucket_hits += sampling_stats.bucket_hits;
        total_bucket_misses += sampling_stats.bucket_misses;
        total_hot_buckets_skipped += sampling_stats.hot_buckets_skipped;
        total_vocab_filtered += sampling_stats.vocab_filtered;
    }

    // Return statistics instead of printing per-file summaries
    Ok(FileProcessingStats {
        timing: cumulative_stats,
        collision_timing: cumulative_collision_timing,
        collision_counters: cumulative_collision_counters,
        total_lines,
        contaminated_lines,
        bucket_hits: total_bucket_hits,
        bucket_misses: total_bucket_misses,
        hot_buckets_skipped: total_hot_buckets_skipped,
        vocab_filtered: total_vocab_filtered,
        training_vocab_size: training_vocabulary.len(),
    })
}

fn insert_with_hot_bucket_detection(toxic_buckets: &ToxicBuckets, bucket_id: u64, doc_id: u32) {
    // Perform the actual insertion
    toxic_buckets.entry(bucket_id).or_default().insert(doc_id);
}

fn print_timing_category(config: &Config, name: &str, duration: Duration, total_ms: f64) {
    let ms = duration.as_secs_f64() * 1000.0;
    let percentage = (ms / total_ms) * 100.0;
    debug_println!(config, "{}: {:8.1}ms ({:5.1}%)", name, ms, percentage);
}

fn print_collision_breakdown(
    config: &Config,
    collision_timing: &CollisionTiming,
    collision_counters: &CollisionCounters,
) {
    let total_collision_ms = collision_timing.total_collision_processing.as_secs_f64() * 1000.0;

    if total_collision_ms > 0.0 {
        debug_println!(config, "\n=== COLLISION DETECTION BREAKDOWN ===");
        debug_println!(
            config,
            "Total collision processing: {:8.1}ms",
            total_collision_ms
        );

        // Timing breakdown
        debug_println!(config, "\nTiming Breakdown:");
        if collision_timing.left_traversal.as_millis() > 0 {
            let ms = collision_timing.left_traversal.as_secs_f64() * 1000.0;
            let pct = (ms / total_collision_ms) * 100.0;
            debug_println!(
                config,
                "  Left traversal:        {:8.1}ms ({:5.1}%)",
                ms,
                pct
            );
        }
        if collision_timing.right_traversal.as_millis() > 0 {
            let ms = collision_timing.right_traversal.as_secs_f64() * 1000.0;
            let pct = (ms / total_collision_ms) * 100.0;
            debug_println!(
                config,
                "  Right traversal:       {:8.1}ms ({:5.1}%)",
                ms,
                pct
            );
        }
        if collision_timing.intersection_computation.as_millis() > 0 {
            let ms = collision_timing.intersection_computation.as_secs_f64() * 1000.0;
            let pct = (ms / total_collision_ms) * 100.0;
            debug_println!(
                config,
                "  Intersection compute:  {:8.1}ms ({:5.1}%)",
                ms,
                pct
            );
        }
        if collision_timing.document_state_updates.as_millis() > 0 {
            let ms = collision_timing.document_state_updates.as_secs_f64() * 1000.0;
            let pct = (ms / total_collision_ms) * 100.0;
            debug_println!(
                config,
                "  Document state:        {:8.1}ms ({:5.1}%)",
                ms,
                pct
            );
        }
        if collision_timing.bucket_access.as_millis() > 0 {
            let ms = collision_timing.bucket_access.as_secs_f64() * 1000.0;
            let pct = (ms / total_collision_ms) * 100.0;
            debug_println!(
                config,
                "  Bucket access:         {:8.1}ms ({:5.1}%)",
                ms,
                pct
            );
        }
        if collision_timing.hot_bucket_filtering.as_millis() > 0 {
            let ms = collision_timing.hot_bucket_filtering.as_secs_f64() * 1000.0;
            let pct = (ms / total_collision_ms) * 100.0;
            debug_println!(
                config,
                "  Hot bucket filtering:  {:8.1}ms ({:5.1}%)",
                ms,
                pct
            );
        }
        if collision_timing.miss_threshold_checking.as_millis() > 0 {
            let ms = collision_timing.miss_threshold_checking.as_secs_f64() * 1000.0;
            let pct = (ms / total_collision_ms) * 100.0;
            debug_println!(
                config,
                "  Miss threshold check:  {:8.1}ms ({:5.1}%)",
                ms,
                pct
            );
        }

        // Algorithmic metrics
        debug_println!(config, "\nTraversal Metrics:");
        debug_println!(
            config,
            "  Total left steps:      {:8}",
            collision_counters.total_left_steps
        );
        debug_println!(
            config,
            "  Total right steps:     {:8}",
            collision_counters.total_right_steps
        );
        debug_println!(
            config,
            "  Max left steps:        {:8}",
            collision_counters.max_left_steps
        );
        debug_println!(
            config,
            "  Max right steps:       {:8}",
            collision_counters.max_right_steps
        );
        let avg_left = if collision_counters.total_left_steps > 0 {
            collision_counters.total_left_steps as f64
                / (collision_counters.total_left_steps.max(1)) as f64
        } else {
            0.0
        };
        let avg_right = if collision_counters.total_right_steps > 0 {
            collision_counters.total_right_steps as f64
                / (collision_counters.total_right_steps.max(1)) as f64
        } else {
            0.0
        };
        debug_println!(config, "  Avg left steps:        {:8.1}", avg_left);
        debug_println!(config, "  Avg right steps:       {:8.1}", avg_right);

        debug_println!(config, "\nIntersection Metrics:");
        debug_println!(
            config,
            "  Intersections computed: {:8}",
            collision_counters.intersection_computations
        );
        debug_println!(
            config,
            "  Total intersection sz:  {:8}",
            collision_counters.total_intersection_size
        );
        debug_println!(
            config,
            "  Empty intersections:    {:8}",
            collision_counters.empty_intersections
        );
        let avg_intersection_size = if collision_counters.intersection_computations > 0 {
            collision_counters.total_intersection_size as f64
                / collision_counters.intersection_computations as f64
        } else {
            0.0
        };
        debug_println!(
            config,
            "  Avg intersection size:  {:8.1}",
            avg_intersection_size
        );

        debug_println!(config, "\nDocument Management:");
        debug_println!(
            config,
            "  Documents processed:    {:8}",
            collision_counters.documents_processed
        );
        debug_println!(
            config,
            "  Documents dropped:      {:8}",
            collision_counters.documents_dropped_misses
        );
        debug_println!(
            config,
            "  Max active documents:   {:8}",
            collision_counters.max_active_documents
        );

        debug_println!(config, "\nBucket Operations:");
        debug_println!(
            config,
            "  Buckets accessed:       {:8}",
            collision_counters.buckets_accessed
        );
        debug_println!(
            config,
            "  Hot buckets skipped:    {:8}",
            collision_counters.hot_buckets_skipped
        );
        debug_println!(
            config,
            "  Bucket entries proc:    {:8}",
            collision_counters.total_bucket_entries_processed
        );
        let avg_bucket_size = if collision_counters.buckets_accessed > 0 {
            collision_counters.total_bucket_entries_processed as f64
                / collision_counters.buckets_accessed as f64
        } else {
            0.0
        };
        debug_println!(config, "  Avg bucket size:        {:8.1}", avg_bucket_size);

        debug_println!(config, "\nHit/Miss Metrics:");
        debug_println!(
            config,
            "  Collision hits:         {:8}",
            collision_counters.collision_hits
        );
        debug_println!(
            config,
            "  Collision misses:       {:8}",
            collision_counters.collision_misses
        );
        debug_println!(
            config,
            "  Miss dropouts:          {:8}",
            collision_counters.consecutive_miss_dropouts
        );
        let total_attempts =
            collision_counters.collision_hits + collision_counters.collision_misses;
        let hit_rate = if total_attempts > 0 {
            (collision_counters.collision_hits as f64 / total_attempts as f64) * 100.0
        } else {
            0.0
        };
        debug_println!(config, "  Hit rate:               {:8.1}%", hit_rate);

        debug_println!(config, "\nLSH Cache Performance:");
        debug_println!(
            config,
            "  Cache hits:             {:8}",
            collision_counters.cache_hits
        );
        debug_println!(
            config,
            "  Cache misses:           {:8}",
            collision_counters.cache_misses
        );
        let total_cache_requests = collision_counters.cache_hits + collision_counters.cache_misses;
        let cache_hit_rate = if total_cache_requests > 0 {
            (collision_counters.cache_hits as f64 / total_cache_requests as f64) * 100.0
        } else {
            0.0
        };
        debug_println!(config, "  Cache hit rate:         {:8.1}%", cache_hit_rate);
        let lsh_computations_saved = collision_counters.cache_hits;
        debug_println!(
            config,
            "  LSH computations saved: {:8}",
            lsh_computations_saved
        );
        debug_println!(config, "=====================================");
    }
}

// Placeholder for future complete evaluation logic
fn complete_evaluation() -> bool {
    true // For now, always passes complete evaluation
}

fn save_bucket_contents(config: &Config, bucket_contents: &BucketContents) -> Result<(), Error> {
    create_dir_all(&config.report_output_dir)?;
    let bucket_file = config.report_output_dir.join("bucket_contents.jsonl");

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

pub fn save_toxic_contamination_results(
    results: &DashMap<String, Vec<ToxicContaminationEntry>>,
    report_output_dir: &PathBuf,
    eval_text_snippets: &EvalTextSnippets,
) -> Result<PathBuf, Error> {
    save_toxic_contamination_results_with_filename(
        results,
        report_output_dir,
        None,
        eval_text_snippets,
    )
}

pub fn save_toxic_contamination_results_with_filename(
    results: &DashMap<String, Vec<ToxicContaminationEntry>>,
    report_output_dir: &PathBuf,
    custom_filename: Option<&str>,
    eval_text_snippets: &EvalTextSnippets,
) -> Result<PathBuf, Error> {
    create_dir_all(report_output_dir)?;
    let default_filename = get_results_filename("toxic");
    let filename = custom_filename.unwrap_or(&default_filename);
    let output_file = report_output_dir.join(filename);

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
                "toxic_score": contamination_entry.toxic_score,
                "ngram_match_cnt": contamination_entry.ngram_match_cnt,
                "method": "toxic"
            });

            // Add matching ngrams, bucket sizes, and bucket IDs if available (debug mode)
            if let Some(ref ngrams) = contamination_entry.matching_ngrams {
                result["matching_ngrams"] = json!(ngrams);
            }
            if let Some(ref sizes) = contamination_entry.bucket_sizes {
                result["bucket_sizes"] = json!(sizes);
            }
            if let Some(ref ids) = contamination_entry.bucket_ids {
                result["bucket_ids"] = json!(ids);
            }

            // Add token indices if available
            if let Some(start_idx) = contamination_entry.contamination_start_idx {
                result["contamination_start_idx"] = json!(start_idx);
            }
            if let Some(end_idx) = contamination_entry.contamination_end_idx {
                result["contamination_end_idx"] = json!(end_idx);
            }

            // Add overlapping text if available
            if let Some(ref overlap_text) = contamination_entry.training_overlap_text {
                result["training_overlap_text"] = json!(overlap_text);
            }

            // Look up eval text snippet
            if let Some(eval_text) = eval_text_snippets.get(&(
                contamination_entry.eval_name.clone(),
                contamination_entry.eval_line,
            )) {
                result["eval_overlap_text"] = json!(eval_text.value());
            } else {
                result["eval_overlap_text"] = json!("");
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
        println!(
            "Found {} contamination instances across {} files",
            total_contaminations,
            results.len()
        );
        println!("Results saved to: {:?}", output_file);
    } else {
        println!("=== NO TOXIC CONTAMINATION DETECTED ===");
        println!("No contamination found in training data");
        println!("Empty results file saved to: {:?}", output_file);
    }

    Ok(output_file)
}

fn print_bucket_statistics(
    config: &Config,
    toxic_buckets: &ToxicBuckets,
    bucket_contents: &Option<BucketContents>,
) {
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
    let avg_size = if total_buckets > 0 {
        total_entries as f64 / total_buckets as f64
    } else {
        0.0
    };

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
        println!(
            "Total potential buckets: 2^{} (extremely large)",
            config.toxic_hyperplanes
        );
        println!("Total occupied buckets: {}", total_buckets);
        println!("Bucket space occupancy: ~0% (negligible)");
    } else {
        println!(
            "Total potential buckets: {}",
            total_potential_buckets as u64
        );
        println!("Total occupied buckets: {}", total_buckets);
        println!("Bucket space occupancy: {:.6}%", occupancy_percentage);
    }
    println!("Total entries: {}", total_entries);
    println!(
        "Empty buckets: {} ({:.1}%)",
        empty_buckets,
        empty_buckets as f64 / total_buckets as f64 * 100.0
    );
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
            let distinct_count = contents
                .get(bucket_id)
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
                println!(
                    "\n{}. Bucket {} ({} entries, {} distinct):",
                    i + 1,
                    bucket_id,
                    size,
                    ngrams.len()
                );
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
            let distinct_count = contents
                .get(&bucket_id)
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
        for (i, (bucket_id, entries, distinct_count)) in diverse_buckets.iter().take(3).enumerate()
        {
            if let Some(ngrams_set) = contents.get(bucket_id) {
                let mut ngrams: Vec<String> = ngrams_set.iter().cloned().collect();
                ngrams.sort();
                let sample_size = std::cmp::min(10, ngrams.len());
                println!(
                    "\n{}. Bucket {} ({} entries, {} distinct):",
                    i + 1,
                    bucket_id,
                    entries,
                    distinct_count
                );
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
    let large_buckets: Vec<_> = bucket_sizes
        .iter()
        .filter(|&&size| size > large_bucket_threshold)
        .collect();

    if !large_buckets.is_empty() {
        println!();
        println!("⚠️  PERFORMANCE WARNING:");
        println!(
            "Found {} buckets with >{}x average size (>{} entries)",
            large_buckets.len(),
            10,
            large_bucket_threshold
        );
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
    toxic_buckets: &ToxicBuckets,
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
        let (eval_name, eval_line, total_ngrams, _old_live, unique_ngrams) =
            doc_entry.value().clone();

        let hot_ngrams = hot_ngram_counts.get(&doc_id).copied().unwrap_or(0);
        let live_ngrams = total_ngrams.saturating_sub(hot_ngrams);

        *doc_entry.value_mut() = (
            eval_name,
            eval_line,
            total_ngrams,
            live_ngrams,
            unique_ngrams,
        );
    }

    let total_docs_with_hot_ngrams = hot_ngram_counts.len();
    println!(
        "Updated live n-gram counts for {} docs affected by hot buckets)",
        total_docs_with_hot_ngrams
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sliding_window_algorithm() {
        // Test the sliding window optimization with 8-item string and 3-grams
        let test_embeddings = EmbeddingMap::new();

        // Create test vectors for 8 tokens
        let test_vectors = vec![
            vec![1.0, 10.0], // a
            vec![2.0, 20.0], // b
            vec![3.0, 30.0], // c
            vec![4.0, 40.0], // d
            vec![5.0, 50.0], // e
            vec![6.0, 60.0], // f
            vec![7.0, 70.0], // g
            vec![8.0, 80.0], // h
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

        let result =
            compute_ngram_embedding_training(&tokens, 3, &test_embeddings, 42, 3.0, &mut timing);

        // Should produce 6 n-grams: ["a","b","c"], ["b","c","d"], ["c","d","e"], ["d","e","f"], ["e","f","g"], ["f","g","h"]
        assert_eq!(result.len(), 6);

        // Expected 3-gram sums:
        let expected = vec![
            vec![1.0 + 2.0 + 3.0, 10.0 + 20.0 + 30.0], // a+b+c = [6, 60]
            vec![2.0 + 3.0 + 4.0, 20.0 + 30.0 + 40.0], // b+c+d = [9, 90]
            vec![3.0 + 4.0 + 5.0, 30.0 + 40.0 + 50.0], // c+d+e = [12, 120]
            vec![4.0 + 5.0 + 6.0, 40.0 + 50.0 + 60.0], // d+e+f = [15, 150]
            vec![5.0 + 6.0 + 7.0, 50.0 + 60.0 + 70.0], // e+f+g = [18, 180]
            vec![6.0 + 7.0 + 8.0, 60.0 + 70.0 + 80.0], // f+g+h = [21, 210]
        ];

        for (i, (actual, exp)) in result.iter().zip(expected.iter()).enumerate() {
            assert!(
                (actual[0] - exp[0]).abs() < 1e-6,
                "N-gram {} dim 0: expected {}, got {}",
                i,
                exp[0],
                actual[0]
            );
            assert!(
                (actual[1] - exp[1]).abs() < 1e-6,
                "N-gram {} dim 1: expected {}, got {}",
                i,
                exp[1],
                actual[1]
            );
        }

        println!("✅ Sliding window optimization test passed!");
        println!("Generated {} 3-grams from 8 tokens", result.len());
        println!(
            "First 3-gram: [{:.0}, {:.0}] (expected: [6, 60])",
            result[0][0], result[0][1]
        );
        println!(
            "Last 3-gram:  [{:.0}, {:.0}] (expected: [21, 210])",
            result[5][0], result[5][1]
        );
    }
}

// Create purified versions of training files with contaminated lines removed
fn create_purified_files(
    config: &Config,
    contamination_results: &DashMap<String, Vec<ToxicContaminationEntry>>,
    training_files: &[PathBuf],
) -> Result<(), Error> {
    println!("\nCreating purified files...");

    // Determine output directory for cleaned files
    let cleaned_dir = config
        .cleaned_output_dir
        .as_ref()
        .unwrap_or(&config.report_output_dir);

    // Process each training file that has contamination
    for file_path in training_files {
        let file_name = if file_path.extension().and_then(|s| s.to_str()) == Some("gz") {
            file_path
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("unknown")
                .to_string()
        } else {
            file_path
                .file_name()
                .and_then(|f| f.to_str())
                .unwrap_or("unknown")
                .to_string()
        };

        // Check if this file has any contamination
        if let Some(contaminations) = contamination_results.get(&file_name) {
            if contaminations.is_empty() {
                continue;
            }

            // Collect contaminated line numbers
            let mut contaminated_lines = HashSet::new();
            for entry in contaminations.iter() {
                contaminated_lines.insert(entry.training_line);
            }

            // Use the shared write_purified_file function
            write_purified_file(file_path, cleaned_dir, &contaminated_lines, config)?;

            println!(
                "Created purified file for {} (removed {} lines)",
                file_name,
                contaminated_lines.len()
            );
        }
    }

    println!("Purification complete.");
    Ok(())
}
