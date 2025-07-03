# TOXIC: Token-Oriented eXclusion via Interference Clustering

TOXIC is a novel contamination detection approach that uses word embeddings and locality-sensitive hashing to detect semantic contamination between training and evaluation datasets. Unlike traditional methods that rely on exact text matching, TOXIC can identify paraphrased and semantically similar content.

## How TOXIC Works

### 1. Word Embedding Foundation
- Loads pre-trained word embeddings (default: 300-dimensional FastText vectors)
- Maps words to dense semantic vector representations
- Handles out-of-vocabulary (OOV) words using "poison tokens"

### 2. Poison Token Mechanics ⚠️

**The Core Innovation**: TOXIC uses semantically destructive vectors for unknown words.

Traditional approaches ignore or nullify OOV words, but this can cause false positives:
```
Training: "If you have 1 apple and I have 1 apple, we have 2 apples"
Eval:     "If you have 2 apples and I have 2 apples, we have 4 apples"
```

Without poison tokens: The embeddings for "apple", "have", "if" make these appear similar.
With poison tokens: The numbers "1", "2", "4" get distinct destructive vectors that break false similarity.

**Poison Token Categories:**
- Numbers: 1, 2, 42, 1990
- Proper nouns: Names, places, brands  
- Domain-specific terms: Technical jargon, IDs
- Misspellings and typos

**Asymmetric Poison Strategy**: 
- **Eval processing**: OOV words get consistent poison vectors (stored for reproducibility)
- **Training processing**: OOV words get chaotic poison vectors (fresh each time, not stored)
- This prevents false matches while maintaining eval dataset consistency

### 3. N-gram Embedding Computation
- Text is cleaned and split into words (not subword tokens)
- Sliding window n-grams are created (default: 4-grams)
- Each n-gram is converted to an embedding by summing constituent word vectors
- Example: "Find all c in" → sum(embed("find") + embed("all") + embed("c") + embed("in"))

### 4. LSH Without Normalization
- N-gram embeddings preserve magnitude information (normalization removed)
- Sign Random Projections create LSH bucket IDs:
  - Random hyperplanes partition the embedding space
  - Dot product with hyperplane determines bit (positive/negative)
  - 64 hyperplanes create 64-bit bucket IDs
- Similar n-grams map to same buckets with high probability

### 5. Contamination Detection with Bucket-Based Overlap
- Reference dataset n-grams are indexed by LSH bucket
- Training documents are processed and checked for bucket collisions
- **Key Innovation**: Measures unique LSH buckets hit rather than raw n-gram counts
- Overlap ratio = (unique eval buckets with matches) / (total eval buckets)
- Contamination flagged if overlap ratio exceeds threshold (default: 0.95)

**Why bucket-based overlap?**
- Accounts for LSH bucket collisions where multiple n-grams hash to same bucket
- Provides more robust similarity measurement than raw n-gram counting
- Naturally handles document length differences through bucket distribution

### 6. Cross-Dataset Contamination Discovery
TOXIC can discover contamination across multiple evaluation datasets simultaneously, revealing unexpected data leakage patterns.

## Key Parameters

- **`toxic_embedding_path`**: Path to pre-trained word vectors (FastText format)
- **`toxic_hyperplanes`**: Number of LSH hyperplanes (default: 64)
- **`toxic_overlap_threshold`**: Minimum overlap ratio for contamination (default: 0.95)
- **`toxic_poison_scale`**: Amplification factor for poison token impact (default: 3.0)
- **`ngram_size`**: N-gram window size (default: 4, higher = more precise)

## Algorithm Complexity

- **Time**: O(D × W × H) where D=documents, W=words/doc, H=hyperplanes
- **Space**: O(V + B) where V=vocabulary size, B=LSH buckets
- **Memory**: Asymmetric OOV handling (eval caching + training chaos), scales to massive datasets

## Strengths

- **Semantic awareness**: Detects paraphrased and reworded contamination
- **Document length robust**: Handles asymmetric document sizes correctly
- **Memory efficient**: Asymmetric poison strategy prevents unbounded growth
- **Cross-domain detection**: Can identify contamination across different datasets
- **Tunable precision**: N-gram size and threshold allow fine-tuning
- **Mathematically principled**: Uses proven LSH techniques with embedding geometry

## Limitations

- **Requires embeddings**: Needs pre-trained word vectors (large file)
- **Computational overhead**: More expensive than pure text-based methods
- **Language specific**: Embeddings tied to specific languages
- **Embedding quality dependent**: Poor embeddings → poor detection
- **Hyperparameter sensitive**: Threshold and n-gram size need tuning

## Use Cases

**Excellent for detecting:**
- **Paraphrased content**: Same meaning, different wording
- **Semantic similarity**: Related concepts and ideas
- **Cross-language leakage**: If multilingual embeddings used
- **Template variations**: Modified versions of base content
- **Academic misconduct**: Reworded plagiarism

**Less effective for:**
- **Exact character matches**: MinHash might be faster
- **Very short documents**: Insufficient context for embeddings
- **Domain-specific jargon**: If not in pre-trained embeddings

## Configuration Example

```yaml
mode: toxic
debug: true
content_key: text
local_input: /path/to/training/data
reference_input: /path/to/eval/data
output_dir: /path/to/output

ngram_size: 4
toxic_embedding_path: /path/to/wiki-news-300d-1M.vec
toxic_hyperplanes: 64
toxic_overlap_threshold: 0.95
toxic_poison_scale: 3.0
```

## Output Format

Results saved to `toxic_contamination_results.jsonl`:
```json
{
  "training_file": "train_batch_1.jsonl",
  "training_line": 42, 
  "eval_dataset": "mmlu_dev",
  "eval_line": 156,
  "overlap_ratio": 0.67,
  "method": "toxic"
}
```

## Advanced Features

### Debug Mode Enhanced Analysis
When `debug: true` is set, TOXIC provides detailed contamination analysis:
- **Matching n-grams**: Shows specific text segments that triggered matches
- **Bucket sizes**: Reveals LSH bucket collision statistics
- **Bucket contents**: Saves detailed bucket analysis to `bucket_contents.jsonl`
- **Timing breakdowns**: Performance profiling for optimization

### Multi-dataset Discovery
Can reveal contamination patterns across multiple evaluation datasets:
```
Training line 42 → MMLU line 156 (0.85 overlap)
Training line 42 → HellaSwag line 23 (0.71 overlap)  
Training line 42 → ARC line 891 (0.63 overlap)
```

### Poison Token Philosophy
The asymmetric poison token approach balances reproducibility with chaos prevention. Eval datasets get consistent poison vectors for reproducible analysis, while training data gets chaotic poison vectors to prevent false matches.

**Asymmetric Principle**: "Eval consistency for analysis, training chaos for precision."