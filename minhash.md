# MinHash Contamination Detection

MinHash is a probabilistic technique for estimating Jaccard similarity between documents using locality-sensitive hashing (LSH). This implementation uses MinHash signatures to efficiently detect potential contamination between training and evaluation datasets.

## How MinHash Works

### 1. Document Preprocessing
- Text is cleaned (lowercased, punctuation removed, whitespace normalized)
- Documents are tokenized using configurable tokenizers:
  - `p50k`: GPT-style BPE tokenization
  - `cl100k`: GPT-4 style BPE tokenization  
  - `uniseg`: Unicode word boundary segmentation
  - Default: Character-level tokenization

### 2. N-gram Generation
- Sliding window n-grams are extracted from token sequences
- Default n-gram size: 3 (configurable via `ngram_size`)
- Example: "hello world test" → ["hello", "world", "test"] → [("hello", "world", "test")]

### 3. MinHash Signature Generation
- Each n-gram is hashed using multiple hash functions
- For each hash function, the minimum hash value across all n-grams becomes part of the signature
- Signature length = `num_bands × band_size` (default: 7 × 8 = 56 hash functions)
- This creates a compact fingerprint representing the document's content

### 4. LSH Banding
- MinHash signatures are divided into bands of fixed size
- Documents with identical values in any band are considered "collision candidates"
- This allows approximate similarity search without comparing all document pairs
- Trade-off: more bands = higher precision, fewer bands = higher recall

### 5. Contamination Detection
- Reference dataset signatures are indexed by band values
- Training documents are processed and checked for band collisions
- When collisions occur, exact Jaccard similarity is computed
- Contamination flagged if similarity exceeds threshold (default: 0.5)

## Key Parameters

- **`num_bands`**: Number of LSH bands (default: 7)
- **`band_size`**: Hash functions per band (default: 8) 
- **`ngram_size`**: N-gram window size (default: 3)
- **`jaccard_similarity_threshold`**: Minimum similarity for contamination (default: 0.5)
- **`exact_override`**: Force exact Jaccard computation vs LSH approximation
- **`tokenizer_str`**: Tokenization method ("p50k", "cl100k", "uniseg", or default)

## Strengths

- **Fast**: O(n) signature generation, sub-linear similarity search
- **Memory efficient**: Compact signatures much smaller than full documents
- **Mathematically principled**: Unbiased estimator of Jaccard similarity
- **Scalable**: Handles large datasets efficiently
- **Tunable precision/recall**: LSH banding allows performance trade-offs

## Limitations

- **Exact matches only**: Struggles with paraphrasing or semantic similarity
- **Sensitive to tokenization**: Different tokenizers can miss contamination
- **Order-dependent**: N-gram approach sensitive to word reordering
- **No semantic understanding**: Treats "car" and "automobile" as completely different

## Use Cases

Best for detecting:
- **Copy-paste contamination**: Exact or near-exact duplicates
- **Template reuse**: Documents following similar patterns
- **Systematic leakage**: Large-scale data copying between datasets

Less effective for:
- **Paraphrased content**: Semantically identical but differently worded
- **Translated content**: Same meaning, different language
- **Summarized content**: Condensed versions of original text

## Output Format

Results saved to `contamination_results.jsonl`:
```json
{
  "training_file": "train_batch_1.jsonl", 
  "training_line": 42,
  "eval_dataset": "mmlu_dev",
  "eval_line": 156, 
  "jaccard_similarity": 0.83,
  "method": "minhash"
}
```

## Example Configuration

```yaml
mode: minhash
ngram_size: 3
num_bands: 7
band_size: 8
tokenizer_str: uniseg
jaccard_similarity_threshold: 0.8
exact_override: true
```