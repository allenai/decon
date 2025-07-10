# SIMPLE: Sampled Index Match with Progressive Length Extension

SIMPLE is a contamination detection approach that uses n-gram indexing with intelligent sampling and progressive cluster expansion to efficiently detect overlapping content between training and evaluation datasets.

## How SIMPLE Works

### 1. N-gram Index Construction
- Processes evaluation datasets to build an n-gram index
- Each n-gram is assigned a unique ID and mapped to document IDs
- Supports both word-level and subword tokenization
- Handles short documents (< n-gram size) by using all available tokens

### 2. Sampling Strategy
- Training data is processed with configurable sampling (default: every M n-grams)
- When a match is found, the algorithm expands around the hit
- Avoids exhaustive n-gram checking while maintaining high detection accuracy
- Sample rate configurable via `sample_every_m_tokens` parameter

### 3. Progressive Cluster Expansion
- Upon finding an n-gram match, expands both left and right
- Uses intersection-based traversal to track which documents remain viable
- Tolerates gaps using `max_consecutive_misses` parameter
- Creates contamination clusters representing contiguous matching regions

### 4. Document-Level Overlap Calculation
- Tracks unique n-grams matched per evaluation document
- Overlap ratio = (unique matched n-grams) / (total unique n-grams in doc)
- Provides document-level granularity for contamination assessment

### 5. Toxic Score Integration
- Calculates IDF-based toxic scores similar to TOXIC mode
- Weights n-grams by their inverse document frequency
- Rare n-grams contribute more to the contamination score
- Requires both overlap ratio and toxic score to exceed thresholds

### 6. Efficient Memory Management
- Uses concurrent data structures (DashMap) for parallel processing
- Maintains separate mappings for display vs computation
- Atomic ID generation for thread-safe n-gram indexing

## Key Parameters

- **`ngram_size`**: Size of n-gram window (default: 13)
- **`sample_every_m_tokens`**: Sampling rate for training data (default: 50)
- **`max_consecutive_misses`**: Gap tolerance during expansion (default: 3)
- **`toxic_overlap_threshold`**: Minimum overlap ratio (default: 0.5)
- **`toxic_score_threshold`**: Minimum toxic score (default: 0.5)
- **`tokenizer_str`**: Tokenization method ("word", "p50k", "cl100k", etc.)

## Algorithm Complexity

- **Time**: O(T/M × E × C) where T=training tokens, M=sample rate, E=expansion length, C=cluster operations
- **Space**: O(N + D) where N=unique n-grams, D=document metadata
- **Sampling benefit**: Reduces time complexity by factor of M while maintaining accuracy

## Strengths

- **Efficiency**: Sampling dramatically reduces computation vs exhaustive search
- **Accuracy**: Progressive expansion captures full contamination regions
- **Flexible tokenization**: Supports multiple tokenizer backends
- **Document awareness**: Tracks contamination at document granularity
- **Parallel processing**: Leverages Rayon for multi-threaded execution
- **Memory efficient**: Indexes only evaluation data, streams training data

## Limitations

- **Exact matching**: Requires identical n-gram sequences
- **Sampling trade-offs**: Very small contamination regions might be missed
- **No semantic understanding**: Cannot detect paraphrased content
- **Token-dependent**: Results vary with tokenization choice

## Use Cases

**Excellent for detecting:**
- **Large-scale contamination**: Efficiently finds substantial overlaps
- **Verbatim copying**: Exact text reuse between datasets  
- **Performance-critical scenarios**: When speed matters more than semantic detection
- **Token-level precision**: When exact token sequences matter

**Less effective for:**
- **Paraphrased content**: Reworded but semantically similar text
- **Very short overlaps**: Smaller than sampling window
- **Semantic plagiarism**: Same ideas, different expression

## Configuration Example

```yaml
mode: simple
debug: false
content_key: text
local_input: /path/to/training/data
reference_input: /path/to/eval/data
report_output_dir: /path/to/output

ngram_size: 13
sample_every_m_tokens: 50
max_consecutive_misses: 3
toxic_overlap_threshold: 0.5
toxic_score_threshold: 0.5
tokenizer_str: cl100k
```

## Output Format

Results saved to `simple_contamination_results.jsonl`:
```json
{
  "training_file": "train_batch_1.jsonl",
  "training_line": 42,
  "eval_dataset": "mmlu_dev", 
  "eval_line": 156,
  "overlap_ratio": 0.82,
  "toxic_score": 3.45,
  "contamination_start_idx": 125,
  "contamination_end_idx": 287,
  "method": "simple"
}
```

## Advanced Features

### Intersection-Based Walking
The cluster expansion algorithm maintains active document sets and uses intersection logic to determine when documents no longer match, preventing false positive expansion.

### Word Vocabulary Building
In word tokenization mode, builds vocabulary dynamically from the reference set, enabling consistent word-to-ID mapping across processing.

### Position Recovery
Tracks token indices for contamination regions, enabling precise location of matching content within documents for further analysis.

### Sampling Philosophy
"Sample sparsely, expand thoroughly" - the algorithm samples at intervals but exhaustively explores around detected matches to capture complete contamination regions.