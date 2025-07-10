# Configuration Guide

This guide covers all configuration options for Decon's contamination detection system. Configuration can be provided via YAML files or command-line flags. Command-line flags always override values from configuration files.

## Overview

Decon uses a flexible configuration system that supports:
- **YAML configuration files** for base settings
- **Command-line overrides** for runtime flexibility
- **Sensible defaults** for most parameters

When running in remote or containerized environments, you can override any configuration option without modifying files.

## Detector Configuration

The detector supports three modes: `simple`, `minhash`, and `toxic`. Each mode has both shared and mode-specific options.

### Shared Options (All Modes)

These options apply regardless of which detection mode you choose:

#### Core Settings
```yaml
# Detection algorithm to use
mode: simple  # Options: simple, minhash, toxic

# JSON field containing text content in your data files
content_key: text

# Enable debug output for troubleshooting
debug: false
```

#### Input/Output Paths
```yaml
# Directory containing training data to check for contamination
local_input: /path/to/training/data

# Directory containing evaluation/reference data to check against
reference_input: /path/to/eval/data

# Output directory for contamination reports
report_output_dir: /path/to/reports

# Optional: Separate directory for cleaned files (defaults to report_output_dir)
cleaned_output_dir: /path/to/cleaned

# Create cleaned versions of files with contaminated lines removed
purify: false
```

#### Text Processing
```yaml
# Characters to treat as punctuation (removed during tokenization)
# Default excludes: * + - = < > ^ _ []
punctuation_chars: "!\"#&'(),./:;?@`{|}~"

# Tokenizer to use (word or BPE-based)
tokenizer_str: word  # Options: word, p50k, cl100k
```

#### Performance
```yaml
# Number of worker threads for parallel processing (0 = use all cores)
worker_threads: 0
```

### SIMPLE Mode Options

SIMPLE mode uses n-gram matching with intelligent sampling for efficient exact contamination detection.

```yaml
# Size of n-grams to match
ngram_size: 13

# Sample every M tokens for efficiency (1 = no sampling)
sample_every_m_tokens: 10

# Maximum consecutive non-matches before stopping cluster expansion
max_consecutive_misses: 3

# Minimum overlap ratio to consider as contamination
toxic_overlap_threshold: 0.3

# Minimum toxic score (IDF-weighted) to report contamination
toxic_score_threshold: 0.0
```

### MinHash Mode Options

MinHash mode uses locality-sensitive hashing for approximate matching.

```yaml
# Number of hash functions per band
band_size: 5

# Number of bands for LSH
num_bands: 20

# Size of n-grams for shingling
ngram_size: 4

# Minimum Jaccard similarity to consider as contamination
jaccard_similarity_threshold: 0.5

# Random seed for hash functions
hash_seed: 0

# Windowing options for handling different document sizes
window_size_increment: 100
num_windows: 5
window_step_size: 50
```

### TOXIC Mode Options

TOXIC mode uses semantic embeddings with "poison tokens" for robust detection of paraphrased content.

```yaml
# Path to word embeddings file (e.g., word2vec format)
toxic_embedding_path: /path/to/embeddings.vec

# Number of random hyperplanes for LSH
toxic_hyperplanes: 64

# Minimum overlap ratio to consider as contamination
toxic_overlap_threshold: 0.85

# Minimum toxic score to report contamination
toxic_score_threshold: 2.0

# Scale factor for poison tokens (numbers, proper nouns, etc.)
toxic_poison_scale: 3.0

# Skip LSH buckets with more than this many documents (-1 to disable)
skip_hot_bucket_threshold: -1

# Sampling parameters (same as SIMPLE mode)
sample_every_m_tokens: 10
max_consecutive_misses: 3

# LRU cache size for n-gram to bucket mappings
ngram_bucket_lru_cache: 1000
```

## Orchestrator Configuration

The orchestrator manages distributed processing across multiple daemon instances.

### Required Settings
```yaml
# S3 path containing training data files
remote_file_input: s3://bucket/training-data/

# S3 path for contamination reports
remote_report_output_dir: s3://bucket/reports/

# Daemon endpoint
daemon_url: http://localhost:8080

# Local working directory for temporary files
local_work_dir: /tmp/decon-work
```

### Optional Settings

#### Output Configuration
```yaml
# S3 path for cleaned files (if purify is enabled)
remote_cleaned_output_dir: s3://bucket/cleaned/
```

#### Performance Tuning
```yaml
# Maximum concurrent jobs (auto-detected from daemon if not set)
max_concurrent_jobs: 100

# How often to poll for job status (seconds)
poll_interval: 5

# Number of s5cmd workers for S3 operations
s5cmd_workers: 50

# Delay before cleaning up completed jobs (seconds)
cleanup_delay: 10
```

#### Batch Processing
```yaml
# Files to download per batch
download_batch_size: 50

# Maximum files in download queue
download_queue_max: 200

# Maximum completed jobs in upload queue
upload_queue_max: 100

# Jobs to upload per batch
upload_batch_size: 50

# Timeout before processing partial upload batch (seconds)
upload_batch_timeout: 10
```

## Command-Line Override Examples

### Daemon Examples

```bash
# Override detection mode and input paths
decon daemon --config base.yaml \
  --mode simple \
  --local-input /data/training \
  --reference-input /data/eval

# Override SIMPLE mode parameters
decon daemon --config base.yaml \
  --ngram-size 15 \
  --sample-every-m-tokens 5 \
  --toxic-overlap-threshold 0.5

# Override TOXIC mode parameters
decon daemon --config base.yaml \
  --mode toxic \
  --toxic-embedding-path /models/embeddings.vec \
  --toxic-hyperplanes 128
```

### Orchestrator Examples

```bash
# Override S3 paths
python orchestration.py --config base.yaml \
  --remote-file-input s3://my-bucket/data/ \
  --remote-report-output-dir s3://my-bucket/output/

# Override performance settings
python orchestration.py --config base.yaml \
  --max-concurrent-jobs 200 \
  --download-batch-size 100 \
  --s5cmd-workers 100

# Override daemon endpoint
python orchestration.py --config base.yaml \
  --daemon-url http://remote-daemon:8080
```

## Best Practices

1. **Start with defaults**: Most parameters have sensible defaults. Only override what you need.

2. **Mode-specific tuning**:
   - **SIMPLE**: Adjust `sample_every_m_tokens` for speed vs accuracy tradeoff
   - **MinHash**: Tune `num_bands` and `band_size` for precision/recall balance
   - **TOXIC**: Ensure `toxic_embedding_path` points to appropriate embeddings for your domain

3. **Performance optimization**:
   - Increase `worker_threads` for CPU-bound operations
   - Adjust batch sizes based on available memory
   - Use sampling (`sample_every_m_tokens` > 1) for large datasets

4. **Threshold tuning**:
   - Start with default thresholds
   - Review results and adjust based on false positive/negative rates
   - Use the review tool to analyze detection quality

5. **Remote execution**:
   - Use command-line overrides instead of modifying config files
   - Set appropriate timeouts and retry settings
   - Monitor daemon health and job completion rates