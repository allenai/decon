# Contamination Detection for ML Datasets

Comprehensive contamination detection tools for training datasets. Identifies when training data contains text that appears in evaluation datasets using multiple detection approaches. Supports both exact matching and semantic similarity detection.

## Table of Contents
- [Overview](#overview)
- [Detection Methods](#detection-methods)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Commands](#commands)
- [Understanding Results](#understanding-results)

## Overview

**Contamination detection** is critical for machine learning evaluation. When training data accidentally contains examples from evaluation datasets, it can lead to artificially inflated performance metrics and unreliable model evaluation.

This tool provides three complementary detection approaches to efficiently identify contamination:
- **Exact matches**: Identical text between training and evaluation data
- **Near-duplicates**: Text with minor modifications (typos, formatting, small edits)
- **Semantic similarity**: Paraphrased or reworded content with similar meaning
- **Document length asymmetry**: Contamination between documents of very different lengths

## Detection Methods

### 📊 [MinHash Detection](minhash.md) (`mode: minhash`)
Fast, memory-efficient detection using Jaccard similarity and LSH.
- **Best for**: Exact matches, copy-paste contamination, template reuse
- **Speed**: Very fast, O(n) processing
- **Memory**: Low memory footprint
- **Accuracy**: High precision for exact/near-exact matches

### 🧬 [TOXIC Detection](toxic.md) (`mode: toxic`)
Semantic contamination detection using word embeddings and poison tokens.
- **Best for**: Paraphrased content, semantic similarity, cross-domain leakage
- **Speed**: Moderate, requires embedding computation
- **Memory**: Bounded vocabulary, scalable to large datasets
- **Accuracy**: Detects subtle semantic contamination that MinHash misses

### 🔍 [SIMPLE Detection](simple.md) (`mode: simple`)
Efficient n-gram matching with intelligent sampling and cluster expansion.
- **Best for**: Large-scale exact contamination, performance-critical scenarios
- **Speed**: Fast with configurable sampling, efficient parallel processing
- **Memory**: Indexes only evaluation data, streams training data
- **Accuracy**: High precision for substantial overlaps, tunable via sampling rate

**Choose your method based on your contamination concerns:**
- Use **MinHash** for fast exact-match detection in large datasets
- Use **TOXIC** when concerned about paraphrasing and semantic leakage
- Use **SIMPLE** for efficient large-scale detection with sampling trade-offs
- Run multiple methods for comprehensive coverage

## Installation

### Prerequisites
- Rust 1.70+
- Sufficient disk space for your datasets

### Build from Source
```bash
git clone https://github.com/your-org/minhash-rs.git
cd minhash-rs
cargo build --release
```

### AWS EC2 Setup (Recommended for Large Datasets)
For large-scale contamination detection, we recommend using EC2 instances with fast local storage:

```bash
# Set up RAID 0 across NVMe drives for maximum I/O
sudo yum install mdadm -y
sudo mdadm --create /dev/md0 --level=0 --raid-devices=8 /dev/nvme1n1 /dev/nvme2n1 /dev/nvme3n1 /dev/nvme4n1 /dev/nvme5n1 /dev/nvme6n1 /dev/nvme7n1 /dev/nvme8n1
sudo mkfs.xfs /dev/md0
sudo mkdir /mnt/raid0
sudo mount /dev/md0 /mnt/raid0
sudo chown -R $USER /mnt/raid0

# Install dependencies
sudo yum install gcc cmake openssl-devel g++ -y

# Download data efficiently
wget https://github.com/peak/s5cmd/releases/download/v2.2.2/s5cmd_2.2.2_Linux-64bit.tar.gz
tar -xvzf s5cmd_2.2.2_Linux-64bit.tar.gz
sudo mv s5cmd /usr/local/bin

# Download your datasets
s5cmd cp -sp s3://bucket/path/to/training/* /mnt/raid0/training_data
s5cmd cp -sp s3://bucket/path/to/eval/* /mnt/raid0/eval_data
```

## Quick Start

### 1. Prepare Your Data
Organize your data into two directories:
```
/path/to/training/     # Training datasets (.jsonl files)
/path/to/evaluation/   # Evaluation datasets (.jsonl files)
```

Use the `python/download_evals.py` script to download and normalize evaluation datasets based on an eval config. See `examples/decontamination/eval_datasets.yaml`.

### 2. Create Configuration
Create a `config.yaml` file. Choose your detection method:

**MinHash Configuration (fast exact matching):**
```yaml
mode: minhash
content_key: text
local_input: /path/to/training
reference_input: /path/to/evaluation
output_dir: /path/to/results

# LSH Parameters
band_size: 8
ngram_size: 3
num_bands: 7
tokenizer_str: uniseg
jaccard_similarity_threshold: 0.8
```

**TOXIC Configuration (semantic detection):**
```yaml
mode: toxic
content_key: text
local_input: /path/to/training
reference_input: /path/to/evaluation
output_dir: /path/to/results

# TOXIC Parameters
ngram_size: 4
toxic_embedding_path: /path/to/wiki-news-300d-1M.vec
toxic_hyperplanes: 64
toxic_overlap_threshold: 0.3
```

**SIMPLE Configuration (efficient sampled detection):**
```yaml
mode: simple
content_key: text
local_input: /path/to/training
reference_input: /path/to/evaluation
output_dir: /path/to/results

# SIMPLE Parameters
ngram_size: 13
sample_every_m_tokens: 50
max_consecutive_misses: 3
toxic_overlap_threshold: 0.5
toxic_score_threshold: 0.5
tokenizer_str: cl100k
```

### 3. Run Contamination Detection
```bash
cargo run --release -- contamination-detect --config config.yaml
```

### 4. Review Results
```bash
# View summary (filename depends on detection method)
cat /path/to/results/contamination_results.jsonl      # MinHash results
cat /path/to/results/toxic_contamination_results.jsonl # TOXIC results

# Interactive review with side-by-side comparison
cargo run --release -- review-contamination --config config.yaml
```

## Configuration

### Shared Parameters

| Parameter | Description | Default | Notes |
|-----------|-------------|---------|-------|
| `mode` | Detection method | `minhash` | `minhash`, `toxic`, or `simple` |
| `content_key` | JSON field containing text | `text` | Supports nested keys like `data.content` |
| `local_input` | Training data directory | - | Path to training datasets |
| `reference_input` | Evaluation data directory | - | Path to evaluation datasets |
| `output_dir` | Results output directory | - | Where to save contamination results |
| `ngram_size` | N-gram window size | 3 | Larger = more precise, smaller = more sensitive |

### MinHash-Specific Parameters

| Parameter | Description | Default | Notes |
|-----------|-------------|---------|-------|
| `jaccard_similarity_threshold` | Minimum similarity to report | 0.5 | Higher = stricter matching |
| `num_bands` | Number of LSH bands | 7 | Fewer = more sensitive |
| `band_size` | Hash functions per band | 8 | More = more precise |
| `tokenizer_str` | Tokenization method | `uniseg` | `p50k`, `cl100k`, `uniseg`, or default |
| `exact_override` | Force exact Jaccard computation | `false` | `true` for higher accuracy |

### TOXIC-Specific Parameters

| Parameter | Description | Default | Notes |
|-----------|-------------|---------|-------|
| `toxic_embedding_path` | Path to word vectors | - | FastText format (e.g., wiki-news-300d-1M.vec) |
| `toxic_hyperplanes` | Number of LSH hyperplanes | 64 | More = more precise buckets |
| `toxic_overlap_threshold` | Minimum overlap ratio | 0.3 | Lower = more sensitive |
| `toxic_poison_scale` | Poison token amplification | 3.0 | Higher = more destructive to false matches |

### SIMPLE-Specific Parameters

| Parameter | Description | Default | Notes |
|-----------|-------------|---------|-------|
| `sample_every_m_tokens` | Sampling rate for training data | 50 | Lower = more thorough but slower |
| `max_consecutive_misses` | Gap tolerance during expansion | 3 | Higher = more permissive clustering |
| `toxic_overlap_threshold` | Minimum overlap ratio | 0.5 | Used for document-level filtering |
| `toxic_score_threshold` | Minimum toxic score | 0.5 | IDF-based contamination score |

### Performance Tuning

**MinHash - For maximum sensitivity** (catch more near-duplicates):
```yaml
mode: minhash
num_bands: 4                    # Fewer bands = more permissive
jaccard_similarity_threshold: 0.7  # Lower threshold
ngram_size: 3                   # Phrase-level matching
```

**MinHash - For maximum precision** (fewer false positives):
```yaml
mode: minhash
num_bands: 14                   # More bands = stricter matching
jaccard_similarity_threshold: 0.9  # Higher threshold
ngram_size: 1                   # Word-level matching
```

**TOXIC - For maximum sensitivity** (catch paraphrases):
```yaml
mode: toxic
ngram_size: 3                   # Shorter n-grams
toxic_overlap_threshold: 0.2    # Lower threshold
toxic_hyperplanes: 32           # Fewer hyperplanes
```

**TOXIC - For maximum precision** (reduce false positives):
```yaml
mode: toxic
ngram_size: 5                   # Longer n-grams
toxic_overlap_threshold: 0.4    # Higher threshold
toxic_hyperplanes: 128          # More hyperplanes
```

## Commands

### `contamination-detect`
Performs contamination detection between training and evaluation datasets.

```bash
cargo run --release -- contamination-detect --config config.yaml
```

**Output**: Creates results file with detected contamination instances:
- MinHash: `contamination_results.jsonl`
- TOXIC: `toxic_contamination_results.jsonl`
- SIMPLE: `simple_contamination_results.jsonl`

### `review-contamination`
Interactive tool for reviewing detected contamination with side-by-side text comparison.

```bash
# Use default results file
cargo run --release -- review-contamination --config config.yaml

# Use custom results file
cargo run --release -- review-contamination --config config.yaml --results-file custom_results.jsonl
```

## Understanding Results

### Results Format
Each detected contamination is saved as a JSON line:

**MinHash Results:**
```json
{
  "training_file": "train_dataset",
  "training_line": 42,
  "eval_dataset": "gsm8k_test", 
  "eval_line": 1,
  "jaccard_similarity": 0.987,
  "method": "minhash"
}
```

**TOXIC Results:**
```json
{
  "training_file": "train_dataset",
  "training_line": 42,
  "eval_dataset": "gsm8k_test",
  "eval_line": 1, 
  "overlap_ratio": 0.673,
  "method": "toxic"
}
```

### Review Tool Output
The review tool shows side-by-side comparison:

```
================================================================================
CONTAMINATION #1 of 3
================================================================================
📁 TRAINING FILE: train_dataset
📋 EVAL DATASET:  gsm8k_test
🎯 OVERLAP RATIO:   0.673

🔍 TRAINING TEXT (line 42):
   "Once upon a time in a magical kingdom, a wise tailor discovered that a robe takes 2 bolts of blue fiber and half that much white fiber. The question that puzzled the tailor was: how many bolts in total does this require?"

🔍 EVAL TEXT (line 1):
   "A robe takes 2 bolts of blue fiber and half that much white fiber. How many bolts in total?"

⚠️  HIGH SIMILARITY - Likely contamination
```
