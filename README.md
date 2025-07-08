# Contamination Detection for ML Datasets

Contamination detection tools for training datasets. Identifies when training data contains text that appears in evaluation datasets using multiple detection approaches. Supports both exact matching and semantic similarity detection.

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

### 🔍 [SIMPLE Detection](simple.md) (`mode: simple`)
Efficient n-gram matching with intelligent sampling and cluster expansion.
- **Best for**: Large-scale exact contamination, performance-critical scenarios
- **Speed**: Fast with configurable sampling, efficient parallel processing
- **Memory**: Indexes only evaluation data, streams training data
- **Accuracy**: High precision for substantial overlaps, tunable via sampling rate

### 📊 [Windowed MinHash Detection](minhash.md) (`mode: minhash`)
Memory-efficient detection using Jaccard similarity and LSH.
- WIP.
- Needs sliding window approach tuned, or other adaptations for dissimilar set sizes.

### 🧬 [TOXIC Detection](toxic.md) (`mode: toxic`)
Semantic contamination detection using word embeddings and poison tokens.
- WIP
- **Best for**: Paraphrased content, semantic similarity, cross-domain leakage
- **Speed**: Moderate, requires embedding computation
- **Memory**: Bounded vocabulary, scalable to large datasets
- **Accuracy**: Detects subtle semantic contamination that MinHash misses


**Use SIMPLE as it is tested and complete**

## Installation

### Prerequisites
- Rust 1.70+
- Sufficient disk space for your datasets

### Build from Source
```bash
git clone https://github.com/allenai/decon
cd decon
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
Create a `config.yaml` file.

See /examples/eval.

### 3. Run Contamination Detection
```bash
cargo run --release detect --config config.yaml
```

### 4. Review Results
```bash
# View summary (filename depends on detection method)
cat /path/to/results/contamination_results.jsonl      # MinHash results
cat /path/to/results/toxic_contamination_results.jsonl # TOXIC results

# Interactive review with side-by-side comparison
cargo run --release review --config config.yaml
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
| `purify` | Create cleaned files | `false` | Remove contaminated lines from training data |
| `cleaned_file_output` | Cleaned files directory | `output_dir` | Optional separate directory for purified files |

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

## Data Purification

The tool can automatically create cleaned versions of your training data with contaminated lines removed. This is useful for:
- Creating decontaminated training datasets
- Removing evaluation data that leaked into training
- Ensuring clean model training

### Enabling Purification

Add these parameters to your configuration:

```yaml
# Enable purification
purify: true
```

### Purification Process

When `purify: true` is set:
1. Detection runs normally and saves contamination results
2. For each training file with contamination:
   - Creates a new file with `.clean.jsonl` suffix
   - Copies all non-contaminated lines
   - Skips lines flagged as contaminated
3. Reports how many lines were removed from each file

### Example

```yaml
# config.yaml
mode: simple
local_input: /data/training
reference_input: /data/evaluation
output_dir: /results/contamination
purify: true
```

Running this will:
- Save contamination results to `/results/contamination/simple_contamination_results.jsonl`
- For example: `train_001.jsonl` → `train_001.clean.jsonl` (with contaminated lines removed)

### Orchestration

The decontamination tool supports distributed processing through orchestration and daemon mode, enabling efficient contamination detection at scale.

#### Daemon Mode
The daemon runs as an HTTP server that processes contamination detection jobs asynchronously:
- Pre-builds the reference index once at startup for faster processing
- Provides REST API endpoints for job submission and status monitoring
- Uses a configurable worker thread pool for parallel processing
- Supports all detection modes (Simple, MinHash, TOXIC)

#### Orchestration Layer
A Python orchestration script manages distributed processing across multiple hosts:
- Downloads training files from S3 in batches
- Distributes work across hosts using consistent hashing
- Submits files to local daemon for processing
- Uploads results (contamination reports and purified files) back to S3
- Tracks progress and handles retries automatically

#### Running Orchestration

**1. Start the daemon on each host:**
```bash
# Using make
make daemon

# Or directly with cargo
cargo run --release daemon --config config.yaml
```

**2. Run orchestration (example for host 1 of 3):**
```bash
export PMR_HOST_INDEX=0
export PMR_HOST_COUNT=3
python python/orchestration.py --config orchestration.yaml
```

**3. Monitor progress:**
The orchestrator will show progress and the daemon provides health/status endpoints:
- `GET /health` - Check daemon health
- `GET /status/:job_id` - Get job results

#### API Response Example
```json
{
  "job_id": "123e4567-e89b-12d3-a456-426614174000",
  "status": "completed",
  "output_path": "fixtures/output/training-simple-0.80.jsonl",
  "purified_path": "fixtures/cleaned/training.clean.jsonl"
}
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
