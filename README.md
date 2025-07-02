# MinHash Contamination Detection

Fast MinHash-based contamination detection for training datasets. Identifies when training data contains text that appears in evaluation datasets using Locality Sensitive Hashing (LSH) and Jaccard similarity. Work in progress.

## Table of Contents
- [Overview](#overview)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Commands](#commands)
- [Understanding Results](#understanding-results)

## Overview

**Contamination detection** is critical for machine learning evaluation. When training data accidentally contains examples from evaluation datasets, it can lead to artificially inflated performance metrics and unreliable model evaluation.

This tool uses **MinHash signatures** and **Locality Sensitive Hashing (LSH)** to efficiently detect:
- **Exact matches**: Identical text between training and evaluation data
- **Near-duplicates**: Text with minor modifications (typos, formatting, small edits)
- **Partial containment**: When evaluation text is contained within longer training examples

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

Each `.jsonl` file should contain one JSON object per line with a `text` field:
```json
{"text": "Your training example text here"}
{"text": "Another training example"}
```

### 2. Create Configuration
Create a `config.yaml` file:

```yaml
name: my-contamination-check
content_key: text
local_input: /path/to/training
reference_input: /path/to/evaluation
working_dir: /tmp/work
output_dir: /path/to/results

# LSH Parameters (adjust for sensitivity vs. speed)
band_size: 8
ngram_size: 1
num_bands: 7

# Required settings
num_docs: 1000
num_sig_chunks: 2
max_lines_per_path: 100
tokenizer_str: uniseg
annotate_only: true
exact_override: true

# Similarity threshold (0.0-1.0)
jaccard_similarity_threshold: 0.8
```

### 3. Run Contamination Detection
```bash
cargo run --release -- contamination-detect --config config.yaml
```

### 4. Review Results
```bash
# View summary
cat /path/to/results/contamination_results.jsonl

# Interactive review with side-by-side comparison
cargo run --release -- review-contamination --config config.yaml
```

## Configuration

### Core Parameters

| Parameter | Description | Default | Notes |
|-----------|-------------|---------|-------|
| `jaccard_similarity_threshold` | Minimum similarity to report | 0.8 | Higher = stricter matching |
| `num_bands` | Number of LSH bands | 7 | Fewer = more sensitive |
| `band_size` | Hash functions per band | 8 | More = more precise |
| `ngram_size` | N-gram size for tokenization | 1 | 1=words, 3=phrases |

### Sensitivity Tuning

**For maximum sensitivity** (catch more near-duplicates):
```yaml
num_bands: 4                    # Fewer bands = more permissive
jaccard_similarity_threshold: 0.7  # Lower threshold
```

**For maximum precision** (fewer false positives):
```yaml
num_bands: 14                   # More bands = stricter matching
jaccard_similarity_threshold: 0.9  # Higher threshold
```

## Commands

### `contamination-detect`
Performs contamination detection between training and evaluation datasets.

```bash
cargo run --release -- contamination-detect --config config.yaml
```

**Output**: Creates `contamination_results.jsonl` with detected contamination instances.

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

```json
{
  "training_file": "train_dataset",
  "training_line": 42,
  "eval_dataset": "gsm8k_test",
  "eval_line": 1,
  "jaccard_similarity": 0.987
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
🎯 JACCARD SIM:   0.987

🔍 TRAINING TEXT (line 42):
   "A robe takes 2 bolts of fiber and half that much white fiber. How many bolts in total?"

🔍 EVAL TEXT (line 1):
   "A robe takes 2 bolts of blue fiber and half that much white fiber. How many bolts in total?"

⚠️  VERY HIGH SIMILARITY - Likely contamination
```
