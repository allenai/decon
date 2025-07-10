# Contamination Detection

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

This tool provides three complementary detection approaches to efficiently identify contamination:
- **Exact matches**: Identical text between training and evaluation data
- **Near-duplicates**: Text with minor modifications (typos, formatting, small edits)
- **Semantic similarity**: Paraphrased or reworded content with similar meaning
- **Document length asymmetry**: Contamination between documents of very different lengths

## Detection Methods

### 🔍 [SIMPLE Detection](doc/simple.md) (`mode: simple`)
Efficient n-gram matching with intelligent sampling and cluster expansion.
- **Best for**: Large-scale exact contamination, performance-critical scenarios
- **Speed**: Fast with configurable sampling, efficient parallel processing
- **Memory**: Indexes only evaluation data, streams training data
- **Accuracy**: High precision for substantial overlaps, tunable via sampling rate

### 📊 [Windowed MinHash Detection](doc/minhash.md) (`mode: minhash`)
Memory-efficient detection using Jaccard similarity and LSH.
- WIP.
- Needs sliding window approach tuned, or other adaptations for dissimilar set sizes.

### 🧬 [TOXIC Detection](doc/toxic.md) (`mode: toxic`)
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

## Configuration

See the [Configuration Guide](doc/configuration.md) for detailed information about all available options, including:
- Detection mode parameters
- Input/output settings
- Performance tuning
- Command-line overrides
- Orchestrator settings

## Quick Start

### 1. Prepare Your Data
Organize your data into two directories:
```
/path/to/training/     # Training datasets (.jsonl files)
/path/to/evaluation/   # Evaluation datasets (.jsonl files)
```

Use the `python/download_evals.py` script to download and normalize evaluation datasets based on an eval config. See `examples/decontamination/eval_datasets.yaml`.

### 2. Create Configuration
Create a configuration file based on your needs. See the [Configuration Guide](doc/configuration.md) for all available options and the `examples/` directory for sample configurations.

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

When enabled, the tool will:
- Save contamination results to your configured output directory
- Create cleaned versions of training files (e.g., `train_001.jsonl` → `train_001.clean.jsonl`)

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

**SIMPLE Results (with overlap text):**
```json
{
  "training_file": "train_dataset",
  "training_line": 42,
  "eval_dataset": "gsm8k_test",
  "eval_line": 1,
  "overlap_ratio": 0.85,
  "toxic_score": 3.2,
  "method": "simple",
  "contamination_start_idx": 15,
  "contamination_end_idx": 25,
  "training_overlap_text": "... quick brown 【fox jumps over the lazy dog】 and ran away ..."
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
