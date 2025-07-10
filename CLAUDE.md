# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Decon is a contamination detection tool for machine learning datasets that identifies when training data contains text from evaluation datasets. It provides three detection algorithms: SIMPLE (n-gram matching), MinHash (near-duplicate detection), and TOXIC (semantic similarity).

## Common Development Commands

### Building and Running
- `cargo build --release` - Build optimized binary
- `cargo run --release detect --config <config.yaml>` - Run contamination detection
- `cargo run --release review --config <config.yaml> --step` - Interactive review mode
- `cargo run --release daemon --config <config.yaml> --port 8080` - Start daemon server

### Testing and Development
- `cargo fmt` - Format Rust code
- `cargo clippy` - Run Rust linter
- `cargo test` - Run unit tests
- `./test_daemon.sh` - Test daemon functionality
- `./test_multi_thread.sh` - Test multi-threading

### Make Targets
- `make simple/minhash/toxic` - Run detection with example configs
- `make review` - Interactive contamination review
- `make tp/tn/fp/fn` - Review true/false positives/negatives
- `make orchestrate CONFIG=<path>` - Run distributed orchestration
- `make daemon` - Start daemon server
- `make evals` - Download evaluation datasets
- `make embeddings` - Prepare word embeddings

### Python Orchestration
- `python python/orchestration.py --config <config.yaml>` - Run distributed processing
- `python python/download_evals.py` - Download evaluation datasets
- `python python/prepare_embeddings.py` - Prepare embeddings for TOXIC mode

## Architecture Overview

### Core Components

1. **Detection Algorithms** (src/)
   - `simple.rs`: SIMPLE n-gram matching with sampling and cluster expansion
   - `minhash.rs`: MinHash LSH-based near-duplicate detection (WIP)
   - `toxic.rs`: TOXIC semantic similarity using word embeddings (WIP)

2. **Entry Points**
   - `main.rs`: CLI with detect/review/daemon commands
   - `daemon.rs`: HTTP server for distributed processing at port 8080

3. **Data Flow**
   - Evaluation data is indexed in memory
   - Training data is streamed for memory efficiency
   - Results written to `report_output_dir` as JSON
   - Cleaned data optionally written to `cleaned_output_dir`

4. **Distributed Processing**
   - Python orchestration layer (`python/orchestration.py`) coordinates multiple daemons
   - Consistent hashing distributes work across nodes
   - S3 integration for cloud-scale processing

### Configuration Structure

Key configuration parameters:
- `mode`: Detection algorithm (simple/minhash/toxic)
- `content_key`: JSON field containing text (default: "text")
- `local_input`: Training data directory
- `reference_input`: Evaluation data directory
- `report_output_dir`: Where to write contamination reports
- `cleaned_output_dir`: Where to write decontaminated data
- `purify`: Whether to create cleaned datasets

Algorithm-specific:
- SIMPLE: `ngram_size`, `sample_every_m_tokens`, `max_consecutive_misses`
- MinHash: `signature_size`, `num_bands`, `jaccard_threshold`
- TOXIC: `poison_ratio`, `embedding_type`, `semantic_threshold`

### Key Design Decisions

1. **Memory Efficiency**: Only evaluation data is indexed; training data is streamed
2. **Parallelism**: Uses Rayon for parallel processing within files
3. **Scalability**: Daemon mode enables distributed processing across machines
4. **Flexibility**: Multiple detection algorithms for different contamination types
5. **Performance**: Configurable sampling rates balance speed vs accuracy

Note: SIMPLE is the only fully tested and complete detection method. MinHash and TOXIC are work in progress.

### Changes

Please favor simplicity and focused changes.
