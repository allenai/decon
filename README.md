# Contamination Detection

Contamination detection tools for training datasets. Identifies when training data contains text that appears in evaluation datasets using multiple detection approaches. Supports both exact matching and semantic similarity detection.

## Overview

This tool provides three **prototype-grade** complementary detection approaches to efficiently identify contamination:

- **Exact matches**: Identical text between training and evaluation data
- **Near-duplicates**: Text with minor modifications (typos, formatting, small edits)
- **Semantic similarity**: Paraphrased or reworded content with similar meaning
- **Document length asymmetry**: Contamination between documents of very different lengths

## Quick Start

### 🚀 Running Decontamination on AWS with poormanray

The fastest way to run large-scale decontamination is using our automated deployment wizard which generates `poormanray` commands:

```bash
# Clone the repository
git clone https://github.com/allenai/decon
cd decon

# Run the deployment wizard
make deploy-wizard
```

See the [Configuration Guide](doc/configuration.md) for customizing detection parameters.


### 💻 Local Development

For development or processing smaller datasets locally:

```bash
# Clone and build
git clone https://github.com/allenai/decon
cd decon
cargo build --release

# Download evaluation datasets
make evals

# Run contamination detection
make simple

# Download your dataset and configure examples/simple.yaml with the source of training data to decontaminate

# Review results interactively
make simple-review
```


## Detection Methods

### 🔍 [SIMPLE Detection](doc/simple.md) (`mode: simple`)
Efficient n-gram matching with ngram sampling and cluster expansion.
- **Best for**: Large-scale exact contamination, performance-critical scenarios
- **Speed**: Fast with configurable sampling, efficient parallel processing
- **Memory**: Indexes only evaluation data, streams training data
- **Accuracy**: High precision for substantial overlaps, tunable via sampling rate

### 📊 [Windowed MinHash Detection](doc/minhash.md) (`mode: minhash`)
Memory-efficient detection using Jaccard similarity and LSH.
- WIP
- Needs sliding window approach tuned, may not outperform simple

### 🧬 [TOXIC Detection](doc/toxic.md) (`mode: toxic`)
Semantic contamination detection using word embeddings and poison tokens.
- WIP
- Not yet satisfied with collision behavior


**Use SIMPLE as it is tested and complete**

## Configuration

See the [Configuration Guide](doc/configuration.md) for detailed information about all available options, including:

