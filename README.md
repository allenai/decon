# Contamination Detection

Contamination detection tools for training datasets. Identifies when training data contains text that appears in evaluation datasets using multiple detection approaches.

## Overview

This tool provides three **prototype-grade** complementary detection approaches to efficiently identify contamination.

You can use this tool to produce **report files** which highlight potential contamination and a **clean copy** of your dataset.

A typical workflow is to run the tool to [generate reports](#-running-decontamination-on-aws-with-poormanray) then [review the reports](#reviewing-results) to determine appropriate thresholds for your contamination preferences.

## Quick Start

### 🚀 Running Decontamination on AWS with poormanray

The fastest way to run large-scale decontamination is using the deployment wizard which generates `poormanray` commands:

```bash
# Clone the repository
git clone https://github.com/allenai/decon
cd decon

# Run the deployment wizard
make deploy-wizard
```

See the [Configuration Guide](doc/configuration.md) for customizing detection parameters.

### Running with CLI

For those that prefer explicitness, directness, and less abstraction, you can run the tool directly.

```bash
# Clone and build
git clone https://github.com/allenai/decon
cd decon

# Download evaluation datasets. The cli will look for these in the configured reference (evals) directory. Below shows the defaults (recommended).
s5cmd sync s3://decon-evals/* fixtures/reference

# Download your data set to the directory of your choice.
# NOTE: you will probably want a different location, this is just the default.
s5cmd sync s3://your-data-set-prefix fixtures/local_input

# Run contamination detection.
cargo run --release detect --config examples/simple.yaml

# For full set of options, help is available.
# Note that the options mix all the different modes (I'll clean this up eventually)
# Also note that each has a sensible default. Performance and outcomes may vary wildly depending on options.
cargo run --release detect --help
```

If you want to manually run the orchestrator to manage downloading, uploading, and running against a persistent server with the index, please study the configuration flag outputs from `make deploy-wizard`. Author anticipates CLI focused users would prefer to manage non-contamination details on their own.

## Reviewing Results

Let's say you've got an s3 bucket which contains report output files. Basic convenient review follows.

```
s5cmd sync s3://ai2-decon-reports/big-reasoning-traces/* /my-results-directory

# To review individual matches.
cd decon
make review /my-results-directory

# To see statistics by eval
make stats /my-results-directory

```

### Tuning

With a batch of results, you might want to experiment with alternative thresholds.

Filtering options are available.

```
# Stats with all three filters combined
cargo run --release -- review --stats --dir /my-results-directory --min-overlap-ratio 0.3 --min-idf-score 1.5 --min-length 8

# Stats with aggressive filtering (high thresholds)
cargo run --release -- review --step --dir /my-results-directory --min-overlap-ratio 0.9

# Restrict to a specific eval
cargo run --release -- review --eval mmlu --min-overlap-ratio 0.99  --step --dir /my-results-directory

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

See the [Configuration Guide](doc/configuration.md) for detailed information about all available options.
