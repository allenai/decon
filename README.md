# Contamination Detection

Contamination detection tools for training datasets. Identifies when training data contains text that appears in evaluation datasets using multiple detection approaches.

## Overview

This tool provides three **prototype-grade** complementary detection approaches to efficiently identify contamination.

You can use this tool to produce **report files** which highlight potential contamination and a **clean copy** of your dataset.

A typical workflow is to run the tool to [generate reports](#-running-decontamination-on-aws-with-poormanray) then [review the reports](#reviewing-results) to determine appropriate thresholds for your contamination preferences.

## Quick Start

### 🚀 Running Decontamination on AWS with poormanray

The fastest way to run large-scale decontamination is using the poormanray-command-generator which generates `poormanray` commands:

```bash
# Clone the repository
git clone https://github.com/allenai/decon
cd decon

make poormanray-command-generator
```

See the [Configuration Guide](doc/configuration.md) for customizing detection parameters.

### Running with CLI

For those that prefer explicitness, directness, and less abstraction, you can run the tool directly.

```bash
# Clone and build
git clone https://github.com/allenai/decon
cd decon
cargo build --release
# optional, add target/release/ to your path to invoke decon directly.

# Download evaluation datasets. The cli will look for these in the configured reference directory.
# The datasets use "best-available" format - includes answers when available
s5cmd sync s3://decon-evals/* fixtures/

# Download your data set to the directory of your choice.
# NOTE: you will probably want a different location, this is just the default.
s5cmd sync s3://your-data-set-prefix fixtures/local_input

# Run contamination detection.
target/release/decon detect --config config/default.yaml

# Pass the purify flag to write a decontaminated copy of your dataset to the
# configured cleaned_output_dir.
target/release/decon detect --config config/default.yaml --purify

# For full set of options, help is available.
# Note that the options mix all the different modes (I'll clean this up eventually)
# Also note that each has a sensible default. Performance and outcomes may vary wildly depending on options.
target/release/decon detect --help
```

If you want to manually run the orchestrator to manage downloading, uploading, and running against a persistent server with the index, please study the configuration flag outputs from `make poormanray-command-generator`. Author anticipates CLI focused users would prefer to manage non-contamination details on their own.

## Reviewing Results

Let's say you've got an s3 bucket which contains report output files. Basic convenient review follows.

```
s5cmd sync s3://ai2-decon-reports/big-reasoning-traces/* /my-results-directory

# To review individual matches.
cd decon
cargo build --release
target/release/decon review  /my-results-directory

# To see statistics by eval
target/release/decon review --stats --eval mmlu /my-results-directory

```

### Tuning

With a batch of results, you might want to experiment with alternative thresholds.

Filtering options are available.

```
# Stats with all three filters combined
decon review /my-results-directory

# Interactive review with aggressive filtering (high thresholds)
decon review /my-results-directory --stats

# Restrict to a specific eval
decon review /my-results-directory --eval mmlu --min-overlap-ratio 0.9

```

## Evaluation Datasets

Preparing high-quality evaluation datasets is crucial for accurate contamination detection. See the [Evaluation Dataset Guide](doc/eval-datasets.md) for information on downloading, examining, and refining evaluation datasets.
