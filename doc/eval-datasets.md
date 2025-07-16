# Evaluation Dataset Management

This document covers downloading, examining, and post-processing evaluation datasets for contamination detection.

## Overview

Decon requires evaluation datasets (reference datasets) to detect contamination in training data. These datasets should be properly prepared and cleaned to ensure accurate detection results.

## Downloading Evaluation Datasets

### Using pre-packaged datasets

These datasets have been prepared using the tools described in the rest of this document. If you don't need to modify eval details, just use this.

```
s5cmd sync s3://decon-evals/* fixtures/
```

### Using the Download Script

In the `python/evals.py` is a big dictionary that serves as an eval configuration. As well as some generated code to extract data from the evals it references.

To generate decon eval datasets from sources directly (mostly hugging face) run the evals script.

```bash
python python/evals.py
```

### Post-Processing with refine-references

The evals script will populate temporary directories under fixtures. These datasets can then be refined using the `decon references` tool.

This can be used for the following.

- Partition the datasets into chunks for faster loading and index building.
- Removes exact duplicates
- Applies some filters like minimum characters

