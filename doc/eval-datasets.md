# Evaluation Dataset Management

This document covers downloading, examining, and post-processing evaluation datasets for contamination detection.

## Overview

Decon requires evaluation datasets (reference datasets) to detect contamination in training data. These datasets should be properly prepared and cleaned to ensure accurate detection results.

## Downloading Evaluation Datasets

### Using the Download Script

```bash
python python/download_evals.py
```

This script downloads common evaluation datasets and places them in the `fixtures/reference` directory. The script handles:
- Automatic downloading from HuggingFace datasets
- Conversion to JSONL format
- Gzip compression for storage efficiency

### Manual Download

For custom evaluation datasets:

1. Place JSONL files in `fixtures/reference/`
2. Each line should be a JSON object with a text field
3. Gzip compression is recommended: `gzip your_dataset.jsonl`

### Make Target

```bash
make evals
```

This downloads a standard set of evaluation datasets used for contamination detection.

## Examining Evaluation Datasets

### Dataset Structure

Each evaluation dataset should be in JSONL format with consistent field names:

```json
{"text": "The actual evaluation text content"}
{"content": "Alternative field name for text"}
{"question": "For Q&A datasets", "answer": "The answer text"}
```

### Quick Examination

To inspect a dataset:

```bash
# View first few lines
zcat fixtures/reference/dataset.jsonl.gz | head -5 | jq .

# Count total examples
zcat fixtures/reference/dataset.jsonl.gz | wc -l

# Check field structure
zcat fixtures/reference/dataset.jsonl.gz | head -1 | jq keys
```

## Post-Processing with refine-references

The `refine-references` command provides powerful dataset cleaning and deduplication capabilities.

### Running Reference Refinement

```bash
# Dry run to see what would be changed
cargo run --release refine-references --dry-run

# Actually refine the datasets
cargo run --release refine-references
```

### What Reference Refinement Does

1. **Exact Deduplication**: Removes identical text entries across all reference files
2. **Near-Duplicate Detection**: Uses MinHash to find and remove near-duplicates (>98% similar)
3. **Quality Filtering**: Removes entries that are too short (< 15 words)
4. **Cross-Dataset Deduplication**: Ensures unique content across multiple eval datasets

### Output

Refined datasets are saved to `fixtures/reference-refined/` with the same filenames as the originals.

### Example Output

```
Starting reference file refinement...
Found 12 reference files to process

Phase 1: Detecting exact duplicates...
Processing files: 100%|████████████████| 12/12

=== DUPLICATE DETECTION SUMMARY ===
Total files processed: 12
Total lines processed: 45000
Unique lines found: 42000
Duplicate lines found: 3000

Phase 2: Analyzing lines to keep after deduplication...
Lines remaining after deduplication: 42000 (removed 3000 duplicates)

Phase 3: Detecting near-duplicates with MinHash...
Computing MinHash signatures: 100%|████████████████| 12/12
Building LSH index...
Detecting near-duplicates with >98% similarity...

=== MINHASH DEDUPLICATION SUMMARY ===
Lines processed for MinHash: 42000
Near-duplicates found (>98% similar): 1500
Lines remaining after MinHash deduplication: 40500 (removed 1500 near-duplicates)

Phase 4: Analyzing filters...
Filtering files: 100%|████████████████| 12/12

=== FILTER STATISTICS ===
Lines after deduplication: 40500
Lines removed by filters:
  - min_length (<15 words): 500
Lines after all filters: 40000

Phase 5: Writing refined files...
Writing files: 100%|████████████████| 12/12

=== OVERALL SUMMARY ===
Original lines: 45000
After exact deduplication: 42000 (removed 3000 duplicates)
After MinHash deduplication: 40500 (removed 1500 near-duplicates)
After filtering: 40000 (removed 500 by filters)
Total reduction: 11.1% (5000 lines removed)
```

## Best Practices

### 1. Dataset Selection
- Include diverse evaluation datasets relevant to your domain
- Ensure datasets are high-quality and well-curated
- Consider dataset size - larger reference sets increase detection accuracy but require more memory

### 2. Pre-Processing
- Always run `refine-references` on new evaluation datasets
- Use `--dry-run` first to preview changes
- Keep original datasets as backup

### 3. Field Mapping
- Ensure your config file's `content_key` matches the field names in your eval datasets
- Common field names: "text", "content", "passage", "question", "answer"

### 4. Storage
- Keep refined datasets in `fixtures/reference-refined/`
- Use gzip compression to save space
- Document dataset sources and versions

## Troubleshooting

### Out of Memory Errors
- Split large evaluation datasets into smaller files
- Use the server mode for distributed processing
- Increase system RAM or use a machine with more memory

### Missing Text Fields
- Check dataset structure with `jq keys`
- Update `content_key` in your configuration
- Use `jq` to transform datasets to expected format

### Slow Processing
- Enable sampling with `sample_every_m_tokens` for SIMPLE mode
- Use fewer reference datasets for initial testing
- Consider using MinHash mode which is more memory-efficient

## Integration with Decon

After preparing evaluation datasets:

1. Update your config file to point to the refined datasets:
   ```yaml
   reference_input: fixtures/reference-refined
   ```

2. Run contamination detection:
   ```bash
   cargo run --release detect --config config.yaml
   ```

3. Review results:
   ```bash
   cargo run --release review --config config.yaml --step
   ```