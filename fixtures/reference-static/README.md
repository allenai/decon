# Reference Static Datasets

This directory contains evaluation datasets that are not available on HuggingFace Hub and need to be stored locally.

## AGIEval

AGIEval (AGI Evaluation) is a benchmark for assessing foundation model AGI capabilities across various reasoning tasks.

### English Datasets:
- `lsat-ar` - LSAT Analytical Reasoning
- `lsat-lr` - LSAT Logical Reasoning  
- `lsat-rc` - LSAT Reading Comprehension
- `logiqa-en` - LogiQA English version
- `sat-math` - SAT Math
- `sat-en` - SAT English
- `aqua-rat` - AQUA-RAT (Algebra Question Answering with Rationales)
- `sat-en-without-passage` - SAT English without passage
- `gaokao-english` - Gaokao English exam

### File Format
The AGIEval datasets should be in JSONL format with the following structure:
- `passage`: Context passage (optional)
- `question`: The question text
- `options`: List of answer choices
- `label`: Correct answer label

Source: https://github.com/ruixiangcui/AGIEval