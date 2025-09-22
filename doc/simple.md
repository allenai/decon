# Sampled Index Match with Progressive Length Extension

SIMPLE detects contamination between training and evaluation datasets using sampled n-gram matching with bidirectional cluster expansion and set overlap tests.

## Core Approach

SIMPLE operates in two phases:

1. **Index Construction**: Build n-gram indices of evaluation data with IDF weighting
2. **Detection**: Sample training data for matches, expand clusters around hits, and score contamination

## Reference Index Construction

The reference index processes evaluation datasets to create searchable n-gram indices:

- **Two-tier lookup**: N-gram hash → ID → document sets (optimized for sparsity of initial matches)
- **Component separation**: Questions, answers, and passages indexed independently
- **IDF pre-computation**: Document frequencies calculated and cached for most n-grams
- **Hot bucket optimization**: High-frequency n-grams replaced with sentinel ID with lazily evaluation of large document sets

## Detection Process

### Sampling Strategy
Training data is processed with configurable sampling intervals (`sample_every_m_tokens`). When an n-gram matches the reference index, the algorithm switches from sampling to exhaustive cluster expansion.

### Bidirectional Cluster Expansion
Upon finding a match:
1. Initialize tracking for all documents containing the matched n-gram
2. Expand left and right from the hit position
3. At each step, compute the intersection of the set of document ids matching the current n-gram and the current set of surviving documents ids
4. Remove documents that accumulate too many consecutive misses (`question_max_consecutive_misses`)
5. Continue until all documents are eliminated or boundaries reached

## Scoring System

### IDF-Weighted Overlap
Contamination scoring uses inverse document frequency (IDF) weighting:

```math
\frac{\sum_{x \in U_t \cap U_e} \text{idf}(x)}{\sum_{y \in U_e} \text{idf}(y)}
```

where U<sub>t</sub> is the set of unique n-grams in the training document segment and U<sub>e</sub> is the set of unique n-grams in the evaluation document.


### Component Weighting
Scores combine question, answer, and passage overlaps with adaptive weights:
- **QAP** (all components): 0.7 question, 0.2 answer, 0.1 passage
- **QA** (no passage): 0.75 question, 0.25 answer
- **QP** (no answer): 0.85 question, 0.15 passage
- **Q** (question only): 1.0 question

Weights adjust based on component confidence (e.g., short questions redistribute weight to answers/passages).

### Answer Proximity
For QA datasets, contamination requires the answer appears near the question cluster. Short answers use exact token matching; long answers use n-gram overlap with IDF weighting.

### Passage Proximity
For datasets with passages, contamination checks if the passage appears within a configurable distance (`min_passage_distance`) from the question cluster. Passages use n-gram overlap with IDF weighting and can tolerate gaps (`passage_max_consecutive_misses`).

### Length-Based Thresholds
Short texts require higher scores through interpolated thresholds to minimize false positives on common texts with small numbers of inserts, edits, or deletes:
- Texts below `perfect_match_decay_start` tokens need perfect matches
- Scores interpolate linearly to normal threshold at `perfect_match_decay_end` tokens
