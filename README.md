# Contamination Detection

Decon identifies documents contaminated with eval instances.

It uses [simple](doc/simple.md) token based sampling and counting methods, making it suitable for large datasets. It is deterministic with interpretable results.

Decon can produce contamination reports and cleaned datasets with contamination removed.

## Contamination Example

> TRAINING DOC:
>
>   "...  for θ 30 c i θ i0 4 for θ 90 d i θ is constant for all values of θ **the plane face of plano convex lens of focal**
> **length 20 cm is silvered this combination is equivalent to the type of mirror and its focal length is** a convex f 20 c
> m b **concave f**  20 cm in a displacement method using convex lens two images are obtained for a separation of d between ..."
>
>EVAL TEXT:
>
>   PROMPT: the plane face of plano convex lens of focal length 20 cm is silvered this combination is equivalent to the type of mirror and its focal length is
>
>   ANSWER: concave f 10 cm

## Quick Start

### Input Data

To prepare a dataset for decontamination, create a directory of jsonl files. Each JSON object in the files should have a single key with a string value for a training document. [[example]](tests/fixtures/training/contaminated_mixed.jsonl).

Decon also expects eval references to be jsonl files in a directory. Decon reference files have a simple normalized format with passage, question, answer keys which can be generated from hf datasets with included tooling. A small reference set is included by default to get started. [[example](bundled-evals/)]

### CLI

```bash
# Clone and build
git clone https://github.com/allenai/decon
cd decon
cargo build --release

# Run contamination detection.
target/release/decon detect --training-dir /path/to/your/training-files

# Create a clean copy (contaminated documents removed) of your dataset
target/release/decon detect --training-dir /path/to/your/training-files --purify

# List current eval datasets in reference (small default set initially)
target/release/decon evals

# For full set of commands and options, help is available.
target/release/decon --help
```

Sensible defaults are provided for [decon parameters](config/default.yaml), with a single `contamination_score_threshold` that can be adjusted to desired sensitivity. Experimenting with these parameters on your own dataset and eval set is recommended.

#### Example Run

The following run was performed on a 30GB sample web dataset with an AMD Ryzen 9 9950X 16-Core Processor.

```
$ decon detect --training-dir ~/sample-data --evals-dir ~/references

Reference files 703/703 [00:00:29/00:00:00] [███████████████████████████████████████]

┌──────────────────────────────────────────────────────────────────┐
│                  Reference Index Building Summary                │
├──────────────────────────────────────────────────────────────────┤
│   Total lines examined                                 3,176,410 │
│   Lines indexed                                        1,577,383 │
│   Lines skipped                                        1,599,027 │
│     - Duplicates                                          59,456 │
│     - Below minimum tokens                             1,539,571 │
├──────────────────────────────────────────────────────────────────┤
│   Unique n-grams                                      33,624,313 │
│   Hot n-grams                                                309 │
│   Eval datasets                                       201 suites │
├──────────────────────────────────────────────────────────────────┤
│   Build time                                              38.59s │
└──────────────────────────────────────────────────────────────────┘

Training files 4,487/4,487 [00:02:55/00:00:00] [████████████████████████████████████]

┌───────────────────────────────────────────┐
│     Contamination Detection Results       │
├───────────────────────────────────────────┤
│ Training lines                  5,162,084 │
│ Processing rate                 34 μs/doc │
├───────────────────────────────────────────┤
│ Index building time                38.59s │
│ Detection time                    175.69s │
│ Total time                        214.28s │
├───────────────────────────────────────────┤
│ Contaminated matches                7,699 │
│ Contaminated documents              1,851 │
└───────────────────────────────────────────┘
```

### Server

Decon can also be run as a server to facilitate distributing workloads.

```bash
# Launch a server
decon server --port 8080
```

An example [orchestration script](python/orchestration.py) is provided which demonstrates one approach to batch retrieve a partition of documents, submit documents to the server, poll for job status, and upload reports and clean documents to a new location.

See [deployment guide](doc/deployment.md) for more details.

## Reviewing Results

Decon includes tools for qualitative review and basic stats which can be filtered to analyze contamination.

```bash
# To qualitatively review individual matches
decon review /my-results-directory

# To see statistics
decon review --stats /my-results-directory

# To review with filters, e.g. specific eval with minimum score
decon review /my-results-directory --eval mmlu --min-score 0.9

# Compare results between different decontamination runs
decon compare /tmp/results-a /tmp/results-b
```

Decon reports are jsonl files which are ready for analysis beyond the provided tooling.

#### Example Run

```
$ decon review --stats /tmp/decon-ffc31fde

Summary:
  Training docs contaminated: 1367
  Total contamination instances: 6702
  Unique eval instances: 1338

=== TRAINING DOCUMENTS CONTAMINATED BY EVAL SUITE ===
(Each count represents unique training documents that need removal)

  sciq                                  652 │███████████████████████████████████████│
  mmlu                                  278 │█████████████████████                  │
  mmlu_pro                              211 │████████████████                       │
  ai2_arc_easy                           83 │██████                                 │
  super_gpqa                             65 │████                                   │
  ai2_arc_challenge                      62 │████                                   │
  medmcqa                                53 │████                                   │
  jeopardy                               43 │███                                    │
  agi_eval_aqua_rat                      30 │██                                     │
  trivia_qa                              21 │█                                      │
  ...

  === CONTAMINATED EVAL INSTANCES BY SUITE ===
(Unique eval examples found in training data)

  medmcqa                               536 │███████████████████████████████████████│
  mmlu                                  226 │█████████████████████                  │
  mmlu_pro                              131 │████████████                           │
  sciq                                   76 │███████                                │
  ai2_arc_easy                           44 │████                                   │
  trivia_qa                              44 │████                                   │
  super_gpqa                             35 │███                                    │
  jeopardy                               26 │██                                     │
  ai2_arc_challenge                      26 │██                                     │
  agi_eval_aqua_rat                      11 │█                                      |
  ...

```

## Evaluation Datasets

Three eval suites are included in the reference dataset by default, gsm8k, mmlu, and agi_eval.

It's likely you will want to build your own reference set with your evals of interest.

The `decon evals` command can process an extensible [declarative yaml file](config/evals.yaml) to normalize huggingface datasets.

To download all the pre-configured evals included in the configuration file, run the following command. This requires python3 with the datasets library installed.

```
decon evals --download
```

See the [Evaluation Dataset Guide](doc/eval-datasets.md) for more information on preparing evaluation datasets.
