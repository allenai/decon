# Contamination Detection

Decon identifies documents contaminated with eval instances.

It uses [simple](doc/simple.md) token based sampling and counting methods, making it suitable for large datasets. It is deterministic with interpretable results.

Decon can produce contamination reports and cleaned datasets.

## How Decon Works

Consider a 30GB web dataset in `~/sample-data` that includes documents containing evaluation question text.

> TRAINING DOC:
>
>   "...  for θ 30 c i θ i0 4 for θ 90 d i θ is constant for all values of θ **the plane face of plano convex lens of focal**
> **length 20 cm is silvered this combination is equivalent to the type of mirror and its focal length is** a convex f 20 c
> m b **concave f** 20 cm in a displacement method using convex lens two images are obtained for a separation of d between ..."
>
>
>EVAL PROMPT: the plane face of plano convex lens of focal length 20 cm is silvered this combination is equivalent to the type of mirror and its focal length is
>
>EVAL ANSWER: concave f 10 cm

We can identify the contamination locations running decon.

```
$ decon detect --training-dir ~/sample-data --evals-dir ~/references

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

$ decon review --stats /tmp/decon-295c0cbd

=== TRAINING DOCUMENTS CONTAMINATED BY EVAL SUITE ===
(Each count represents unique training documents that need removal)

  sciq                                  652 │███████████████████████████████████████│
  mmlu                                  278 │█████████████████████                  │
  mmlu_pro                              211 │████████████████                       │
  ai2_arc_easy                           83 │██████                                 │
  super_gpqa                             65 │████                                   │

  ...
```

## Quick Start

### CLI

```bash
# Clone and build. Requires rust 1.88
git clone https://github.com/allenai/decon
cd decon

# For full set of commands and options, help is available.
cargo run --release -- --help

# List current eval datasets in reference (small default set initially).
# See advanced section below for instructions to curate your own reference set.
cargo run --release -- evals

# Run contamination detection.
cargo run --release -- detect --training-dir tests/fixtures/training/

# Create a clean copy (contaminated documents removed) of your dataset.
cargo run --release -- detect --training-dir tests/fixtures/training/ --purify

# Review report output. A decon detect run will report an output directory.
cargo run --release -- review /tmp/decon-output-directory
```

Sensible defaults are provided for [decon parameters](config/default.yaml), with a single `contamination_score_threshold` that can be adjusted to desired sensitivity. Experimenting with these parameters on your own dataset and eval set is recommended.

## Advanced Usage

### Preparing Datasets

#### Training Documents

Decon operates on a directory containing jsonl files.

Each JSON object in the files must contain a single field with a string value representing a training document [[example]](tests/fixtures/training/contaminated_mixed.jsonl).

#### Eval Suites

Decon runs against a reference set of eval suites that is also expected be a directory containing jsonl files [[example](bundled-evals/)].

Decon eval reference files have a simple normalized format with passage, question, answer keys which can be generated from hf datasets with included tooling. A small reference set is included by default to get started.

#### Eval Reference Set Curation

Three eval suites are included in the reference dataset by default, gsm8k, mmlu, and agi_eval.

It's likely you will want to build your own reference set with your evals of interest.

The `decon evals` command can process an extensible [declarative yaml file](config/evals.yaml) to normalize huggingface datasets.

To download all the pre-configured evals included in the configuration file, run the following command. This requires python3 with the datasets library installed.

```
# Review current set of evals in reference
cargo run --release -- evals --download

# Download and normalize all evals configured in a config file
cargo run --release -- evals --download --config config/evals.yaml
```

See the [Evaluation Dataset Guide](doc/eval-datasets.md) for more information on preparing evaluation datasets.

### Server

Decon can also be run as a server to facilitate distributing workloads.

```bash
# Launch a server
decon server --port 8080
```

An example orchestration script is provided which demonstrates one approach to batch retrieve a partition of documents, submit documents to the server, poll for job status, and upload reports and clean documents to a new location.

See [deployment guide](doc/deployment.md) for details.

### Reviewing Results

Decon includes tools for qualitative review and basic stats which can be filtered to analyze contamination.

```bash
# To qualitatively review individual matches
cargo run --release -- review /my-results-directory

# To see statistics
cargo run --release -- review --stats /my-results-directory

# To review with filters, e.g. specific eval with minimum score
cargo run --release -- review /my-results-directory --eval mmlu --min-score 0.9

# Compare results between different decontamination runs
cargo run --release -- compare /tmp/results-a /tmp/results-b
```

Decon reports are jsonl files which are ready for analysis beyond the provided tooling.
