# Evaluation Dataset Management

Decon builds a reference index of eval instances to efficiently detect contamination.

This document describes how to prepare your evaluation data set for use with decon.

## Background

Decon expects eval references to be in a normalized format. Evals come in many shapes and sizes.

The core idea is to normalize eval instances into passages, questions, and answers. Where any given eval may have a combination of Q, Q+A, Q+P, Q+P+A. Only the question is required.

In some situations, an eval may include multiple choice answers. In general, the transform for these involves unpacking all Q+A combinations from a multiple choice into a distinct reference entry to be decontaminated against.

## Preparing Eval References

Decon reads a directory provided by the `--evals-dir` option or, if not specified, reads from a default directory which has a few pre-bundled and formatted evals.

It ingests all of the jsonl files in the directory.

**You can produce these records any way that is convenient for you, just make sure they have the appropriate keys and are placed in the provided `--evals-dir` directory.**

Decon includes some tooling for downloading and transforming hugging face datasets in a declarative fashion for convenience, discussed later.

### Record Example
Below is a concrete example of an eval reference record.

```json
{
  "eval_key": "mmlu",
  "config": "all",
  "split": "test",
  "eval_instance_index": 0,
  "doc_id": 1766765,
  "question": "Find the degree for the given field extension Q(sqrt(2), sqrt(3), sqrt(18)) over Q.",
  "answer": "0",
  "is_correct": false,
  "fingerprint": "32957242b0602972"
}
```

### Record Format

Each evaluation record is a JSON object with the following structure:

#### Required Fields

- **`eval_key`** (string, required): Dataset identifier (e.g., 'gsm8k', 'mmlu', 'humaneval')
- **`eval_instance_index`** (integer, required): Zero-based index of the example within its (eval, config, split)
- **`split`** (string, required): Dataset split name ('train', 'test', 'validation', etc.)
- **`question`** (string, required): The prompt/question text extracted from the dataset
- **`doc_id`** (integer, required): Unique document ID (incremental across all processed datasets)
- **`fingerprint`** (string, required): SHA256 hash (first 16 chars) of passage+question+answer for deduplication cross reference set comparison

#### Optional Fields

- **`config`** (string/null, optional): HuggingFace config name if applicable (e.g., 'all' for MMLU subsets)
- **`passage`** (string, optional): Supporting context or reference text for the question
- **`answer`** (string, optional): The answer or expected response
- **`sub_index`** (integer, optional): Sub-index when a single dataset example expands to multiple records (e.g., parallel Q&A arrays)
- **`is_correct`** (boolean, optional): For multiple-choice questions, indicates if this choice is the correct answer

### Default Reference Set

Decon includes [3 pre-formatted eval suites](/bundled-evals) to assist in getting started and to serve as a reference.

## Decon Tooling

To check the status of a reference set you can run the following commands.

```bash
# Display overview of the default eval reference set
$ decon evals

Reference Dataset Statistics
============================
Note: Q = Questions, A = Answers, P = Passages. Use --stats flag to show detailed length statistics.

┌───────────┬───────────┬───────────┬───────────┐
│ Eval Name │         Q │         A │         P │
├───────────┼───────────┼───────────┼───────────┤
│ agi_eval  │     13909 │     13769 │      9149 │
│ gsm8k     │      8792 │      8792 │         0 │
│ mmlu      │     63432 │     63432 │         0 │
└───────────┴───────────┴───────────┴───────────┘

Total evals: 3
Total questions: 86133
Total answers: 85993
Total passages: 9149

# Display overview of a specific non-default directory
decon evals --dir /path

# Display overview of with additional statistics on entry lengths
decon evals --stats
```

### Downloading

Decon includes an evals.yaml file that offers a declarative format for including huggingface datasets.

There are substantial transformations that are applied which are discussed in [evals.yaml](config/evals.yaml).

```bash
# To download all entries in the default yaml file
decon evals --download

# To download all entries in a custom configuration
decon evals --download --config /path/my-evals.yaml

# To download a specific key from evals yaml file
decon evals --eval mmlu

# To download evals to a specific directory for a custom reference set path
decon evals --download --output-dir /path/my-eval-set
```

### Reproducible exports

Set `hf_revision` to a full Hugging Face commit SHA in an eval entry to pin its source.
The downloader forwards it as `load_dataset(..., revision=...)` and records it in
normalized rows. Existing entries without `hf_revision` retain their unpinned behavior.
See [the supplemental config](../config/post-training-evals.yaml) for pinned AIME 2026,
HMMT February 2026, single-turn IFBench, and both VGraf IFEval multi-turn
configurations. It can be passed to `--config` like any other eval config.

Exports are staged before publication. A missing dataset, requested split, or empty
normalized split makes the command exit nonzero and leaves previous references intact.
`--download` replaces the output directory with the complete configured set only after
all exports succeed. `--eval NAME` replaces that eval's chunks and preserves other
references, numbering its `doc_id`s after the largest one already present; decon
requires `doc_id`s to be unique across a reference directory. To add a supplemental eval
to an existing set, run `--eval NAME --config` with that eval's config rather than
merging separately exported directories. The output path must be a directory, not a
symlink, and its parent must be writable so the staged directory can be published by rename.
