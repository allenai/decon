# Decon Deployment Guide

This guide discusses a reference implementation for running decon across many hosts to decontaminate large datasets.

The reference implementation is designed to work on EC2 instances with data stored in S3.

In the future we hope to make this more convenient, but hope that this is straightforward to adapt.

## Overview

Decontamination is an embarrassingly parallel problem.

By partitioning a large data set such we map a given training data file to a host, we can simply process files and upload contamination reports and cleaned datasets on each host independently.

We use two components to achieve this.

#### 1. Decon Server

This is simply decon running with a simple http server which receives local filenames as post data, maintains a queue of filenames to process, returns a unique job idea to the poster, and can be queried on job status. It then processes its queue producing cleaned data files and contamination reports.

#### 2. Orchestrator

This is a [python script](/python/orchestration.py) which using s5cmd to batch download training data files to the local filesystem. It submits these files to the server, then polls for job status, and when a file has been processed it uploads the report and cleaned data to a specified destination. It uses environment variables for the total number of hosts and its own host index to select the partition of training files it will process based on a hash modulo. It also regulates the rate at which it downloads files, submits jobs, and uploads files to keep CPUs busy but not exhaust filesystem resources.

### How it works

```
┌─────────────────┐
│   S3 Input      │
│ Training Data   │
│ (JSONL files)   │
└────────┬────────┘
         │
         │ Download
         ▼
┌─────────────────────────────────────────────────────────────┐
│                      EC2 Instance(s)                        │
│                                                             │
│  ┌──────────────┐          ┌───────────────────────────┐    │
│  │ Orchestrator │ ────────▶│        Server             │    │
│  │              │  Submit  │                           │    │
│  │ • Downloads  │  Jobs    │  • Loads reference data   │    │
│  │ • Distributes│          │  • Processes files        │    │
│  │ • Uploads    │◀──────── │  • Detects contamination  │    │
│  │              │  Results │  • Creates cleaned files  │    │
│  └──────┬───────┘          └───────────────────────────┘    │
│         │                                                   │
└─────────┼───────────────────────────────────────────────────┘
          │-------------------------
          │ Upload                 | Upload
          ▼                        ▼
    ┌─────────────┐       ┌──────────────────┐
    │ S3 Reports  │       │ S3 Cleaned Files │
    │   Output    │       │    (Optional)    │
    │ (JSONL)     │       │ (JSONL.gz)       │
    └─────────────┘       └──────────────────┘
```

## Command Examples

To prepare a host running decon a variation of the following steps might be appropriate.

```bash
git clone https://github.com/allenai/decon.git
cd decon
cargo build --release
decon evals --download  # Or otherwise download a prepared directory for the reference set

pip install -r python/requirements.txt

# On a specific host record the index/count to select which files to process
export PMR_HOST_INDEX=0
export PMR_HOST_COUNT=1
```

Next launch the server which will run decontamination. Carefully select the appropriate options. While file names to decontaminate will be posted, it's important to pay particular attention to appropriate directories for evals, reports, and clean outputs.

```bash
decon server \
    --content-key text \
    --contamination-score-threshold 0.8 \
    --evals-dir fixtures/reference \
    --report-output-dir /mnt/decon-work/results \
    --cleaned-output-dir /mnt/decon-work/cleaned \
    --purify
```

Next launch an orchestrator to process a specific dataset. Be careful to get your source and destination correct to avoid overwriting any data.

```bash
python python/orchestration.py \
    --config config/orchestration.yaml \
    --remote-file-input s3://my-training-data/source-1 \
    --remote-report-output-dir s3://my-decon-report-bucket/my-training-data-source-1 \
    --remote-cleaned-output-dir s3://my-training-data/source-1-decon \
    --local-work-dir /mnt/decon-work
```

## Notes

You can run many orchestrator processes on a host, each handling a different dataset.

This is a bare bones direct approach to processing large datasets. This approach does not discuss validating success or monitoring, beyond mentioning it is worthwhile to check cleaned file contents and reviewing orchestrator/server logs.

## Scaling

The memory requirements vary depending on the size of your dataset and the size of your eval reference set.

A typical run with the full reference set outlined in evals.yaml requires about 14GB of RAM.

Likewise runtime can vary depending on your specific dataset and some experimentation is necessary to determine a good expectation.



