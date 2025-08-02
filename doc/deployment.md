# Decon Deployment Guide

Deploy Decon on EC2 clusters using the poormanray (pmr) tool.

This guide explains the architecture and provides tips for processing multiple datasets efficiently.

## Prerequisites

1. **Install poormanray**: Follow the instructions at [Poor Man's Ray CLI](https://github.com/allenai/olmo-cookbook/blob/main/README.md#poor-mans-ray-cli)
  - **AWS Credentials**: Ensure your AWS credentials are configured
  - **SSH Key**: Have an SSH key ready (default: `~/.ssh/id_rsa`)

2. **GitHub Token**: Get a GitHub personal access token (the decon repository is private)

## Overview

### Poormanray

Poormanray runs commands across EC2 instances.

Run decon on a dataset in 4 steps:

1. Create the cluster
2. Install decon software
3. Launch a decon server
4. Launch an orchestrator to download files, submit to server, and upload results

Generate poormanray commands with sensible defaults:

```bash
make poormanray-command-generator
```

Example outputs for a dataset are shown below to inform the later discussion.

#### Creating a cluster

```
poormanray create \
  --name decon \
  --owner robert \
  --number 12 \
  --instance-type i4i.4xlarge \
  --region us-east-1
```

#### Set up decon

```
poormanray setup-decon \
  --name decon \
  --ssh-key-path ~/.ssh/id_rsa \
  --github-token xxxx
```

#### Launch server

```
poormanray run \
  --name decon \
  --command "cd decon && nohup cargo run --release -- server \
    --config config/default.yaml \
    --content-key text \
    --question-threshold 0.815 \
    --answer-threshold 0.8 \
    --reference-input fixtures/reference \
    --report-output-dir /mnt/decon-work/results \
    --cleaned-output-dir /mnt/decon-work/cleaned \
    --purify \
    > server.log 2>&1 & disown" \
  --ssh-key-path ~/.ssh/id_rsa \
  --detach
```

#### Run an orchestrator to process a dataset

```
poormanray run \
  --name decon \
  --command "cd decon && nohup python python/orchestration.py \
    --config config/orchestration.yaml \
    --remote-file-input s3://ai2-llm/pretraining-data/sources/thinking-data/llama-nemotron-processed-chinese-filtered-ngram-filtered-with-token-counts/ \
    --remote-report-output-dir s3://ai2-decon-reports/8-1/ai2-llm-pretraining-data-sources-thinking-data-llama-nemotron-processed-chinese-filtered-ngram-filtered-with-token-counts \
    --remote-cleaned-output-dir s3://ai2-llm/pretraining-data/sources/thinking-data/llama-nemotron-processed-chinese-filtered-ngram-filtered-with-token-counts-decon-2 \
    --local-work-dir /mnt/decon-work \
    > orchestrator.log 2>&1 & disown" \
  --ssh-key-path ~/.ssh/id_rsa \
  --detach
```

### How it works

Decon runs as either a standalone CLI or a server. The orchestrator script downloads source files, submits them to the decon server, and uploads results.

This guide focuses on running decon as a server with the orchestrator on poormanray.

#### Overview

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

#### Server

The HTTP server receives POST requests with local file paths and produces contamination reports and optionally cleaned files.

Launch the server with a configuration file (--config flag) and override options via command line arguments. Use config/default.yaml as the baseline.

A server instance runs until killed. Run multiple orchestrators against the same server to process different datasets with identical configuration.

#### Orchestrator

The orchestrator launches with input and output targets on S3.

It maintains a download queue and monitors the server's work queue, creating a pipeline for downloading and processing files.

It also monitors job status reported by the server, and upon completion uploads the output files to the S3 targets and cleans up local files after the upload completes.

Orchestrators know the number of hosts in a cluster and the current host's index through environment variables set when running the decon setup script on poormanray. This partitions the work by S3 file, and only files that hash modulo to an orchestrator's host index get processed by that orchestrator.

Orchestrators produce their own scoped workspace and only process output files based on the job IDs they receive when submitting jobs to the server. This allows you to run many orchestrators, one per dataset being decontaminated, at the same time. Just remember that a server process only has one set of decontamination parameters.

### Practical Tips

A single i4i.4xlarge host processes ~ 26B tokens/hour.

Work distributes by S3 file hash, so small datasets on large clusters may have uneven workloads.

There are two make targets to help with tracking the status of a cluster's work. An orchestrator creates a lock file in /mnt/decon-work/<workspace-hash>, limiting one orchestrator per input source hash per host. The lock file is deleted when an orchestrator's work completes. To check the status you can run `make polling-auto-status NAME=<your-cluster-name>` which will print out "Running" or "Finished". Alternatively, you can call `make polling-auto-terminate NAME=<your-cluster-name>` which will terminate the poormanray cluster when "Finished" state is detected.

Since launching a large cluster takes time, when processing datasets with different contamination parameters, restart the server with a new config after completing a batch of datasets against a specific decontamination parameter set. The server responds to SIGTERM by shutting down, so stop servers with `poormanray run -n <your-cluster-name> -c "pkill decon"`. Then run a new server launch command on the cluster with alternative configuration override options.

The orchestrator polls when first launched in case the server is still processing the eval index (takes ~1 minute), so you can launch orchestrators quickly after launching servers.

The i4i.4xlarge is over-provisioned. This happens partly because the purification step reads files directly into memory before writing clean versions. With datasets containing large files, e.g. 50 7GB files, it could reach the purification step at the same time, and would consume a lot of RAM. This needs optimization (TODO), but until then, the i4i.4xlarge is over provisioned for one size fits all simplicity. You can of course use instances with a larger CPU/RAM ratio if you know your dataset's shape. Over provisioning also allows for running many orchestrators simultaneously, e.g., processing a full training dataset at once.

When processing many datasets, update the generate_orchestrator_command.py file with your cluster and output base options and run `make generate-orchestrator-command s3://new-input-prefix` to quickly generate orchestrator commands. This lacks polish, but manually updating a first orchestrator launch command generated from `make poormanray-command-generator` works fine.

The generated and example command can write to log files for any debugging and tracking purposes.
