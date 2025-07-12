# Decon Deployment Guide

This guide covers deploying Decon on EC2 clusters using the poormanray (pmr) tool.

## Prerequisites

1. **Install poormanray**: Follow the instructions at [Poor Man's Ray CLI](https://github.com/allenai/olmo-cookbook/blob/main/README.md#poor-mans-ray-cli)

2. **AWS Credentials**: Ensure your AWS credentials are configured

3. **SSH Key**: Have an SSH key ready (default: `~/.ssh/id_rsa`)

4. **GitHub Token**: The decon repository is private, you'll need a GitHub personal access token

## Quick Start

### Poormanray

The easiest way to deploy Decon is using the interactive wizard:

```bash
make poormanray-command-generator
```

The wizard will guide you through:

**Cluster Configuration:**
- Cluster name
- Number of instances (default: 2)
- Instance type (default: i4i.2xlarge)
- SSH key path (default: ~/.ssh/id_rsa)
- GitHub token (required for private repo)

**Daemon Configuration:**
- Detection mode (simple/minhash/toxic, default: simple)
- Content key field (default: text)
- Mode-specific parameters (n-gram size, sampling rate, thresholds)
- Tokenizer choice (word/p50k/cl100k, default: word)
- Debug mode (default: off)
- Data purification (default: off)
- Daemon port (default: 8080)

**Orchestrator Configuration (optional):**
- S3 input data path
- S3 report output path
- S3 cleaned files path (if purification enabled)

The wizard outputs a set of poormanray commands to decontaminate with poormanray, which you can use as a reference in the future and forget about the wizard if you don't like software terms from the 90s.


## Managing Deployments

### Check Status

```bash
# Check cluster and daemon status
make deploy-status NAME=my-decon

# View instance list
poormanray list --name my-decon
```

### View Logs

```bash
# View daemon logs
make deploy-logs NAME=my-decon LOG=daemon

# View orchestrator logs
make deploy-logs NAME=my-decon LOG=orchestrator

# Follow logs in real-time
python python/deploy.py logs --name my-decon --log-type daemon --follow
```

### Terminate Cluster

#### Manual Termination

```bash
# Terminate all instances
make deploy-terminate NAME=my-decon
```

#### Automatic Termination (Recommended)

To avoid unnecessary charges, use the auto-terminate feature that monitors the orchestrator and automatically shuts down the cluster when work completes:

```bash
# Start monitoring and auto-terminate when done
make polling-auto-terminate NAME=my-decon
```

This command will:
- Poll the orchestrator logs every 60 seconds
- Look for the completion marker "WORK COMPLETE EXITING"
- Automatically terminate the cluster when detected
- Handle connection failures gracefully (up to 5 consecutive failures)

You can customize the polling interval:
```bash
python python/deploy.py polling-auto-terminate --name my-decon --poll-interval 30
```

**Best Practice**: Start the auto-terminate monitor in a separate terminal after launching your orchestrator. This ensures your cluster is automatically cleaned up when processing completes, preventing runaway costs from forgotten clusters.

## Deployment Architecture

When you deploy Decon:

1. **EC2 Instances**: Created with specified instance type and count
2. **Environment Setup**: Each instance gets:
   - Python 3.12, Rust, and required tools
   - AWS credentials from your local environment
   - PMR_HOST_INDEX and PMR_HOST_COUNT environment variables
3. **Decon Installation**:
   - Repository cloned
   - Rust project built
   - Python dependencies installed
   - Evaluation datasets downloaded
4. **Services Started**:
   - **Daemon**: HTTP server for processing contamination detection jobs
   - **Orchestrator**: Manages S3 file downloads, job distribution, and result uploads

### Data Flow Architecture

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
│  ┌──────────────┐         ┌────────────────────────────┐  │
│  │ Orchestrator │ ────────▶│        Daemon              │  │
│  │              │  Submit  │                            │  │
│  │ • Downloads  │  Jobs    │  • Loads reference data   │  │
│  │ • Distributes│          │  • Processes files        │  │
│  │ • Uploads    │◀──────── │  • Detects contamination │  │
│  │              │  Results │  • Creates cleaned files  │  │
│  └──────┬───────┘         └────────────────────────────┘  │
│         │                                                   │
└─────────┼───────────────────────────────────────────────────┘
          │
          │ Upload
          ▼
    ┌─────────────┐       ┌──────────────────┐
    │ S3 Reports  │       │ S3 Cleaned Files │
    │   Output    │       │    (Optional)    │
    │ (JSONL)     │       │ (JSONL.gz)       │
    └─────────────┘       └──────────────────┘

Key Components:
• S3 Input: Your training data files (supports .jsonl, .jsonl.gz, etc.)
• Orchestrator: Python process that coordinates the workflow
• Daemon: Rust HTTP server that performs the actual contamination detection
• S3 Reports: Contamination detection results
• S3 Cleaned: Purified datasets with contaminated lines removed (if --purify enabled)

Note: Multiple EC2 instances can run in parallel, each processing different files.

## Troubleshooting

### poormanray not found
Install it following the guide at: https://github.com/allenai/olmo-cookbook

### SSH connection issues
- Ensure your SSH key has proper permissions: `chmod 600 ~/.ssh/id_rsa`
- Check that the key matches the one used to create instances

### Daemon not responding
- Check logs: `make deploy-logs NAME=my-decon LOG=daemon`
- Ensure the daemon port (default 8080) is not blocked
- Verify the build completed successfully during setup

### Orchestrator issues
- Check orchestration config matches your S3 bucket setup
- Verify AWS credentials are properly configured
- Check logs: `make deploy-logs NAME=my-decon LOG=orchestrator`

## Cost Optimization

- Use appropriate instance types for your workload
- Remember to terminate clusters when done: `make deploy-terminate NAME=my-decon`
- Consider using spot instances for cost savings (configure in poormanray)
