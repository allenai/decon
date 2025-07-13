# Decon Deployment Guide

This guide covers deploying Decon on EC2 clusters using the poormanray (pmr) tool.

## Prerequisites

1. **Install poormanray**: Follow the instructions at [Poor Man's Ray CLI](https://github.com/allenai/olmo-cookbook/blob/main/README.md#poor-mans-ray-cli)

2. **AWS Credentials**: Ensure your AWS credentials are configured

3. **SSH Key**: Have an SSH key ready (default: `~/.ssh/id_rsa`)

4. **GitHub Token**: The decon repository is private, you'll need a GitHub personal access token

## Quick Start

### Poormanray

The easiest way to deploy Decon is using the interactive tool. After doing this once, you will get a sense for the basic structure of how to setup your own configuration and workflow.

```bash
make poormanray-command-generator
```

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

