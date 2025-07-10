# Decon Deployment Guide

This guide covers deploying Decon on EC2 clusters using the poormanray (pmr) tool.

## Prerequisites

1. **Install poormanray**: Follow the instructions at [Poor Man's Ray CLI](https://github.com/allenai/olmo-cookbook/blob/main/README.md#poor-mans-ray-cli)

2. **AWS Credentials**: Ensure your AWS credentials are configured

3. **SSH Key**: Have an SSH key ready (default: `~/.ssh/id_rsa`)

4. **GitHub Token**: The decon repository is private, you'll need a GitHub personal access token

## Quick Start

### Interactive Deployment Wizard

The easiest way to deploy Decon is using the interactive wizard:

```bash
make deploy-wizard
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

The wizard provides helpful descriptions for each option and sensible defaults that you can accept by pressing Enter.

### Manual Deployment Steps

If you prefer to run commands manually:

```bash
# 1. Create cluster
poormanray create --name my-decon --number 2 --instance-type i4i.2xlarge

# 2. Setup Decon
poormanray setup-decon --name my-decon --ssh-key-path ~/.ssh/id_rsa --github-token YOUR_TOKEN --detach

# 3. Start daemon
poormanray run --name my-decon --command "cd decon; nohup make daemon > daemon.log 2>&1 & disown" --ssh-key-path ~/.ssh/id_rsa --detach

# 4. Start orchestrator
poormanray run --name my-decon --command "cd decon; nohup make orchestrate > orchestrator.log 2>&1 & disown" --ssh-key-path ~/.ssh/id_rsa --detach
```

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

```bash
# Terminate all instances
make deploy-terminate NAME=my-decon
```

## Advanced Usage

### Non-Interactive Deployment

Basic deployment with defaults:
```bash
python python/deploy.py deploy \
  --name my-decon \
  --instances 3 \
  --instance-type i4i.2xlarge \
  --ssh-key ~/.ssh/id_rsa \
  --github-token YOUR_TOKEN
```

Full deployment with custom configuration:
```bash
python python/deploy.py deploy \
  --name my-decon \
  --instances 3 \
  --github-token YOUR_TOKEN \
  --mode simple \
  --ngram-size 15 \
  --sample-every-m-tokens 5 \
  --tokenizer word \
  --purify \
  --remote-file-input s3://bucket/training-data \
  --remote-report-output-dir s3://bucket/reports \
  --remote-cleaned-output-dir s3://bucket/cleaned
```

### Custom Configuration

You can override daemon settings during deployment:

```bash
python python/deploy.py deploy \
  --name my-decon \
  --daemon-port 9090 \
  --instances 2
```

### Programmatic API

For integration with other tools (e.g., MCP servers):

```python
from deploy import create_deployment_manager, DeploymentStage

# Create manager
manager = create_deployment_manager(
    cluster_name="my-decon",
    instance_count=2,
    instance_type="i4i.2xlarge",
    github_token="YOUR_TOKEN"
)

# Execute individual stages
manager.create_cluster()
manager.setup_decon()
manager.start_daemon()
manager.start_orchestrator()

# Check status
status = manager.check_status()
print(status)

# Clean up
manager.terminate_cluster()
```

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
