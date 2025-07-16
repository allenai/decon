# Contamination Detection Orchestration

This directory contains the Python orchestration scripts for running distributed contamination detection across multiple hosts.

## Overview

The orchestration system allows you to:
- Distribute contamination detection across multiple hosts
- Download training files from S3 (supports .jsonl, .jsonl.gz, .jsonl.zst, .jsonl.bz2, .jsonl.xz)
- Submit files to the local server for processing
- Upload results and cleaned files back to S3
- Handle failures gracefully

## Setup

1. Install Python dependencies:
```bash
pip install -r requirements.txt
```

2. Configure AWS credentials (if not already done):
```bash
aws configure
```

3. Install s5cmd (high-performance S3 client):
```bash
# Download from https://github.com/peak/s5cmd/releases
wget https://github.com/peak/s5cmd/releases/download/v2.2.2/s5cmd_2.2.2_Linux-64bit.tar.gz
tar -xvzf s5cmd_2.2.2_Linux-64bit.tar.gz
sudo mv s5cmd /usr/local/bin/
```

## Configuration

Create an orchestration configuration file (see `config/orchestration.yaml`):

```yaml
# S3 locations
remote_file_input: s3://your-bucket/training-data/
remote_report_output_dir: s3://your-bucket/output/  # Contamination reports go here
remote_cleaned_output_dir: s3://your-bucket/cleaned/  # Cleaned files go here (when purify=true)

# Server configuration
server_url: http://localhost:8080

# Local working directory
local_work_dir: /tmp/decon-work

# Performance settings
max_concurrent_jobs: 10
poll_interval: 5
s5cmd_workers: 50
cleanup_delay: 10
```

## Running the Orchestrator

### Single Host

```bash
# Start the server first
make server

# Run orchestration
python python/orchestration.py --config config/orchestration.yaml
```

### Multiple Hosts

Set environment variables to distribute work across hosts:

```bash
# On host 1 of 3
export PMR_HOST_INDEX=0
export PMR_HOST_COUNT=3
python python/orchestration.py --config orchestration.yaml

# On host 2 of 3
export PMR_HOST_INDEX=1
export PMR_HOST_COUNT=3
python python/orchestration.py --config orchestration.yaml

# On host 3 of 3
export PMR_HOST_INDEX=2
export PMR_HOST_COUNT=3
python python/orchestration.py --config orchestration.yaml
```

### Debug Mode

For development and testing, you can limit the number of files processed:

```bash
# Process only 5 files for debugging
MAX_FILES_DEBUG=5 python python/orchestration.py --config orchestration.yaml

# Or use the make target
make orchestrate-debug
```

When `MAX_FILES_DEBUG` is set:
- Only the specified number of files will be processed
- **No files will be uploaded to S3**
- **No local files will be cleaned up**
- Results remain in the local working directory for inspection

## How It Works

1. **File Distribution**: Each host is assigned files based on a hash of the filename modulo the host count
2. **Download**: Uses s5cmd for efficient batch downloads from S3
3. **Processing**: Submits files to the local server and tracks job status
4. **Upload**: Uploads contamination results and cleaned files to S3
5. **Cleanup**: Removes local files after successful processing

## Output Structure

All outputs go to configured S3 directories:

- **Contamination reports**: `{basename}.report.jsonl`
- **Cleaned files**: `{basename}.clean.jsonl{compression}`
- **Clean markers**: `{basename}.clean` (for uncontaminated files)

Example for input file `sponge-text-2024-08-00001.jsonl.gz`:
- Report: `s3://bucket/output/sponge-text-2024-08-00001.report.jsonl`
- Cleaned: `s3://bucket/output/sponge-text-2024-08-00001.clean.jsonl.gz`
- Or if no contamination: `s3://bucket/output/sponge-text-2024-08-00001.clean`

## Error Handling

- Graceful shutdown on SIGINT/SIGTERM (Ctrl+C)
- Force shutdown on second interrupt
- Failed files are logged but don't stop processing
- Automatic retries for transient S3/HTTP errors

## Monitoring

Check the logs for:
- Number of files to process
- Job submission status
- Completion statistics
- Any errors or warnings
