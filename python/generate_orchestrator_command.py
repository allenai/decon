#!/usr/bin/env python3
"""
Generate poormanray orchestrator command from S3 input path.

This script takes an S3 input path and generates a complete poormanray command
for running the decon orchestrator, automatically computing cluster name and
output paths.
"""

import sys
import argparse


def shorten_name(name, max_length=50):
    """Shorten a name if it exceeds max_length by taking first and last parts."""
    if len(name) <= max_length:
        return name

    # Keep some chars from start and end
    keep_start = max_length // 2 - 2
    keep_end = max_length // 2 - 2
    shortened = f"{name[:keep_start]}--{name[-keep_end:]}"

    # Clean up any accidental double dashes
    while '--' in shortened:
        shortened = shortened.replace('--', '-')

    return shortened


def generate_orchestrator_command(s3_input_path):
    """Generate poormanray orchestrator command from S3 input path."""

    # Validate input
    if not s3_input_path.startswith("s3://"):
        print(f"Error: Input path must start with 's3://', got: {s3_input_path}", file=sys.stderr)
        sys.exit(1)

    # Extract dataset name from S3 path
    # Remove s3:// prefix and replace / with -
    dataset_name = s3_input_path[5:].replace('/', '-').strip('-')

    # Remove file extensions if present
    if dataset_name.endswith('.jsonl') or dataset_name.endswith('.json'):
        dataset_name = dataset_name.rsplit('.', 1)[0]

    # Shorten the cluster name to avoid exceeding limits
    cluster_name = 'decon'

    # Generate report output path with shortened name
    # First remove the common prefix if present
    report_name = dataset_name
    if report_name.startswith('ai2-llm-pretraining-data-sources-'):
        report_name = report_name[len('ai2-llm-pretraining-data-sources-'):]

    variant = "low-precision-anr"

    report_name = shorten_name(report_name, max_length=80)
    report_output_dir = f"s3://ai2-decon-reports/8-1/{report_name}"

    # Generate cleaned output path
    # Remove trailing slash from input path if present
    input_path_clean = s3_input_path.rstrip('/')
    cleaned_output_dir = f"{input_path_clean}-decon-2"
    
    # Ensure remote-file-input has trailing slash
    remote_file_input = s3_input_path.rstrip('/') + '/'
    
    # Ensure output directories don't have trailing slashes
    report_output_dir = report_output_dir.rstrip('/')
    cleaned_output_dir = cleaned_output_dir.rstrip('/')
    
    # Validate that cleaned output dir is not the same as input dir
    # Compare without trailing slashes for accurate comparison
    input_dir_normalized = s3_input_path.rstrip('/')
    cleaned_dir_normalized = cleaned_output_dir.rstrip('/')
    
    if input_dir_normalized == cleaned_dir_normalized:
        print(f"Error: Cleaned output directory cannot be the same as input directory!", file=sys.stderr)
        print(f"  Input: {input_dir_normalized}", file=sys.stderr)
        print(f"  Cleaned output: {cleaned_dir_normalized}", file=sys.stderr)
        print(f"This would overwrite your input data! Please choose a different output location.", file=sys.stderr)
        sys.exit(1)

    # Generate the poormanray command
    command = f"""poormanray run \\
  --name {cluster_name} \\
  --command "cd decon && nohup python python/orchestration.py \\
    --config config/orchestration.yaml \\
    --remote-file-input {remote_file_input} \\
    --remote-report-output-dir {report_output_dir} \\
    --remote-cleaned-output-dir {cleaned_output_dir} \\
    --local-work-dir /mnt/decon-work \\
    > orchestrator.log 2>&1 & disown" \\
  --ssh-key-path ~/.ssh/id_rsa \\
  --detach"""

    return command


def main():
    parser = argparse.ArgumentParser(
        description="Generate poormanray orchestrator command from S3 input path"
    )
    parser.add_argument(
        "s3_input_path",
        help="S3 path for input data to decontaminate (e.g., s3://bucket/training-data/)"
    )

    args = parser.parse_args()

    command = generate_orchestrator_command(args.s3_input_path)
    print(command)


if __name__ == "__main__":
    main()
