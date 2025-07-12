#!/usr/bin/env python3
"""
Decon Deployment Utilities
=========================

This module provides utilities for deploying and managing Decon on remote clusters
using the poormanray (pmr) CLI tool.

It includes both a wizard interface for interactive deployment and low-level
functions for programmatic control (useful for MCP server integration).
"""

import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import List, Optional, Dict, Any
import shutil
import click


class DeploymentStage(Enum):
    """Stages of the deployment process"""
    CREATE_CLUSTER = "create_cluster"
    SETUP_DECON = "setup_decon"
    START_DAEMON = "start_daemon"
    START_ORCHESTRATOR = "start_orchestrator"
    MONITOR = "monitor"
    TERMINATE = "terminate"


@dataclass
class DeploymentConfig:
    """Configuration for a Decon deployment"""
    cluster_name: str
    owner: Optional[str] = None
    instance_count: int = 2
    instance_type: str = "i4i.2xlarge"
    region: str = "us-east-1"
    ssh_key_path: str = "~/.ssh/id_rsa"
    github_token: Optional[str] = None
    config_path: Optional[str] = None
    daemon_port: int = 8080
    orchestration_config: Optional[str] = None

    # Daemon configuration
    mode: str = "simple"
    content_key: str = "text"
    ngram_size: int = 4
    sample_every_m_tokens: int = 10
    toxic_overlap_threshold: float = 0.8
    toxic_score_threshold: float = 2.0
    tokenizer_str: str = "word"
    debug: bool = False
    purify: bool = False

    # Orchestrator configuration
    remote_file_input: Optional[str] = None
    remote_report_output_dir: Optional[str] = None
    remote_cleaned_output_dir: Optional[str] = None
    local_work_dir: str = "/mnt/decon-work"

    def __post_init__(self):
        # Expand paths
        self.ssh_key_path = os.path.expanduser(self.ssh_key_path)
        if self.config_path:
            self.config_path = os.path.expanduser(self.config_path)
        if self.orchestration_config:
            self.orchestration_config = os.path.expanduser(self.orchestration_config)


class PoorManRayError(Exception):
    """Raised when poormanray is not installed"""
    pass


class DeploymentManager:
    """Manages Decon deployments on remote clusters"""

    def __init__(self, config: DeploymentConfig):
        self.config = config
        self._validate_prerequisites()

    def _validate_prerequisites(self):
        """Check that poormanray is installed"""
        if shutil.which("poormanray") is None:
            raise PoorManRayError(
                "poormanray is not installed. Please install it from: "
                "https://github.com/allenai/olmo-cookbook/blob/main/README.md#poor-mans-ray-cli"
            )

    def _run_pmr_command(self, command: List[str], check: bool = True) -> subprocess.CompletedProcess:
        """Run a poormanray command"""
        full_command = ["poormanray"] + command
        print(f"Running: {' '.join(full_command)}")

        result = subprocess.run(
            full_command,
            capture_output=True,
            text=True,
            check=False
        )

        if check and result.returncode != 0:
            print(f"Error: {result.stderr}")
            raise subprocess.CalledProcessError(result.returncode, full_command, result.stdout, result.stderr)

        return result

    def create_cluster(self) -> bool:
        """Create EC2 instances for the cluster"""
        print(f"\n🚀 Creating cluster '{self.config.cluster_name}' with {self.config.instance_count} instances...")

        command = [
            "create",
            "--name", self.config.cluster_name,
            "--number", str(self.config.instance_count),
            "--instance-type", self.config.instance_type,
            "--region", self.config.region,
            "--detach"
        ]

        if self.config.owner:
            command.extend(["--owner", self.config.owner])

        try:
            self._run_pmr_command(command)
            print("✅ Cluster created successfully")
            return True
        except subprocess.CalledProcessError:
            print("❌ Failed to create cluster")
            return False

    def setup_decon(self) -> bool:
        """Set up Decon on all instances"""
        print(f"\n🔧 Setting up Decon on cluster '{self.config.cluster_name}'...")

        command = [
            "setup-decon",
            "--name", self.config.cluster_name,
            "--ssh-key-path", self.config.ssh_key_path,
            "--detach"
        ]

        if self.config.github_token:
            command.extend(["--github-token", self.config.github_token])

        try:
            self._run_pmr_command(command)
            print("✅ Decon setup completed")
            return True
        except subprocess.CalledProcessError:
            print("❌ Failed to setup Decon")
            return False

    def start_daemon(self) -> bool:
        """Start the Decon daemon on all instances"""
        print(f"\n🎯 Starting Decon daemon on cluster '{self.config.cluster_name}'...")

        # Build daemon command with config overrides
        daemon_cmd = "cd decon && nohup cargo run --release -- daemon --config examples/simple.yaml"

        # Add all configuration overrides
        daemon_cmd += f" --mode {self.config.mode}"
        daemon_cmd += f" --content-key {self.config.content_key}"
        daemon_cmd += f" --ngram-size {self.config.ngram_size}"
        daemon_cmd += f" --sample-every-m-tokens {self.config.sample_every_m_tokens}"
        daemon_cmd += f" --toxic-overlap-threshold {self.config.toxic_overlap_threshold}"
        daemon_cmd += f" --toxic-score-threshold {self.config.toxic_score_threshold}"
        daemon_cmd += f" --tokenizer {self.config.tokenizer_str}"

        # Set output directories within the local work directory mount
        daemon_cmd += f" --report-output-dir {self.config.local_work_dir}/results"
        daemon_cmd += f" --cleaned-output-dir {self.config.local_work_dir}/cleaned"

        if self.config.daemon_port != 8080:
            daemon_cmd += f" --port {self.config.daemon_port}"

        if self.config.debug:
            daemon_cmd += " --debug true"

        if self.config.purify:
            daemon_cmd += " --purify true"

        daemon_cmd += " > daemon.log 2>&1 & disown"

        command = [
            "run",
            "--name", self.config.cluster_name,
            "--command", daemon_cmd,
            "--ssh-key-path", self.config.ssh_key_path,
            "--detach"
        ]

        try:
            self._run_pmr_command(command)
            print("✅ Daemon started successfully")
            time.sleep(5)  # Give daemon time to start
            return True
        except subprocess.CalledProcessError:
            print("❌ Failed to start daemon")
            return False

    def start_orchestrator(self) -> bool:
        """Start the orchestrator on all instances"""
        print(f"\n🎼 Starting orchestrator on cluster '{self.config.cluster_name}'...")

        # Use custom orchestration config if provided
        config_file = self.config.orchestration_config or "examples/orchestration.yaml"

        orchestrator_cmd = f"cd decon && nohup python python/orchestration.py --config {config_file}"

        # Add daemon URL override if using non-default port
        if self.config.daemon_port != 8080:
            orchestrator_cmd += f" --daemon-url http://localhost:{self.config.daemon_port}"

        # Add S3 path overrides if provided
        if self.config.remote_file_input:
            orchestrator_cmd += f" --remote-file-input {self.config.remote_file_input}"

        if self.config.remote_report_output_dir:
            orchestrator_cmd += f" --remote-report-output-dir {self.config.remote_report_output_dir}"

        if self.config.remote_cleaned_output_dir:
            orchestrator_cmd += f" --remote-cleaned-output-dir {self.config.remote_cleaned_output_dir}"

        # Add local work directory
        orchestrator_cmd += f" --local-work-dir {self.config.local_work_dir}"

        orchestrator_cmd += " > orchestrator.log 2>&1 & disown"

        command = [
            "run",
            "--name", self.config.cluster_name,
            "--command", orchestrator_cmd,
            "--ssh-key-path", self.config.ssh_key_path,
            "--detach"
        ]

        try:
            self._run_pmr_command(command)
            print("✅ Orchestrator started successfully")
            return True
        except subprocess.CalledProcessError:
            print("❌ Failed to start orchestrator")
            return False

    def check_status(self) -> Dict[str, Any]:
        """Check the status of the deployment"""
        print(f"\n🔍 Checking status of cluster '{self.config.cluster_name}'...")

        # List instances
        list_command = [
            "list",
            "--name", self.config.cluster_name,
            "--region", self.config.region
        ]

        try:
            result = self._run_pmr_command(list_command)

            # Check daemon health on each instance
            health_command = [
                "run",
                "--name", self.config.cluster_name,
                "--command", f"curl -s http://localhost:{self.config.daemon_port}/health || echo 'Daemon not responding'",
                "--ssh-key-path", self.config.ssh_key_path
            ]

            health_result = self._run_pmr_command(health_command, check=False)

            return {
                "instances": result.stdout,
                "daemon_health": health_result.stdout,
                "success": health_result.returncode == 0
            }
        except subprocess.CalledProcessError as e:
            return {
                "instances": e.stdout or "Failed to list instances",
                "daemon_health": "Unknown",
                "success": False
            }

    def view_logs(self, log_type: str = "daemon") -> str:
        """View logs from the deployment"""
        log_file = f"{log_type}.log"
        command = [
            "run",
            "--name", self.config.cluster_name,
            "--command", f"cd decon && tail -n 100 {log_file}",
            "--ssh-key-path", self.config.ssh_key_path
        ]

        try:
            result = self._run_pmr_command(command)
            return result.stdout
        except subprocess.CalledProcessError:
            return f"Failed to retrieve {log_type} logs"

    def terminate_cluster(self) -> bool:
        """Terminate the cluster"""
        print(f"\n🔴 Terminating cluster '{self.config.cluster_name}'...")

        command = [
            "terminate",
            "--name", self.config.cluster_name,
            "--region", self.config.region
        ]

        try:
            self._run_pmr_command(command)
            print("✅ Cluster terminated successfully")
            return True
        except subprocess.CalledProcessError:
            print("❌ Failed to terminate cluster")
            return False

    def full_deployment(self) -> bool:
        """Execute the full deployment pipeline"""
        stages = [
            (self.create_cluster, "Creating cluster"),
            (self.setup_decon, "Setting up Decon"),
            (self.start_daemon, "Starting daemon"),
            (self.start_orchestrator, "Starting orchestrator"),
        ]

        for stage_func, stage_name in stages:
            if not stage_func():
                print(f"\n❌ Deployment failed at stage: {stage_name}")
                return False

        print("\n✅ Full deployment completed successfully!")

        # Show status
        status = self.check_status()
        print("\n📊 Deployment Status:")
        print(status["instances"])

        return True


@click.group()
def cli():
    """Decon deployment management utilities"""
    pass


@cli.command()
def wizard():
    # Clear the screen for a clean interface
    os.system('clear' if os.name == 'posix' else 'cls')
    print("\n\n💡 \033[1;93mSensible defaults are provided in [brackets]. Press Enter to accept.\033[0m\n")

    # Dataset configuration first
    print("=== Dataset Configuration ===\n")

    while True:
        remote_file_input = click.prompt(
            "S3 path for input data to decontaminate (e.g., s3://bucket/training-data)",
            type=str
        )
        # Validate that it starts with s3://
        if remote_file_input.startswith("s3://"):
            break
        else:
            print("❌ Error: Input data path must start with 's3://'. Please try again.")

    # Extract dataset name from S3 path by removing s3:// prefix and replacing / with -
    dataset_name = remote_file_input[5:].replace('/', '-').strip('-')
    if dataset_name.endswith('.jsonl') or dataset_name.endswith('.json'):
        # Remove file extension if present
        dataset_name = dataset_name.rsplit('.', 1)[0]

    # Data purification question immediately after
    print("\n\nDecon always produces contamination report files including per-file contamination matches and scores.")
    # Report output configuration
    default_report_path = f"s3://ai2-decon-reports/{dataset_name}"
    remote_report_output_dir = click.prompt(
        "S3 path for contamination reports",
        default=default_report_path,
        type=str
    )

    purify = click.confirm("\nWould you like to also create a clean copy of the dataset with contaminated documents removed?", default=False)

    remote_cleaned_output_dir = None
    if purify:
        # Generate default cleaned dataset path
        # Split the input path and replace the last segment with {segment}-decon
        path_parts = remote_file_input.rstrip('/').split('/')
        if len(path_parts) > 3:  # Ensure we have at least s3://bucket/something
            last_segment = path_parts[-1]
            # Remove file extension if present
            if last_segment.endswith('.jsonl') or last_segment.endswith('.json'):
                last_segment = last_segment.rsplit('.', 1)[0]
            path_parts[-1] = f"{last_segment}-decon"
            default_cleaned_path = '/'.join(path_parts)
        else:
            # Fallback if path is too short
            default_cleaned_path = f"{remote_file_input.rstrip('/')}-decon"

        # Ask for cleaned dataset destination immediately after purification question
        while True:
            remote_cleaned_output_dir = click.prompt(
                "S3 bucket for cleaned dataset copy",
                default=default_cleaned_path,
                type=str
            )
            # Validate that it starts with s3://
            if remote_cleaned_output_dir.startswith("s3://"):
                break
            else:
                print("❌ Error: Cleaned dataset destination must start with 's3://'. Please try again.")


    # Validate that it starts with s3://
    while not remote_report_output_dir.startswith("s3://"):
        print("❌ Error: Report output path must start with 's3://'. Please try again.")
        remote_report_output_dir = click.prompt(
            "S3 path for contamination reports",
            default=default_report_path,
            type=str
        )

    # Basic cluster configuration
    print("\n=== Poorman Ray Cluster Configuration ===")
    cluster_name = click.prompt("Cluster name", default=dataset_name, type=str)

    # Get current username as default for owner
    import getpass
    default_owner = getpass.getuser()
    owner = click.prompt("Owner", default=default_owner, type=str)
    instance_count = click.prompt("Number of instances", type=int, default=2)
    instance_type = click.prompt("Instance type", default="i4i.2xlarge")
    ssh_key = click.prompt("SSH key path", default="~/.ssh/id_rsa")
    github_token = click.prompt("GitHub token (only read access for private decon repo required)")

    # Detection configuration
    print("\n=== Detection Configuration ===")

    # Detection mode - temporarily forced to simple
    # print("\nDetection modes:")
    # print("  simple  - Fast n-gram matching (other options might run, but are works in progress and will not outperform simple)")
    # print("  minhash - Near-duplicate detection using MinHash")
    # print("  toxic   - Semantic similarity using word embeddings\n")
    # mode = click.prompt("Detection mode", default="simple",
    #                    type=click.Choice(['simple', 'minhash', 'toxic']))
    mode = "simple"  # Force simple mode for now

    # Common parameters
    content_key = click.prompt("JSON field containing text in jsonl files to decontaminate", default="text")

    print("\n\n💡 \033[1;93mSensible defaults are provided in [brackets]. See https://github.com/allenai/decon/blob/main/doc/simple.md for context.\033[0m\n")

    # Mode-specific parameters - always use simple mode configuration
    # if mode == "simple":
    print("\n--- SIMPLE Mode Configuration ---")
    ngram_size = click.prompt("N-gram size (values < 4 not recommended)", type=int, default=4)
    sample_every_m_tokens = click.prompt("Sample every M tokens (1 = no sampling)", type=int, default=10)
    toxic_overlap_threshold = click.prompt("Overlap ratio threshold (Matches must have (num_unique_ngrams_matched / num_unique_ngrams_in_eval) > threshold)", type=float, default=0.85)
    toxic_score_threshold = click.prompt("IDF score threshold (Matches must have sum matched n-gram IDF > threshold)", type=float, default=2.0)
    # else:
    #     # Use defaults for other modes
    #     ngram_size = 4
    #     sample_every_m_tokens = 10
    #     toxic_overlap_threshold = 0.8
    #     toxic_score_threshold = 2.0

    # Tokenizer
    print("\nTokenizer options:")
    print("  word    - Simple word tokenization (fastest)")
    print("  p50k    - OpenAI's P50K tokenizer")
    print("  cl100k  - OpenAI's CL100K tokenizer")
    tokenizer_str = click.prompt("Tokenizer", default="word",
                                type=click.Choice(['word', 'p50k', 'cl100k']))

    # Other options
    debug = False  # Debug mode hardcoded to false
    daemon_port = 8080  # Daemon port hardcoded to default

    # Force local work directory to /mnt/decon-work
    local_work_dir = "/mnt/decon-work"

    # Create configuration
    deploy_config = DeploymentConfig(
        cluster_name=cluster_name,
        owner=owner,
        instance_count=instance_count,
        instance_type=instance_type,
        ssh_key_path=ssh_key,
        github_token=github_token,
        daemon_port=daemon_port,
        # Daemon settings
        mode=mode,
        content_key=content_key,
        ngram_size=ngram_size,
        sample_every_m_tokens=sample_every_m_tokens,
        toxic_overlap_threshold=toxic_overlap_threshold,
        toxic_score_threshold=toxic_score_threshold,
        tokenizer_str=tokenizer_str,
        debug=debug,
        purify=purify,
        # Orchestrator settings
        remote_file_input=remote_file_input if remote_file_input else None,
        remote_report_output_dir=remote_report_output_dir if remote_report_output_dir else None,
        remote_cleaned_output_dir=remote_cleaned_output_dir if remote_cleaned_output_dir else None,
        local_work_dir=local_work_dir,
    )

    # Show configuration summary
    print("\n📋 Deployment Summary:")
    print(f"  Cluster: {deploy_config.cluster_name} ({deploy_config.instance_count} x {deploy_config.instance_type})")
    print(f"  Mode: {deploy_config.mode}")
    print(f"  Tokenizer: {deploy_config.tokenizer_str}")
    print(f"  N-gram size: {deploy_config.ngram_size}")
    print(f"  Sampling: every {deploy_config.sample_every_m_tokens} tokens")
    print(f"  Purification: {'enabled' if deploy_config.purify else 'disabled'}")

    if remote_file_input:
        print(f"\nOrchestrator:")
        print(f"  Input: {deploy_config.remote_file_input}")
        print(f"  Reports: {deploy_config.remote_report_output_dir}")
        if deploy_config.remote_cleaned_output_dir:
            print(f"  Cleaned: {deploy_config.remote_cleaned_output_dir}")
        print(f"  Work dir: {deploy_config.local_work_dir}")

    print("\n")
    print("━" * 80)
    print("🚀 ✨ DEPLOYMENT COMMANDS ✨ 🚀".center(80))
    print("━" * 80)
    print("\nCopy and run these commands to deploy your cluster:\n")

    print("\033[1m # 1. Create cluster\033[0m")
    print("\npoormanray create \\")
    print(f"  --name {cluster_name} \\")
    print(f"  --owner {owner} \\")
    print(f"  --number {instance_count} \\")
    print(f"  --instance-type {instance_type} \\")
    print("  --region us-east-1 \n")

    print("\033[1m # 2. Install dependencies and setup environment\033[0m")
    print("\npoormanray setup-decon \\")
    print(f"  --name {cluster_name} \\")
    print(f"  --ssh-key-path {ssh_key} \\")
    print(f"  --github-token {github_token} \n")

    print("\033[1m # 3. Launch decontamination server that builds reference index and receives work.\033[0m")
    print("\npoormanray run \\")
    print(f"  --name {cluster_name} \\")
    print("  --command \"cd decon && nohup cargo run --release -- daemon \\")
    print("    --config examples/simple.yaml \\")
    print(f"    --mode {mode} --content-key {content_key} \\")
    print(f"    --ngram-size {ngram_size} --sample-every-m-tokens {sample_every_m_tokens} \\")
    print(f"    --toxic-overlap-threshold {toxic_overlap_threshold} \\")
    print(f"    --toxic-score-threshold {toxic_score_threshold} \\")
    print(f"    --tokenizer {tokenizer_str} \\")
    print(f"    --report-output-dir {local_work_dir}/results \\")
    print(f"    --cleaned-output-dir {local_work_dir}/cleaned", end="")
    if purify:
        print(" \\")
        print("    --purify true", end="")
    print(" \\")
    print("    > daemon.log 2>&1 & disown\" \\")
    print(f"  --ssh-key-path {ssh_key} \\")
    print("  --detach\n")

    print("\033[1m # 4. Launch orchestrator which downloads files, submits to daemon, and uploads results.\033[0m")
    print("\npoormanray run \\")
    print(f"  --name {cluster_name} \\")
    print("  --command \"cd decon && nohup python python/orchestration.py \\")
    print("    --config examples/orchestration.yaml \\")
    print(f"    --remote-file-input {remote_file_input} \\")
    print(f"    --remote-report-output-dir {remote_report_output_dir} \\")
    if remote_cleaned_output_dir:
        print(f"    --remote-cleaned-output-dir {remote_cleaned_output_dir} \\")
    print(f"    --local-work-dir {local_work_dir} \\")
    print("    > orchestrator.log 2>&1 & disown\" \\")
    print(f"  --ssh-key-path {ssh_key} \\")
    print("  --detach\n")

    print("\n" + "━" * 80)
    print("📝 USEFUL COMMANDS".center(80))
    print("━" * 80)
    print(f"  View daemon logs:   make deploy-logs NAME={cluster_name} LOG=daemon")
    print(f"  View orchestrator logs:  make deploy-logs NAME={cluster_name} LOG=orchestrator")
    print(f"  Auto-terminate:     make polling-auto-terminate NAME={cluster_name}")
    print("⚠️  Remember to terminate your cluster when done to avoid unnecessary charges\n")


@cli.command()
@click.option("--name", required=True, help="Cluster name")
@click.option("--owner", required=True, help="Owner username")
@click.option("--instances", default=2, type=int, help="Number of instances")
@click.option("--instance-type", default="i4i.2xlarge", help="EC2 instance type")
@click.option("--ssh-key", default="~/.ssh/id_rsa", help="Path to SSH private key")
@click.option("--github-token", required=True, help="GitHub token for private repo access")
@click.option("--daemon-port", default=8080, type=int, help="Daemon port")
# Daemon options
@click.option("--mode", default="simple", type=click.Choice(['simple', 'minhash', 'toxic']), help="Detection mode")
@click.option("--content-key", default="text", help="JSON field containing text")
@click.option("--ngram-size", default=4, type=int, help="N-gram size")
@click.option("--sample-every-m-tokens", default=10, type=int, help="Sampling rate")
@click.option("--toxic-overlap-threshold", default=0.8, type=float, help="Overlap threshold")
@click.option("--toxic-score-threshold", default=2.0, type=float, help="Toxic score threshold")
@click.option("--tokenizer", default="word", type=click.Choice(['word', 'p50k', 'cl100k']), help="Tokenizer")
@click.option("--debug/--no-debug", default=False, help="Enable debug mode")
@click.option("--purify/--no-purify", default=False, help="Enable data purification")
# Orchestrator options
@click.option("--remote-file-input", help="S3 path for input data")
@click.option("--remote-report-output-dir", help="S3 path for reports")
@click.option("--remote-cleaned-output-dir", help="S3 path for cleaned files")
@click.option("--local-work-dir", default="/mnt/decon-work", help="Local working directory for orchestrator")
def deploy(name, owner, instances, instance_type, ssh_key, github_token, daemon_port,
          mode, content_key, ngram_size, sample_every_m_tokens,
          toxic_overlap_threshold, toxic_score_threshold, tokenizer,
          debug, purify, remote_file_input, remote_report_output_dir,
          remote_cleaned_output_dir, local_work_dir):
    """Deploy Decon cluster (non-interactive)"""
    deploy_config = DeploymentConfig(
        cluster_name=name,
        owner=owner,
        instance_count=instances,
        instance_type=instance_type,
        ssh_key_path=ssh_key,
        github_token=github_token,
        daemon_port=daemon_port,
        # Daemon settings
        mode=mode,
        content_key=content_key,
        ngram_size=ngram_size,
        sample_every_m_tokens=sample_every_m_tokens,
        toxic_overlap_threshold=toxic_overlap_threshold,
        toxic_score_threshold=toxic_score_threshold,
        tokenizer_str=tokenizer,
        debug=debug,
        purify=purify,
        # Orchestrator settings
        remote_file_input=remote_file_input,
        remote_report_output_dir=remote_report_output_dir,
        remote_cleaned_output_dir=remote_cleaned_output_dir,
        local_work_dir=local_work_dir,
    )

    try:
        manager = DeploymentManager(deploy_config)
        success = manager.full_deployment()
        sys.exit(0 if success else 1)
    except PoorManRayError as e:
        print(f"❌ {e}")
        sys.exit(1)


@cli.command()
@click.option("--name", required=True, help="Cluster name")
@click.option("--ssh-key", default="~/.ssh/id_rsa", help="Path to SSH private key")
def status(name, ssh_key):
    """Check status of a Decon deployment"""
    deploy_config = DeploymentConfig(
        cluster_name=name,
        ssh_key_path=ssh_key
    )

    try:
        manager = DeploymentManager(deploy_config)
        status = manager.check_status()

        print("📊 Cluster Status:")
        print(status["instances"])
        print("\n🏥 Daemon Health:")
        print(status["daemon_health"])

    except PoorManRayError as e:
        print(f"❌ {e}")
        sys.exit(1)


@cli.command()
@click.option("--name", required=True, help="Cluster name")
@click.option("--ssh-key", default="~/.ssh/id_rsa", help="Path to SSH private key")
@click.option("--log-type", type=click.Choice(["daemon", "orchestrator"]), default="daemon", help="Which log to view")
@click.option("--follow", is_flag=True, help="Follow log output")
def logs(name, ssh_key, log_type, follow):
    """View logs from a Decon deployment"""
    deploy_config = DeploymentConfig(
        cluster_name=name,
        ssh_key_path=ssh_key
    )

    try:
        manager = DeploymentManager(deploy_config)

        if follow:
            # Use poormanray directly for following logs
            cmd = f"cd decon && tail -f {log_type}.log"
            subprocess.run([
                "poormanray", "run",
                "--name", name,
                "--command", cmd,
                "--ssh-key-path", ssh_key
            ])
        else:
            logs = manager.view_logs(log_type)
            print(logs)

    except PoorManRayError as e:
        print(f"❌ {e}")
        sys.exit(1)


@cli.command()
@click.option("--name", required=True, help="Cluster name")
@click.confirmation_option(prompt="Are you sure you want to terminate this cluster?")
def terminate(name):
    """Terminate a Decon cluster"""
    deploy_config = DeploymentConfig(cluster_name=name)

    try:
        manager = DeploymentManager(deploy_config)
        success = manager.terminate_cluster()
        sys.exit(0 if success else 1)
    except PoorManRayError as e:
        print(f"❌ {e}")
        sys.exit(1)


@cli.command(name="polling-auto-terminate")
@click.option("--name", required=True, help="Cluster name")
@click.option("--poll-interval", default=60, help="Poll interval in seconds (default: 60)")
@click.option("--ssh-key", default="~/.ssh/id_rsa", help="Path to SSH private key")
def polling_auto_terminate(name, poll_interval, ssh_key):
    """Monitor orchestrator logs and auto-terminate when complete"""
    print("=====================================")
    print("Polling Auto-Terminate Monitor")
    print("=====================================")
    print(f"Cluster: {name}")
    print(f"Poll interval: {poll_interval}s")
    print(f"Monitoring for: WORK COMPLETE EXITING")
    print(f"Started at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=====================================\n")

    deploy_config = DeploymentConfig(
        cluster_name=name,
        ssh_key_path=ssh_key
    )

    try:
        manager = DeploymentManager(deploy_config)

        # Track consecutive failures
        consecutive_failures = 0
        max_failures = 5

        while True:
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Checking orchestrator logs...")

            try:
                # Get orchestrator logs
                logs = manager.view_logs("orchestrator")

                # Check for completion marker
                if "WORK COMPLETE EXITING" in logs:
                    print("\n=====================================")
                    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] DETECTED: WORK COMPLETE EXITING")
                    print("=====================================")
                    print("Work is complete! Initiating cluster termination...")
                    print()

                    # Small delay to ensure everything is properly flushed
                    time.sleep(5)

                    # Terminate the cluster (without confirmation prompt)
                    print(f"Executing cluster termination for '{name}'")
                    if manager.terminate_cluster():
                        print("\n=====================================")
                        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] SUCCESS: Cluster terminated")
                        print("=====================================")
                        sys.exit(0)
                    else:
                        print("\n=====================================")
                        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] ERROR: Failed to terminate cluster")
                        print("=====================================")
                        print(f"Please manually run: poormanray terminate --name {name}")
                        sys.exit(1)
                else:
                    # Reset failure counter on successful log retrieval
                    consecutive_failures = 0
                    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Work still in progress...")

            except subprocess.CalledProcessError as e:
                consecutive_failures += 1
                print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] WARNING: Failed to retrieve logs (attempt {consecutive_failures}/{max_failures})")

                if consecutive_failures >= max_failures:
                    print("\n=====================================")
                    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] ERROR: Too many consecutive failures")
                    print("=====================================")
                    print(f"Unable to retrieve logs after {max_failures} attempts.")
                    print("The cluster may have already been terminated or there may be a connection issue.")
                    print(f"Please check manually with: poormanray list --name {name}")
                    sys.exit(1)

            except Exception as e:
                print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] ERROR: Unexpected error: {e}")
                consecutive_failures += 1

                if consecutive_failures >= max_failures:
                    print("Too many errors, exiting...")
                    sys.exit(1)

            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Sleeping for {poll_interval}s...\n")
            time.sleep(poll_interval)

    except PoorManRayError as e:
        print(f"❌ {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n\nMonitoring cancelled by user")
        sys.exit(0)


# Low-level API functions for programmatic use (e.g., MCP server)

def create_deployment_manager(
    cluster_name: str,
    instance_count: int = 2,
    instance_type: str = "i4i.2xlarge",
    ssh_key_path: str = "~/.ssh/id_rsa",
    github_token: Optional[str] = None,
    **kwargs
) -> DeploymentManager:
    """
    Create a deployment manager instance for programmatic use.

    This is useful for MCP server integration or other automated tools.
    """
    config = DeploymentConfig(
        cluster_name=cluster_name,
        instance_count=instance_count,
        instance_type=instance_type,
        ssh_key_path=ssh_key_path,
        github_token=github_token,
        **kwargs
    )
    return DeploymentManager(config)


def get_deployment_stages() -> List[DeploymentStage]:
    """Get list of deployment stages for programmatic control"""
    return list(DeploymentStage)


if __name__ == "__main__":
    cli()
