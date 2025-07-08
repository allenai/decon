#!/usr/bin/env python3
"""
Orchestration script for distributed contamination detection.
Manages downloading files from S3, submitting to daemon, and uploading results.
"""

import os
import sys
import json
import yaml
import time
import signal
import hashlib
import logging
import subprocess
import multiprocessing
from pathlib import Path
from typing import Dict, List, Set, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict
from urllib.parse import urlparse

import boto3
import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry


# Configuration
@dataclass
class OrchestrationConfig:
    # Required fields
    remote_file_input: str  # s3://bucket/training-data/
    remote_report_output_dir: str  # s3://bucket/output/
    remote_cleaned_output_dir: Optional[str] = None  # s3://bucket/cleaned/
    daemon_url: str  # http://localhost:8080
    local_work_dir: str  # /tmp/decon-work/
    
    # Optional fields with defaults
    max_concurrent_jobs: Optional[int] = None  # Will be set based on daemon worker threads
    poll_interval: int = 5
    s5cmd_workers: int = 50
    cleanup_delay: int = 10
    
    @classmethod
    def from_yaml(cls, path: str) -> 'OrchestrationConfig':
        with open(path) as f:
            data = yaml.safe_load(f)
        
        # Extract required fields
        config = cls(
            remote_file_input=data['remote_file_input'],
            remote_report_output_dir=data['remote_report_output_dir'],
            remote_cleaned_output_dir=data.get('remote_cleaned_output_dir'),
            daemon_url=data.get('daemon_url', 'http://localhost:8080'),
            local_work_dir=data.get('local_work_dir', '/tmp/decon-work')
        )
        
        # Apply optional fields
        for field in ['max_concurrent_jobs', 'poll_interval', 's5cmd_workers', 'cleanup_delay']:
            if field in data:
                setattr(config, field, data[field])
        
        return config


@dataclass
class Job:
    job_id: str
    file_path: str
    s3_key: str
    status: str = 'submitted'
    output_path: Optional[str] = None
    purified_path: Optional[str] = None
    error: Optional[str] = None


class ContaminationOrchestrator:
    def __init__(self, config: OrchestrationConfig):
        self.config = config
        self.logger = self._setup_logging()
        self.session = self._setup_session()
        self.s3_client = boto3.client('s3')
        
        # Job tracking
        self.active_jobs: Dict[str, Job] = {}
        self.completed_files: Set[str] = set()
        self.failed_files: Set[str] = set()
        
        # Signal handling
        self.shutdown_requested = False
        self.force_shutdown = False
        self._setup_signal_handlers()
        
        # Host identification
        self.host_index = int(os.environ.get('PMR_HOST_INDEX', '0'))
        self.host_count = int(os.environ.get('PMR_HOST_COUNT', '1'))
        
        # Warn if using defaults (single host mode)
        if 'PMR_HOST_INDEX' not in os.environ and 'PMR_HOST_COUNT' not in os.environ:
            self.logger.warning("Using default host configuration (single host mode: PMR_HOST_INDEX=0, PMR_HOST_COUNT=1)")
            self.logger.warning("For distributed processing, set PMR_HOST_INDEX and PMR_HOST_COUNT environment variables")
        
        # Debug limit for development
        self.max_files_debug = None
        if 'MAX_FILES_DEBUG' in os.environ:
            self.max_files_debug = int(os.environ['MAX_FILES_DEBUG'])
            self.logger.warning(f"DEBUG MODE: Limited to processing {self.max_files_debug} files")
            self.logger.warning(f"DEBUG MODE: Uploads and cleanup are DISABLED")
        
        self.logger.info(f"Orchestrator initialized: host {self.host_index} of {self.host_count}")
    
    def _setup_logging(self) -> logging.Logger:
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger('decon-orchestrator')
    
    def _setup_session(self) -> requests.Session:
        session = requests.Session()
        retry = Retry(
            total=3,
            backoff_factor=0.3,
            status_forcelist=[500, 502, 503, 504]
        )
        adapter = HTTPAdapter(max_retries=retry)
        session.mount('http://', adapter)
        session.mount('https://', adapter)
        return session
    
    def _setup_signal_handlers(self):
        def signal_handler(signum, frame):
            if self.shutdown_requested:
                self.logger.warning("Force shutdown requested")
                self.force_shutdown = True
            else:
                self.logger.info("Graceful shutdown requested")
                self.shutdown_requested = True
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    def run(self):
        """Main orchestration loop."""
        start_time = time.time()
        
        self.logger.info("="*60)
        self.logger.info("Starting contamination detection orchestration")
        self.logger.info("="*60)
        
        # Validate daemon is running
        if not self._check_daemon_health():
            self.logger.error("Daemon is not running. Please start the daemon with: make daemon")
            sys.exit(1)
        
        # Setup working directories
        self._setup_directories()
        
        # Get list of files to process
        files_to_process = self._get_files_to_process()
        if not files_to_process:
            self.logger.info("No files to process")
            return
        
        self.logger.info(f"Starting processing of {len(files_to_process)} files")
        
        # Download and process files
        self._process_files(files_to_process)
        
        # Wait for remaining jobs to complete
        self._wait_for_completion()
        
        elapsed = time.time() - start_time
        self.logger.info("="*60)
        self.logger.info(f"Orchestration complete in {elapsed:.2f} seconds")
        self.logger.info("="*60)
    
    def _check_daemon_health(self) -> bool:
        """Check if the daemon is running and healthy."""
        try:
            response = self.session.get(f"{self.config.daemon_url}/health", timeout=5)
            if response.status_code == 200:
                data = response.json()
                self.logger.info(f"Daemon is healthy: {data['status']}")
                
                # Set max_concurrent_jobs based on daemon worker threads if not explicitly configured
                if self.config.max_concurrent_jobs is None:
                    worker_threads = data.get('worker_threads', 4)
                    self.config.max_concurrent_jobs = worker_threads * 4
                    self.logger.info(f"Setting max_concurrent_jobs to {self.config.max_concurrent_jobs} "
                                   f"(4x daemon worker threads: {worker_threads})")
                
                return True
        except Exception as e:
            self.logger.error(f"Failed to connect to daemon: {e}")
        return False
    
    def _setup_directories(self):
        """Create local working directories."""
        self.logger.info(f"Setting up working directories in {self.config.local_work_dir}")
        
        for subdir in ['download', 'results', 'cleaned']:
            path = Path(self.config.local_work_dir) / subdir
            path.mkdir(parents=True, exist_ok=True)
            self.logger.debug(f"Created directory: {path}")
    
    def _parse_s3_uri(self, uri: str) -> Tuple[str, str]:
        """Parse S3 URI into bucket and prefix."""
        parsed = urlparse(uri)
        if parsed.scheme != 's3':
            raise ValueError(f"Invalid S3 URI: {uri}")
        bucket = parsed.netloc
        prefix = parsed.path.lstrip('/')
        return bucket, prefix
    
    def _list_s3_files(self, s3_uri: str, suffixes: List[str] = None) -> List[str]:
        """List all files in S3 location with given suffixes."""
        if suffixes is None:
            suffixes = ['.jsonl', '.jsonl.gz', '.jsonl.zst', '.jsonl.bz2', '.jsonl.xz']
        
        bucket, prefix = self._parse_s3_uri(s3_uri)
        
        self.logger.info(f"Starting S3 file list for bucket={bucket}, prefix={prefix}")
        self.logger.info(f"Looking for files with suffixes: {suffixes}")
        
        files = []
        paginator = self.s3_client.get_paginator('list_objects_v2')
        pages = paginator.paginate(Bucket=bucket, Prefix=prefix)
        
        page_num = 0
        suffix_counts = {suffix: 0 for suffix in suffixes}
        
        for page in pages:
            page_num += 1
            if 'Contents' not in page:
                self.logger.debug(f"Page {page_num}: No contents")
                continue
            
            page_files = 0
            for obj in page['Contents']:
                key = obj['Key']
                for suffix in suffixes:
                    if key.endswith(suffix):
                        files.append(key)
                        page_files += 1
                        suffix_counts[suffix] += 1
                        break
            
            self.logger.info(f"Page {page_num}: Found {page_files} files (total so far: {len(files)})")
        
        self.logger.info(f"S3 listing complete: {len(files)} total files found")
        for suffix, count in suffix_counts.items():
            if count > 0:
                self.logger.info(f"  {suffix}: {count} files")
        
        return files
    
    def _get_processed_files(self) -> Set[str]:
        """Get set of already processed files from output directories."""
        processed = set()
        
        self.logger.info("Checking for already processed files...")
        
        # Check for clean markers in report directory
        clean_markers = self._list_s3_files(self.config.remote_report_output_dir, ['.clean'])
        
        # Also check for cleaned files if cleaned directory is configured
        cleaned_files = []
        if self.config.remote_cleaned_output_dir:
            cleaned_files = self._list_s3_files(self.config.remote_cleaned_output_dir, ['.clean.jsonl', '.clean.jsonl.gz', '.clean.jsonl.zst', '.clean.jsonl.bz2', '.clean.jsonl.xz'])
            self.logger.info(f"Found {len(cleaned_files)} cleaned files in cleaned directory")
        
        self.logger.info(f"Found {len(clean_markers)} clean markers")
        
        # Process clean markers
        for key in clean_markers:
            filename = os.path.basename(key)
            if filename.endswith('.clean'):
                # Clean file marker format: basename.clean
                basename = filename[:-6]  # Remove .clean
                # Add all possible variants of this file
                processed.add(f"{basename}.jsonl")
                for ext in ['.gz', '.zst', '.bz2', '.xz']:
                    processed.add(f"{basename}.jsonl{ext}")
        
        
        # Process cleaned files
        for key in cleaned_files:
            filename = os.path.basename(key)
            # Extract base name from cleaned filename
            # Expected format: basename.clean.jsonl or basename.clean.jsonl.gz
            basename = None
            for ext in ['.clean.jsonl.gz', '.clean.jsonl.zst', '.clean.jsonl.bz2', '.clean.jsonl.xz', '.clean.jsonl']:
                if filename.endswith(ext):
                    basename = filename[:-len(ext)]
                    break
            
            if basename:
                # Add all possible variants of this file
                processed.add(f"{basename}.jsonl")
                for ext in ['.gz', '.zst', '.bz2', '.xz']:
                    processed.add(f"{basename}.jsonl{ext}")
        
        self.logger.info(f"Total already processed files: {len(processed)}")
        return processed
    
    def _should_process_file(self, s3_key: str) -> bool:
        """Check if this host should process the given file."""
        # Hash the S3 key to get a consistent number
        hash_value = int(hashlib.md5(s3_key.encode()).hexdigest(), 16)
        return (hash_value % self.host_count) == self.host_index
    
    def _get_files_to_process(self) -> List[str]:
        """Get list of files this host should process."""
        self.logger.info("Determining files to process...")
        
        # Get all training files
        all_files = self._list_s3_files(self.config.remote_file_input)
        self.logger.info(f"Found {len(all_files)} total training files")
        
        # Get already processed files
        processed = self._get_processed_files()
        
        # Filter for this host
        files_to_process = []
        skipped_processed = 0
        skipped_other_host = 0
        
        self.logger.info(f"Filtering files for host {self.host_index} of {self.host_count}...")
        
        for i, s3_key in enumerate(all_files):
            if i > 0 and i % 1000 == 0:
                self.logger.info(f"  Processed {i}/{len(all_files)} files...")
            
            filename = os.path.basename(s3_key)
            
            # Skip if already processed
            if filename in processed:
                skipped_processed += 1
                continue
            
            # Skip if not assigned to this host
            if not self._should_process_file(s3_key):
                skipped_other_host += 1
                continue
            
            files_to_process.append(s3_key)
            
            # Check debug limit
            if self.max_files_debug and len(files_to_process) >= self.max_files_debug:
                self.logger.warning(f"DEBUG MODE: Limiting to {self.max_files_debug} files")
                break
        
        self.logger.info(f"File filtering complete:")
        self.logger.info(f"  - Files for this host: {len(files_to_process)}")
        self.logger.info(f"  - Already processed: {skipped_processed}")
        self.logger.info(f"  - Assigned to other hosts: {skipped_other_host}")
        
        return files_to_process
    
    def _create_s5cmd_file(self, s3_keys: List[str], command_file: Path) -> Path:
        """Create s5cmd command file for downloading files."""
        bucket, _ = self._parse_s3_uri(self.config.remote_file_input)
        download_dir = Path(self.config.local_work_dir) / 'download'
        
        with open(command_file, 'w') as f:
            for s3_key in s3_keys:
                local_path = download_dir / os.path.basename(s3_key)
                f.write(f"cp s3://{bucket}/{s3_key} {local_path}\n")
        
        return command_file
    
    def _download_batch(self, s3_keys: List[str]) -> List[str]:
        """Download a batch of files using s5cmd."""
        command_file = Path(self.config.local_work_dir) / f"download_batch_{time.time()}.txt"
        self._create_s5cmd_file(s3_keys, command_file)
        
        # Run s5cmd
        cmd = ['s5cmd', '--numworkers', str(self.config.s5cmd_workers), 'run', str(command_file)]
        self.logger.info(f"Starting batch download of {len(s3_keys)} files with {self.config.s5cmd_workers} workers")
        
        try:
            start_time = time.time()
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            elapsed = time.time() - start_time
            
            self.logger.info(f"s5cmd batch download completed in {elapsed:.2f} seconds")
            if result.stdout:
                lines = result.stdout.strip().split('\n')
                if len(lines) > 10:
                    self.logger.debug(f"s5cmd output (first 10 lines):\n{chr(10).join(lines[:10])}")
                else:
                    self.logger.debug(f"s5cmd output:\n{result.stdout}")
            
            # Return list of downloaded files
            download_dir = Path(self.config.local_work_dir) / 'download'
            downloaded = []
            missing = []
            
            for s3_key in s3_keys:
                local_file = download_dir / os.path.basename(s3_key)
                if local_file.exists():
                    downloaded.append((str(local_file), s3_key))
                else:
                    missing.append(s3_key)
            
            self.logger.info(f"Download verification: {len(downloaded)} successful, {len(missing)} missing")
            if missing:
                self.logger.error(f"Failed to download {len(missing)} files:")
                for m in missing[:5]:  # Show first 5
                    self.logger.error(f"  - {m}")
                if len(missing) > 5:
                    self.logger.error(f"  ... and {len(missing) - 5} more")
            
            return downloaded
        
        except subprocess.CalledProcessError as e:
            self.logger.error(f"s5cmd failed with exit code {e.returncode}")
            if e.stderr:
                self.logger.error(f"s5cmd stderr: {e.stderr}")
            raise
        
        finally:
            # Cleanup command file
            if command_file.exists():
                command_file.unlink()
    
    def _submit_to_daemon(self, file_path: str, s3_key: str) -> Optional[Job]:
        """Submit a file to the daemon for processing."""
        try:
            response = self.session.post(
                f"{self.config.daemon_url}/submit",
                json={"file_path": file_path}
            )
            
            if response.status_code == 200:
                data = response.json()
                job = Job(
                    job_id=data['job_id'],
                    file_path=file_path,
                    s3_key=s3_key,
                    status='submitted'
                )
                self.logger.info(f"Submitted job {job.job_id} for {os.path.basename(s3_key)}")
                return job
            else:
                self.logger.error(f"Failed to submit {file_path}: {response.text}")
                return None
        
        except Exception as e:
            self.logger.error(f"Error submitting {file_path}: {e}")
            return None
    
    def _poll_job_status(self, job: Job) -> bool:
        """Poll job status and update job object. Returns True if complete."""
        try:
            response = self.session.get(f"{self.config.daemon_url}/status/{job.job_id}")
            
            if response.status_code == 200:
                data = response.json()
                job.status = data['status']
                
                if job.status == 'completed':
                    job.output_path = data.get('output_path')
                    job.purified_path = data.get('purified_path')
                    self.logger.info(f"Job {job.job_id} completed")
                    return True
                
                elif job.status == 'failed':
                    job.error = data.get('error', 'Unknown error')
                    self.logger.error(f"Job {job.job_id} failed: {job.error}")
                    return True
            
            else:
                self.logger.error(f"Failed to get status for job {job.job_id}: {response.text}")
        
        except Exception as e:
            self.logger.error(f"Error polling job {job.job_id}: {e}")
        
        return False
    
    def _upload_results(self, job: Job):
        """Upload job results to S3."""
        basename = os.path.basename(job.s3_key)
        basename_no_ext = basename
        
        # Remove compression extensions to get base name
        for ext in ['.gz', '.zst', '.bz2', '.xz']:
            if basename_no_ext.endswith('.jsonl' + ext):
                basename_no_ext = basename_no_ext[:-len(ext)]
                break
        
        # Remove .jsonl to get pure base name
        if basename_no_ext.endswith('.jsonl'):
            basename_no_ext = basename_no_ext[:-6]
        
        # Skip uploads in debug mode
        if self.max_files_debug:
            self.logger.info(f"DEBUG MODE: Skipping upload for {basename}")
            if job.status == 'completed':
                self.logger.info(f"  Output would be at: {job.output_path}")
                if job.purified_path:
                    self.logger.info(f"  Purified would be at: {job.purified_path}")
            return
        
        # Handle successful completion
        if job.status == 'completed':
            # Check if contamination was found
            if job.output_path and os.path.exists(job.output_path):
                # Check if file has content
                if os.path.getsize(job.output_path) > 0:
                    # Upload contamination report with new naming
                    report_filename = f"{basename_no_ext}.report.jsonl"
                    self._upload_file(
                        job.output_path,
                        self.config.remote_report_output_dir,
                        report_filename
                    )
                    
                    # Upload purified file if it exists
                    if job.purified_path and os.path.exists(job.purified_path):
                        # Determine compression based on original file
                        compression_ext = ""
                        for ext in ['.gz', '.zst', '.bz2', '.xz']:
                            if basename.endswith('.jsonl' + ext):
                                compression_ext = ext
                                break
                        
                        cleaned_filename = f"{basename_no_ext}.clean.jsonl{compression_ext}"
                        # Use cleaned output dir if configured, otherwise use report dir
                        cleaned_output_dir = self.config.remote_cleaned_output_dir or self.config.remote_report_output_dir
                        self._upload_file(
                            job.purified_path,
                            cleaned_output_dir,
                            cleaned_filename
                        )
                else:
                    # No contamination found - create clean marker
                    self._create_clean_marker(basename_no_ext)
            else:
                # No output file means no contamination
                self._create_clean_marker(basename_no_ext)
        
        # Handle failure
        elif job.status == 'failed':
            self.failed_files.add(basename)
            self.logger.error(f"Job failed for {basename}: {job.error}")
    
    def _create_clean_marker(self, filename: str):
        """Create a marker file indicating no contamination."""
        marker_path = Path(self.config.local_work_dir) / 'results' / f"{filename}.clean"
        with open(marker_path, 'w') as f:
            f.write("clean!\n")
        
        self._upload_file(
            str(marker_path),
            self.config.remote_report_output_dir,
            f"{filename}.clean"
        )
        
        # Cleanup local marker
        marker_path.unlink()
    
    def _upload_file(self, local_path: str, s3_base: str, key_suffix: str):
        """Upload a single file to S3 using s5cmd."""
        bucket, prefix = self._parse_s3_uri(s3_base)
        s3_key = f"{prefix}/{key_suffix}" if prefix else key_suffix
        s3_uri = f"s3://{bucket}/{s3_key}"
        
        cmd = ['s5cmd', 'cp', local_path, s3_uri]
        
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            self.logger.info(f"Uploaded {os.path.basename(local_path)} to {s3_uri}")
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Failed to upload {local_path}: {e.stderr}")
            raise
    
    def _cleanup_job_files(self, job: Job):
        """Clean up local files for a completed job."""
        # Skip cleanup in debug mode
        if self.max_files_debug:
            self.logger.info(f"DEBUG MODE: Skipping cleanup for job {job.job_id}")
            return
        
        files_to_remove = [job.file_path]
        
        if job.output_path and os.path.exists(job.output_path):
            files_to_remove.append(job.output_path)
        
        if job.purified_path and os.path.exists(job.purified_path):
            files_to_remove.append(job.purified_path)
        
        for file_path in files_to_remove:
            try:
                os.unlink(file_path)
                self.logger.debug(f"Cleaned up {file_path}")
            except Exception as e:
                self.logger.warning(f"Failed to cleanup {file_path}: {e}")
    
    def _process_files(self, files_to_process: List[str]):
        """Main processing loop for files."""
        # Ensure max_concurrent_jobs has a value (fallback to 10 if somehow still None)
        if self.config.max_concurrent_jobs is None:
            self.logger.warning("max_concurrent_jobs not set, using default of 10")
            self.config.max_concurrent_jobs = 10
            
        batch_size = self.config.max_concurrent_jobs
        total_batches = (len(files_to_process) + batch_size - 1) // batch_size
        
        self.logger.info(f"Starting file processing: {len(files_to_process)} files in {total_batches} batches")
        
        for batch_num, i in enumerate(range(0, len(files_to_process), batch_size), 1):
            if self.shutdown_requested:
                self.logger.info("Shutdown requested, stopping file processing")
                break
            
            # Download batch
            batch = files_to_process[i:i+batch_size]
            self.logger.info(f"Processing batch {batch_num}/{total_batches} ({len(batch)} files)")
            
            downloaded_files = self._download_batch(batch)
            
            # Submit downloaded files
            submitted_count = 0
            for local_path, s3_key in downloaded_files:
                if self.shutdown_requested:
                    break
                
                job = self._submit_to_daemon(local_path, s3_key)
                if job:
                    self.active_jobs[job.job_id] = job
                    submitted_count += 1
            
            self.logger.info(f"Submitted {submitted_count} jobs from batch {batch_num}")
            
            # Poll and process active jobs
            while len(self.active_jobs) >= self.config.max_concurrent_jobs:
                if self.force_shutdown:
                    self.logger.warning("Force shutdown, abandoning active jobs")
                    return
                
                self.logger.debug(f"Active jobs at capacity ({len(self.active_jobs)}), polling...")
                self._poll_and_process_jobs()
                time.sleep(self.config.poll_interval)
    
    def _poll_and_process_jobs(self):
        """Poll active jobs and process completed ones."""
        completed_jobs = []
        
        self.logger.debug(f"Polling {len(self.active_jobs)} active jobs...")
        
        for job_id, job in self.active_jobs.items():
            if self._poll_job_status(job):
                completed_jobs.append(job_id)
        
        if completed_jobs:
            self.logger.info(f"Found {len(completed_jobs)} completed jobs")
        
        # Process completed jobs
        for job_id in completed_jobs:
            job = self.active_jobs.pop(job_id)
            filename = os.path.basename(job.s3_key)
            
            # Upload results
            try:
                self._upload_results(job)
                self.completed_files.add(filename)
                self.logger.info(f"Successfully processed: {filename}")
            except Exception as e:
                self.logger.error(f"Failed to upload results for job {job_id}: {e}")
                self.failed_files.add(filename)
            
            # Cleanup with delay
            if not self.max_files_debug:
                time.sleep(self.config.cleanup_delay)
            self._cleanup_job_files(job)
        
        # Log progress
        if completed_jobs:
            self.logger.info(f"Progress: {len(self.completed_files)} completed, "
                           f"{len(self.active_jobs)} active, {len(self.failed_files)} failed")
    
    def _wait_for_completion(self):
        """Wait for all remaining jobs to complete."""
        if not self.active_jobs:
            self.logger.info("No remaining jobs to wait for")
            return
        
        self.logger.info(f"Waiting for {len(self.active_jobs)} remaining jobs to complete")
        
        wait_cycles = 0
        while self.active_jobs and not self.force_shutdown:
            wait_cycles += 1
            if wait_cycles % 12 == 0:  # Log every minute (assuming 5 second poll interval)
                self.logger.info(f"Still waiting for {len(self.active_jobs)} jobs...")
            
            self._poll_and_process_jobs()
            
            if self.active_jobs:
                time.sleep(self.config.poll_interval)
        
        if self.force_shutdown and self.active_jobs:
            self.logger.warning(f"Force shutdown with {len(self.active_jobs)} active jobs")
        
        # Final summary
        self.logger.info(f"Completed: {len(self.completed_files)} files")
        if self.failed_files:
            self.logger.warning(f"Failed: {len(self.failed_files)} files")
            for filename in self.failed_files:
                self.logger.warning(f"  - {filename}")
        
        # Debug mode summary
        if self.max_files_debug:
            self.logger.warning(f"DEBUG MODE: Results are in {self.config.local_work_dir}")
            self.logger.warning(f"DEBUG MODE: No files were uploaded or cleaned up")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Contamination detection orchestrator')
    parser.add_argument('--config', required=True, help='Path to configuration YAML file')
    args = parser.parse_args()
    
    # Load configuration
    try:
        config = OrchestrationConfig.from_yaml(args.config)
    except Exception as e:
        print(f"Failed to load configuration: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Run orchestrator
    orchestrator = ContaminationOrchestrator(config)
    orchestrator.run()


if __name__ == '__main__':
    main()