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
import threading
from queue import Queue, Empty
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
    # Required fields (no defaults)
    remote_file_input: str  # s3://bucket/training-data/
    remote_report_output_dir: str  # s3://bucket/output/
    daemon_url: str  # http://localhost:8080
    local_work_dir: str  # /tmp/decon-work/

    # Optional fields with defaults
    remote_cleaned_output_dir: Optional[str] = None  # s3://bucket/cleaned/
    max_concurrent_jobs: Optional[int] = None  # Will be set based on daemon worker threads
    poll_interval: int = 5
    s5cmd_workers: int = 50
    cleanup_delay: int = 10
    download_batch_size: int = 50  # Files per download batch
    download_queue_max: int = 200  # Max files in download queue
    upload_queue_max: int = 100  # Max files in upload queue
    upload_batch_size: int = 50  # Jobs per upload batch
    upload_batch_timeout: int = 10  # Seconds to wait before processing partial batch

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
        for field in ['max_concurrent_jobs', 'poll_interval', 's5cmd_workers', 'cleanup_delay',
                      'download_batch_size', 'download_queue_max', 'upload_queue_max',
                      'upload_batch_size', 'upload_batch_timeout']:
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

        # Summary statistics
        self.stats_files_processed: int = 0
        self.stats_files_clean: int = 0
        self.stats_files_contaminated: int = 0
        self.stats_report_files_uploaded: int = 0
        self.stats_cleaned_files_uploaded: int = 0

        # Thread-safe queues
        self.download_queue = Queue(maxsize=config.download_queue_max)  # (local_path, s3_key) tuples
        self.upload_queue = Queue(maxsize=config.upload_queue_max)  # Completed Job objects

        # Tracking
        self.files_remaining: int = 0  # Total files left to process
        self.files_to_download: List[str] = []  # S3 keys still to download
        self.download_index = 0  # Current position in files_to_download

        # Thread control
        self.threads_running = False
        self.download_thread = None
        self.upload_thread = None

        # Performance metrics
        self.metrics_start_time = time.time()
        self.last_metrics_time = time.time()
        self.total_downloaded = 0
        self.total_submitted = 0
        self.total_uploaded = 0

        # Daemon configuration (loaded from health check)
        self.daemon_report_dir = None
        self.daemon_cleaned_dir = None
        self.daemon_purify = False


        # Host identification
        self.host_index = int(os.environ.get('PMR_HOST_INDEX', '0'))
        self.host_count = int(os.environ.get('PMR_HOST_COUNT', '1'))

        # Warn if using defaults (single host mode)
        if 'PMR_HOST_INDEX' not in os.environ and 'PMR_HOST_COUNT' not in os.environ:
            self.logger.warning("Single host mode: Set PMR_HOST_INDEX and PMR_HOST_COUNT for distributed processing")

        # Debug limit for development
        self.max_files_debug = None
        if 'MAX_FILES_DEBUG' in os.environ:
            self.max_files_debug = int(os.environ['MAX_FILES_DEBUG'])
            self.logger.warning(f"DEBUG MODE: Limited to processing {self.max_files_debug} files")

        # Orchestrator initialized

    def _setup_logging(self) -> logging.Logger:
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
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


    def _log_performance_metrics(self, force=False):
        """Log performance metrics periodically."""
        current_time = time.time()
        elapsed_since_last = current_time - self.last_metrics_time

        # Log every 30 seconds or when forced
        if not force and elapsed_since_last < 30:
            return

        total_elapsed = current_time - self.metrics_start_time

        # Calculate rates
        download_rate = self.total_downloaded / total_elapsed if total_elapsed > 0 else 0
        submit_rate = self.total_submitted / total_elapsed if total_elapsed > 0 else 0
        upload_rate = self.total_uploaded / total_elapsed if total_elapsed > 0 else 0

        self.logger.info("PERFORMANCE METRICS:")
        self.logger.info(f"  Elapsed time: {total_elapsed:.1f}s")
        self.logger.info(f"  Files remaining: {self.files_remaining}")
        self.logger.info(f"  Download queue: {self.download_queue.qsize()} files")
        self.logger.info(f"  Active jobs: {len(self.active_jobs)}")
        self.logger.info(f"  Upload queue: {self.upload_queue.qsize()} files")
        self.logger.info(f"  Rates: Download={download_rate:.1f}/s, Submit={submit_rate:.1f}/s, Upload={upload_rate:.1f}/s")
        self.logger.info(f"  Total: Downloaded={self.total_downloaded}, Submitted={self.total_submitted}, Uploaded={self.total_uploaded}")
        self.logger.info("=" * 60)

        self.last_metrics_time = current_time

    def run(self):
        """Main orchestration loop."""
        start_time = time.time()

        self.logger.warning("Starting contamination detection orchestration")

        # Validate daemon is running
        if not self._check_daemon_health():
            self.logger.error("Daemon is not running. Please start the daemon with: make daemon")
            sys.exit(1)

        # Setup working directories
        self._setup_directories()

        # Get list of files to process
        files_to_process = self._get_files_to_process()
        if not files_to_process:
            self.logger.warning("No files to process")
            return

        self.logger.warning(f"Processing {len(files_to_process)} files")

        # Initialize metrics
        self.files_remaining = len(files_to_process)
        self.metrics_start_time = time.time()

        # Download and process files
        self._process_files(files_to_process)

        # Wait for remaining jobs to complete
        self._wait_for_completion()

        elapsed = time.time() - start_time

        # Print summary statistics
        self.logger.warning(f"\nSUMMARY: Processed={self.stats_files_processed} Clean={self.stats_files_clean} Contaminated={self.stats_files_contaminated} Failed={len(self.failed_files)} Time={elapsed:.2f}s")

    def _check_daemon_health(self) -> bool:
        """Check if the daemon is running and healthy."""
        try:
            response = self.session.get(f"{self.config.daemon_url}/health", timeout=5)
            if response.status_code == 200:
                data = response.json()
                # Daemon is healthy

                # Set max_concurrent_jobs based on daemon worker threads if not explicitly configured
                if self.config.max_concurrent_jobs is None:
                    worker_threads = data.get('worker_threads', 4)
                    self.config.max_concurrent_jobs = worker_threads * 4
                    # Set max_concurrent_jobs

                # Extract daemon's output directories
                self.daemon_report_dir = data.get('report_output_dir')
                self.daemon_cleaned_dir = data.get('cleaned_output_dir')
                self.daemon_purify = data.get('purify', False)

                # Daemon output directories configured

                return True
        except Exception as e:
            self.logger.error(f"Failed to connect to daemon: {e}")
        return False

    def _setup_directories(self):
        """Create local working directories."""
        # Setting up working directories

        for subdir in ['download', 'results', 'cleaned']:
            path = Path(self.config.local_work_dir) / subdir
            path.mkdir(parents=True, exist_ok=True)
            # Created directory

    def _parse_s3_uri(self, uri: str) -> Tuple[str, str]:
        """Parse S3 URI into bucket and prefix."""
        parsed = urlparse(uri)
        if parsed.scheme != 's3':
            raise ValueError(f"Invalid S3 URI: {uri}")
        bucket = parsed.netloc
        prefix = parsed.path.lstrip('/')
        # Parsed S3 URI
        return bucket, prefix

    def _list_s3_files(self, s3_uri: str, suffixes: List[str] = None) -> List[str]:
        """List all files in S3 location with given suffixes."""
        if suffixes is None:
            suffixes = ['.jsonl', '.jsonl.gz', '.jsonl.zst', '.jsonl.bz2', '.jsonl.xz']

        bucket, prefix = self._parse_s3_uri(s3_uri)

        # Listing S3 files

        files = []
        paginator = self.s3_client.get_paginator('list_objects_v2')
        pages = paginator.paginate(Bucket=bucket, Prefix=prefix)

        page_num = 0
        suffix_counts = {suffix: 0 for suffix in suffixes}

        for page in pages:
            page_num += 1
            if 'Contents' not in page:
                # No contents in page
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

            # Processing page

        # S3 listing complete

        return files

    def _get_processed_files(self) -> Set[str]:
        """Get set of already processed files from output directories."""
        processed = set()

        # Checking for already processed files

        # Check for local clean markers
        clean_markers = []
        clean_markers_file = Path(self.config.local_work_dir) / 'clean_markers.txt'
        if clean_markers_file.exists():
            with open(clean_markers_file, 'r') as f:
                clean_markers = [line.strip() for line in f if line.strip()]

        # Also check for cleaned files if cleaned directory is configured
        cleaned_files = []
        if self.config.remote_cleaned_output_dir:
            cleaned_files = self._list_s3_files(self.config.remote_cleaned_output_dir, ['.clean.jsonl', '.clean.jsonl.gz', '.clean.jsonl.zst', '.clean.jsonl.bz2', '.clean.jsonl.xz'])
            # Found cleaned files

        # Found clean markers

        # Process clean markers (now just filenames)
        for basename in clean_markers:
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

        # Already processed files checked
        return processed

    def _should_process_file(self, s3_key: str) -> bool:
        """Check if this host should process the given file."""
        # Hash the S3 key to get a consistent number
        hash_value = int(hashlib.md5(s3_key.encode()).hexdigest(), 16)
        return (hash_value % self.host_count) == self.host_index

    def _get_files_to_process(self) -> List[str]:
        """Get list of files this host should process."""
        # Determining files to process

        # Get all training files
        all_files = self._list_s3_files(self.config.remote_file_input)
        # Found training files

        # Get already processed files
        processed = self._get_processed_files()

        # Filter for this host
        files_to_process = []
        skipped_processed = 0
        skipped_other_host = 0

        # Filtering files

        for i, s3_key in enumerate(all_files):
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

        self.logger.warning(f"Files to process: {len(files_to_process)} (skipped: {skipped_processed} already processed, {skipped_other_host} for other hosts)")

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
        # Starting batch download

        try:
            start_time = time.time()
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            elapsed = time.time() - start_time

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

            # Download verified
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
                # Job submitted
                self.total_submitted += 1
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
                    # Job completed
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


    def _add_clean_marker(self, filename: str):
        """Add filename to local clean markers list."""
        clean_markers_file = Path(self.config.local_work_dir) / 'clean_markers.txt'
        with open(clean_markers_file, 'a') as f:
            f.write(f"{filename}\n")


    def _upload_file(self, local_path: str, s3_base: str, key_suffix: str):
        """Upload a single file to S3 using s5cmd."""
        bucket, prefix = self._parse_s3_uri(s3_base)
        s3_key = f"{prefix}/{key_suffix}" if prefix else key_suffix
        s3_uri = f"s3://{bucket}/{s3_key}"

        cmd = ['s5cmd', 'cp', local_path, s3_uri]

        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            # File uploaded
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Failed to upload {local_path}: {e.stderr}")
            raise

    def _batch_upload_files(self, upload_commands: List[Tuple[str, str, str]]):
        """Upload multiple files using a single s5cmd command with parallel workers."""
        if not upload_commands:
            return

        # Create command file for batch upload
        commands_file = Path(self.config.local_work_dir) / 'upload_commands.txt'

        try:
            with open(commands_file, 'w') as f:
                for local_path, s3_base, key_suffix in upload_commands:
                    bucket, prefix = self._parse_s3_uri(s3_base)
                    s3_key = f"{prefix}/{key_suffix}" if prefix else key_suffix
                    s3_uri = f"s3://{bucket}/{s3_key}"
                    f.write(f"cp {local_path} {s3_uri}\n")

            # Execute batch upload with parallel workers
            cmd = [
                's5cmd',
                f'--numworkers={self.config.s5cmd_workers}',
                'run',
                str(commands_file)
            ]

            # Executing batch upload

            result = subprocess.run(cmd, check=True, capture_output=True, text=True)

            # s5cmd completed

            # Batch upload successful

        except subprocess.CalledProcessError as e:
            self.logger.error(f"Batch upload failed: {e.stderr}")
            # Fall back to individual uploads for debugging
            for local_path, s3_base, key_suffix in upload_commands:
                try:
                    self._upload_file(local_path, s3_base, key_suffix)
                except Exception as upload_error:
                    self.logger.error(f"Individual upload failed for {local_path}: {upload_error}")
                    raise
        finally:
            # Clean up command file
            if commands_file.exists():
                commands_file.unlink()

    def _cleanup_job_files(self, job: Job):
        """Clean up local files for a completed job."""

        files_to_remove = [job.file_path]

        if job.output_path and os.path.exists(job.output_path):
            files_to_remove.append(job.output_path)

        if job.purified_path and os.path.exists(job.purified_path):
            files_to_remove.append(job.purified_path)

        for file_path in files_to_remove:
            try:
                os.unlink(file_path)
            except Exception as e:
                self.logger.warning(f"Failed to cleanup {file_path}: {e}")

    def _download_worker(self):
        """Worker thread that continuously downloads files."""
        # Download worker started

        while self.threads_running:
            # Check if we need to download more files
            if self.download_queue.qsize() >= self.config.download_queue_max * 0.75:
                time.sleep(5)  # Queue is getting full, wait
                continue

            # Get next batch of files to download
            batch_size = min(self.config.download_batch_size, len(self.files_to_download) - self.download_index)
            if batch_size <= 0:
                # No more files to download
                break

            batch = self.files_to_download[self.download_index:self.download_index + batch_size]
            self.download_index += batch_size

            # Download the batch
            try:
                downloaded = self._download_batch(batch)

                # Add successfully downloaded files to queue
                for local_path, s3_key in downloaded:
                    # Verify file before adding to queue
                    if os.path.exists(local_path) and os.path.getsize(local_path) > 0:
                        self.download_queue.put((local_path, s3_key))
                        self.total_downloaded += 1
                        self.files_remaining -= 1
                    else:
                        self.logger.error(f"Downloaded file invalid: {local_path}")

            except Exception as e:
                self.logger.error(f"Download worker error: {e}")
                time.sleep(5)  # Wait before retrying

        # Download worker finished

    def _upload_worker(self):
        """Worker thread that continuously uploads results in batches."""
        # Upload worker started

        batch_size = self.config.upload_batch_size
        job_batch = []
        last_batch_time = time.time()

        while self.threads_running or not self.upload_queue.empty():
            try:
                # Try to collect a batch of jobs
                timeout = 1 if job_batch else 5  # Shorter timeout if we have pending jobs

                try:
                    job = self.upload_queue.get(timeout=timeout)
                    job_batch.append(job)
                    # Added job to batch
                except Empty:
                    pass  # No new jobs available

                # Process batch if we have enough jobs or it's been too long
                current_time = time.time()
                time_since_last_batch = current_time - last_batch_time
                should_process = (
                    len(job_batch) >= batch_size or
                    (job_batch and time_since_last_batch > self.config.upload_batch_timeout) or
                    (not self.threads_running and job_batch)  # Final batch on shutdown
                )

                if should_process and job_batch:
                    # Processing upload batch
                    self._process_job_batch(job_batch)
                    job_batch.clear()
                    last_batch_time = current_time

            except Exception as e:
                self.logger.error(f"Upload worker error: {e}")

        # Process any remaining jobs
        if job_batch:
            self._process_job_batch(job_batch)

        # Upload worker finished

    def _process_job_batch(self, jobs: List[Job]):
        """Process a batch of completed jobs."""
        upload_commands = []
        clean_markers = []

        # Processing completed jobs

        for job in jobs:
            filename = os.path.basename(job.s3_key)

            try:
                # Collect upload commands for this job
                job_uploads, job_markers = self._collect_upload_commands(job)
                # Job uploads collected

                upload_commands.extend(job_uploads)

                clean_markers.extend(job_markers)

                self.completed_files.add(filename)
                self.stats_files_processed += 1

            except Exception as e:
                self.logger.error(f"Failed to collect uploads for job {job.job_id}: {e}")
                self.failed_files.add(filename)

        # Execute batch upload
        # Upload commands collected
        upload_successful = False
        cleanup_allowed = False  # Separate flag for cleanup

        if upload_commands:
            # Executing batch upload
            try:
                self._batch_upload_files(upload_commands)
                self.total_uploaded += len(upload_commands)
                # Batch upload completed
                upload_successful = True
                cleanup_allowed = True  # Allow cleanup after successful upload
            except Exception as e:
                self.logger.error(f"Batch upload failed: {e}", exc_info=True)
                # Mark all jobs in this batch as failed
                for job in jobs:
                    self.failed_files.add(os.path.basename(job.s3_key))
                # DO NOT clean up files if upload failed!
                self.logger.error("Skipping cleanup due to upload failure - files retained for retry")
                return
        else:
            # All jobs were clean
            # For clean files, we should NOT delete the output files
            # They might be needed for verification or other purposes
            cleanup_allowed = False
            # Skipping cleanup for clean files

        # Add clean markers to local list (only if upload was successful or files were clean)
        if upload_successful or (not upload_commands and clean_markers):
            for marker_data in clean_markers:
                try:
                    self._add_clean_marker(marker_data)
                    self.stats_files_clean += 1
                except Exception as e:
                    self.logger.error(f"Failed to add clean marker: {e}")

        # Cleanup job files (only if cleanup is allowed)
        if cleanup_allowed:
            # Cleaning up files
            for job in jobs:
                time.sleep(0.1)  # Brief delay for cleanup
                self._cleanup_job_files(job)
        else:
            self.logger.warning("Skipping file cleanup - files retained")

    def _collect_upload_commands(self, job: Job) -> Tuple[List[Tuple[str, str, str]], List[str]]:
        """Collect upload commands for a completed job.
        Returns: (upload_commands, clean_markers)
        """
        upload_commands = []
        clean_markers = []

        basename = os.path.basename(job.s3_key)
        basename_no_ext = basename

        # Collecting upload commands

        # Remove compression extensions to get base name
        for ext in ['.gz', '.zst', '.bz2', '.xz']:
            if basename_no_ext.endswith('.jsonl' + ext):
                basename_no_ext = basename_no_ext[:-len(ext)]
                break

        # Remove .jsonl to get pure base name
        if basename_no_ext.endswith('.jsonl'):
            basename_no_ext = basename_no_ext[:-6]

        if job.status == 'completed':
            # Use the output paths provided by the daemon
            report_path = job.output_path
            cleaned_path = job.purified_path

            # Job completed

            # Verify the files exist before trying to upload
            if report_path and not os.path.exists(report_path):
                self.logger.error(f"Daemon reported output_path {report_path} but file does not exist!")
                report_path = None

            if cleaned_path and not os.path.exists(cleaned_path):
                self.logger.error(f"Daemon reported purified_path {cleaned_path} but file does not exist!")
                cleaned_path = None

            # Add upload commands
            if report_path:
                # Use the original filename from the daemon
                report_filename = os.path.basename(report_path)
                upload_commands.append((report_path, self.config.remote_report_output_dir, report_filename))
                self.stats_files_contaminated += 1
                self.stats_report_files_uploaded += 1
                # Will upload report
            else:
                # No contamination found - mark for clean marker creation
                clean_markers.append(basename_no_ext)
                # No report file, marking as clean

            # Always check for cleaned file and upload if it exists
            if cleaned_path:
                # Use the original filename from the daemon
                cleaned_filename = os.path.basename(cleaned_path)
                cleaned_output_dir = self.config.remote_cleaned_output_dir or self.config.remote_report_output_dir
                upload_commands.append((cleaned_path, cleaned_output_dir, cleaned_filename))
                self.stats_cleaned_files_uploaded += 1
                # Will upload cleaned file

        elif job.status == 'failed':
            self.logger.error(f"Job failed for {basename}: {job.error}")

        return upload_commands, clean_markers

    def _process_files(self, files_to_process: List[str]):
        """Main processing loop with concurrent download/upload threads."""
        # Ensure max_concurrent_jobs has a value
        if self.config.max_concurrent_jobs is None:
            self.logger.warning("max_concurrent_jobs not set, using default of 10")
            self.config.max_concurrent_jobs = 10

        # Initialize files to download
        self.files_to_download = files_to_process
        self.download_index = 0

        # Start worker threads
        self.threads_running = True
        self.download_thread = threading.Thread(target=self._download_worker, name="DownloadWorker")
        self.upload_thread = threading.Thread(target=self._upload_worker, name="UploadWorker")

        self.download_thread.start()
        self.upload_thread.start()

        # Started concurrent processing

        # Main loop: submit files and poll for completion
        last_metrics_log = time.time()

        while (self.files_remaining > 0 or
               not self.download_queue.empty() or
               len(self.active_jobs) > 0 or
               not self.upload_queue.empty()):


            # Submit files from download queue if daemon has capacity
            while len(self.active_jobs) < self.config.max_concurrent_jobs:
                try:
                    local_path, s3_key = self.download_queue.get_nowait()

                    # Submit to daemon
                    job = self._submit_to_daemon(local_path, s3_key)
                    if job:
                        self.active_jobs[job.job_id] = job
                    else:
                        # Submission failed, update tracking
                        self.failed_files.add(os.path.basename(s3_key))

                except Empty:
                    # No files ready to submit
                    break

            # Poll active jobs
            self._poll_and_process_jobs()

            # Log metrics periodically
            self._log_performance_metrics()

            # Brief sleep to prevent CPU spinning
            time.sleep(1)

        # Stop worker threads
        self.threads_running = False

        # Waiting for worker threads
        if self.download_thread.is_alive():
            self.download_thread.join(timeout=30)
        if self.upload_thread.is_alive():
            self.upload_thread.join(timeout=60)  # Give more time for uploads

        # Worker threads finished

    def _poll_and_process_jobs(self):
        """Poll active jobs and process completed ones."""
        completed_jobs = []

        # Polling active jobs

        for job_id, job in self.active_jobs.items():
            if self._poll_job_status(job):
                completed_jobs.append(job_id)

        # Move completed jobs to upload queue
        for job_id in completed_jobs:
            job = self.active_jobs.pop(job_id)
            # self.logger.info(f"Job {job_id} completed with status '{job.status}', moving to upload queue")
            self.upload_queue.put(job)

    def _wait_for_completion(self):
        """Wait for all remaining jobs to complete."""
        # The main processing loop handles everything now
        # Just log final metrics
        self._log_performance_metrics(force=True)

        # Final summary
        if self.failed_files:
            self.logger.warning(f"Failed: {len(self.failed_files)} files")
            for filename in list(self.failed_files)[:10]:  # Show first 10
                self.logger.warning(f"  - {filename}")
            if len(self.failed_files) > 10:
                self.logger.warning(f"  ... and {len(self.failed_files) - 10} more")

        # Debug mode summary
        if self.max_files_debug:
            self.logger.warning(f"DEBUG MODE: Processing was limited to {self.max_files_debug} files")


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
