#!/usr/bin/env python3
"""
Get evaluation dataset split sizes from evals.yaml configuration.

This script reads the evals.yaml configuration and retrieves the number of instances
in each split for each benchmark. It handles:
- HuggingFace datasets: Uses metadata API, streaming, or Hub API
- Bundled/local datasets: Counts lines in actual .jsonl.gz files

Output is JSON mapping: benchmark_alias -> split_name -> instance_count
"""

import argparse
import json
import gzip
import yaml
from pathlib import Path
from datasets import load_dataset_builder, load_dataset
from collections import defaultdict
import sys
import requests
from typing import Optional, Tuple, Dict


def count_jsonl_lines(file_path):
    """Count lines in a JSONL file (plain or gzipped)."""
    try:
        if file_path.suffix == '.gz' or file_path.name.endswith('.jsonl.gz'):
            with gzip.open(file_path, 'rt', encoding='utf-8') as f:
                return sum(1 for line in f if line.strip())
        else:
            with open(file_path, 'r', encoding='utf-8') as f:
                return sum(1 for line in f if line.strip())
    except Exception as e:
        print(f"Error counting lines in {file_path}: {e}", file=sys.stderr)
        return None


def get_bundled_split_sizes(eval_name, eval_config, script_dir):
    """Get split sizes for bundled/local datasets by counting file lines."""
    local_path = eval_config.get('local_path')
    
    if not local_path:
        return None, "No local_path"
    
    # Handle bundled prefix pattern (e.g., bundled_evals/agi_eval_)
    if local_path.startswith('bundled_evals/') and local_path.endswith('_'):
        bundled_dir = script_dir.parent / 'bundled-evals'
        
        if not bundled_dir.exists():
            return None, f"Bundled directory not found: {bundled_dir}"
        
        # Extract prefix pattern
        prefix = local_path.split('/')[-1]
        
        # Find all matching files
        pattern = f"{prefix}*.jsonl.gz"
        matching_files = list(bundled_dir.glob(pattern))
        
        if not matching_files:
            return None, f"No bundled files found matching: {pattern}"
        
        # Parse filenames to extract splits and count lines
        split_counts = defaultdict(int)
        
        for file_path in matching_files:
            # Parse filename: e.g., agi_eval_train-1.jsonl.gz
            # Extract split name (everything between prefix and the chunk number)
            filename = file_path.stem.replace('.jsonl', '')  # Remove .jsonl from .jsonl.gz
            
            # For agi_eval, files are named like: agi_eval_train-1, logiqa_en_train-1, etc.
            # We need to extract the split name
            if filename.startswith(prefix):
                # Remove prefix
                remainder = filename[len(prefix):]
            else:
                # For agi_eval where files have different dataset names
                # Parse to find the split
                parts = filename.split('_')
                if len(parts) >= 2:
                    # Last part might be split-chunk (e.g., "train-1")
                    last_part = parts[-1]
                    if '-' in last_part:
                        split_name = last_part.split('-')[0]
                        remainder = split_name
                    else:
                        remainder = last_part
                else:
                    remainder = filename
            
            # Extract split name (before the dash if chunk number exists)
            if '-' in remainder:
                split_name = remainder.split('-')[0]
            else:
                split_name = remainder
            
            # Count lines in file
            line_count = count_jsonl_lines(file_path)
            if line_count is not None:
                split_counts[split_name] += line_count
        
        if split_counts:
            return dict(split_counts), None
        else:
            return None, "No lines counted in bundled files"
    
    else:
        # Handle regular local file
        local_file = Path(local_path)
        if not local_file.is_absolute():
            local_file = script_dir.parent / local_file
        
        if not local_file.exists():
            return None, f"Local file not found: {local_file}"
        
        line_count = count_jsonl_lines(local_file)
        if line_count is not None:
            # Assume single file is 'train' split unless specified
            splits = eval_config.get('splits', ['train'])
            return {splits[0]: line_count}, None
        else:
            return None, "Failed to count lines"


def try_hf_hub_api(hf_path: str, hf_config: Optional[str], configured_splits: list) -> Tuple[Optional[Dict], Optional[str]]:
    """Try to get split sizes using HuggingFace Hub API directly."""
    try:
        # Construct API URL
        url = f"https://huggingface.co/api/datasets/{hf_path}"
        response = requests.get(url, timeout=10)
        
        if response.status_code != 200:
            return None, f"Hub API returned {response.status_code}"
        
        data = response.json()
        
        # Try to extract split info from various locations in the response
        split_info = {}
        
        # Look for splits in dataset info
        if 'cardData' in data and data['cardData']:
            card_data = data['cardData']
            if 'dataset_info' in card_data:
                dataset_info = card_data['dataset_info']
                # Handle both single config and multiple configs
                configs_to_check = []
                if isinstance(dataset_info, dict):
                    if hf_config and 'config_name' in dataset_info:
                        if dataset_info['config_name'] == hf_config:
                            configs_to_check.append(dataset_info)
                    elif not hf_config:
                        configs_to_check.append(dataset_info)
                elif isinstance(dataset_info, list):
                    if hf_config:
                        configs_to_check = [c for c in dataset_info if c.get('config_name') == hf_config]
                    else:
                        configs_to_check = dataset_info
                
                for config_info in configs_to_check:
                    if 'splits' in config_info:
                        splits = config_info['splits']
                        # Handle both dict and list of splits
                        if isinstance(splits, dict):
                            for split_name, split_data in splits.items():
                                if split_name in configured_splits:
                                    if isinstance(split_data, dict) and 'num_examples' in split_data:
                                        split_info[split_name] = split_data['num_examples']
                        elif isinstance(splits, list):
                            for split_data in splits:
                                if isinstance(split_data, dict):
                                    split_name = split_data.get('name')
                                    if split_name and split_name in configured_splits:
                                        if 'num_examples' in split_data:
                                            split_info[split_name] = split_data['num_examples']
        
        if split_info:
            return split_info, None
        
        return None, "No split info in Hub API response"
        
    except Exception as e:
        return None, f"Hub API error: {str(e)[:50]}"


def try_streaming_count(hf_path: str, hf_config: Optional[str], split: str, max_count: int = 100000) -> Optional[int]:
    """Try to count examples using streaming mode.
    
    Args:
        hf_path: HuggingFace dataset path
        hf_config: Optional config name
        split: Split to count
        max_count: Maximum items to count (safety limit)
    
    Returns:
        Count if successful, None if failed
    """
    try:
        import warnings
        import os
        
        # Suppress all output during streaming
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            
            # Temporarily suppress stderr to hide HF download bars
            original_stderr = sys.stderr
            try:
                sys.stderr = open(os.devnull, 'w')
                
                # Load in streaming mode (doesn't download full dataset)
                if hf_config:
                    dataset = load_dataset(
                        hf_path, hf_config, 
                        split=split, 
                        streaming=True, 
                        trust_remote_code=True
                    )
                else:
                    dataset = load_dataset(
                        hf_path, 
                        split=split, 
                        streaming=True, 
                        trust_remote_code=True
                    )
                
            finally:
                sys.stderr.close()
                sys.stderr = original_stderr
            
            # Count by iterating through the stream
            count = 0
            for _ in dataset:
                count += 1
                # Safety limit to prevent infinite loops
                if count >= max_count:
                    break
            
            return count
            
    except Exception as e:
        # Could log the actual error if needed for debugging
        return None


def get_hf_split_sizes(eval_name, eval_config, verbose=False, use_streaming=True):
    """Get split sizes for HuggingFace datasets using multiple fallback strategies."""
    hf_path = eval_config.get('hf_path')
    hf_config = eval_config.get('hf_config')
    
    if not hf_path:
        return None, "No hf_path"
    
    configured_splits = eval_config.get('splits', [])
    
    # Strategy 1: Try metadata API (fast, no download)
    try:
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            warnings.filterwarnings("ignore", message=".*pattern.*")
            
            if hf_config:
                builder = load_dataset_builder(hf_path, hf_config, trust_remote_code=True)
            else:
                builder = load_dataset_builder(hf_path, trust_remote_code=True)
        
        # Extract split sizes for configured splits
        split_info = {}
        for split in configured_splits:
            if split in builder.info.splits:
                split_info[split] = builder.info.splits[split].num_examples
        
        if split_info:
            if verbose:
                print(f"      ✓ Metadata API", file=sys.stderr)
            return split_info, None
        
    except Exception as e:
        if verbose:
            print(f"      ✗ Metadata API: {str(e)[:50]}...", file=sys.stderr)
    
    # Strategy 2: Try HuggingFace Hub API
    if verbose:
        print(f"      Trying Hub API...", file=sys.stderr)
    
    split_info, error = try_hf_hub_api(hf_path, hf_config, configured_splits)
    if split_info:
        if verbose:
            print(f"      ✓ Hub API", file=sys.stderr)
        return split_info, None
    elif verbose:
        print(f"      ✗ Hub API: {error}", file=sys.stderr)
    
    # Strategy 3: Try streaming and count (slower but reliable)
    if not use_streaming:
        return None, "All fast methods failed (streaming disabled)"
    
    if verbose:
        print(f"      Trying streaming count (may be slow)...", file=sys.stderr)
    
    split_info = {}
    all_failed = True
    
    for split in configured_splits:
        if verbose:
            print(f"         Counting {split}...", end="", flush=True, file=sys.stderr)
        count = try_streaming_count(hf_path, hf_config, split)
        if count is not None:
            split_info[split] = count
            all_failed = False
            if verbose:
                print(f" {count}", file=sys.stderr)
        else:
            split_info[split] = None
            if verbose:
                print(f" failed", file=sys.stderr)
    
    if not all_failed:
        if verbose:
            print(f"      ✓ Streaming count", file=sys.stderr)
        return split_info, None
    
    return None, "All methods failed"


def get_eval_split_sizes(eval_name, eval_config, script_dir, verbose=False, use_streaming=True):
    """Get split sizes for any evaluation dataset."""
    # Try bundled/local first
    if 'local_path' in eval_config:
        return get_bundled_split_sizes(eval_name, eval_config, script_dir)
    
    # Try HuggingFace with multiple strategies
    if 'hf_path' in eval_config:
        return get_hf_split_sizes(eval_name, eval_config, verbose=verbose, use_streaming=use_streaming)
    
    return None, "No valid path configuration"


def main():
    parser = argparse.ArgumentParser(
        description="Get evaluation dataset split sizes from evals.yaml"
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config/evals.yaml',
        help='Path to evals.yaml configuration file (default: config/evals.yaml)'
    )
    parser.add_argument(
        '--pretty',
        action='store_true',
        help='Pretty-print JSON output'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Print progress to stderr'
    )
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress all stderr output (including HuggingFace progress bars)'
    )
    parser.add_argument(
        '--no-streaming',
        action='store_true',
        help='Skip streaming fallback (faster but may miss some datasets)'
    )
    parser.add_argument(
        '--debug-api',
        type=str,
        metavar='DATASET',
        help='Debug mode: dump Hub API response for specific dataset'
    )
    
    args = parser.parse_args()
    
    # Debug mode: dump API response and exit
    if args.debug_api:
        import json
        url = f"https://huggingface.co/api/datasets/{args.debug_api}"
        print(f"Fetching: {url}", file=sys.stderr)
        try:
            response = requests.get(url, timeout=10)
            print(f"Status: {response.status_code}", file=sys.stderr)
            if response.status_code == 200:
                print("\nAPI Response:", file=sys.stderr)
                print(json.dumps(response.json(), indent=2))
            else:
                print(f"Error: {response.text}", file=sys.stderr)
        except Exception as e:
            print(f"Exception: {e}", file=sys.stderr)
        sys.exit(0)
    
    # Suppress HuggingFace logging if quiet mode
    if args.quiet:
        import logging
        logging.getLogger("datasets").setLevel(logging.ERROR)
        logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
        import warnings
        warnings.filterwarnings("ignore")
        args.verbose = False
    
    # Resolve config path
    script_dir = Path(__file__).parent
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = script_dir.parent / config_path
    
    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}", file=sys.stderr)
        sys.exit(1)
    
    # Load evals.yaml
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    if args.verbose:
        print(f"Loading dataset split sizes from {config_path}...", file=sys.stderr)
        print(f"Found {len(config['evals'])} datasets\n", file=sys.stderr)
    
    results = {}
    errors = {}
    
    for eval_name in sorted(config['evals'].keys()):
        eval_config = config['evals'][eval_name]
        
        if args.verbose:
            print(f"Processing {eval_name}...", file=sys.stderr)
        
        split_info, error = get_eval_split_sizes(
            eval_name, eval_config, script_dir, 
            verbose=args.verbose, use_streaming=not args.no_streaming
        )
        
        if error:
            if args.verbose:
                print(f"   ⚠️  {error}", file=sys.stderr)
            errors[eval_name] = error
        elif split_info:
            if args.verbose:
                print(f"   ✓ Got counts: {split_info}", file=sys.stderr)
            results[eval_name] = split_info
        else:
            if args.verbose:
                print("   ? Unknown error", file=sys.stderr)
            errors[eval_name] = "Unknown error"
    
    # Output JSON to stdout (direct mapping of benchmark -> splits -> counts)
    if args.pretty:
        print(json.dumps(results, indent=2))
    else:
        print(json.dumps(results))
    
    if args.verbose:
        print(f"\n✓ Successfully processed {len(results)} datasets", file=sys.stderr)
        if errors:
            print(f"⚠️  {len(errors)} datasets had errors:", file=sys.stderr)
            for dataset, error in sorted(errors.items()):
                print(f"     {dataset}: {error}", file=sys.stderr)


if __name__ == "__main__":
    main()

