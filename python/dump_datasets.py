#!/usr/bin/env python3
"""
Dump HuggingFace eval datasets to raw JSONL format for debugging.
"""

import os
import json
from datasets import load_dataset, Dataset
from pathlib import Path
from datetime import datetime
from evals import EVAL_CONFIG

def load_local_jsonl(file_path):
    """Load a local JSONL file as a HuggingFace Dataset."""
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return Dataset.from_list(data)

def dump_dataset(dataset_name, dataset_config):
    """Dump a single dataset to JSONL format."""
    print(f"\nProcessing {dataset_name}...")
    
    try:
        # Load dataset
        if 'local_path' in dataset_config:
            # Load from local JSONL file
            local_file = Path(dataset_config['local_path'])
            if not local_file.exists():
                print(f"Error: Local file not found: {local_file}")
                return
            # Create a dataset dict with a single 'train' split
            dataset = {'train': load_local_jsonl(local_file)}
        elif 'hf_config' in dataset_config:
            dataset = load_dataset(dataset_config['hf_path'], dataset_config['hf_config'])
        else:
            dataset = load_dataset(dataset_config['hf_path'])
    except Exception as e:
        print(f"Error loading {dataset_name}: {e}")
        return
    
    # Process each split
    for split in dataset_config['splits']:
        if split not in dataset:
            print(f"Warning: Split '{split}' not found in {dataset_name}")
            continue
            
        print(f"  - Dumping {split} split ({len(dataset[split])} examples)...")
        
        # Output file path directly in dumps directory
        output_file = Path('fixtures/dumps') / f"{dataset_name}_{split}.jsonl"
        
        # Custom JSON encoder to handle datetime objects
        def json_serial(obj):
            """JSON serializer for objects not serializable by default."""
            if isinstance(obj, datetime):
                return obj.isoformat()
            raise TypeError(f"Type {type(obj)} not serializable")
        
        # Write raw data to JSONL
        with open(output_file, 'w') as f:
            for idx, example in enumerate(dataset[split]):
                # Write the raw example with custom serializer
                f.write(json.dumps(example, default=json_serial) + '\n')
        
        print(f"    Saved to: {output_file}")

def main():
    """Main function to dump all datasets."""
    # Create base dumps directory
    dumps_dir = Path('fixtures/dumps')
    dumps_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Dumping {len(EVAL_CONFIG['evals'])} evaluation datasets to {dumps_dir}...")
    
    # Process each eval dataset
    for dataset_name, dataset_config in EVAL_CONFIG['evals'].items():
        dump_dataset(dataset_name, dataset_config)
    
    print("\nDone! All datasets have been dumped to fixtures/dumps/")

if __name__ == "__main__":
    main()