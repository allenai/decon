#!/usr/bin/env python3
"""
Download and transform HuggingFace eval datasets for contamination detection.
"""

import yaml
import json
from datasets import load_dataset
from pathlib import Path
import os

def download_and_transform_eval(eval_name, eval_config, global_config):
    """Download HF dataset and transform to our JSONL format"""

    print(f"Loading {eval_name} from {eval_config['hf_path']}...")

    # Load dataset from HuggingFace
    try:
        if 'hf_config' in eval_config:
            dataset = load_dataset(eval_config['hf_path'], eval_config['hf_config'])
        else:
            dataset = load_dataset(eval_config['hf_path'])
    except Exception as e:
        print(f"Error loading {eval_name}: {e}")
        return

    # Create output directory
    output_dir = Path(global_config['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)

    # Process each split
    for split in eval_config['splits']:
        if split not in dataset:
            print(f"Warning: Split '{split}' not found in {eval_name}")
            continue

        print(f"Processing {eval_name} {split} split ({len(dataset[split])} examples)...")

        # Output file path
        output_file = output_dir / f"{eval_name}_{split}.jsonl"

        with open(output_file, 'w') as f:
            for idx, example in enumerate(dataset[split]):
                # Extract text field
                text_field = eval_config['transform']['text_field']
                text = example[text_field]

                # Build output record
                record = {
                    global_config['jsonl_format']['text_field']: text,
                    global_config['jsonl_format']['eval_field']: eval_name,
                    global_config['jsonl_format']['index_field']: idx,
                    global_config['jsonl_format']['split_field']: split
                }

                # Add any extra fields
                if 'extra_fields' in eval_config['transform']:
                    for field in eval_config['transform']['extra_fields']:
                        if field in example:
                            record[field] = example[field]

                # Write to JSONL
                f.write(json.dumps(record) + '\n')

        print(f"Saved {len(dataset[split])} examples to {output_file}")

def main():
    """Main function to process all eval datasets"""

    # Load configuration
    config_path = Path(__file__).parent.parent / "examples" / "eval" / "eval_datasets.yaml"

    if not config_path.exists():
        print(f"Config file not found: {config_path}")
        return

    with open(config_path) as f:
        config = yaml.safe_load(f)

    print(f"Processing {len(config['evals'])} eval datasets...")

    # Process each eval dataset
    for eval_name, eval_config in config['evals'].items():
        download_and_transform_eval(eval_name, eval_config, config)

    print("Done!")

if __name__ == "__main__":
    main()
