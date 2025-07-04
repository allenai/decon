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

                # Generate records based on answer field configuration
                records_to_write = []
                
                # Always create the base question record
                base_record = {
                    global_config['jsonl_format']['text_field']: text,
                    global_config['jsonl_format']['eval_field']: eval_name,
                    global_config['jsonl_format']['index_field']: idx,
                    global_config['jsonl_format']['split_field']: split,
                    'type': 'question'
                }
                
                # Add any extra fields
                if 'extra_fields' in eval_config['transform']:
                    for field in eval_config['transform']['extra_fields']:
                        if field in example:
                            base_record[field] = example[field]
                
                records_to_write.append(base_record)
                
                # Handle answer fields if configured
                if 'answer_field' in eval_config['transform']:
                    answer_field = eval_config['transform']['answer_field']
                    if answer_field in example:
                        answer_value = example[answer_field]
                        
                        # Handle different answer field types
                        if isinstance(answer_value, list):
                            # Array of answers - create separate record for each
                            for answer in answer_value:
                                answer_record = {
                                    global_config['jsonl_format']['text_field']: answer,
                                    global_config['jsonl_format']['eval_field']: eval_name,
                                    global_config['jsonl_format']['index_field']: idx,
                                    global_config['jsonl_format']['split_field']: split,
                                    'type': 'answer'
                                }
                                # Add extra fields to answer record too
                                if 'extra_fields' in eval_config['transform']:
                                    for field in eval_config['transform']['extra_fields']:
                                        if field in example:
                                            answer_record[field] = example[field]
                                records_to_write.append(answer_record)
                        else:
                            # Single answer - create separate record
                            answer_record = {
                                global_config['jsonl_format']['text_field']: answer_value,
                                global_config['jsonl_format']['eval_field']: eval_name,
                                global_config['jsonl_format']['index_field']: idx,
                                global_config['jsonl_format']['split_field']: split,
                                'type': 'answer'
                            }
                            # Add extra fields to answer record too
                            if 'extra_fields' in eval_config['transform']:
                                for field in eval_config['transform']['extra_fields']:
                                    if field in example:
                                        answer_record[field] = example[field]
                            records_to_write.append(answer_record)
                
                # Handle choices field if configured (e.g., multiple choice questions)
                if 'choices_field' in eval_config['transform']:
                    choices_field = eval_config['transform']['choices_field']
                    if choices_field in example:
                        choices = example[choices_field]
                        
                        # Handle choices structure: {'text': [...], 'label': [...]}
                        if isinstance(choices, dict) and 'text' in choices:
                            for choice_text in choices['text']:
                                choice_record = {
                                    global_config['jsonl_format']['text_field']: choice_text,
                                    global_config['jsonl_format']['eval_field']: eval_name,
                                    global_config['jsonl_format']['index_field']: idx,
                                    global_config['jsonl_format']['split_field']: split,
                                    'type': 'answer'
                                }
                                # Add extra fields to choice record too
                                if 'extra_fields' in eval_config['transform']:
                                    for field in eval_config['transform']['extra_fields']:
                                        if field in example:
                                            choice_record[field] = example[field]
                                records_to_write.append(choice_record)
                        elif isinstance(choices, list):
                            # Handle simple list of choices
                            for choice in choices:
                                choice_record = {
                                    global_config['jsonl_format']['text_field']: choice,
                                    global_config['jsonl_format']['eval_field']: eval_name,
                                    global_config['jsonl_format']['index_field']: idx,
                                    global_config['jsonl_format']['split_field']: split,
                                    'type': 'answer'
                                }
                                # Add extra fields to choice record too
                                if 'extra_fields' in eval_config['transform']:
                                    for field in eval_config['transform']['extra_fields']:
                                        if field in example:
                                            choice_record[field] = example[field]
                                records_to_write.append(choice_record)

                # Write all records to JSONL
                for record in records_to_write:
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
