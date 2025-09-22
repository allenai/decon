#!/usr/bin/env python3

# Output Record Format Documentation
# ==================================
#
# This script processes evaluation datasets and outputs JSONL files where each line
# is a JSON record representing a single evaluation instance. The output format is
# designed for use with the Decon contamination detection tool.
#
# Standard Fields (always present):
# ---------------------------------
# - eval_key: String identifier for the evaluation dataset (e.g., 'gsm8k', 'mmlu')
# - eval_instance_index: Integer index of the example within its dataset split
# - split: Dataset split name ('train', 'test', 'validation', etc.)
# - question: The prompt/question text extracted from the dataset
# - doc_id: Unique document ID (incremental across all processed datasets)
# - config: HuggingFace config name if applicable (may be null)
# - fingerprint: SHA256 hash (first 16 chars) of passage+question+answer for deduplication
#
# Optional Fields (present when available):
# ------------------------------------------
# - passage: Supporting context or reference text for the question
# - answer: An answer or expected response
# - sub_index: When a single dataset example expands to multiple records (e.g., parallel arrays)
# - is_correct: Boolean flag for multiple-choice questions indicating the correct option
#
# Special Processing:
# -------------------
# - Multiple choice questions with expand_choices create one record per choice
# - Parallel arrays (e.g., multiple Q&A pairs) create multiple records from one example
# - Records with empty/meaningless answers are filtered out
# - Large outputs are automatically chunked into 5MB files by ChunkedFileWriter
# - Each record gets a fingerprint for tracking and deduplication
#
# Usage in Contamination Detection:
# ----------------------------------
# These records serve as the reference/evaluation data that training datasets are
# checked against. The Decon tool indexes these records and searches for overlapping
# n-grams between training text and evaluation questions/answers/passages.

import argparse
import json
import yaml
import gzip
import hashlib
from datasets import load_dataset, Dataset
from pathlib import Path
import os
import shutil


# Global config that will be loaded in main()
EVAL_CONFIG = None


class ChunkedFileWriter:
    """Handles writing JSONL files in chunks to avoid creating very large files."""

    CHUNK_SIZE_BYTES = 5 * 1024 * 1024  # 5MB

    def __init__(self, base_path):
        """
        Initialize a chunked file writer.

        Args:
            base_path: Path to the output file (without chunk number).
                       E.g., "output/eval_train.jsonl"
        """
        self.base_path = Path(base_path)
        self.base_name = self.base_path.stem
        self.extension = ''.join(self.base_path.suffixes)
        self.output_dir = self.base_path.parent

        self.chunk_number = 0
        self.current_chunk_size = 0
        self.writer = None
        self.buffer = []
        self.total_records = 0
        self.chunk_files = []

    def write(self, record):
        """Write a record to the chunked file."""
        record = self._add_fingerprint(record)

        line = json.dumps(record) + '\n'
        line_bytes = len(line.encode('utf-8'))

        if self.current_chunk_size + line_bytes > self.CHUNK_SIZE_BYTES or self.writer is None:
            self._start_new_chunk()

        self.buffer.append(line)
        self.current_chunk_size += line_bytes
        self.total_records += 1

        if len(self.buffer) >= 1000:
            self._flush_buffer()

    def _start_new_chunk(self):
        """Start writing to a new chunk file."""
        if self.writer is not None:
            self._flush_buffer()
            self.writer.close()

        self.chunk_number += 1
        chunk_filename = f"{self.base_name}-{self.chunk_number}{self.extension}"
        chunk_path = self.output_dir / chunk_filename
        self.chunk_files.append(chunk_path)

        if self.extension.endswith('.gz'):
            self.writer = gzip.open(chunk_path, 'wt', encoding='utf-8')
        else:
            self.writer = open(chunk_path, 'w', encoding='utf-8')

        self.current_chunk_size = 0

    def _flush_buffer(self):
        """Write buffered lines to the current file."""
        if self.writer and self.buffer:
            for line in self.buffer:
                self.writer.write(line)
            self.buffer = []

    def close(self):
        """Close the writer and flush any remaining data."""
        if self.writer:
            self._flush_buffer()
            self.writer.close()
            self.writer = None

    def get_chunk_files(self):
        """Get list of chunk files created."""
        return self.chunk_files

    def get_total_records(self):
        """Get total number of records written."""
        return self.total_records

    def _add_fingerprint(self, record):
        """Add a fingerprint hash to the record based on passage, prompt, and answer fields."""
        passage = record.get('passage', '')
        prompt = record.get('prompt') or record.get('question', '')
        answer = record.get('answer', '')

        if passage is None:
            passage = ''
        if prompt is None:
            prompt = ''
        if answer is None:
            answer = ''

        content = f"{passage}{prompt}{answer}"

        hash_obj = hashlib.sha256(content.encode('utf-8'))
        fingerprint = hash_obj.hexdigest()[:16]  # Use first 16 chars for brevity

        record['fingerprint'] = fingerprint
        return record


def load_local_jsonl(file_path):
    """Load a local JSONL file as a HuggingFace Dataset."""
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return Dataset.from_list(data)


def is_meaningful_answer(value):
    """
    Check if an answer value is meaningful (non-empty).

    Args:
        value: The answer value to check

    Returns:
        True if the value contains meaningful content, False otherwise
    """
    if value is None:
        return False
    if isinstance(value, str) and not value.strip():
        return False
    if isinstance(value, dict) and all(not v for v in value.values()):
        return False
    return True


def process_answers(eval_entry, example):
    """
    Process answer configurations to create multiple entries from a single example.

    Args:
        eval_entry: The eval configuration entry
        example: The dataset example

    Returns:
        List of (example, answer_info) tuples to process
    """
    # Check for parallel arrays first (special case for prompt/answer arrays)
    prompts_config = eval_entry.get('prompts', eval_entry.get('prompt'))
    answers_config = eval_entry.get('answers', eval_entry.get('answer'))  # Support both old and new format

    # Handle parallel arrays if prompts has 'array' field
    if isinstance(prompts_config, dict) and 'array' in prompts_config:
        questions = get_nested_field(example, prompts_config['array'])

        # Find the array answer if it exists
        answer_array = None
        if isinstance(answers_config, list):
            for answer_item in answers_config:
                if isinstance(answer_item, dict) and 'array' in answer_item:
                    answer_array = get_nested_field(example, answer_item['array'])
                    break
        elif isinstance(answers_config, dict) and 'array' in answers_config:
            answer_array = get_nested_field(example, answers_config['array'])

        if questions:
            expanded = []
            if not isinstance(questions, list):
                questions = [questions]
            if answer_array and not isinstance(answer_array, list):
                answer_array = [answer_array]

            for idx, q in enumerate(questions):
                if not q or (isinstance(q, str) and not q.strip()):
                    continue

                answer_info = {
                    'sub_index': idx,
                    'override_prompt': str(q),
                    'override_answer': str(answer_array[idx]) if answer_array and idx < len(answer_array) else None
                }
                expanded.append((example, answer_info))

            return expanded if expanded else [(example, {})]

    # Process regular answers configuration
    answers_config = eval_entry.get('answers', eval_entry.get('answer'))  # Support both old and new format

    if not answers_config:
        return [(example, {})]

    # Convert old single answer format to list
    if not isinstance(answers_config, list):
        answers_config = [answers_config]

    expanded = []

    for answer_item in answers_config:
        if isinstance(answer_item, str):
            # Simple field reference
            answer_value = get_nested_field(example, answer_item)
            if answer_value is not None:
                # Special handling for date objects from DROP dataset
                if isinstance(answer_value, dict) and 'answer.date' in answer_item:
                    # Try to extract a meaningful date string
                    date_parts = []
                    if answer_value.get('month'):
                        date_parts.append(answer_value['month'])
                    if answer_value.get('day'):
                        date_parts.append(answer_value['day'])
                    if answer_value.get('year'):
                        date_parts.append(answer_value['year'])

                    if date_parts:  # Only if we have at least one non-empty part
                        answer_info = {
                            'answer_config': answer_item,
                            'override_answer': ' '.join(date_parts)
                        }
                        expanded.append((example, answer_info))
                    # Skip if all date parts are empty
                elif is_meaningful_answer(answer_value):
                    # Only include answers with meaningful content
                    answer_info = {
                        'answer_config': answer_item,
                        'override_answer': str(answer_value)
                    }
                    expanded.append((example, answer_info))

        elif isinstance(answer_item, dict):
            if 'field' in answer_item and 'transform' in answer_item and not 'key_field' in answer_item:
                # Simple field with transformation
                field = answer_item.get('field')
                transform = answer_item.get('transform')
                answer_value = get_nested_field(example, field)

                if answer_value is not None:
                    # Apply transformation
                    if transform == 'binary_to_bool':
                        # Convert 0/1 to false/true
                        try:
                            val_int = int(answer_value)
                            answer_value = "true" if val_int == 1 else "false"
                        except (ValueError, TypeError):
                            pass

                    # Only include if the answer is meaningful after transformation
                    if is_meaningful_answer(answer_value):
                        answer_info = {
                            'answer_config': answer_item,
                            'override_answer': str(answer_value)
                        }
                        expanded.append((example, answer_info))

            elif 'expand_choices' in answer_item:
                # Expand choices configuration
                choices_config = answer_item['expand_choices']
                choices_field = choices_config.get('field')
                correct_index_field = choices_config.get('correct_index')

                if choices_field:
                    choices = get_nested_field(example, choices_field)
                    correct_index = get_nested_field(example, correct_index_field) if correct_index_field else None

                    if choices is not None:
                        if isinstance(choices, dict) and 'text' in choices:
                            # Handle dict format with text and labels
                            for idx, choice_text in enumerate(choices['text']):
                                answer_info = {
                                    'choice_index': idx,
                                    'override_answer': str(choice_text),
                                    'is_correct': (choices.get('label', [None])[idx] == correct_index) if correct_index else None
                                }
                                expanded.append((example, answer_info))
                        elif isinstance(choices, list):
                            # Handle simple list format
                            for idx, choice in enumerate(choices):
                                # Check if this is the correct answer
                                is_correct = None
                                if correct_index is not None:
                                    # Convert index to letter if needed (A=0, B=1, etc.)
                                    if isinstance(correct_index, str) and len(correct_index) == 1 and correct_index.isalpha():
                                        correct_idx = ord(correct_index.upper()) - ord('A')
                                        is_correct = (idx == correct_idx)
                                    elif isinstance(correct_index, int):
                                        is_correct = (idx == correct_index)

                                answer_info = {
                                    'choice_index': idx,
                                    'override_answer': str(choice),
                                    'is_correct': is_correct
                                }
                                expanded.append((example, answer_info))

            elif 'indexed' in answer_item:
                # Indexed answer without expansion
                indexed_config = answer_item['indexed']
                field = indexed_config.get('field')
                key_field = indexed_config.get('key_field')
                prefix = indexed_config.get('prefix')

                if prefix and key_field:
                    # Prefix-based indexing (e.g., answerA, answerB, answerC)
                    answer_key = get_nested_field(example, key_field)

                    if answer_key is not None:
                        # Apply label transformation if configured
                        transformed_key = answer_key
                        if 'label_transform' in indexed_config:
                            transform_type = indexed_config['label_transform']
                            if transform_type == 'numbers_to_letters':
                                # Convert "1" -> "A", "2" -> "B", etc.
                                try:
                                    key_int = int(answer_key)
                                    if key_int >= 1:
                                        transformed_key = chr(ord('A') + key_int - 1)
                                    else:
                                        transformed_key = 'A'
                                except (ValueError, TypeError):
                                    pass

                        # Construct field name using prefix + transformed key
                        answer_field_name = f"{prefix}{transformed_key}"
                        answer_value = get_nested_field(example, answer_field_name)

                        if answer_value is not None and is_meaningful_answer(answer_value):
                            answer_info = {
                                'answer_config': answer_item,
                                'override_answer': str(answer_value)
                            }
                            expanded.append((example, answer_info))

                elif field and key_field:
                    # Array-based indexing
                    answer_array = get_nested_field(example, field)
                    answer_key = get_nested_field(example, key_field)

                    if answer_array is not None and answer_key is not None:
                        answer_value = None
                        if isinstance(answer_array, list):
                            # Convert key to index
                            if isinstance(answer_key, str) and len(answer_key) == 1 and answer_key.isalpha():
                                # Letter index (A=0, B=1, etc.)
                                idx = ord(answer_key.upper()) - ord('A')
                                if 0 <= idx < len(answer_array):
                                    answer_value = answer_array[idx]
                            else:
                                # Numeric index
                                try:
                                    idx = int(answer_key)
                                    if 0 <= idx < len(answer_array):
                                        answer_value = answer_array[idx]
                                except (ValueError, TypeError):
                                    pass

                        if answer_value is not None and is_meaningful_answer(answer_value):
                            answer_info = {
                                'answer_config': answer_item,
                                'override_answer': str(answer_value)
                            }
                            expanded.append((example, answer_info))

            elif 'expand_prefix' in answer_item:
                # Expand fields by prefix pattern
                prefix_config = answer_item['expand_prefix']
                prefix = prefix_config.get('prefix')
                suffix_match_field = prefix_config.get('suffix_match_field')

                if prefix and suffix_match_field:
                    # Find all fields that start with the prefix
                    matching_fields = []
                    for key in example.keys():
                        if key.startswith(prefix):
                            suffix = key[len(prefix):]
                            if suffix:  # Only if there's a suffix after the prefix
                                matching_fields.append((key, suffix))

                    # Sort by suffix to ensure consistent ordering
                    matching_fields.sort(key=lambda x: x[1])

                    # Get the correct suffix value
                    correct_suffix = get_nested_field(example, suffix_match_field)

                    # Apply label transformation if configured
                    if correct_suffix is not None and 'label_transform' in prefix_config:
                        transform_type = prefix_config['label_transform']
                        if transform_type == 'numbers_to_letters':
                            # Convert "1" -> "A", "2" -> "B", etc.
                            try:
                                suffix_int = int(correct_suffix)
                                if suffix_int >= 1:
                                    correct_suffix = chr(ord('A') + suffix_int - 1)
                            except (ValueError, TypeError):
                                pass
                        elif transform_type == 'add_one':
                            # Convert 0 -> "1", 1 -> "2", etc. for 0-based to 1-based
                            try:
                                suffix_int = int(correct_suffix)
                                correct_suffix = str(suffix_int + 1)
                            except (ValueError, TypeError):
                                pass
                        elif transform_type == 'index_to_letter':
                            # Convert 0 -> "a", 1 -> "b", 2 -> "c", 3 -> "d"
                            try:
                                suffix_int = int(correct_suffix)
                                correct_suffix = chr(ord('a') + suffix_int)
                            except (ValueError, TypeError):
                                pass
                        elif transform_type == 'binary_to_bool':
                            # Convert 0 -> "false", 1 -> "true"
                            try:
                                suffix_int = int(correct_suffix)
                                correct_suffix = "true" if suffix_int == 1 else "false"
                            except (ValueError, TypeError):
                                pass

                    # Create entries for each matching field
                    for field_name, suffix in matching_fields:
                        field_value = get_nested_field(example, field_name)
                        if field_value is not None:
                            is_correct = (str(suffix) == str(correct_suffix)) if correct_suffix is not None else None

                            answer_info = {
                                'choice_suffix': suffix,
                                'override_answer': str(field_value),
                                'is_correct': is_correct
                            }
                            expanded.append((example, answer_info))

            elif 'array' in answer_item:
                # Array field (handled above in parallel arrays)
                pass

            elif 'field' in answer_item:
                # Legacy format with field and key_field
                field = answer_item.get('field')
                key_field = answer_item.get('key_field')

                if field:
                    if key_field:
                        # Indexed access
                        answer_array = get_nested_field(example, field)
                        answer_key = get_nested_field(example, key_field)

                        if answer_array is not None and answer_key is not None and isinstance(answer_array, list):
                            answer_value = None
                            # Convert key to index
                            if isinstance(answer_key, str) and len(answer_key) == 1 and answer_key.isalpha():
                                # Letter index (A=0, B=1, etc.)
                                idx = ord(answer_key.upper()) - ord('A')
                                if 0 <= idx < len(answer_array):
                                    answer_value = answer_array[idx]
                            else:
                                # Numeric index
                                try:
                                    idx = int(answer_key)
                                    if 0 <= idx < len(answer_array):
                                        answer_value = answer_array[idx]
                                except (ValueError, TypeError):
                                    pass

                            if answer_value is not None and is_meaningful_answer(answer_value):
                                answer_info = {
                                    'answer_config': answer_item,
                                    'override_answer': str(answer_value)
                                }
                                expanded.append((example, answer_info))
                    else:
                        # Direct field access
                        answer_value = get_nested_field(example, field)
                        if answer_value is not None and is_meaningful_answer(answer_value):
                            answer_info = {
                                'answer_config': answer_item,
                                'override_answer': str(answer_value)
                            }
                            expanded.append((example, answer_info))

    return expanded if expanded else [(example, {})]


def get_nested_field(obj, field_path):
    """Access nested fields using dot notation (e.g., 'answer.value' or 'choices.text.0')"""
    if not field_path:
        return None

    fields = field_path.split('.')
    value = obj
    for field in fields:
        if field.isdigit():
            # Array index access
            index = int(field)
            if isinstance(value, (list, tuple)) and 0 <= index < len(value):
                value = value[index]
            else:
                return None
        else:
            # Object property access
            if isinstance(value, dict) and field in value:
                value = value[field]
            else:
                return None
    return value


def extract_passage(example, config, answer_info=None):
    """
    Extract passage from example based on configuration.

    Args:
        example: The dataset example
        config: The passage configuration (can be string or dict)
        answer_info: Additional answer information

    Returns:
        Extracted passage text or None
    """
    if not config:
        return None

    if isinstance(config, str):
        # Simple field reference
        passage = get_nested_field(example, config)
        return str(passage) if passage is not None else None

    return None


def extract_prompt(example, config, answer_info=None):
    """
    Extract prompt/question from example based on configuration.

    Args:
        example: The dataset example
        config: The prompt configuration (can be string or dict)
        answer_info: Additional answer information

    Returns:
        Extracted prompt text or None
    """
    if not config:
        return None

    # Check if we have an override prompt
    if answer_info and 'override_prompt' in answer_info:
        return answer_info['override_prompt']

    if isinstance(config, str):
        # Simple field reference
        prompt = get_nested_field(example, config)
        return str(prompt) if prompt is not None else None

    if isinstance(config, dict):
        # Complex configuration
        if 'concat_messages' in config:
            # Concatenate messages from a conversation
            messages_config = config['concat_messages']
            messages_field = messages_config.get('field', 'messages')
            content_field = messages_config.get('content_field', 'content')

            messages = get_nested_field(example, messages_field)
            if messages and isinstance(messages, list):
                contents = []
                for msg in messages:
                    if isinstance(msg, dict):
                        content = msg.get(content_field)
                        if content:
                            contents.append(str(content))

                # Join all message contents
                prompt_text = " ".join(contents) if contents else None

                # Append additional field if specified
                if 'append_field' in config:
                    append_value = get_nested_field(example, config['append_field'])
                    if append_value:
                        if prompt_text:
                            prompt_text = f"{prompt_text} {append_value}"
                        else:
                            prompt_text = str(append_value)

                return prompt_text

        elif 'field' in config:
            prompt = get_nested_field(example, config['field'])
            return str(prompt) if prompt is not None else None

    return None


def extract_answer(example, config, answer_info=None):
    """
    Extract answer from example based on configuration.

    Args:
        example: The dataset example
        config: The answer configuration (can be string, dict, or list)
        answer_info: Additional answer information

    Returns:
        Extracted answer text or None
    """
    # Check if we have an override answer
    if answer_info and 'override_answer' in answer_info:
        return answer_info['override_answer']

    if not config:
        return None

    # Use the answer config from answer_info if available
    if answer_info and 'answer_config' in answer_info:
        config = answer_info['answer_config']

    if isinstance(config, str):
        # Simple field reference
        answer = get_nested_field(example, config)
        return str(answer) if answer is not None else None

    if isinstance(config, list):
        # For multi_answer, just use the first field if no transform
        # (shouldn't happen if multi_answer transform is used)
        if config:
            first_item = config[0]
            if isinstance(first_item, str):
                answer = get_nested_field(example, first_item)
                return str(answer) if answer is not None else None
            elif isinstance(first_item, dict):
                # Let dict handling below process it
                config = first_item
            else:
                return None

    if isinstance(config, dict):
        # Complex configuration with key lookup
        if 'field' in config:
            answer_value = get_nested_field(example, config['field'])

            # Check if we need to use a key field to index into the answer
            if 'key_field' in config and answer_value is not None:
                answer_key = get_nested_field(example, config['key_field'])

                if answer_key is not None and isinstance(answer_value, list):
                    try:
                        key_int = int(answer_key) if isinstance(answer_key, str) else answer_key
                        if 0 <= key_int < len(answer_value):
                            return str(answer_value[key_int])
                    except (ValueError, TypeError):
                        pass

            # Return the answer value as-is if no key lookup
            if answer_value is not None:
                if isinstance(answer_value, list) and answer_value:
                    return str(answer_value[0])
                return str(answer_value)

    return None


def process_eval_dataset(eval_name, eval_config, output_dir, document_id_counter):
    """Process a single evaluation dataset. Returns True if successful, False if failed."""

    if 'local_path' in eval_config:
        local_path = eval_config['local_path']

        # Check if this is a bundled prefix pattern (e.g., bundled_evals/agi_eval_)
        if local_path.startswith('bundled_evals/') and local_path.endswith('_'):
            # Handle bundled files by copying them directly
            print(f"Copying bundled files for {eval_name} with prefix: {local_path}...")

            # Find the bundled-evals directory relative to the script
            script_dir = Path(__file__).parent.parent
            bundled_dir = script_dir / 'bundled-evals'

            if not bundled_dir.exists():
                print(f"Error: Bundled directory not found: {bundled_dir}")
                return False

            # Extract the prefix pattern (e.g., 'agi_eval_' from 'bundled_evals/agi_eval_')
            prefix = local_path.split('/')[-1]

            # Find all matching files
            pattern = f"{prefix}*.jsonl.gz"
            matching_files = list(bundled_dir.glob(pattern))

            if not matching_files:
                print(f"Warning: No bundled files found matching pattern: {pattern}")
                return False

            # Ensure output directory exists
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            # Copy all matching files to output directory
            files_copied = 0
            for src_file in matching_files:
                # For agi_eval, just copy with the same name since the files already have the full dataset names
                # For other datasets, replace the prefix with eval_name in the destination filename
                if eval_name == 'agi_eval':
                    dest_filename = src_file.name
                else:
                    # e.g., some_prefix_train-1.jsonl.gz -> some_dataset_train-1.jsonl.gz
                    dest_filename = src_file.name.replace(prefix, f"{eval_name}_", 1)
                dest_file = output_dir / dest_filename

                shutil.copy2(src_file, dest_file)
                files_copied += 1
                print(f"  Copied {src_file.name} -> {dest_filename}")

            print(f"  Total files copied: {files_copied}")
            return True
        else:
            # Handle regular local file
            print(f"Loading {eval_name} from local file: {local_path}...")
            local_file = Path(local_path)
            if not local_file.exists():
                print(f"Error: Local file not found: {local_file}")
                return False
            dataset = {'train': load_local_jsonl(local_file)}
    else:
        print(f"Loading {eval_name} from {eval_config['hf_path']}...")
        # Load dataset from HuggingFace
        try:
            if 'hf_config' in eval_config:
                dataset = load_dataset(eval_config['hf_path'], eval_config['hf_config'])
            else:
                dataset = load_dataset(eval_config['hf_path'])
        except Exception as e:
            print(f"ERROR: Failed to load {eval_name}: {e}")
            return False

    # If we handled bundled files, we're done
    if 'local_path' in eval_config and eval_config['local_path'].startswith('bundled_evals/') and eval_config['local_path'].endswith('_'):
        return True

    # For non-bundled datasets, continue with normal processing
    if 'dataset' not in locals():
        try:
            if 'local_path' in eval_config:
                # Already loaded above
                pass
            elif 'hf_path' in eval_config:
                # Already loaded above
                pass
        except Exception as e:
            print(f"Error loading {eval_name}: {e}")
            return False

    # Ensure output directory exists
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Process each split
    for split in eval_config.get('splits', []):
        if split not in dataset:
            print(f"Warning: Split '{split}' not found in {eval_name}")
            continue

        print(f"Processing {eval_name} {split} split ({len(dataset[split])} examples)...")

        # Output file path
        output_file = output_dir / f"{eval_name}_{split}.jsonl"
        writer = ChunkedFileWriter(output_file)

        records_written = 0

        try:
            for idx, example in enumerate(dataset[split]):
                # Process answers to potentially get multiple entries
                entries = process_answers(eval_config, example)

                for entry_example, answer_info in entries:
                    # Extract fields using the three methods
                    passage = extract_passage(entry_example, eval_config.get('passage'), answer_info)
                    prompt = extract_prompt(entry_example, eval_config.get('prompt', eval_config.get('prompts')), answer_info)

                    # Check if this split has no answers
                    no_answer_splits = eval_config.get('no_answer_splits', [])
                    if split in no_answer_splits:
                        answer = None
                    else:
                        answer = extract_answer(entry_example, eval_config.get('answer', eval_config.get('answers')), answer_info)

                    # Skip if no prompt
                    if not prompt or (isinstance(prompt, str) and not prompt.strip()):
                        continue

                    # Create record
                    record = {
                        'eval_key': eval_name,  # Dataset identifier
                        'eval_instance_index': idx,  # Index within the split
                        'split': split,  # Dataset split (train/test/validation)
                        'question': prompt,  # Using 'question' for compatibility
                        'passage': passage,
                        'answer': answer,
                        'config': eval_config.get('hf_config'),
                        'doc_id': document_id_counter[0]
                    }

                    # Add sub_index if this came from parallel arrays
                    if 'sub_index' in answer_info:
                        record['sub_index'] = answer_info['sub_index']

                    # Add is_correct if this came from expand_choices
                    if 'is_correct' in answer_info:
                        record['is_correct'] = answer_info['is_correct']

                    document_id_counter[0] += 1

                    # Remove None fields
                    if answer is None:
                        record.pop('answer', None)
                    if passage is None:
                        record.pop('passage', None)

                    writer.write(record)
                    records_written += 1

        finally:
            writer.close()

        print(f"  Saved {records_written} records to {output_file}")

        # Print chunk information if multiple chunks
        chunk_files = writer.get_chunk_files()
        if len(chunk_files) > 1:
            print(f"  Split into {len(chunk_files)} chunks:")
            for chunk_file in chunk_files:
                print(f"      - {chunk_file.name}")

    return True


def main():
    """Main function with CLI."""
    parser = argparse.ArgumentParser(
        description="Process eval datasets with simplified configuration"
    )

    parser.add_argument(
        "--download",
        action="store_true",
        help="Download and process all evaluation datasets"
    )

    parser.add_argument(
        "--eval",
        type=str,
        help="Process a specific evaluation dataset by name"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        help="Output directory for downloaded datasets (default: ~/.local/share/decon/references)"
    )

    parser.add_argument(
        "--config",
        type=str,
        help="Configuration file for evaluation datasets (default: config/evals.yaml)"
    )

    args = parser.parse_args()

    if not any(vars(args).values()):
        parser.print_help()
        return

    # Load configuration from specified or default path
    global EVAL_CONFIG
    if args.config:
        config_path = Path(args.config).expanduser().absolute()
    else:
        # Default to bundled config
        config_path = Path(__file__).parent.parent / 'config' / 'evals.yaml'

    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}")
        return

    with open(config_path, 'r') as f:
        EVAL_CONFIG = yaml.safe_load(f)

    print(f"Loaded eval configuration from {config_path}")

    # Initialize document ID counter
    document_id_counter = [1]

    # Determine output directory
    if args.output_dir:
        output_dir = Path(args.output_dir).expanduser().absolute()
    else:
        # Default to ~/.local/share/decon/references
        output_dir = Path.home() / ".local" / "share" / "decon" / "references"

    if args.download:
        print(f"Processing {len(EVAL_CONFIG['evals'])} eval datasets...")
        print(f"Output directory: {output_dir}")

        # Clear output directory
        if output_dir.exists():
            print(f"Clearing existing directory: {output_dir}")
            shutil.rmtree(output_dir)

        output_dir.mkdir(parents=True, exist_ok=True)
        print("Created fresh output directory")

        # Process all datasets
        failed_datasets = []
        successful_datasets = []

        for eval_name, eval_config in EVAL_CONFIG['evals'].items():
            success = process_eval_dataset(eval_name, eval_config, output_dir, document_id_counter)
            if success:
                successful_datasets.append(eval_name)
            else:
                failed_datasets.append(eval_name)

        print(f"\nProcessing complete!")
        print(f"  Total documents: {document_id_counter[0] - 1}")
        print(f"  Successful datasets: {len(successful_datasets)}")
        print(f"  Failed datasets: {len(failed_datasets)}")

        if failed_datasets:
            print(f"\nFailed to process the following datasets:")
            for dataset in sorted(failed_datasets):
                print(f"  - {dataset}")

    elif args.eval:
        if args.eval not in EVAL_CONFIG['evals']:
            print(f"Error: Dataset '{args.eval}' not found.")
            print("\nAvailable datasets:")
            for name in sorted(EVAL_CONFIG['evals'].keys()):
                print(f"  - {name}")
            return

        print(f"Output directory: {output_dir}")
        output_dir.mkdir(parents=True, exist_ok=True)

        eval_config = EVAL_CONFIG['evals'][args.eval]
        process_eval_dataset(args.eval, eval_config, output_dir, document_id_counter)


if __name__ == "__main__":
    main()
