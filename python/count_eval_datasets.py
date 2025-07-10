#!/usr/bin/env python3

import json
import glob
from collections import Counter

def count_eval_datasets():
    """Count records grouped by eval_dataset (suite) across all JSONL files."""
    
    # Find all .jsonl files in current directory
    jsonl_files = glob.glob("*.jsonl")
    
    eval_suite_counts = Counter()
    total_records = 0
    
    print(f"Processing {len(jsonl_files)} JSONL files...")
    
    for file_path in jsonl_files:
        print(f"Processing: {file_path}")
        
        try:
            with open(file_path, 'r') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    
                    try:
                        record = json.loads(line)
                        eval_dataset = record.get('eval_dataset')
                        
                        if eval_dataset:
                            # Strip the split suffix (everything after last underscore)
                            eval_suite = '_'.join(eval_dataset.split('_')[:-1])
                            if eval_suite:  # Only count if there was something before the last underscore
                                eval_suite_counts[eval_suite] += 1
                                total_records += 1
                        
                    except json.JSONDecodeError as e:
                        print(f"  JSON decode error in {file_path} line {line_num}: {e}")
                        
        except Exception as e:
            print(f"  Error processing {file_path}: {e}")
    
    return eval_suite_counts, total_records

if __name__ == "__main__":
    counts, total = count_eval_datasets()
    
    print(f"\n=== EVAL_SUITE STATISTICS ===")
    print(f"Total contamination incidents processed: {total:,}")
    print(f"Unique eval_suites: {len(counts)}")
    print(f"\nCounts by eval_suite:")
    
    # Sort by count (descending)
    for suite, count in counts.most_common():
        print(f"  {suite:<30} {count:>8,} records")
