#!/usr/bin/env python3

import json
import glob
import sys
import argparse
from pathlib import Path

def get_suite_name(eval_dataset):
    """Extract suite name by removing the split suffix (last underscore part)."""
    parts = eval_dataset.split('_')
    if len(parts) > 1:
        return '_'.join(parts[:-1])
    return eval_dataset

def get_similarity_level(overlap_ratio):
    """Get similarity assessment based on overlap ratio."""
    if overlap_ratio >= 0.95:
        return "⚠️  VERY HIGH SIMILARITY - Likely contamination"
    elif overlap_ratio >= 0.8:
        return "🔶 HIGH SIMILARITY - Probable contamination"
    elif overlap_ratio >= 0.6:
        return "🔸 MODERATE SIMILARITY - Possible contamination"
    else:
        return "🔹 LOW SIMILARITY - Unlikely contamination"

def truncate_text(text, max_lines=25):
    """Truncate text to max_lines and add truncation notice if needed."""
    lines = text.split('\n')
    if len(lines) <= max_lines:
        return text, False
    
    truncated = '\n'.join(lines[:max_lines])
    return truncated, True

def display_contamination_record(record, index, total):
    """Display a single contamination record in nice format."""
    
    print("=" * 80)
    print(f"CONTAMINATION #{index} of {total}")
    print("=" * 80)
    
    # Extract fields with defaults
    training_file = record.get('training_file', 'Unknown')
    eval_dataset = record.get('eval_dataset', 'Unknown')
    overlap_ratio = record.get('overlap_ratio', 0.0)
    toxic_score = record.get('toxic_score', 0.0)
    training_line = record.get('training_line', 'Unknown')
    eval_line = record.get('eval_line', 'Unknown')
    training_overlap_text = record.get('training_overlap_text', '')
    eval_overlap_text = record.get('eval_overlap_text', '')
    
    # Header info
    print(f"📁 TRAINING FILE: {training_file}")
    print(f"📋 EVAL DATASET:  {eval_dataset}")
    print(f"🎯 OVERLAP RATIO: {overlap_ratio:.3f}")
    print(f"🧪 IDF SCORE:     {toxic_score:.3f}")
    print()
    
    # Training text
    print(f"🔍 TRAINING TEXT (line {training_line}):")
    training_text, was_truncated = truncate_text(training_overlap_text)
    print(f'   "{training_text}"')
    if was_truncated:
        print("   ... [truncated: use --full to see complete text]")
    print()
    
    # Eval text  
    print(f"🔍 EVAL TEXT (line {eval_line}):")
    print(f'   "{eval_overlap_text}"')
    print()
    
    # Similarity assessment
    similarity = get_similarity_level(overlap_ratio)
    print(similarity)
    print()

def find_contamination_records(suite_name, show_full=False):
    """Find all contamination records for a given eval suite."""
    
    jsonl_files = glob.glob("*.jsonl")
    records = []
    
    print(f"🔍 Searching for contamination records in suite: {suite_name}")
    print(f"📁 Processing {len(jsonl_files)} JSONL files...")
    print()
    
    for file_path in jsonl_files:
        try:
            with open(file_path, 'r') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    
                    try:
                        record = json.loads(line)
                        eval_dataset = record.get('eval_dataset', '')
                        
                        if eval_dataset:
                            record_suite = get_suite_name(eval_dataset)
                            if record_suite == suite_name:
                                records.append(record)
                                
                    except json.JSONDecodeError:
                        continue
                        
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            continue
    
    return records

def main():
    parser = argparse.ArgumentParser(description='Display contamination records for an eval suite')
    parser.add_argument('suite', help='Eval suite name (e.g., mmlu, sciq)')
    parser.add_argument('--full', action='store_true', help='Show full text without truncation')
    parser.add_argument('--limit', type=int, default=10, help='Maximum number of records to display (default: 10)')
    parser.add_argument('--no-pause', action='store_true', help='Don\'t pause between records')
    parser.add_argument('--min-idf', type=float, help='Minimum IDF score threshold')
    parser.add_argument('--min-overlap', type=float, help='Minimum overlap ratio threshold (0.0-1.0)')
    parser.add_argument('--max-idf', type=float, help='Maximum IDF score threshold')
    parser.add_argument('--max-overlap', type=float, help='Maximum overlap ratio threshold (0.0-1.0)')
    
    args = parser.parse_args()
    
    records = find_contamination_records(args.suite, args.full)
    
    if not records:
        print(f"❌ No contamination records found for suite: {args.suite}")
        return
    
    # Apply filters
    original_count = len(records)
    filtered_records = []
    
    for record in records:
        overlap_ratio = record.get('overlap_ratio', 0.0)
        idf_score = record.get('toxic_score', 0.0)
        
        # Check filters
        if args.min_overlap is not None and overlap_ratio < args.min_overlap:
            continue
        if args.max_overlap is not None and overlap_ratio > args.max_overlap:
            continue
        if args.min_idf is not None and idf_score < args.min_idf:
            continue
        if args.max_idf is not None and idf_score > args.max_idf:
            continue
            
        filtered_records.append(record)
    
    records = filtered_records
    
    if not records:
        filter_desc = []
        if args.min_overlap: filter_desc.append(f"overlap >= {args.min_overlap}")
        if args.max_overlap: filter_desc.append(f"overlap <= {args.max_overlap}")
        if args.min_idf: filter_desc.append(f"IDF >= {args.min_idf}")
        if args.max_idf: filter_desc.append(f"IDF <= {args.max_idf}")
        
        print(f"❌ No records found for suite '{args.suite}' with filters: {', '.join(filter_desc)}")
        print(f"📊 Original count: {original_count} records")
        return
    
    # Sort by overlap ratio (highest first)
    records.sort(key=lambda x: x.get('overlap_ratio', 0), reverse=True)
    
    total_records = len(records)
    display_count = min(args.limit, total_records)
    
    # Show filter info if any filters were applied
    filter_info = []
    if args.min_overlap: filter_info.append(f"overlap >= {args.min_overlap}")
    if args.max_overlap: filter_info.append(f"overlap <= {args.max_overlap}")  
    if args.min_idf: filter_info.append(f"IDF >= {args.min_idf}")
    if args.max_idf: filter_info.append(f"IDF <= {args.max_idf}")
    
    if filter_info:
        print(f"📊 Found {total_records} contamination records for suite '{args.suite}' (filtered from {original_count})")
        print(f"🔍 Filters applied: {', '.join(filter_info)}")
    else:
        print(f"📊 Found {total_records} contamination records for suite '{args.suite}'")
    
    print(f"📋 Displaying top {display_count} records (sorted by overlap ratio)")
    print()
    
    for i, record in enumerate(records[:display_count], 1):
        display_contamination_record(record, i, display_count)
        
        # Add pause between records (except for last one) unless --no-pause is used
        if i < display_count and not args.no_pause:
            try:
                input("Press Enter to continue to next record...")
                print()
            except EOFError:
                # Non-interactive mode, just continue
                print()
        elif i < display_count:
            # No pause mode, just add spacing
            print()
    
    if total_records > display_count:
        print(f"📝 Note: {total_records - display_count} more records available. Use --limit to see more.")

if __name__ == "__main__":
    main()