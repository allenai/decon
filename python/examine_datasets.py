#!/usr/bin/env python3
"""
Script to examine HuggingFace datasets and identify their splits and fields.
"""

import argparse
from datasets import load_dataset, get_dataset_config_names, DatasetInfo
from huggingface_hub import dataset_info
import sys

# Mapping from TARGET_EVAL_DATASETS patterns to EVAL_PATHS entries
TARGET_TO_EVAL_MAPPING = {
    # ARC variants
    "arc_*:mc::xlarge": "ai2_arc",
    "arc_*:bpb::full": "ai2_arc",
    "arc_*:rc::olmes:full": "ai2_arc",
    
    # MMLU variants
    "mmlu_*:mc::olmes": "cais/mmlu",
    "mmlu_*:cot::hamish_zs_reasoning": "cais/mmlu",
    "mmlu_*:bpb": "cais/mmlu",
    "mmlu_*:rc::olmes": "cais/mmlu",
    
    # CommonsenseQA variants
    "csqa:mc::xlarge": "commonsense_qa",
    "csqa:bpb::olmes:full": "commonsense_qa",
    "csqa:rc::olmes:full": "commonsense_qa",
    
    # PIQA variants
    "piqa:mc::xlarge": "piqa",
    "piqa:bpb::olmes:full": "piqa",
    "piqa:rc::olmes:full": "piqa",
    
    # SocialIQA variants
    "socialiqa:mc::xlarge": "social_i_qa",
    "socialiqa:bpb::olmes:full": "social_i_qa",
    "socialiqa:rc::olmes:full": "social_i_qa",
    
    # DROP variants
    "drop:mc::gen2mc": "allenai/drop_mc",
    "drop::xlarge": "EleutherAI/drop",
    "drop:bpb::gen2mc": "allenai/drop_mc",
    "drop:rc::gen2mc": "allenai/drop_mc",
    
    # Jeopardy variants
    "jeopardy:mc::gen2mc": "allenai/jeopardy_mc",
    "jeopardy::xlarge": "soldni/jeopardy",
    "jeopardy:bpb::gen2mc": "allenai/jeopardy_mc",
    "jeopardy:rc::gen2mc": "allenai/jeopardy_mc",
    
    # Natural Questions variants
    "naturalqs:mc::gen2mc": "allenai/nq_open_mc",
    "naturalqs::xlarge": "google-research-datasets/nq_open",
    "naturalqs:bpb::gen2mc": "allenai/nq_open_mc",
    "naturalqs:rc::gen2mc": "allenai/nq_open_mc",
    
    # SQuAD variants
    "squad:mc::gen2mc": "allenai/squad_mc",
    "squad::xlarge": "rajpurkar/squad",
    "squad:bpb::gen2mc": "allenai/squad_mc",
    "squad:rc::gen2mc": "allenai/squad_mc",
    
    # CoQA variants
    "coqa:mc::gen2mc": "allenai/coqa_mc",
    "coqa::xlarge": "EleutherAI/coqa",
    "coqa:bpb::gen2mc": "allenai/coqa_mc",
    "coqa:rc::gen2mc": "allenai/coqa_mc",
    
    # Basic Skills variants
    "basic_skills_*:mc::olmes": "allenai/basic-skills",
    "basic_skills_*:rc::olmes": "allenai/basic-skills",
    "basic_skills:bpb::olmes": "allenai/basic-skills",
    
    # Medical datasets
    "medmcqa:mc::none": "openlifescienceai/medmcqa",
    "medmcqa:bpb::none": "openlifescienceai/medmcqa",
    "medmcqa:rc::none": "openlifescienceai/medmcqa",
    "medqa_en:mc::none": "davidheineman/medqa-en",
    "medqa_en:bpb::none": "davidheineman/medqa-en",
    "medqa_en:rc::none": "davidheineman/medqa-en",
    
    # Science datasets
    "sciq:mc::xlarge": "sciq",
    "sciq:bpb::olmo3": "sciq",
    "sciq:rc::olmo3": "sciq",
    
    # HellaSwag variants
    "hellaswag:rc::xlarge": "hellaswag",
    "hellaswag:bpb::olmes:full": "hellaswag",
    "hellaswag:rc::olmes:full": "hellaswag",
    
    # Winogrande variants
    "winogrande:rc::xlarge": "winogrande",
    "winogrande:bpb::olmes:full": "winogrande",
    "winogrande:rc::olmes:full": "winogrande",
    
    # Lambda variants
    "lambada": "EleutherAI/lambada_openai",
    "lambada:bpb": "EleutherAI/lambada_openai",
    
    # GSM8K variants
    "gsm8k::olmes": "gsm8k",
    "gsm8k::zs_cot_latex": "gsm8k",
    
    # GSM Symbolic
    "gsm_symbolic*::olmo3": "apple/GSM-Symbolic",
    
    # Math datasets
    "minerva_math_*::olmes": "EleutherAI/hendrycks_math",
    "minerva_math_*::hamish_zs_reasoning": "EleutherAI/hendrycks_math",
    "minerva_math_500::hamish_zs_reasoning": "HuggingFaceH4/MATH-500",
    "minerva_*:bpb::olmes": "EleutherAI/hendrycks_math",
    "aime::hamish_zs_reasoning": "AI-MO/aimo-validation-aime",
    
    # Code datasets
    "bigcodebench:3shot::olmo3": "bigcode/bigcodebench",
    "codex_humaneval:3shot::olmo3": "openai_humaneval",
    "codex_humaneval:3shot:bpb::none": "openai_humaneval",
    "codex_humanevalfim_single:temp0.2": "loubnabnl/humaneval_infilling",
    "codex_humanevalfim_multi:temp0.2": "loubnabnl/humaneval_infilling",
    "codex_humanevalfim_random:temp0.2": "loubnabnl/humaneval_infilling",
    "codex_humanevalplus:0-shot-chat::tulu-thinker": "evalplus/humanevalplus",
    "deepseek_leetcode::olmo3": "davidheineman/deepseek-leetcode",
    "ds1000:3shot::olmo3": "xlangai/DS-1000",
    "mbpp:3shot::olmo3": "google-research-datasets/mbpp",
    "mbpp:3shot:bpb::none": "google-research-datasets/mbpp",
    "mbppplus:0-shot-chat::tulu-thinker": "evalplus/mbppplus",
    "multipl_e_humaneval:*::olmo3": "nuprl/MultiPL-E",
    "multipl_e_mbpp:*::olmo3": "nuprl/MultiPL-E",
    "mt_mbpp_v2fix:*": "google-research-datasets/mbpp",
    "livecodebench_codegeneration::tulu-thinker": "livecodebench/code_generation_lite",
    
    # Evaluation datasets
    "alpaca_eval_v3::hamish_zs_reasoning": "tatsu-lab/alpaca_eval",
    "styled_alpacaeval::tulu-thinker": "tatsu-lab/alpaca_eval",
    "multiturn_alpacaeval_*::tulu": "tatsu-lab/alpaca_eval",
    "ifeval::hamish_zs_reasoning": "HuggingFaceH4/ifeval",
    "ifeval_ood::tulu-thinker": "HuggingFaceH4/ifeval",
    "ifeval_mt_*::tulu": "HuggingFaceH4/ifeval",
    
    # Logic and reasoning datasets
    "zebralogic::hamish_zs_reasoning": "allenai/ZebraLogicBench-private",
    "popqa::hamish_zs_reasoning": "akariasai/PopQA",
    "styled_popqa::tulu-thinker": "akariasai/PopQA",
    "bbh_*:cot::hamish_zs_reasoning": "lukaemon/bbh",
    "gpqa:0shot_cot::hamish_zs_reasoning": "Idavidrein/gpqa",
    "agi_eval_*:0shot_cot::hamish_zs_reasoning": None,  # Local files
    
    # Omega datasets
    "omega_*:0-shot-chat": ["allenai/omega-compositional", "allenai/omega-explorative", "allenai/omega-transformative"],
    
    # Other datasets
    "simpleqa::tulu-thinker": "lighteval/SimpleQA",
    "styled_math500::tulu-thinker": "HuggingFaceH4/MATH-500",
    "qasper_yesno:bpb::olmes": "allenai/qasper-yesno",
    "qasper_yesno:rc::olmes": "allenai/qasper-yesno",
    "sciriff_yesno:bpb::olmes": "allenai/sciriff-yesno",
    "sciriff_yesno:rc::olmes": "allenai/sciriff-yesno",
    
    # LAB-Bench datasets
    "lab_bench_dbqa:bpb": "futurehouse/lab-bench",
    "lab_bench_dbqa": "futurehouse/lab-bench",
    "lab_bench_protocolqa:bpb": "futurehouse/lab-bench",
    "lab_bench_protocolqa": "futurehouse/lab-bench",
    
    # Chat datasets
    "ultrachat_masked_ppl": "HuggingFaceH4/ultrachat_200k",
    "wildchat_masked_ppl": "allenai/WildChat",
}


def get_fully_qualified_name(path):
    """Get the fully qualified dataset name from HuggingFace."""
    try:
        info = dataset_info(path)
        return info.id  # This gives the fully qualified name
    except:
        return path  # Fallback to original path

def examine_dataset(path):
    """Examine a dataset and return its splits and fields."""
    # For paths without "/", try them directly first
    if "/" not in path:
        actual_path = path
        fallback_path = f"allenai/{path}"
    else:
        actual_path = path
        fallback_path = None

    def try_load_dataset(dataset_path):
        """Helper function to try loading a dataset with all config handling."""
        # Get the fully qualified name
        qualified_name = get_fully_qualified_name(dataset_path)

        # First try loading without config (for datasets that don't need one)
        try:
            dataset = load_dataset(dataset_path, streaming=True, trust_remote_code=True)

            results = {
                "path": qualified_name,
                "original_path": path,
                "configs": {"default": {}},
                "status": "success"
            }

            # Examine only the first split
            split_names = list(dataset.keys())
            if split_names:
                split_name = split_names[0]
                try:
                    # Get first sample to examine
                    sample = next(iter(dataset[split_name]))
                    fields = list(sample.keys())
                    
                    # Truncate each field to first 200 characters for display
                    truncated_sample = {}
                    for field, value in sample.items():
                        if isinstance(value, str):
                            truncated_sample[field] = value[:200] + "..." if len(value) > 200 else value
                        elif isinstance(value, list):
                            str_value = str(value)
                            truncated_sample[field] = str_value[:200] + "..." if len(str_value) > 200 else str_value
                        else:
                            str_value = str(value)
                            truncated_sample[field] = str_value[:200] + "..." if len(str_value) > 200 else str_value
                    
                    results["configs"]["default"][split_name] = {
                        "fields": fields,
                        "sample": truncated_sample,
                        "all_splits": split_names
                    }
                except Exception as e:
                    results["configs"]["default"][split_name] = f"ERROR: {str(e)}"

            return results

        except Exception as e:
            if "Config name is missing" in str(e):
                # Dataset requires config - get all available configs
                try:
                    config_names = get_dataset_config_names(dataset_path)

                    results = {
                        "path": qualified_name,
                        "original_path": path,
                        "configs": {},
                        "status": "success"
                    }

                    # Examine each config
                    for config_name in config_names:
                        try:
                            dataset = load_dataset(dataset_path, config_name, streaming=True, trust_remote_code=True)
                            results["configs"][config_name] = {}

                            # Examine only the first split in this config
                            split_names = list(dataset.keys())
                            if split_names:
                                split_name = split_names[0]
                                try:
                                    # Get first sample to examine
                                    sample = next(iter(dataset[split_name]))
                                    fields = list(sample.keys())
                                    
                                    # Truncate each field to first 200 characters for display
                                    truncated_sample = {}
                                    for field, value in sample.items():
                                        if isinstance(value, str):
                                            truncated_sample[field] = value[:200] + "..." if len(value) > 200 else value
                                        elif isinstance(value, list):
                                            str_value = str(value)
                                            truncated_sample[field] = str_value[:200] + "..." if len(str_value) > 200 else str_value
                                        else:
                                            str_value = str(value)
                                            truncated_sample[field] = str_value[:200] + "..." if len(str_value) > 200 else str_value
                                    
                                    results["configs"][config_name][split_name] = {
                                        "fields": fields,
                                        "sample": truncated_sample,
                                        "all_splits": split_names
                                    }
                                except Exception as split_e:
                                    results["configs"][config_name][split_name] = f"ERROR: {str(split_e)}"

                        except Exception as config_e:
                            results["configs"][config_name] = f"ERROR: {str(config_e)}"

                    return results

                except Exception as config_list_e:
                    raise e  # Fall back to original error
            else:
                raise e  # Re-raise the original error

    # Try the primary path first
    try:
        return try_load_dataset(actual_path)
    except Exception as e:
        # If we have a fallback path, try it
        if fallback_path:
            try:
                return try_load_dataset(fallback_path)
            except Exception as fallback_e:
                # Return error for the original path attempt
                return {
                    "path": actual_path,
                    "original_path": path,
                    "status": "error",
                    "error": str(e)
                }
        else:
            return {
                "path": actual_path,
                "original_path": path,
                "status": "error",
                "error": str(e)
            }

def display_dataset_result(result, dataset_num=None, total_datasets=None):
    """Display examination result for a single dataset with improved formatting."""
    if dataset_num and total_datasets:
        print(f"\n[{dataset_num:2d}/{total_datasets}] Dataset: {result['original_path']}")
    else:
        print(f"\nDataset: {result['original_path']}")
    print("=" * 80)
    
    if result["status"] == "success":
        print(f"✓ Status: SUCCESS")
        print(f"  HuggingFace Path: {result['path']}")
        if result["path"] != result["original_path"]:
            print(f"  (resolved from: {result['original_path']})")
        
        print(f"\n  Available Configurations: {len(result['configs'])}")
        
        for config_idx, (config_name, splits) in enumerate(result["configs"].items(), 1):
            print(f"\n  [{config_idx}] Configuration: '{config_name}'")
            print("  " + "-" * 60)
            
            if isinstance(splits, dict):
                for split_name, split_data in splits.items():
                    if isinstance(split_data, dict) and "fields" in split_data:
                        fields = split_data["fields"]
                        sample = split_data["sample"]
                        all_splits = split_data.get("all_splits", [split_name])
                        
                        print(f"      Available splits: {', '.join(all_splits)}")
                        print(f"      Example from split '{split_name}':")
                        print(f"      Fields ({len(fields)}): {', '.join(fields)}")
                        print(f"\n      Sample data:")
                        
                        # Display sample data in a more readable format
                        for field, value in sample.items():
                            value_str = str(value)
                            if len(value_str) > 100:
                                value_str = value_str[:100] + "..."
                            print(f"        • {field}: {value_str}")
                    else:
                        print(f"      Error in split '{split_name}': {split_data}")
            else:
                print(f"      Error: {splits}")
    else:
        print(f"✗ Status: ERROR")
        print(f"  Attempted path: {result['path']}")
        print(f"  Error: {result['error']}")
    
    print()


def get_unique_eval_paths():
    """Get unique evaluation paths from TARGET_TO_EVAL_MAPPING."""
    unique_paths = set()
    
    for eval_path in TARGET_TO_EVAL_MAPPING.values():
        if eval_path is None:
            continue  # Skip unmapped entries (local files)
        elif isinstance(eval_path, list):
            unique_paths.update(eval_path)
        else:
            unique_paths.add(eval_path)
    
    return sorted(list(unique_paths))


def examine_all_datasets():
    """Examine all datasets from TARGET_TO_EVAL_MAPPING."""
    print("=" * 80)
    print("DATASET EXAMINATION RESULTS")
    print("=" * 80)
    
    unique_paths = get_unique_eval_paths()
    
    for i, path in enumerate(unique_paths, 1):
        result = examine_dataset(path)
        display_dataset_result(result, i, len(unique_paths))


def analyze_mapping_coverage():
    """Analyze the coverage of TARGET_TO_EVAL_MAPPING."""
    print("=" * 80)
    print("TARGET TO EVAL MAPPING ANALYSIS")
    print("=" * 80)
    
    # Group by unique EVAL_PATHS values
    eval_path_usage = {}
    unmapped_count = 0
    
    for target, eval_path in TARGET_TO_EVAL_MAPPING.items():
        if eval_path is None:
            unmapped_count += 1
        elif isinstance(eval_path, list):
            for path in eval_path:
                if path not in eval_path_usage:
                    eval_path_usage[path] = []
                eval_path_usage[path].append(target)
        else:
            if eval_path not in eval_path_usage:
                eval_path_usage[eval_path] = []
            eval_path_usage[eval_path].append(target)
    
    # Display mapping statistics
    print(f"\nTotal mappings: {len(TARGET_TO_EVAL_MAPPING)}")
    print(f"Unmapped entries (local files): {unmapped_count}")
    print(f"Unique HuggingFace datasets: {len(eval_path_usage)}")
    
    # Show most used datasets
    print("\nMost frequently used datasets:")
    sorted_usage = sorted(eval_path_usage.items(), key=lambda x: len(x[1]), reverse=True)
    for eval_path, targets in sorted_usage[:10]:
        print(f"  {eval_path}: {len(targets)} variants")
    
    # Show all unique paths
    print(f"\nAll unique dataset paths ({len(eval_path_usage)}):")
    for path in sorted(eval_path_usage.keys()):
        print(f"  - {path}")


def main():
    """Main function with CLI support."""
    parser = argparse.ArgumentParser(
        description="Examine HuggingFace datasets to identify their splits and fields"
    )
    
    parser.add_argument(
        "--ds",
        type=str,
        help="Examine a specific dataset by name or HuggingFace path"
    )
    
    parser.add_argument(
        "--all-configs",
        action="store_true",
        help="Examine all configurations even if a default exists"
    )
    
    parser.add_argument(
        "--mapping",
        action="store_true",
        help="Analyze the TARGET_TO_EVAL_MAPPING coverage"
    )
    
    args = parser.parse_args()
    
    if args.mapping:
        analyze_mapping_coverage()
    elif args.ds:
        # Examine a single dataset
        print(f"Examining single dataset: {args.ds}")
        result = examine_dataset(args.ds)
        display_dataset_result(result)
    else:
        # Examine all datasets
        examine_all_datasets()
    
    if not args.mapping:
        print("=" * 80)
        print("EXAMINATION COMPLETE")
        print("=" * 80)

if __name__ == "__main__":
    main()
