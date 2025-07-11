#!/usr/bin/env python3
"""
Script to examine HuggingFace datasets and identify their splits and fields.
"""

import argparse
from datasets import load_dataset, get_dataset_config_names, DatasetInfo
from huggingface_hub import dataset_info
import sys

EVAL_PATHS = [
    "AI-MO/aimo-validation-aime",
    "ai2_arc",
    "akariasai/PopQA",
    "allenai/basic-skills",
    "allenai/coqa_mc",
    "allenai/drop_mc",
    "allenai/jeopardy_mc",
    "allenai/multilingual_mbpp",
    "allenai/nq_open_mc",
    "allenai/omega-compositional",
    "allenai/omega-explorative",
    "allenai/omega-transformative",
    "allenai/paloma",
    "allenai/qasper-yesno",
    "allenai/sciriff-yesno",
    "allenai/SimpleToM",
    "allenai/squad_mc",
    "allenai/ZebraLogicBench-private",
    "aps/super_glue",
    "apple/GSM-Symbolic",
    "bigcode/bigcodebench",
    "bigcode/bigcodebench-hard",
    "cais/mmlu",
    "commonsense_qa",
    "cosmos_qa",
    "cruxeval-org/cruxeval",
    "davidheineman/deepseek-leetcode",
    "davidheineman/medqa-en",
    "EleutherAI/coqa",
    "EleutherAI/drop",
    "EleutherAI/hendrycks_math",
    "EleutherAI/lambada_openai",
    "evalplus/humanevalplus",
    "evalplus/mbppplus",
    "google-research-datasets/mbpp",
    "google-research-datasets/nq_open",
    "google-research-datasets/tydiqa",
    "gsm8k",
    "hellaswag",
    "HuggingFaceH4/ifeval",
    "HuggingFaceH4/MATH-500",
    "Idavidrein/gpqa",
    "LEXam-Benchmark/LEXam",
    "lighteval/SimpleQA",
    "livecodebench/code_generation_lite",
    "loubnabnl/humaneval_infilling",
    "lucasmccabe/logiqa",
    "lukaemon/bbh",
    "mandarjoshi/trivia_qa",
    "nuprl/MultiPL-E",
    "openai_humaneval",
    "openai/mrcr",
    "openbookqa",
    "openlifescienceai/medmcqa",
    "piqa",
    "qintongli/GSM-Plus",
    "rajpurkar/squad",
    "rajpurkar/squad_v2",
    "sarahwie/copycolors_mcqa",
    "sciq",
    "social_i_qa",
    "soldni/jeopardy",
    "tatsu-lab/alpaca_eval",
    "tau/zero_scrolls",
    "truthful_qa",
    "wckwan/MT-Eval",
    "winogrande",
    "xlangai/DS-1000",
    "HuggingFaceH4/ultrachat_200k",
    "allenai/WildChat",
    "futurehouse/lab-bench"
]

# Target evaluation datasets to ensure coverage
# Format: dataset_name:variant::source
TARGET_EVAL_DATASETS = [
    "arc_*:mc::xlarge",
    "mmlu_*:mc::olmes",
    "csqa:mc::xlarge",
    "piqa:mc::xlarge",
    "socialiqa:mc::xlarge",
    "drop:mc::gen2mc",
    "jeopardy:mc::gen2mc",
    "naturalqs:mc::gen2mc",
    "squad:mc::gen2mc",
    "coqa:mc::gen2mc",
    "basic_skills_*:mc::olmes",
    "medmcqa:mc::none",
    "medqa_en:mc::none",
    "sciq:mc::xlarge",
    "hellaswag:rc::xlarge",
    "winogrande:rc::xlarge",
    "lambada",
    "basic_skills_*:rc::olmes",
    "drop::xlarge",
    "jeopardy::xlarge",
    "naturalqs::xlarge",
    "squad::xlarge",
    "coqa::xlarge",
    "gsm8k::olmes",
    "gsm_symbolic*::olmo3",
    "minerva_math_*::olmes",
    "bigcodebench:3shot::olmo3",
    "codex_humaneval:3shot::olmo3",
    "deepseek_leetcode::olmo3",
    "ds1000:3shot::olmo3",
    "mbpp:3shot::olmo3",
    "multipl_e_humaneval:*::olmo3",
    "multipl_e_mbpp:*::olmo3",
    "codex_humanevalfim_single:temp0.2",
    "codex_humanevalfim_multi:temp0.2",
    "codex_humanevalfim_random:temp0.2",
    "alpaca_eval_v3::hamish_zs_reasoning",
    "ifeval::hamish_zs_reasoning",
    "gsm8k::zs_cot_latex",
    "minerva_math_*::hamish_zs_reasoning",
    "minerva_math_500::hamish_zs_reasoning",
    "aime::hamish_zs_reasoning",
    "codex_humanevalplus:0-shot-chat::tulu-thinker",
    "mbppplus:0-shot-chat::tulu-thinker",
    "livecodebench_codegeneration::tulu-thinker",
    "zebralogic::hamish_zs_reasoning",
    "popqa::hamish_zs_reasoning",
    "bbh_*:cot::hamish_zs_reasoning",
    "gpqa:0shot_cot::hamish_zs_reasoning",
    "agi_eval_*:0shot_cot::hamish_zs_reasoning",
    "mmlu_*:cot::hamish_zs_reasoning",
    "ifeval_ood::tulu-thinker",
    "ifeval_mt_*::tulu",
    "styled_alpacaeval::tulu-thinker",
    "multiturn_alpacaeval_*::tulu",
    "omega_*:0-shot-chat",
    "simpleqa::tulu-thinker",
    "styled_math500::tulu-thinker",
    "styled_popqa::tulu-thinker",
    "minerva_*:bpb::olmes",
    "codex_humaneval:3shot:bpb::none",
    "mbpp:3shot:bpb::none",
    "mt_mbpp_v2fix:*",
    "arc_*:bpb::full",
    "mmlu_*:bpb",
    "csqa:bpb::olmes:full",
    "hellaswag:bpb::olmes:full",
    "winogrande:bpb::olmes:full",
    "socialiqa:bpb::olmes:full",
    "piqa:bpb::olmes:full",
    "coqa:bpb::gen2mc",
    "drop:bpb::gen2mc",
    "jeopardy:bpb::gen2mc",
    "naturalqs:bpb::gen2mc",
    "squad:bpb::gen2mc",
    "sciq:bpb::olmo3",
    "qasper_yesno:bpb::olmes",
    "basic_skills:bpb::olmes",
    "lab_bench_dbqa:bpb",
    "lab_bench_protocolqa:bpb",
    "lambada:bpb",
    "medmcqa:bpb::none",
    "medqa_en:bpb::none",
    "sciriff_yesno:bpb::olmes",
    "arc_*:rc::olmes:full",
    "mmlu_*:rc::olmes",
    "csqa:rc::olmes:full",
    "hellaswag:rc::olmes:full",
    "winogrande:rc::olmes:full",
    "socialiqa:rc::olmes:full",
    "piqa:rc::olmes:full",
    "coqa:rc::gen2mc",
    "drop:rc::gen2mc",
    "jeopardy:rc::gen2mc",
    "naturalqs:rc::gen2mc",
    "squad:rc::gen2mc",
    "sciq:rc::olmo3",
    "qasper_yesno:rc::olmes",
    "basic_skills_*:rc::olmes",
    "lab_bench_dbqa",
    "lab_bench_protocolqa",
    "lambada",
    "medmcqa:rc::none",
    "medqa_en:rc::none",
    "sciriff_yesno:rc::olmes",
    "ultrachat_masked_ppl",
    "wildchat_masked_ppl"
]

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


def examine_all_datasets():
    """Examine all datasets in EVAL_PATHS."""
    print("=" * 80)
    print("DATASET EXAMINATION RESULTS")
    print("=" * 80)
    
    for i, path in enumerate(EVAL_PATHS, 1):
        result = examine_dataset(path)
        display_dataset_result(result, i, len(EVAL_PATHS))


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
    
    args = parser.parse_args()
    
    if args.ds:
        # Examine a single dataset
        print(f"Examining single dataset: {args.ds}")
        result = examine_dataset(args.ds)
        display_dataset_result(result)
    else:
        # Examine all datasets
        examine_all_datasets()
    
    print("=" * 80)
    print("EXAMINATION COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    main()
