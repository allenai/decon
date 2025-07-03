#!/usr/bin/env python3
"""
Script to examine HuggingFace datasets and identify their splits and fields.
"""

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
    "apple/GSM-Symbolic",
    "bigcode/bigcodebench",
    "bigcode/bigcodebench-hard",
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
    "super_glue",
    "tatsu-lab/alpaca_eval",
    "tau/zero_scrolls",
    "truthful_qa",
    "wckwan/MT-Eval",
    "winogrande",
    "xlangai/DS-1000"
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

            # Examine each split
            for split_name in dataset.keys():
                try:
                    # Get first sample to examine
                    sample = next(iter(dataset[split_name]))
                    fields = list(sample.keys())
                    
                    # Truncate each field to first 500 characters for display
                    truncated_sample = {}
                    for field, value in sample.items():
                        if isinstance(value, str):
                            truncated_sample[field] = value[:500] + "..." if len(value) > 500 else value
                        elif isinstance(value, list):
                            truncated_sample[field] = str(value)[:500] + "..." if len(str(value)) > 500 else value
                        else:
                            truncated_sample[field] = str(value)[:500] + "..." if len(str(value)) > 500 else value
                    
                    results["configs"]["default"][split_name] = {
                        "fields": fields,
                        "sample": truncated_sample
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

                            # Examine each split in this config
                            for split_name in dataset.keys():
                                try:
                                    # Get first sample to examine
                                    sample = next(iter(dataset[split_name]))
                                    fields = list(sample.keys())
                                    
                                    # Truncate each field to first 500 characters for display
                                    truncated_sample = {}
                                    for field, value in sample.items():
                                        if isinstance(value, str):
                                            truncated_sample[field] = value[:500] + "..." if len(value) > 500 else value
                                        elif isinstance(value, list):
                                            truncated_sample[field] = str(value)[:500] + "..." if len(str(value)) > 500 else value
                                        else:
                                            truncated_sample[field] = str(value)[:500] + "..." if len(str(value)) > 500 else value
                                    
                                    results["configs"][config_name][split_name] = {
                                        "fields": fields,
                                        "sample": truncated_sample
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

def main():
    """Main function to examine all datasets."""
    print("=" * 80)
    print("DATASET EXAMINATION RESULTS")
    print("=" * 80)

    for i, path in enumerate(EVAL_PATHS, 1):
        print(f"\n[{i:2d}/{len(EVAL_PATHS)}] Examining: {path}")
        print("-" * 60)

        result = examine_dataset(path)

        if result["status"] == "success":
            print(f"✓ Dataset: {result['path']}")
            if result["path"] != result["original_path"]:
                print(f"  (resolved from: {result['original_path']})")

            total_configs = len(result["configs"])
            print(f"  Configs found: {total_configs}")

            for config_name, splits in result["configs"].items():
                if isinstance(splits, dict):
                    print(f"    Config '{config_name}': {len(splits)} splits")
                    for split_name, split_data in splits.items():
                        if isinstance(split_data, dict) and "fields" in split_data:
                            fields = split_data["fields"]
                            sample = split_data["sample"]
                            print(f"      {split_name}: {len(fields)} fields")
                            print(f"        Fields: {', '.join(fields)}")
                            print(f"        Sample data:")
                            for field, value in sample.items():
                                print(f"          {field}: {value}")
                        else:
                            print(f"      {split_name}: {split_data}")
                else:
                    print(f"    Config '{config_name}': {splits}")
        else:
            print(f"✗ UNKNOWN: {result['original_path']}")
            print(f"  Attempted path: {result['path']}")
            print(f"  Error: {result['error']}")

    print("\n" + "=" * 80)
    print("EXAMINATION COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    main()
