#!/usr/bin/env python3
"""
Download and transform HuggingFace eval datasets for contamination detection.
"""

import json
from datasets import load_dataset
from pathlib import Path
import os

# Configuration for downloading and transforming eval datasets
EVAL_CONFIG = {
    'output_dir': 'fixtures/reference',
    'jsonl_format': {
        'text_field': 'text',
        'eval_field': 'eval_name',
        'index_field': 'index',
        'split_field': 'split'
    },
    'evals': {
        'gsm8k': {
            'hf_path': 'openai/gsm8k',
            'hf_config': 'main',
            'splits': ['train', 'test'],
            'transform': {
                'text_field': 'question',
                'answer_field': 'answer'
            }
        },
        'mmlu': {
            'hf_path': 'cais/mmlu',
            'hf_config': 'all',
            'splits': ['test', 'validation', 'dev'],
            'transform': {
                'text_field': 'question',
                'answer_field': 'choices',
                'extra_fields': ['subject']
            }
        },

        'aimo_validation': {
            'hf_path': 'AI-MO/aimo-validation-aime',
            'splits': ['train'],
            'transform': {
                'text_field': 'problem',
                'answer_field': 'answer'
            }
        },

        'ai2_arc_challenge': {
            'hf_path': 'allenai/ai2_arc',
            'hf_config': 'ARC-Challenge',
            'splits': ['train', 'test', 'validation'],
            'transform': {
                'text_field': 'question',
                'answer_field': 'answerKey',
                'choices_field': 'choices'
            }
        },

        'ai2_arc_easy': {
            'hf_path': 'allenai/ai2_arc',
            'hf_config': 'ARC-Easy',
            'splits': ['train', 'test', 'validation'],
            'transform': {
                'text_field': 'question',
                'answer_field': 'answerKey',
                'choices_field': 'choices'
            }
        },

        'popqa': {
            'hf_path': 'akariasai/PopQA',
            'splits': ['test'],
            'transform': {
                'text_field': 'question',
                'answer_field': 'possible_answers'
            }
        },

        'basic_skills_arithmetic': {
            'hf_path': 'allenai/basic-skills',
            'hf_config': 'arithmetic',
            'splits': ['validation'],
            'transform': {
                'text_field': 'question'
            }
        },

        'basic_skills_coding': {
            'hf_path': 'allenai/basic-skills',
            'hf_config': 'coding',
            'splits': ['validation'],
            'transform': {
                'text_field': 'question'
            }
        },

        'coqa_mc': {
            'hf_path': 'allenai/coqa_mc',
            'splits': ['validation'],
            'transform': {
                'text_field': 'query_original'
            }
        },

        'drop_mc': {
            'hf_path': 'allenai/drop_mc',
            'splits': ['validation'],
            'transform': {
                'text_field': 'question_original'
            }
        },

        'jeopardy_mc': {
            'hf_path': 'allenai/jeopardy_mc',
            'splits': ['test'],
            'transform': {
                'text_field': 'context_original'
            }
        },

        'multilingual_mbpp_python': {
            'hf_path': 'allenai/multilingual_mbpp',
            'hf_config': 'python',
            'splits': ['test', 'train', 'validation'],
            'transform': {
                'text_field': 'text'
            }
        },

        'nq_open_mc': {
            'hf_path': 'allenai/nq_open_mc',
            'splits': ['validation'],
            'transform': {
                'text_field': 'question'
            }
        },

        'qasper_yesno': {
            'hf_path': 'allenai/qasper-yesno',
            'splits': ['train', 'validation', 'test'],
            'transform': {
                'text_field': 'question'
            }
        },

        'sciriff_yesno': {
            'hf_path': 'allenai/sciriff-yesno',
            'splits': ['train', 'validation', 'test'],
            'transform': {
                'text_field': 'question'
            }
        },

        'simple_tom_mental_state': {
            'hf_path': 'allenai/SimpleToM',
            'hf_config': 'mental-state-qa',
            'splits': ['test'],
            'transform': {
                'text_field': 'question'
            }
        },

        'squad_mc': {
            'hf_path': 'allenai/squad_mc',
            'splits': ['validation'],
            'transform': {
                'text_field': 'question_original'
            }
        },

        'zebra_logic_grid': {
            'hf_path': 'allenai/ZebraLogicBench-private',
            'hf_config': 'grid_mode',
            'splits': ['test'],
            'transform': {
                'text_field': 'puzzle'
            }
        },

        'gsm_symbolic': {
            'hf_path': 'apple/GSM-Symbolic',
            'splits': ['test'],
            'transform': {
                'text_field': 'question'
            }
        },

        'bigcodebench': {
            'hf_path': 'bigcode/bigcodebench',
            'splits': ['v0.1.4'],
            'transform': {
                'text_field': 'complete_prompt'
            }
        },

        'bigcodebench_hard': {
            'hf_path': 'bigcode/bigcodebench-hard',
            'splits': ['v0.1.4'],
            'transform': {
                'text_field': 'complete_prompt'
            }
        },

        'cosmos_qa': {
            'hf_path': 'allenai/cosmos_qa',
            'splits': ['train', 'test', 'validation'],
            'transform': {
                'text_field': 'question'
            }
        },

        'cruxeval': {
            'hf_path': 'cruxeval-org/cruxeval',
            'splits': ['test'],
            'transform': {
                'text_field': 'code'
            }
        },

        'deepseek_leetcode': {
            'hf_path': 'davidheineman/deepseek-leetcode',
            'splits': ['test'],
            'transform': {
                'text_field': 'prompt'
            }
        },

        'medqa_en': {
            'hf_path': 'davidheineman/medqa-en',
            'splits': ['train', 'test', 'dev'],
            'transform': {
                'text_field': 'question'
            }
        },

        'coqa': {
            'hf_path': 'EleutherAI/coqa',
            'splits': ['train', 'validation'],
            'transform': {
                'text_field': 'story'
            }
        },

        'drop': {
            'hf_path': 'EleutherAI/drop',
            'splits': ['train', 'validation'],
            'transform': {
                'text_field': 'question'
            }
        },

        'hendrycks_math_algebra': {
            'hf_path': 'EleutherAI/hendrycks_math',
            'hf_config': 'algebra',
            'splits': ['train', 'test'],
            'transform': {
                'text_field': 'problem'
            }
        },

        'lambada_openai': {
            'hf_path': 'EleutherAI/lambada_openai',
            'splits': ['test'],
            'transform': {
                'text_field': 'text'
            }
        },

        'humaneval_plus': {
            'hf_path': 'evalplus/humanevalplus',
            'splits': ['test'],
            'transform': {
                'text_field': 'prompt'
            }
        },

        'mbpp_plus': {
            'hf_path': 'evalplus/mbppplus',
            'splits': ['test'],
            'transform': {
                'text_field': 'prompt'
            }
        },

        'mbpp': {
            'hf_path': 'google-research-datasets/mbpp',
            'splits': ['train', 'test', 'validation'],
            'transform': {
                'text_field': 'text'
            }
        },

        'nq_open': {
            'hf_path': 'google-research-datasets/nq_open',
            'splits': ['train', 'validation'],
            'transform': {
                'text_field': 'question'
            }
        },

        'tydiqa_primary': {
            'hf_path': 'google-research-datasets/tydiqa',
            'hf_config': 'primary_task',
            'splits': ['train', 'validation'],
            'transform': {
                'text_field': 'question_text'
            }
        },

        'ifeval': {
            'hf_path': 'HuggingFaceH4/ifeval',
            'splits': ['train'],
            'transform': {
                'text_field': 'prompt'
            }
        },

        'math_500': {
            'hf_path': 'HuggingFaceH4/MATH-500',
            'splits': ['test'],
            'transform': {
                'text_field': 'problem'
            }
        },

        'lexam_mcq': {
            'hf_path': 'LEXam-Benchmark/LEXam',
            'hf_config': 'mcq_4_choices',
            'splits': ['test'],
            'transform': {
                'text_field': 'question'
            }
        },

        'simple_qa': {
            'hf_path': 'lighteval/SimpleQA',
            'splits': ['test'],
            'transform': {
                'text_field': 'problem'
            }
        },

        'livecodebench': {
            'hf_path': 'livecodebench/code_generation_lite',
            'splits': ['test'],
            'transform': {
                'text_field': 'question_content'
            }
        },

        'humaneval_infilling': {
            'hf_path': 'loubnabnl/humaneval_infilling',
            'splits': ['test'],
            'transform': {
                'text_field': 'prompt'
            }
        },

        'logiqa': {
            'hf_path': 'lucasmccabe/logiqa',
            'splits': ['train', 'validation', 'test'],
            'transform': {
                'text_field': 'query'
            }
        },

        'bbh_boolean_expressions': {
            'hf_path': 'lukaemon/bbh',
            'hf_config': 'boolean_expressions',
            'splits': ['test'],
            'transform': {
                'text_field': 'input'
            }
        },

        'trivia_qa': {
            'hf_path': 'mandarjoshi/trivia_qa',
            'hf_config': 'rc',
            'splits': ['train', 'validation', 'test'],
            'transform': {
                'text_field': 'question'
            }
        },

        'multipl_e_humaneval_python': {
            'hf_path': 'nuprl/MultiPL-E',
            'hf_config': 'humaneval-js',
            'splits': ['test'],
            'transform': {
                'text_field': 'prompt'
            }
        },

        # 'mrcr': {
        #     'hf_path': 'openai/mrcr',
        #     'splits': ['train'],
        #     'transform': {
        #         'text_field': 'prompt'
        #     }
        # },

        'openbookqa': {
            'hf_path': 'allenai/openbookqa',
            'splits': ['train', 'validation', 'test'],
            'transform': {
                'text_field': 'question_stem'
            }
        },

        'medmcqa': {
            'hf_path': 'openlifescienceai/medmcqa',
            'splits': ['train', 'test', 'validation'],
            'transform': {
                'text_field': 'question'
            }
        },

        'gsm_plus': {
            'hf_path': 'qintongli/GSM-Plus',
            'splits': ['test'],
            'transform': {
                'text_field': 'question'
            }
        },

        'squad': {
            'hf_path': 'rajpurkar/squad',
            'splits': ['train', 'validation'],
            'transform': {
                'text_field': 'question'
            }
        },

        'squad_v2': {
            'hf_path': 'rajpurkar/squad_v2',
            'splits': ['train', 'validation'],
            'transform': {
                'text_field': 'question'
            }
        },

        'copycolors_mcqa': {
            'hf_path': 'sarahwie/copycolors_mcqa',
            'hf_config': '4_answer_choices',
            'splits': ['validation', 'test'],
            'transform': {
                'text_field': 'question'
            }
        },

        'sciq': {
            'hf_path': 'allenai/sciq',
            'splits': ['train', 'validation', 'test'],
            'transform': {
                'text_field': 'question'
            }
        },

        'social_i_qa': {
            'hf_path': 'allenai/social_i_qa',
            'splits': ['train', 'validation'],
            'transform': {
                'text_field': 'question'
            }
        },

        'jeopardy': {
            'hf_path': 'soldni/jeopardy',
            'hf_config': 'all_questions',
            'splits': ['train'],
            'transform': {
                'text_field': 'question'
            }
        },

        'zero_scrolls_qasper': {
            'hf_path': 'tau/zero_scrolls',
            'hf_config': 'qasper',
            'splits': ['validation', 'test'],
            'transform': {
                'text_field': 'input'
            }
        },

        'mt_eval_refinement': {
            'hf_path': 'wckwan/MT-Eval',
            'hf_config': 'refinement_single',
            'splits': ['test'],
            'transform': {
                'text_field': 'conv'
            }
        },

        'winogrande': {
            'hf_path': 'allenai/winogrande',
            'hf_config': 'winogrande_l',
            'splits': ['train', 'test', 'validation'],
            'transform': {
                'text_field': 'sentence'
            }
        },

        'ds_1000': {
            'hf_path': 'xlangai/DS-1000',
            'splits': ['test'],
            'transform': {
                'text_field': 'prompt'
            }
        }
    }
}

def download_and_transform_eval(eval_name, eval_config, global_config, document_id_counter):
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

    # Create output directory (resolve relative to project root)
    project_root = Path(__file__).parent.parent
    output_dir = project_root / global_config['output_dir']
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
                
                # Handle cases where text might be a list
                if isinstance(text, list):
                    # Join list elements into a string
                    text = ' '.join(str(item) for item in text)
                elif not isinstance(text, str):
                    # Convert to string if it's not already
                    text = str(text)
                
                # Skip empty or None text
                if not text or text.strip() == '':
                    continue

                # Generate records based on answer field configuration
                records_to_write = []

                # Create base record template
                def create_record_template():
                    record = {
                        global_config['jsonl_format']['eval_field']: eval_name,
                        global_config['jsonl_format']['index_field']: idx,
                        global_config['jsonl_format']['split_field']: split,
                    }
                    # Add any extra fields
                    if 'extra_fields' in eval_config['transform']:
                        for field in eval_config['transform']['extra_fields']:
                            if field in example:
                                record[field] = example[field]
                    return record

                # Handle answer fields if configured
                if 'answer_field' in eval_config['transform']:
                    answer_field = eval_config['transform']['answer_field']
                    if answer_field in example:
                        answer_value = example[answer_field]

                        # Always create a question-only record first
                        record_question_only = create_record_template()
                        record_question_only[global_config['jsonl_format']['text_field']] = text
                        records_to_write.append(record_question_only)

                        # Handle different answer field types
                        if isinstance(answer_value, list):
                            # Array of answers - create record for question + each answer
                            for answer in answer_value:
                                if answer is not None:  # Skip None answers
                                    record = create_record_template()
                                    record[global_config['jsonl_format']['text_field']] = text + " " + str(answer)
                                    records_to_write.append(record)
                        else:
                            # Single answer - create record with question + answer
                            if answer_value is not None:  # Skip None answers
                                record = create_record_template()
                                record[global_config['jsonl_format']['text_field']] = text + " " + str(answer_value)
                                records_to_write.append(record)
                    else:
                        # No answer value found - just use question
                        record = create_record_template()
                        record[global_config['jsonl_format']['text_field']] = text
                        records_to_write.append(record)
                else:
                    # No answer field configured - just use question
                    record = create_record_template()
                    record[global_config['jsonl_format']['text_field']] = text
                    records_to_write.append(record)

                # Handle choices field if configured (e.g., multiple choice questions)
                if 'choices_field' in eval_config['transform']:
                    choices_field = eval_config['transform']['choices_field']
                    if choices_field in example:
                        choices = example[choices_field]

                        # Handle choices structure: {'text': [...], 'label': [...]}
                        if isinstance(choices, dict) and 'text' in choices:
                            for choice_text in choices['text']:
                                record = create_record_template()
                                record[global_config['jsonl_format']['text_field']] = text + " " + str(choice_text)
                                records_to_write.append(record)
                        elif isinstance(choices, list):
                            # Handle simple list of choices
                            for choice in choices:
                                record = create_record_template()
                                record[global_config['jsonl_format']['text_field']] = text + " " + str(choice)
                                records_to_write.append(record)

                # Write all records to JSONL (filter out records with < 8 words)
                for record in records_to_write:
                    text = record[global_config['jsonl_format']['text_field']]
                    word_count = len(text.split())

                    # Skip records with fewer than 8 words
                    if word_count >= 8:
                        # Add unique document ID
                        record['doc_id'] = document_id_counter[0]
                        document_id_counter[0] += 1

                        f.write(json.dumps(record) + '\n')

        print(f"Saved {len(dataset[split])} examples to {output_file}")

def main():
    """Main function to process all eval datasets"""

    print(f"Processing {len(EVAL_CONFIG['evals'])} eval datasets...")

    # Initialize global document ID counter (using list for mutable reference)
    document_id_counter = [1]

    # Process each eval dataset
    for eval_name, eval_config in EVAL_CONFIG['evals'].items():
        download_and_transform_eval(eval_name, eval_config, EVAL_CONFIG, document_id_counter)

    print(f"Done! Generated {document_id_counter[0] - 1} total document IDs.")

if __name__ == "__main__":
    main()
