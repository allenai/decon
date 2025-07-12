#!/usr/bin/env python3
"""
Download and transform HuggingFace eval datasets for contamination detection.
"""

import argparse
import json
from datasets import load_dataset, Dataset
from pathlib import Path
import os

# Document splitting configuration
# Set to -1 to disable splitting, or a positive number for the character threshold
DOCUMENT_SPLIT_THRESHOLD = 2500

# Configuration for downloading and transforming eval datasets
EVAL_CONFIG = {
    'output_dir': 'fixtures/reference-download',
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
                'text_field': 'question',
                'answer_field': 'answer'
            }
        },

        'basic_skills_coding': {
            'hf_path': 'allenai/basic-skills',
            'hf_config': 'coding',
            'splits': ['validation'],
            'transform': {
                'text_field': 'question',
                'answer_field': 'answer'
            }
        },

        'basic_skills_common_knowledge': {
            'hf_path': 'allenai/basic-skills',
            'hf_config': 'common_knowledge',
            'splits': ['validation'],
            'transform': {
                'text_field': 'question',
                'answer_field': 'answer'
            }
        },

        'basic_skills_logical_reasoning': {
            'hf_path': 'allenai/basic-skills',
            'hf_config': 'logical_reasoning',
            'splits': ['validation'],
            'transform': {
                'text_field': 'question',
                'answer_field': 'answer'
            }
        },

        'basic_skills_pattern': {
            'hf_path': 'allenai/basic-skills',
            'hf_config': 'pattern',
            'splits': ['validation'],
            'transform': {
                'text_field': 'question',
                'answer_field': 'answer'
            }
        },

        'basic_skills_string_operations': {
            'hf_path': 'allenai/basic-skills',
            'hf_config': 'string_operations',
            'splits': ['validation'],
            'transform': {
                'text_field': 'question',
                'answer_field': 'answer'
            }
        },

        'coqa_mc': {
            'hf_path': 'allenai/coqa_mc',
            'splits': ['validation'],
            'transform': {
                'text_field': 'query_original',
                'answer_field': 'choices_original'
            }
        },

        'drop_mc': {
            'hf_path': 'allenai/drop_mc',
            'splits': ['validation'],
            'transform': {
                'text_field': 'question_original',
                'context_field': 'passage_original',
                'answer_field': 'answer_original'
            }
        },

        'jeopardy_mc': {
            'hf_path': 'allenai/jeopardy_mc',
            'splits': ['test'],
            'transform': {
                'text_field': 'context_original',
                'answer_field': 'continuation_original'
            }
        },

        # Please pick one among the available configs: ['cpp', 'c', 'javascript', 'java', 'python', 'php', 'csharp', 'typescript', 'bash', 'swift', 'go', 'rust', 'ruby', 'r', 'matlab', 'scala', 'haskell']
        # TODO iterate over these ^
        'multilingual_mbpp_python': {
            'hf_path': 'allenai/multilingual_mbpp',
            'hf_config': 'python',
            'splits': ['test', 'train', 'validation'],
            'transform': {
                'text_field': 'text',
                'answer_field': 'code'
            }
        },

        'nq_open_mc': {
            'hf_path': 'allenai/nq_open_mc',
            'splits': ['validation'],
            'transform': {
                'text_field': 'question',
                'answer_field': 'answer_original'
            }
        },

        'qasper_yesno': {
            'hf_path': 'allenai/qasper-yesno',
            'splits': ['train', 'validation', 'test'],
            'transform': {
                'text_field': 'question',
                'context_field': 'context',
                'answer_field': 'answer'
            }
        },

        'sciriff_yesno': {
            'hf_path': 'allenai/sciriff-yesno',
            'splits': ['train', 'validation', 'test'],
            'transform': {
                'text_field': 'question',
                'context_field': 'context',
                'answer_field': 'answer'
            }
        },

        'simple_tom_mental_state': {
            'hf_path': 'allenai/SimpleToM',
            'hf_config': 'mental-state-qa',
            'splits': ['test'],
            'transform': {
                'text_field': 'question',
                'context_field': 'story',
                'answer_field': 'answerKey'
            }
        },

        'squad_mc': {
            'hf_path': 'allenai/squad_mc',
            'splits': ['validation'],
            'transform': {
                'text_field': 'question_original',
                'context_field': 'context_original',
                'answer_field': 'answers_original'
            }
        },

        'zebra_logic_grid': {
            'hf_path': 'allenai/ZebraLogicBench-private',
            'hf_config': 'grid_mode',
            'splits': ['test'],
            'transform': {
                'text_field': 'puzzle',
                'answer_field': 'solution'
            }
        },

        'zebra_logic_mc': {
            'hf_path': 'allenai/ZebraLogicBench-private',
            'hf_config': 'mc_mode',
            'splits': ['test'],
            'transform': {
                'text_field': 'puzzle',
                'answer_field': 'solution'
            }
        },

        'gsm_symbolic': {
            'hf_path': 'apple/GSM-Symbolic',
            'splits': ['test'],
            'transform': {
                'text_field': 'question',
                'answer_field': 'answer'
            }
        },

        'bigcodebench': {
            'hf_path': 'bigcode/bigcodebench',
            'splits': ['v0.1.4'],
            'transform': {
                'text_field': 'complete_prompt',
                'answer_field': 'canonical_solution'
            }
        },

        'bigcodebench_hard': {
            'hf_path': 'bigcode/bigcodebench-hard',
            'splits': ['v0.1.4'],
            'transform': {
                'text_field': 'complete_prompt',
                'answer_field': 'canonical_solution'
            }
        },

        'cosmos_qa': {
            'hf_path': 'allenai/cosmos_qa',
            'splits': ['train', 'test', 'validation'],
            'transform': {
                'text_field': 'question',
                'context_field': 'context',
                'answer_field': 'answer'
            }
        },

        'cruxeval': {
            'hf_path': 'cruxeval-org/cruxeval',
            'splits': ['test'],
            'transform': {
                'text_field': 'code',
                'answer_field': 'output'
            }
        },

        'deepseek_leetcode': {  # Settled
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
                'text_field': 'question',
                'answer_field': 'answer'
            }
        },

        'coqa': {
            'hf_path': 'EleutherAI/coqa',
            'splits': ['train', 'validation'],
            'transform': 'auto'
        },

        'drop': {
            'hf_path': 'EleutherAI/drop',
            'splits': ['train', 'validation'],
            'transform': {
                'text_field': 'question',
                'context_field': 'passage'
            }
        },

        'hendrycks_math_algebra': {
            'hf_path': 'EleutherAI/hendrycks_math',
            'hf_config': 'algebra',
            'splits': ['train', 'test'],
            'transform': {
                'text_field': 'problem',
                'answer_field': 'solution'
            }
        },

        'hendrycks_math_counting_and_probability': {
            'hf_path': 'EleutherAI/hendrycks_math',
            'hf_config': 'counting_and_probability',
            'splits': ['train', 'test'],
            'transform': {
                'text_field': 'problem',
                'answer_field': 'solution'
            }
        },

        'hendrycks_math_geometry': {
            'hf_path': 'EleutherAI/hendrycks_math',
            'hf_config': 'geometry',
            'splits': ['train', 'test'],
            'transform': {
                'text_field': 'problem',
                'answer_field': 'solution'
            }
        },

        'hendrycks_math_intermediate_algebra': {
            'hf_path': 'EleutherAI/hendrycks_math',
            'hf_config': 'intermediate_algebra',
            'splits': ['train', 'test'],
            'transform': {
                'text_field': 'problem',
                'answer_field': 'solution'
            }
        },

        'hendrycks_math_number_theory': {
            'hf_path': 'EleutherAI/hendrycks_math',
            'hf_config': 'number_theory',
            'splits': ['train', 'test'],
            'transform': {
                'text_field': 'problem',
                'answer_field': 'solution'
            }
        },

        'hendrycks_math_prealgebra': {
            'hf_path': 'EleutherAI/hendrycks_math',
            'hf_config': 'prealgebra',
            'splits': ['train', 'test'],
            'transform': {
                'text_field': 'problem',
                'answer_field': 'solution'
            }
        },

        'hendrycks_math_precalculus': {
            'hf_path': 'EleutherAI/hendrycks_math',
            'hf_config': 'precalculus',
            'splits': ['train', 'test'],
            'transform': {
                'text_field': 'problem',
                'answer_field': 'solution'
            }
        },

        'lambada_openai': {  # Settled
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
                'text_field': 'prompt',
                'answer_field': 'canonical_solution'
            }
        },

        'mbpp_plus': {
            'hf_path': 'evalplus/mbppplus',
            'splits': ['test'],
            'transform': 'auto'
        },

        'mbpp': {
            'hf_path': 'google-research-datasets/mbpp',
            'splits': ['train', 'test', 'validation'],
            'transform': 'auto'
        },

        'nq_open': {
            'hf_path': 'google-research-datasets/nq_open',
            'splits': ['train', 'validation'],
            'transform': {
                'text_field': 'question',
                'answer_field': 'answer'
            }
        },

        # Massive with bad chars and unusual language.
        # 'tydiqa_primary': {
        #     'hf_path': 'google-research-datasets/tydiqa',
        #     'hf_config': 'primary_task',
        #     'splits': ['train', 'validation'],
        #     'transform': {
        #         'text_field': 'question_text',
        #         'context_field': 'document_plaintext'
        #     }
        # },

        'ifeval': {  # Settled
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
                'text_field': 'problem',
                'answer_field': 'solution'
            }
        },

        # Please pick one among the available configs: ['mcq_4_choices', 'mcq_perturbation', 'open_question']
        'lexam_mcq': {
            'hf_path': 'LEXam-Benchmark/LEXam',
            'hf_config': 'mcq_4_choices',
            'splits': ['test'],
            'transform': {
                'text_field': 'question',
                'answer_field': 'choices'
            }
        },

        'simple_qa': {
            'hf_path': 'lighteval/SimpleQA',
            'splits': ['test'],
            'transform': {
                'text_field': 'problem',
                'answer_field': 'answer'
            }
        },

        'livecodebench': {  # Settled
            'hf_path': 'livecodebench/code_generation_lite',
            'splits': ['test'],
            'transform': {
                'text_field': 'question_content'
            }
        },

        'humaneval_infilling_multiline': {
            'hf_path': 'loubnabnl/humaneval_infilling',
            'hf_config': 'HumanEval-MultiLineInfilling',
            'splits': ['test'],
            'transform': {
                'text_field': 'prompt',
                'answer_field': 'canonical_solution'
            }
        },

        'humaneval_infilling_singleline': {
            'hf_path': 'loubnabnl/humaneval_infilling',
            'hf_config': 'HumanEval-SingleLineInfilling',
            'splits': ['test'],
            'transform': {
                'text_field': 'prompt',
                'answer_field': 'canonical_solution'
            }
        },

        'humaneval_infilling_randomspan': {
            'hf_path': 'loubnabnl/humaneval_infilling',
            'hf_config': 'HumanEval-RandomSpanInfilling',
            'splits': ['test'],
            'transform': {
                'text_field': 'prompt',
                'answer_field': 'canonical_solution'
            }
        },

        'logiqa': {
            'hf_path': 'lucasmccabe/logiqa',
            'splits': ['train', 'validation', 'test'],
            'transform': {
                'text_field': 'query',
                'context_field': 'context',
                'answer_field': 'correct_option'
            }
        },

        # ['boolean_expressions', 'causal_judgement', 'date_understanding', 'disambiguation_qa', 'dyck_languages', 'formal_fallacies', 'geometric_shapes', 'hyperbaton', 'logical_deduction_five_objects', 'logical_deduction_seven_objects', 'logical_deduction_three_objects', 'movie_recommendation', 'multistep_arithmetic_two', 'navigate', 'object_counting', 'penguins_in_a_table', 'reasoning_about_colored_objects', 'ruin_names', 'salient_translation_error_detection', 'snarks', 'sports_understanding', 'temporal_sequences', 'tracking_shuffled_objects_five_objects', 'tracking_shuffled_objects_seven_objects', 'tracking_shuffled_objects_three_objects', 'web_of_lies', 'word_sorting']
        'bbh_boolean_expressions': {
            'hf_path': 'lukaemon/bbh',
            'hf_config': 'boolean_expressions',
            'splits': ['test'],
            'transform': {
                'text_field': 'input',
                'answer_field': 'target'
            }
        },

        'bbh_causal_judgement': {
            'hf_path': 'lukaemon/bbh',
            'hf_config': 'causal_judgement',
            'splits': ['test'],
            'transform': {
                'text_field': 'input',
                'answer_field': 'target'
            }
        },

        'bbh_date_understanding': {
            'hf_path': 'lukaemon/bbh',
            'hf_config': 'date_understanding',
            'splits': ['test'],
            'transform': {
                'text_field': 'input',
                'answer_field': 'target'
            }
        },

        'bbh_disambiguation_qa': {
            'hf_path': 'lukaemon/bbh',
            'hf_config': 'disambiguation_qa',
            'splits': ['test'],
            'transform': {
                'text_field': 'input',
                'answer_field': 'target'
            }
        },

        'bbh_formal_fallacies': {
            'hf_path': 'lukaemon/bbh',
            'hf_config': 'formal_fallacies',
            'splits': ['test'],
            'transform': {
                'text_field': 'input',
                'answer_field': 'target'
            }
        },

        'bbh_logical_deduction_three_objects': {
            'hf_path': 'lukaemon/bbh',
            'hf_config': 'logical_deduction_three_objects',
            'splits': ['test'],
            'transform': {
                'text_field': 'input',
                'answer_field': 'target'
            }
        },

        'trivia_qa': {
            'hf_path': 'mandarjoshi/trivia_qa',
            'hf_config': 'rc',
            'splits': ['train', 'validation', 'test'],
            'transform': {
                'text_field': 'question',
                'answer_field': 'answer.value'
            }
        },

        'multipl_e_humaneval_python': {
            'hf_path': 'nuprl/MultiPL-E',
            'hf_config': 'humaneval-python',
            'splits': ['test'],
            'transform': 'auto'
        },

        'multipl_e_humaneval_js': {
            'hf_path': 'nuprl/MultiPL-E',
            'hf_config': 'humaneval-js',
            'splits': ['test'],
            'transform': 'auto'
        },

        'multipl_e_humaneval_java': {
            'hf_path': 'nuprl/MultiPL-E',
            'hf_config': 'humaneval-java',
            'splits': ['test'],
            'transform': 'auto'
        },

        'multipl_e_humaneval_go': {
            'hf_path': 'nuprl/MultiPL-E',
            'hf_config': 'humaneval-go',
            'splits': ['test'],
            'transform': 'auto'
        },

        'multipl_e_humaneval_cpp': {
            'hf_path': 'nuprl/MultiPL-E',
            'hf_config': 'humaneval-cpp',
            'splits': ['test'],
            'transform': 'auto'
        },

        'multipl_e_mbpp_python': {
            'hf_path': 'nuprl/MultiPL-E',
            'hf_config': 'mbpp-python',
            'splits': ['test'],
            'transform': 'auto'
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
                'text_field': 'question_stem',
                'answer_field': 'choices'
            }
        },

        'medmcqa': {
            'hf_path': 'openlifescienceai/medmcqa',
            'splits': ['train', 'test', 'validation'],
            'transform': {
                'text_field': 'question',
                'answer_field': 'exp'
            }
        },

        'gsm_plus': {
            'hf_path': 'qintongli/GSM-Plus',
            'splits': ['test', 'testmini'],
            'transform': {
                'text_field': 'question',
                'answer_field': 'solution',
                'extra_fields': ['perturbation_type']
            }
        },

        'squad': {
            'hf_path': 'squad',
            'splits': ['train', 'validation'],
            'transform': {
                'text_field': 'question',
                'context_field': 'context',
                'answer_field': 'answers'
            }
        },

        'squad_v2': {
            'hf_path': 'rajpurkar/squad_v2',
            'splits': ['train', 'validation'],
            'transform': {
                'text_field': 'question',
                'context_field': 'context',
                'answer_field': 'answers'
            }
        },

        'copycolors_mcqa': {
            'hf_path': 'sarahwie/copycolors_mcqa',
            'hf_config': '4_answer_choices',
            'splits': ['validation', 'test'],
            'transform': 'auto'
        },

        'sciq': {
            'hf_path': 'allenai/sciq',
            'splits': ['train', 'validation', 'test'],
            'transform': {
                'text_field': 'question',
                'context_field': 'support',
                'answer_field': 'correct_answer'
            }
        },

        'social_i_qa': {
            'hf_path': 'allenai/social_i_qa',
            'splits': ['train', 'validation'],
            'transform': {
                'context_field': 'context',
                'text_field': 'question'
            }
        },

        'jeopardy': {
            'hf_path': 'soldni/jeopardy',
            'hf_config': 'all_questions',
            'splits': ['train'],
            'transform': {
                'text_field': 'question',
                'answer_field': 'continuation'
            }
        },

        'zero_scrolls_qasper': {  # Settled
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
            'transform': 'auto'
        },

        'winogrande': { # Settled
            'hf_path': 'allenai/winogrande',
            'hf_config': 'winogrande_l',
            'splits': ['train', 'test', 'validation'],
            'transform': {
                'text_field': 'sentence'
            }
        },

        'ds_1000': { # Settled
            'hf_path': 'xlangai/DS-1000',
            'splits': ['test'],
            'transform': {
                'text_field': 'prompt',
                'answer': 'reference_code'
            }
        },

        'omega_compositional': {
            'hf_path': 'allenai/omega-compositional',
            'splits': ['train', 'test'],
            'transform': 'auto'
        },

        'omega_explorative': {
            'hf_path': 'allenai/omega-explorative',
            'splits': ['train', 'test_in', 'test_out'],
            'transform': 'auto'
        },

        'omega_transformative': {
            'hf_path': 'allenai/omega-transformative',
            'splits': ['train', 'test'],
            'transform': 'auto'
        },

        'commonsense_qa': {
            'hf_path': 'commonsense_qa',
            'splits': ['train', 'validation', 'test'],
            'transform': {
                'text_field': 'question',
                'answer_field': 'answerKey'
            }
        },

        'hellaswag': {
            'hf_path': 'hellaswag',
            'splits': ['train', 'test', 'validation'],
            'transform': {
                'text_field': 'ctx',
                'answer_field': 'label'
            }
        },

        'piqa': {
            'hf_path': 'piqa',
            'splits': ['train', 'test', 'validation'],
            'transform': {
                'text_field': 'goal',
                'answer_field': 'label'
            }
        },

        'super_glue_boolq': {
            'hf_path': 'aps/super_glue',
            'hf_config': 'boolq',
            'splits': ['train', 'validation', 'test'],
            'transform': {
                'text_field': 'question',
                'context_field': 'passage',
                'answer_field': 'label'
            }
        },

        'super_glue_rte': {
            'hf_path': 'aps/super_glue',
            'hf_config': 'rte',
            'splits': ['train', 'validation', 'test'],
            'transform': {
                'text_field': 'hypothesis',
                'context_field': 'premise',
                'answer_field': 'label'
            }
        },

        'truthful_qa': {
            'hf_path': 'truthfulqa/truthful_qa',
            'hf_config': 'generation',
            'splits': ['validation'],
            'transform': {
                'text_field': 'question',
                'answer_field': 'best_answer'
            }
        },

        # 'ultrachat_200k': {
        #     'hf_path': 'HuggingFaceH4/ultrachat_200k',
        #     'splits': ['train_sft', 'test_sft', 'train_gen', 'test_gen'],
        #     'transform': 'auto'  # Will use auto extraction to handle the messages field
        # },

        # 'wildchat': {
        #     'hf_path': 'allenai/WildChat',
        #     'splits': ['train'],  # Only has train split
        #     'transform': {
        #         'text_field': 'text',
        #     }
        # },

        'lab_bench_dbqa': {
            'hf_path': 'futurehouse/lab-bench',
            'hf_config': 'DbQA',
            'splits': ['train'],
            'transform': {
                'text_field': 'question',
                'answer_field': 'ideal',
                'choices_field': 'distractors'
            }
        },

        'lab_bench_protocolqa': {
            'hf_path': 'futurehouse/lab-bench',
            'hf_config': 'ProtocolQA',
            'splits': ['train'],
            'transform': {
                'text_field': 'question',
                'answer_field': 'ideal',
                'choices_field': 'distractors'
            }
        },

        # New datasets from TARGET_TO_EVAL_MAPPING
        # Note: cais/hle is a gated dataset requiring access approval
        'mmlu_pro': {
            'hf_path': 'TIGER-Lab/MMLU-Pro',
            'splits': ['test', 'validation'],
            'transform': {
                'text_field': 'question',
                'answer_field': 'answer',
                'choices_field': 'options',
                'extra_fields': ['category']
            }
        },

        'super_gpqa': {
            'hf_path': 'm-a-p/SuperGPQA',
            'splits': ['train'],
            'transform': {
                'text_field': 'question',
                'answer_field': 'answer',
                'choices_field': 'options',
                'extra_fields': ['discipline', 'field', 'subfield']
            }
        },

        'bbeh': {
            'hf_path': 'hubert233/BigBenchExtraHard',
            'splits': ['boardgame_qa', 'boolean_expressions', 'buggy_tables', 'causal_understanding',
                      'disambiguation_qa', 'dyck_languages', 'geometric_shapes', 'hyperbaton', 'linguini',
                      'movie_recommendation', 'multistep_arithmetic', 'nycc', 'object_counting',
                      'object_properties', 'sarc_triples', 'shuffled_objects', 'spatial_reasoning',
                      'sportqa', 'temporal_sequence', 'time_arithmetic', 'web_of_lies', 'word_sorting',
                      'zebra_puzzles'],
            'transform': {
                'text_field': 'input',
                'answer_field': 'target'
            }
        },

        'aime_2025_i': {
            'hf_path': 'opencompass/AIME2025',
            'hf_config': 'AIME2025-I',
            'splits': ['test'],
            'transform': {
                'text_field': 'question',
                'answer_field': 'answer'
            }
        },

        'aime_2025_ii': {
            'hf_path': 'opencompass/AIME2025',
            'hf_config': 'AIME2025-II',
            'splits': ['test'],
            'transform': {
                'text_field': 'question',
                'answer_field': 'answer'
            }
        },

        'humaneval_pro': {
            'hf_path': 'CodeEval-Pro/humaneval-pro',
            'splits': ['train'],
            'transform': {
                'text_field': 'new_problem',
                'answer_field': 'new_solution'
            }
        },

        'mbpp_pro': {
            'hf_path': 'CodeEval-Pro/mbpp-pro',
            'splits': ['train'],
            'transform': {
                'text_field': 'new_problem',
                'answer_field': 'new_solution'
            }
        },

        'codeforces': {
            'hf_path': 'open-r1/codeforces',
            'splits': ['test'],  # Only decontaminate test split as requested
            'transform': {
                'text_field': 'description',
                'context_field': 'title',
                'extra_fields': ['contest_name', 'rating', 'tags']
            }
        },

        'ifbench_multiturn_constraints': {
            'hf_path': 'allenai/IFBench_multi-turn',
            'hf_config': 'ifbench_constraints',
            'splits': ['test'],
            'transform': {
                'text_field': 'prompt'
            }
        },

        'ifbench_multiturn_ifeval': {
            'hf_path': 'allenai/IFBench_multi-turn',
            'hf_config': 'ifeval_constraints',
            'splits': ['test'],
            'transform': {
                'text_field': 'prompt'
            }
        },

        'truthfulqa': {
            'hf_path': 'domenicrosati/TruthfulQA',
            'splits': ['train'],
            'transform': {
                'text_field': 'Question',
                'answer_field': 'Best Answer',
                'extra_fields': ['Type', 'Category']
            }
        },

        'lbpp': {
            'hf_path': 'CohereLabs/lbpp',
            'splits': ['test'],
            'transform': {
                'text_field': 'instruction',
                'answer_field': 'completion'
            }
        },

        'repobench_python_cross_file_first': {
            'hf_path': 'tianyang/repobench_python_v1.1',
            'splits': ['cross_file_first'],
            'transform': {
                'text_field': 'cropped_code',
                'answer_field': 'next_line',
                'extra_fields': ['repo_name', 'file_path', 'level']
            }
        },

        'repobench_python_cross_file_random': {
            'hf_path': 'tianyang/repobench_python_v1.1',
            'splits': ['cross_file_random'],
            'transform': {
                'text_field': 'cropped_code',
                'answer_field': 'next_line',
                'extra_fields': ['repo_name', 'file_path', 'level']
            }
        },

        'repobench_python_in_file': {
            'hf_path': 'tianyang/repobench_python_v1.1',
            'splits': ['in_file'],
            'transform': {
                'text_field': 'cropped_code',
                'answer_field': 'next_line',
                'extra_fields': ['repo_name', 'file_path', 'level']
            }
        },

        'xstest': {
            'hf_path': 'walledai/XSTest',
            'splits': ['test'],
            'transform': {
                'text_field': 'prompt',
                'extra_fields': ['focus', 'type', 'label']
            }
        },

        'harmbench_contextual': {
            'hf_path': 'walledai/HarmBench',
            'hf_config': 'contextual',
            'splits': ['train'],
            'transform': {
                'text_field': 'prompt',
                'context_field': 'context',
                'extra_fields': ['category']
            }
        },

        'harmbench_copyright': {
            'hf_path': 'walledai/HarmBench',
            'hf_config': 'copyright',
            'splits': ['train'],
            'transform': {
                'text_field': 'prompt',
                'extra_fields': ['tags']
            }
        },

        'harmbench_standard': {
            'hf_path': 'walledai/HarmBench',
            'hf_config': 'standard',
            'splits': ['train'],
            'transform': {
                'text_field': 'prompt',
                'extra_fields': ['category']
            }
        },

        'tulu3_do_anything_now': {
            'hf_path': 'allenai/tulu-3-do-anything-now-eval',
            'splits': ['test'],
            'transform': {
                'text_field': 'adversarial',
                'extra_fields': ['vanilla', 'jailbreak', 'platform', 'source']
            }
        },

        'tulu3_trustllm_jailbreak': {
            'hf_path': 'allenai/tulu-3-trustllm-jailbreaktrigger-eval',
            'splits': ['test'],
            'transform': {
                'text_field': 'prompt',
                'extra_fields': ['label', 'source']
            }
        },

        'wildjailbreak_train': {
            'hf_path': 'allenai/wildjailbreak',
            'hf_config': 'train',
            'splits': ['train'],
            'transform': {
                'text_field': 'vanilla',
                'extra_fields': ['data_type']
            }
        },

        'wildjailbreak_eval': {
            'hf_path': 'allenai/wildjailbreak',
            'hf_config': 'eval',
            'splits': ['train'],
            'transform': {
                'text_field': 'adversarial',
                'extra_fields': ['label', 'data_type']
            }
        },

        'wildguardtest': {
            'hf_path': 'walledai/WildGuardTest',
            'splits': ['train'],
            'transform': {
                'text_field': 'prompt',
                'extra_fields': ['adversarial', 'label']
            }
        },

        # AGIEval English datasets (loaded from local files)
        'agi_eval_aqua_rat': {
            'local_path': 'fixtures/reference-static/aqua-rat.jsonl',
            'splits': ['train'],
            'transform': {
                'text_field': 'question',
                'context_field': 'passage',
                'answer_field': 'label',
                'choices_field': 'options'
            }
        },

        'agi_eval_gaokao_english': {
            'local_path': 'fixtures/reference-static/gaokao-english.jsonl',
            'splits': ['train'],
            'transform': {
                'text_field': 'question',
                'context_field': 'passage',
                'answer_field': 'label',
                'choices_field': 'options'
            }
        },

        'agi_eval_logiqa_en': {
            'local_path': 'fixtures/reference-static/logiqa-en.jsonl',
            'splits': ['train'],
            'transform': {
                'text_field': 'question',
                'context_field': 'passage',
                'answer_field': 'label',
                'choices_field': 'options'
            }
        },

        'agi_eval_lsat_ar': {
            'local_path': 'fixtures/reference-static/lsat-ar.jsonl',
            'splits': ['train'],
            'transform': {
                'text_field': 'question',
                'context_field': 'passage',
                'answer_field': 'label',
                'choices_field': 'options'
            }
        },

        'agi_eval_lsat_lr': {
            'local_path': 'fixtures/reference-static/lsat-lr.jsonl',
            'splits': ['train'],
            'transform': {
                'text_field': 'question',
                'context_field': 'passage',
                'answer_field': 'label',
                'choices_field': 'options'
            }
        },

        'agi_eval_lsat_rc': {
            'local_path': 'fixtures/reference-static/lsat-rc.jsonl',
            'splits': ['train'],
            'transform': {
                'text_field': 'question',
                'context_field': 'passage',
                'answer_field': 'label',
                'choices_field': 'options'
            }
        },

        'agi_eval_math': {
            'local_path': 'fixtures/reference-static/math.jsonl',
            'splits': ['train'],
            'transform': {
                'text_field': 'question',
                'context_field': 'passage',
                'answer_field': 'label'
            }
        },

        'agi_eval_sat_en': {
            'local_path': 'fixtures/reference-static/sat-en.jsonl',
            'splits': ['train'],
            'transform': {
                'text_field': 'question',
                'context_field': 'passage',
                'answer_field': 'label',
                'choices_field': 'options'
            }
        },

        'agi_eval_sat_en_without_passage': {
            'local_path': 'fixtures/reference-static/sat-en-without-passage.jsonl',
            'splits': ['train'],
            'transform': {
                'text_field': 'question',
                'context_field': 'passage',
                'answer_field': 'label',
                'choices_field': 'options'
            }
        },

        'agi_eval_sat_math': {
            'local_path': 'fixtures/reference-static/sat-math.jsonl',
            'splits': ['train'],
            'transform': {
                'text_field': 'question',
                'context_field': 'passage',
                'answer_field': 'label',
                'choices_field': 'options'
            }
        }
    }
}


def split_document(text, threshold):
    """Split a document into chunks if it exceeds the threshold.

    Args:
        text: The text to potentially split
        threshold: Character threshold for splitting (-1 to disable)

    Returns:
        List of text chunks
    """
    if threshold == -1 or len(text) <= threshold:
        return [text]

    chunks = []
    current_chunk = ""

    # Split on whitespace
    words = text.split()

    for word in words:
        # Check if adding this word would exceed threshold
        if current_chunk and len(current_chunk) + len(word) + 1 > threshold:
            # Save current chunk and start new one
            chunks.append(current_chunk.strip())
            current_chunk = word
        else:
            # Add word to current chunk
            if current_chunk:
                current_chunk += " " + word
            else:
                current_chunk = word

    # Don't forget the last chunk
    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks


def load_local_jsonl(file_path):
    """Load a local JSONL file as a HuggingFace Dataset."""
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return Dataset.from_list(data)


def auto_extract(example):
    """Automatically extract text fields from complex dataset structures.

    Returns:
        tuple: (longest_text, second_longest_text) or (longest_text, None) if only one found
    """
    text_candidates = []

    def extract_text_fields(obj, path=""):
        """Recursively traverse the data structure to find text fields."""
        if isinstance(obj, str):
            # Check if this string has at least 8 words
            word_count = len(obj.split())
            if word_count >= 8:
                text_candidates.append((obj, word_count, path))
        elif isinstance(obj, (list, tuple)):
            # For lists, check each element
            for i, item in enumerate(obj):
                extract_text_fields(item, f"{path}[{i}]")
        elif isinstance(obj, dict):
            # For dicts, check each value
            for key, value in obj.items():
                new_path = f"{path}.{key}" if path else key
                extract_text_fields(value, new_path)
        # For other types (int, float, bool, etc.), we skip them
        # unless you want to convert numbers to strings

    # Start extraction
    extract_text_fields(example)

    # Sort by word count (descending)
    text_candidates.sort(key=lambda x: x[1], reverse=True)

    # Return the longest and second longest
    if len(text_candidates) == 0:
        return None, None
    elif len(text_candidates) == 1:
        return text_candidates[0][0], None
    else:
        return text_candidates[0][0], text_candidates[1][0]


def get_nested_field(obj, field_path):
    """Access nested fields using dot notation (e.g., 'answer.value')"""
    fields = field_path.split('.')
    value = obj
    for field in fields:
        if isinstance(value, dict) and field in value:
            value = value[field]
        else:
            return None
    return value


def download_and_transform_eval(eval_name, eval_config, global_config, document_id_counter):
    """Download HF dataset and transform to our JSONL format"""

    if 'local_path' in eval_config:
        print(f"Loading {eval_name} from local file: {eval_config['local_path']}...")
    else:
        print(f"Loading {eval_name} from {eval_config['hf_path']}...")

    # Load dataset from HuggingFace or local file
    try:
        if 'local_path' in eval_config:
            # Load from local JSONL file
            local_file = Path(eval_config['local_path'])
            if not local_file.exists():
                print(f"Error: Local file not found: {local_file}")
                return
            # Create a dataset dict with a single 'train' split
            dataset = {'train': load_local_jsonl(local_file)}
        elif 'hf_config' in eval_config:
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

        # Track seen contexts to avoid duplicates
        seen_contexts = set()
        skipped_duplicates = 0

        with open(output_file, 'w') as f:
            for idx, example in enumerate(dataset[split]):
                # Check if we should use auto extraction
                if eval_config['transform'] == 'auto':
                    # Use auto extraction
                    text, answer_value = auto_extract(example)

                    # Skip if no valid text found
                    if text is None:
                        continue

                    # Generate records
                    records_to_write = []

                    # Create base record template
                    record = {
                        global_config['jsonl_format']['eval_field']: eval_name,
                        global_config['jsonl_format']['index_field']: idx,
                        global_config['jsonl_format']['split_field']: split,
                    }

                    # If we have an answer, create record with text + answer
                    # Otherwise just use text
                    if answer_value is not None:
                        record[global_config['jsonl_format']['text_field']] = text + " " + answer_value
                    else:
                        record[global_config['jsonl_format']['text_field']] = text
                    records_to_write.append(record)

                else:
                    # Use the existing manual extraction logic
                    # Extract text field
                    text_field = eval_config['transform']['text_field']
                    text = get_nested_field(example, text_field)

                    # Skip if field not found
                    if text is None:
                        continue

                    # Handle cases where text might be a list
                    if isinstance(text, list):
                        # Take the first element if it's a list
                        if text:  # Check if list is not empty
                            text = str(text[0])
                        else:
                            text = ""
                    elif not isinstance(text, str):
                        # Convert to string if it's not already
                        text = str(text)

                    # Handle context field if configured
                    original_context = None
                    if 'context_field' in eval_config['transform']:
                        context_field = eval_config['transform']['context_field']
                        context = get_nested_field(example, context_field)
                        if context is not None:
                            # Handle cases where context might be a list
                            if isinstance(context, list):
                                # Take the first element if it's a list
                                if context:  # Check if list is not empty
                                    context = str(context[0])
                                else:
                                    context = ""
                            elif not isinstance(context, str):
                                context = str(context)

                            # Store original context for deduplication
                            if context and context.strip():
                                original_context = context
                                # Check if we've seen this context before
                                context_hash = hash(context)
                                if context_hash in seen_contexts:
                                    skipped_duplicates += 1
                                    continue  # Skip this record
                                else:
                                    seen_contexts.add(context_hash)

                                # Concatenate context with text
                                text = context + " " + text

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
                        answer_value = get_nested_field(example, answer_field)
                        if answer_value is not None:
                            # Handle different answer field types
                            if isinstance(answer_value, list):
                                # Array of answers - just take the first one
                                if answer_value and answer_value[0] is not None:  # Check list is not empty and first answer is not None
                                    record = create_record_template()
                                    record[global_config['jsonl_format']['text_field']] = text + " " + str(answer_value[0])
                                    records_to_write.append(record)
                                else:
                                    # Empty or None answer list - just use question
                                    record = create_record_template()
                                    record[global_config['jsonl_format']['text_field']] = text
                                    records_to_write.append(record)
                            else:
                                # Single answer - create record with question + answer
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
                        choices = get_nested_field(example, choices_field)
                        if choices is not None:

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

                    # Split long documents if enabled
                    text_chunks = split_document(text, DOCUMENT_SPLIT_THRESHOLD)

                    for chunk_idx, chunk_text in enumerate(text_chunks):
                        word_count = len(chunk_text.split())

                        # Skip records with fewer than 8 words
                        if word_count >= 8:
                            # Create a new record for this chunk
                            chunk_record = record.copy()
                            chunk_record[global_config['jsonl_format']['text_field']] = chunk_text

                            # Add unique document ID
                            chunk_record['doc_id'] = document_id_counter[0]
                            document_id_counter[0] += 1

                            # Add chunk info if document was split
                            if len(text_chunks) > 1:
                                chunk_record['chunk_idx'] = chunk_idx
                                chunk_record['total_chunks'] = len(text_chunks)

                            f.write(json.dumps(chunk_record) + '\n')

        if skipped_duplicates > 0:
            print(f"Saved {len(dataset[split]) - skipped_duplicates} examples to {output_file} (skipped {skipped_duplicates} duplicates)")
        else:
            print(f"Saved {len(dataset[split])} examples to {output_file}")

def download_all_evals():
    """Download all eval datasets"""
    print(f"Processing {len(EVAL_CONFIG['evals'])} eval datasets...")

    # Initialize global document ID counter (using list for mutable reference)
    document_id_counter = [1]

    # Process each eval dataset
    for eval_name, eval_config in EVAL_CONFIG['evals'].items():
        download_and_transform_eval(eval_name, eval_config, EVAL_CONFIG, document_id_counter)

    print(f"Done! Generated {document_id_counter[0] - 1} total document IDs.")


def list_evals():
    """List all available eval datasets"""
    print(f"Available eval datasets ({len(EVAL_CONFIG['evals'])} total):\n")

    for eval_name in sorted(EVAL_CONFIG['evals'].keys()):
        print(f"  - {eval_name}")


def main():
    """Main function with CLI"""
    parser = argparse.ArgumentParser(
        description="Manage HuggingFace eval datasets for contamination detection"
    )

    parser.add_argument(
        "--download",
        action="store_true",
        help="Download all evaluation datasets"
    )

    parser.add_argument(
        "--list",
        action="store_true",
        help="List all available evaluation datasets"
    )

    args = parser.parse_args()

    # If no arguments provided, show help
    if not any(vars(args).values()):
        parser.print_help()
        return

    if args.list:
        list_evals()
    elif args.download:
        download_all_evals()


if __name__ == "__main__":
    main()
