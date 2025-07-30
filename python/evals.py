#!/usr/bin/env python3
"""
Download and transform HuggingFace eval datasets for contamination detection.
"""

import argparse
import json
from datasets import load_dataset, Dataset
from pathlib import Path
import os
import shutil
import gzip

# Document splitting configuration - REMOVED (no longer needed without concatenation)
# We now store question, context, and answer separately

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
        line = json.dumps(record) + '\n'
        line_bytes = len(line.encode('utf-8'))

        # Check if we need to start a new chunk
        if self.current_chunk_size + line_bytes > self.CHUNK_SIZE_BYTES or self.writer is None:
            self._start_new_chunk()

        self.buffer.append(line)
        self.current_chunk_size += line_bytes
        self.total_records += 1

        # Flush buffer if it gets large
        if len(self.buffer) >= 1000:
            self._flush_buffer()

    def _start_new_chunk(self):
        """Start writing to a new chunk file."""
        # Close current chunk if exists
        if self.writer is not None:
            self._flush_buffer()
            self.writer.close()

        # Create new chunk file
        self.chunk_number += 1
        chunk_filename = f"{self.base_name}-{self.chunk_number}{self.extension}"
        chunk_path = self.output_dir / chunk_filename
        self.chunk_files.append(chunk_path)

        # Open new file
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

EVAL_CONFIG = {
    "output_dir": "fixtures/reference",
    "jsonl_format": {
        "eval_field": "eval_name",
        "index_field": "index",
        "split_field": "split"
    },
    "evals": {
        "agi_eval_aqua_rat": {
            "local_path": "fixtures/reference-static/aqua-rat.jsonl",
            "splits": [
                "train"
            ],
            "transform": {
                "text_field": "question",
                "context_field": "passage",
                "answer_field": "other.solution"
            }
        },
        "agi_eval_gaokao_english": {
            "local_path": "fixtures/reference-static/gaokao-english.jsonl",
            "splits": [
                "train"
            ],
            "transform": {
                "text_field": "question",
                "context_field": "passage",
                "answer_field": "label",
                "combine_context_and_question": True
            }
        },
        "agi_eval_logiqa_en": {
            "local_path": "fixtures/reference-static/logiqa-en.jsonl",
            "splits": [
                "train"
            ],
            "transform": {
                "text_field": "question",
                "context_field": "passage",
                "answer_field": "label",
                "choices_field": "options",
                "combine_context_and_question": True
            }
        },
        "agi_eval_lsat_ar": {
            "local_path": "fixtures/reference-static/lsat-ar.jsonl",
            "splits": [
                "train"
            ],
            "transform": {
                "text_field": "question",
                "context_field": "passage",
                "answer_field": "label",
                "choices_field": "options"
            }
        },
        "agi_eval_lsat_lr": {
            "local_path": "fixtures/reference-static/lsat-lr.jsonl",
            "splits": [
                "train"
            ],
            "transform": {
                "text_field": "question",
                "context_field": "passage",
                "answer_field": "label",
                "combine_context_and_question": True
            }
        },
        "agi_eval_lsat_rc": {
            "local_path": "fixtures/reference-static/lsat-rc.jsonl",
            "splits": [
                "train"
            ],
            "transform": {
                "text_field": "question",
                "context_field": "passage",
                "answer_field": "label",
                "combine_context_and_question": True
            }
        },
        "agi_eval_math": {
            "local_path": "fixtures/reference-static/math.jsonl",
            "splits": [
                "train"
            ],
            "transform": {
                "text_field": "question",
                "answer_field": "other.solution"
            }
        },
        "agi_eval_sat_en": {
            "local_path": "fixtures/reference-static/sat-en.jsonl",
            "splits": [
                "train"
            ],
            "transform": {
                "text_field": "question",
                "context_field": "passage",
                "answer_field": "other.solution"
            }
        },
        "agi_eval_sat_en_without_passage": {
            "local_path": "fixtures/reference-static/sat-en-without-passage.jsonl",
            "splits": [
                "train"
            ],
            "transform": {
                "text_field": "question",
                "answer_field": "other.solution"
            }
        },
        "agi_eval_sat_math": {
            "local_path": "fixtures/reference-static/sat-math.jsonl",
            "splits": [
                "train"
            ],
            "transform": {
                "text_field": "question",
                "context_field": "passage",
                "answer_field": "other.solution"
            }
        },
        "ai2_arc_challenge": {
            "hf_path": "allenai/ai2_arc",
            "hf_config": "ARC-Challenge",
            "splits": [
                "train",
                "test",
                "validation"
            ],
            "transform": {
                "text_field": "question",
                "answer_field": "choices.text",
                "answer_key_field": "answerKey",
                "choices_field": "choices",
                "answer_lookup_field": "choices.label"
            }
        },
        "ai2_arc_easy": {
            "hf_path": "allenai/ai2_arc",
            "hf_config": "ARC-Easy",
            "splits": [
                "train",
                "test",
                "validation"
            ],
            "transform": {
                "text_field": "question",
                "answer_field": "choices.text",
                "answer_key_field": "answerKey",
                "choices_field": "choices",
                "answer_lookup_field": "choices.label"
            }
        },
        "aime_2025_i": {
            "hf_path": "opencompass/AIME2025",
            "hf_config": "AIME2025-I",
            "splits": [
                "test"
            ],
            "transform": {
                "text_field": "question",
                "answer_field": "answer"
            }
        },
        "aime_2025_ii": {
            "hf_path": "opencompass/AIME2025",
            "hf_config": "AIME2025-II",
            "splits": [
                "test"
            ],
            "transform": {
                "text_field": "question",
                "answer_field": "answer"
            }
        },
        "alpaca": {
            "hf_path": "tatsu-lab/alpaca",
            "splits": [
                "eval"
            ],
            "transform": {
                "text_field": "instruction",
                "answer_field": "output"
            }

        },
        "alpaca-multiturn": {
            "hf_path": "VGraf/TurnWise",
            "splits": [
                "eval"
            ],
            "transform": {
                "text_field": "instruction",
                "answer_field": "output"
            }

        },
        "aimo_validation": {
            "hf_path": "AI-MO/aimo-validation-aime",
            "splits": [
                "train"
            ],
            "transform": {
                "text_field": "problem",
                "answer_field": "answer"
            }
        },
        "basic_skills_arithmetic": {
            "hf_path": "allenai/basic-skills",
            "hf_config": "arithmetic",
            "splits": [
                "validation"
            ],
            "transform": {
                "text_field": "question",
                "answer_field": "answer"
            }
        },
        "basic_skills_coding": {
            "hf_path": "allenai/basic-skills",
            "hf_config": "coding",
            "splits": [
                "validation"
            ],
            "transform": {
                "text_field": "question",
                "answer_field": "answer"
            }
        },
        "basic_skills_common_knowledge": {
            "hf_path": "allenai/basic-skills",
            "hf_config": "common_knowledge",
            "splits": [
                "validation"
            ],
            "transform": {
                "text_field": "question",
                "answer_field": "answer"
            }
        },
        "basic_skills_logical_reasoning": {
            "hf_path": "allenai/basic-skills",
            "hf_config": "logical_reasoning",
            "splits": [
                "validation"
            ],
            "transform": {
                "text_field": "question",
                "answer_field": "answer"
            }
        },
        "basic_skills_pattern": {
            "hf_path": "allenai/basic-skills",
            "hf_config": "pattern",
            "splits": [
                "validation"
            ],
            "transform": {
                "text_field": "question",
                "answer_field": "answer"
            }
        },
        "basic_skills_string_operations": {
            "hf_path": "allenai/basic-skills",
            "hf_config": "string_operations",
            "splits": [
                "validation"
            ],
            "transform": {
                "text_field": "question",
                "answer_field": "answer"
            }
        },
        "bbeh": {
            "hf_path": "hubert233/BigBenchExtraHard",
            "splits": [
                "boardgame_qa",
                "boolean_expressions",
                "buggy_tables",
                "causal_understanding",
                "disambiguation_qa",
                "dyck_languages",
                "geometric_shapes",
                "hyperbaton",
                "linguini",
                "movie_recommendation",
                "multistep_arithmetic",
                "nycc",
                "object_counting",
                "object_properties",
                "sarc_triples",
                "shuffled_objects",
                "spatial_reasoning",
                "sportqa",
                "temporal_sequence",
                "time_arithmetic",
                "web_of_lies",
                "word_sorting",
                "zebra_puzzles"
            ],
            "transform": {
                "text_field": "input",
                "answer_field": "target"
            }
        },
        # ['boolean_expressions', 'causal_judgement', 'date_understanding', 'disambiguation_qa', 'dyck_languages', 'formal_fallacies', 'geometric_shapes', 'hyperbaton', 'logical_deduction_five_objects', 'logical_deduction_seven_objects', 'logical_deduction_three_objects', 'movie_recommendation', 'multistep_arithmetic_two', 'navigate', 'object_counting', 'penguins_in_a_table', 'reasoning_about_colored_objects', 'ruin_names', 'salient_translation_error_detection', 'snarks', 'sports_understanding', 'temporal_sequences', 'tracking_shuffled_objects_five_objects', 'tracking_shuffled_objects_seven_objects', 'tracking_shuffled_objects_three_objects', 'web_of_lies', 'word_sorting']
        "bbh_boolean_expressions": {
            "hf_path": "lukaemon/bbh",
            "hf_config": "boolean_expressions",
            "splits": [
                "test"
            ],
            "transform": {
                "text_field": "input",
                "answer_field": "target"
            }
        },
        "bbh_causal_judgement": {
            "hf_path": "lukaemon/bbh",
            "hf_config": "causal_judgement",
            "splits": [
                "test"
            ],
            "transform": {
                "text_field": "input",
                "answer_field": "target"
            }
        },
        "bbh_date_understanding": {
            "hf_path": "lukaemon/bbh",
            "hf_config": "date_understanding",
            "splits": [
                "test"
            ],
            "transform": {
                "text_field": "input",
                "answer_field": "target"
            }
        },
        "bbh_disambiguation_qa": {
            "hf_path": "lukaemon/bbh",
            "hf_config": "disambiguation_qa",
            "splits": [
                "test"
            ],
            "transform": {
                "text_field": "input",
                "answer_field": "target"
            }
        },
        "bbh_formal_fallacies": {
            "hf_path": "lukaemon/bbh",
            "hf_config": "formal_fallacies",
            "splits": [
                "test"
            ],
            "transform": {
                "text_field": "input",
                "answer_field": "target"
            }
        },
        "bbh_logical_deduction_three_objects": {
            "hf_path": "lukaemon/bbh",
            "hf_config": "logical_deduction_three_objects",
            "splits": [
                "test"
            ],
            "transform": {
                "text_field": "input",
                "answer_field": "target"
            }
        },
        "bigcodebench": {
            "hf_path": "bigcode/bigcodebench",
            "splits": [
                "v0.1.4"
            ],
            "transform": {
                "text_field": "complete_prompt",
                "answer_field": "canonical_solution"
            }
        },
        "bigcodebench_hard": {
            "hf_path": "bigcode/bigcodebench-hard",
            "splits": [
                "v0.1.4"
            ],
            "transform": {
                "text_field": "complete_prompt",
                "answer_field": "canonical_solution"
            }
        },
        "codeforces": {
            "hf_path": "open-r1/codeforces",
            "splits": [
                "test"
            ],
            "transform": {
                "text_field": "description",
                "context_field": "title",
                "extra_fields": [
                    "contest_name",
                    "rating",
                    "tags"
                ]
            },
            "no_answer_splits": [
                "test"
            ]
        },
        "commonsense_qa": {  # csqa
            "hf_path": "commonsense_qa",
            "splits": [
                "train",
                "validation",
                "test"
            ],
            "transform": {
                "text_field": "question",
                "answer_field": "answerKey"
            },
            "no_answer_splits": [
                "test"
            ]
        },
        # "copycolors_mcqa": {
        #     "hf_path": "sarahwie/copycolors_mcqa",
        #     "hf_config": "4_answer_choices",
        #     "splits": [
        #         "validation",
        #         "test"
        #     ],
        #     "transform": {
        #         "text_field": "question",
        #         "answer_field": "choices.text",
        #         "answer_key_field": "answerKey"
        #     },
        #     "no_answer_splits": [
        #         "test"
        #     ]
        # },
        "coqa": {
            "hf_path": "EleutherAI/coqa",
            "splits": [
                "train",
                "validation"
            ],
            "transform": {
                "context_field": "story",
                "parallel_arrays": {
                    "question_array": "questions.input_text",
                    "answer_array": "answers.input_text"
                }
            }
        },
        "coqa_mc": {
            "hf_path": "allenai/coqa_mc",
            "splits": [
                "validation"
            ],
            "transform": {
                "text_field": "query_original",
                "answer_field": "choices_original"
            }
        },
        # "cosmos_qa": {
        #     "hf_path": "allenai/cosmos_qa",
        #     "splits": [
        #         "train",
        #         "test",
        #         "validation"
        #     ],
        #     "transform": {
        #         "text_field": "question",
        #         "context_field": "context",
        #         "answer_field": "answer", # This is now ignored when answer_prefix is set
        #         "answer_key_field": "label",
        #         "answer_prefix": "answer"
        #     },
        #     "no_answer_splits": [
        #         "test"
        #     ]
        # },
        "cruxeval": {
            "hf_path": "cruxeval-org/cruxeval",
            "splits": [
                "test"
            ],
            "transform": {
                "text_field": "code",
                "answer_field": "output"
            }
        },
        "deepseek_leetcode": {  # David kind of anticipates some contamination here!
            "hf_path": "davidheineman/deepseek-leetcode",
            "splits": [
                "test"
            ],
            "transform": {
                "text_field": "prompt"
            },
            "no_answer_splits": [
                "test"
            ]
        },
        "drop": {  # TODO: REVIEW for a good candidate for adding passage support at a more tolerant overlap ratio.
            "hf_path": "EleutherAI/drop",
            "splits": [
                "train",
                "validation"
            ],
            "transform": {
                "text_field": "question",
                "context_field": "passage",
                "answer_field": "answer.spans.0"
            }
        },
        "drop_mc": {  # TODO: REVIEW for a good candidate for adding passage support at a more tolerant overlap ratio.
            "hf_path": "allenai/drop_mc",
            "splits": [
                "validation"
            ],
            "transform": {
                "text_field": "question_original",
                "context_field": "passage_original",
                "answer_field": "answer_original.spans.0"
            }
        },
        "ds_1000": {
            "hf_path": "xlangai/DS-1000",
            "splits": [
                "test"
            ],
            "transform": {
                "text_field": "prompt",
                "answer_field": "reference_code"
            }
        },
        "gpqa_main": {
            "hf_path": "Idavidrein/gpqa",
            "hf_config": "gpqa_main",
            "splits": [
                "train",
            ],
            "transform": {
                "text_field": "Question",
                "answer_field": "Correct Answer"
            }
        },
        "gpqa_diamond": {
            "hf_path": "Idavidrein/gpqa",
            "hf_config": "gpqa_diamond",
            "splits": [
                "train",
            ],
            "transform": {
                "text_field": "Question",
                "answer_field": "Correct Answer"
            }
        },
        "gpqa_experts": {
            "hf_path": "Idavidrein/gpqa",
            "hf_config": "gpqa_experts",
            "splits": [
                "train",
            ],
            "transform": {
                "text_field": "Question",
                "answer_field": "Correct Answer"
            }
        },
        "gpqa_extended": {
            "hf_path": "Idavidrein/gpqa",
            "hf_config": "gpqa_extended",
            "splits": [
                "train",
            ],
            "transform": {
                "text_field": "Question",
                "answer_field": "Correct Answer"
            }
        },
        "gsm8k": {
            "hf_path": "openai/gsm8k",
            "hf_config": "main",
            "splits": [
                "train",
                "test"
            ],
            "transform": {
                "text_field": "question",
                "answer_field": "answer"
            }
        },
        "gsm_plus": {
            "hf_path": "qintongli/GSM-Plus",
            "splits": [
                "test",
                "testmini"
            ],
            "transform": {
                "text_field": "question",
                "answer_field": "solution",
                "extra_fields": [
                    "perturbation_type"
                ]
            }
        },
        "gsm_symbolic": {
            "hf_path": "apple/GSM-Symbolic",
            "hf_config": "main",
            "splits": [
                "test"
            ],
            "transform": {
                "text_field": "question",
                "answer_field": "answer"
            }
        },
        "gsm_symbolic": {
            "hf_path": "apple/GSM-Symbolic",
            "hf_config": "p1",
            "splits": [
                "test"
            ],
            "transform": {
                "text_field": "question",
                "answer_field": "answer"
            }
        },
        "gsm_symbolic": {
            "hf_path": "apple/GSM-Symbolic",
            "hf_config": "p2",
            "splits": [
                "test"
            ],
            "transform": {
                "text_field": "question",
                "answer_field": "answer"
            }
        },
        "harmbench_contextual": {
            "hf_path": "walledai/HarmBench",
            "hf_config": "contextual",
            "splits": [
                "train"
            ],
            "transform": {
                "text_field": "prompt",
                "context_field": "context",
                "extra_fields": [
                    "category"
                ]
            },
            "no_answer_splits": [
                "train"
            ]
        },
        "harmbench_copyright": {
            "hf_path": "walledai/HarmBench",
            "hf_config": "copyright",
            "splits": [
                "train"
            ],
            "transform": {
                "text_field": "prompt",
                "extra_fields": [
                    "tags"
                ]
            },
            "no_answer_splits": [
                "train"
            ]
        },
        "harmbench_standard": {
            "hf_path": "walledai/HarmBench",
            "hf_config": "standard",
            "splits": [
                "train"
            ],
            "transform": {
                "text_field": "prompt",
                "extra_fields": [
                    "category"
                ]
            },
            "no_answer_splits": [
                "train"
            ]
        },
        "hellaswag": {
            "hf_path": "hellaswag",
            "splits": [
                "train",
                "test",
                "validation"
            ],
            "transform": {
                "text_field": "ctx",
                "answer_field": "label"
            },
            "no_answer_splits": [
                "test"
            ]
        },
        "hendrycks_math_algebra": {
            "hf_path": "EleutherAI/hendrycks_math",
            "hf_config": "algebra",
            "splits": [
                "train",
                "test"
            ],
            "transform": {
                "text_field": "problem",
                "answer_field": "solution"
            }
        },
        "hendrycks_math_counting_and_probability": {
            "hf_path": "EleutherAI/hendrycks_math",
            "hf_config": "counting_and_probability",
            "splits": [
                "train",
                "test"
            ],
            "transform": {
                "text_field": "problem",
                "answer_field": "solution"
            }
        },
        "hendrycks_math_geometry": {
            "hf_path": "EleutherAI/hendrycks_math",
            "hf_config": "geometry",
            "splits": [
                "train",
                "test"
            ],
            "transform": {
                "text_field": "problem",
                "answer_field": "solution"
            }
        },
        "hendrycks_math_intermediate_algebra": {
            "hf_path": "EleutherAI/hendrycks_math",
            "hf_config": "intermediate_algebra",
            "splits": [
                "train",
                "test"
            ],
            "transform": {
                "text_field": "problem",
                "answer_field": "solution"
            }
        },
        "hendrycks_math_number_theory": {
            "hf_path": "EleutherAI/hendrycks_math",
            "hf_config": "number_theory",
            "splits": [
                "train",
                "test"
            ],
            "transform": {
                "text_field": "problem",
                "answer_field": "solution"
            }
        },
        "hendrycks_math_prealgebra": {
            "hf_path": "EleutherAI/hendrycks_math",
            "hf_config": "prealgebra",
            "splits": [
                "train",
                "test"
            ],
            "transform": {
                "text_field": "problem",
                "answer_field": "solution"
            }
        },
        "hendrycks_math_precalculus": {
            "hf_path": "EleutherAI/hendrycks_math",
            "hf_config": "precalculus",
            "splits": [
                "train",
                "test"
            ],
            "transform": {
                "text_field": "problem",
                "answer_field": "solution"
            }
        },
        "humaneval_infilling_multiline": {
            "hf_path": "loubnabnl/humaneval_infilling",
            "hf_config": "HumanEval-MultiLineInfilling",
            "splits": [
                "test"
            ],
            "transform": {
                "text_field": "prompt",
                "answer_field": "canonical_solution"
            }
        },
        "humaneval_infilling_randomspan": {
            "hf_path": "loubnabnl/humaneval_infilling",
            "hf_config": "HumanEval-RandomSpanInfilling",
            "splits": [
                "test"
            ],
            "transform": {
                "text_field": "prompt",
                "answer_field": "canonical_solution"
            }
        },
        "humaneval_infilling_singleline": {
            "hf_path": "loubnabnl/humaneval_infilling",
            "hf_config": "HumanEval-SingleLineInfilling",
            "splits": [
                "test"
            ],
            "transform": {
                "text_field": "prompt",
                "answer_field": "canonical_solution"
            }
        },
        "humaneval_plus": {
            "hf_path": "evalplus/humanevalplus",
            "splits": [
                "test"
            ],
            "transform": {
                "text_field": "prompt",
                "answer_field": "canonical_solution"
            }
        },
        "humaneval_pro": {
            "hf_path": "CodeEval-Pro/humaneval-pro",
            "splits": [
                "train"
            ],
            "transform": {
                "text_field": "new_problem",
                "answer_field": "new_solution"
            }
        },
        "ifbench_multiturn_constraints": {  # TODO: This a good example for multi-turn complexity.
            "hf_path": "allenai/IFBench_multi-turn",
            "hf_config": "ifbench_constraints",
            "splits": [
                "test"
            ],
            "transform": {
                "text_field": "prompt",
                # "answer_field": "messages.1.content"  # Davidh suggests that we exclude "generated responses as answert"
            }
        },
        "ifbench_multiturn_ifeval": {
            "hf_path": "allenai/IFBench_multi-turn",
            "hf_config": "ifeval_constraints",
            "splits": [
                "test"
            ],
            "transform": {
                "text_field": "prompt"
            },
            "no_answer_splits": [
                "test"
            ]
        },
        "ifeval": {  # note all if evals do not have answer
            "hf_path": "HuggingFaceH4/ifeval",
            "splits": [
                "train"
            ],
            "transform": {
                "text_field": "prompt"
            },
            "no_answer_splits": [
                "train"
            ]
        },
        "if_ood": {
            "hf_path": "valpy/ifeval_ood3",
            "splits": [
                "train"
            ],
            "transform": {
                "text_field": "prompt"
            },
            "no_answer_splits": [
                "train"
            ]
        },
        "jeopardy": {
            "hf_path": "soldni/jeopardy",
            "hf_config": "all_questions",
            "splits": [
                "train"
            ],
            "transform": {
                "text_field": "question",
                "answer_field": "continuation"
            }
        },
        "jeopardy_mc": {
            "hf_path": "allenai/jeopardy_mc",
            "splits": [
                "test"
            ],
            "transform": {
                "text_field": "context_original",
                "answer_field": "continuation_original"
            }
        },
        "lab_bench_dbqa": {
            "hf_path": "futurehouse/lab-bench",
            "hf_config": "DbQA",
            "splits": [
                "train"
            ],
            "transform": {
                "text_field": "question",
                "answer_field": "ideal",
                "choices_field": "distractors"
            }
        },
        "lab_bench_protocolqa": {
            "hf_path": "futurehouse/lab-bench",
            "hf_config": "ProtocolQA",
            "splits": [
                "train"
            ],
            "transform": {
                "text_field": "question",
                "answer_field": "ideal",
                "choices_field": "distractors"
            }
        },
        "lambada_openai": {
            "hf_path": "EleutherAI/lambada_openai",
            "splits": [
                "test"
            ],
            "transform": {
                "text_field": "text"
            },
            "no_answer_splits": [
                "test"
            ]
        },
        "lbpp": {
            "hf_path": "CohereLabs/lbpp",
            "splits": [
                "test"
            ],
            "transform": {
                "text_field": "instruction",
                "answer_field": "completion"
            }
        },
        #  ['mcq_4_choices', 'mcq_perturbation', 'open_question']
        # "lexam_mcq": {
        #     "hf_path": "LEXam-Benchmark/LEXam",
        #     "hf_config": "mcq_4_choices",
        #     "splits": [
        #         "test"
        #     ],
        #     "transform": {
        #         "text_field": "question",
        #         "answer_field": "choices"
        #     }
        # },
        "livecodebench": {
            "hf_path": "livecodebench/code_generation_lite",
            "splits": [
                "test"
            ],
            "transform": {
                "text_field": "question_content"
            },
            "no_answer_splits": [
                "test"
            ]
        },
        "logiqa": {
            "hf_path": "lucasmccabe/logiqa",
            "splits": [
                "train",
                "validation",
                "test"
            ],
            "transform": {
                "text_field": "query",
                "context_field": "context",
                "answer_field": "correct_option"
            }
        },
        "math_500": {
            "hf_path": "HuggingFaceH4/MATH-500",
            "splits": [
                "test"
            ],
            "transform": {
                "text_field": "problem",
                "answer_field": "solution"
            }
        },
        "mbpp": {
            "hf_path": "google-research-datasets/mbpp",
            "splits": [
                "train",
                "test",
                "validation"
            ],
            "transform": {
                "text_field": "text",
                "answer_field": "code"
            }
        },
        "mbpp_plus": {
            "hf_path": "evalplus/mbppplus",
            "splits": [
                "test"
            ],
            "transform": {
                "text_field": "prompt",
                "answer_field": "code"
            }
        },
        "mbpp_pro": {
            "hf_path": "CodeEval-Pro/mbpp-pro",
            "splits": [
                "train"
            ],
            "transform": {
                "text_field": "new_problem",
                "answer_field": "new_solution"
            }
        },
        "medmcqa": {
            "hf_path": "openlifescienceai/medmcqa",
            "splits": [
                "train",
                "test",
                "validation"
            ],
            "transform": {
                "text_field": "question",
                "answer_field": "exp"
            },
            "no_answer_splits": [
                "test"
            ]
        },
        "medqa_en": {
            "hf_path": "davidheineman/medqa-en",
            "splits": [
                "train",
                "test",
                "dev"
            ],
            "transform": {
                "text_field": "question",
                "answer_field": "answer"
            }
        },
        "mmlu": {  # 57 subsets, that each have a train/test/validation split, prefer that. OMIT ALL. maybe double check one day. TODO
            "hf_path": "cais/mmlu",
            "hf_config": "all",
            "splits": [
                "test",
                "validation",
                "dev"
            ],
            "transform": {
                "text_field": "question",
                "answer_field": "choices",
                "extra_fields": [
                    "subject"
                ]
            }
        },
        "mmlu_pro": {
            "hf_path": "TIGER-Lab/MMLU-Pro",
            "splits": [
                "test",
                "validation"
            ],
            "transform": {
                "text_field": "question",
                "answer_field": "options",
                "answer_key_field": "answer_index",
                "choices_field": "options",
                "extra_fields": [
                    "category"
                ]
            }
        },
        "mt_eval_refinement": {
            "hf_path": "wckwan/MT-Eval",
            "hf_config": "refinement_single",
            "splits": [
                "test"
            ],
            "transform": {
                "text_field": "conv.0.user"
            },
            "no_answer_splits": [
                "test"
            ]
        },
        "multilingual_mbpp_cpp": {
            "hf_path": "allenai/multilingual_mbpp",
            "hf_config": "cpp",
            "splits": [
                "test",
                "train",
                "validation"
            ],
            "transform": {
                "text_field": "text",
                "answer_field": "code"
            }
        },
        "multilingual_mbpp_c": {
            "hf_path": "allenai/multilingual_mbpp",
            "hf_config": "c",
            "splits": [
                "test",
                "train",
                "validation"
            ],
            "transform": {
                "text_field": "text",
                "answer_field": "code"
            }
        },
        "multilingual_mbpp_javascript": {
            "hf_path": "allenai/multilingual_mbpp",
            "hf_config": "javascript",
            "splits": [
                "test",
                "train",
                "validation"
            ],
            "transform": {
                "text_field": "text",
                "answer_field": "code"
            }
        },
        "multilingual_mbpp_java": {
            "hf_path": "allenai/multilingual_mbpp",
            "hf_config": "java",
            "splits": [
                "test",
                "train",
                "validation"
            ],
            "transform": {
                "text_field": "text",
                "answer_field": "code"
            }
        },
        "multilingual_mbpp_python": {
            "hf_path": "allenai/multilingual_mbpp",
            "hf_config": "python",
            "splits": [
                "test",
                "train",
                "validation"
            ],
            "transform": {
                "text_field": "text",
                "answer_field": "code"
            }
        },
        "multilingual_mbpp_php": {
            "hf_path": "allenai/multilingual_mbpp",
            "hf_config": "php",
            "splits": [
                "test",
                "train",
                "validation"
            ],
            "transform": {
                "text_field": "text",
                "answer_field": "code"
            }
        },
        "multilingual_mbpp_csharp": {
            "hf_path": "allenai/multilingual_mbpp",
            "hf_config": "csharp",
            "splits": [
                "test",
                "train",
                "validation"
            ],
            "transform": {
                "text_field": "text",
                "answer_field": "code"
            }
        },
        "multilingual_mbpp_typescript": {
            "hf_path": "allenai/multilingual_mbpp",
            "hf_config": "typescript",
            "splits": [
                "test",
                "train",
                "validation"
            ],
            "transform": {
                "text_field": "text",
                "answer_field": "code"
            }
        },
        "multilingual_mbpp_bash": {
            "hf_path": "allenai/multilingual_mbpp",
            "hf_config": "bash",
            "splits": [
                "test",
                "train",
                "validation"
            ],
            "transform": {
                "text_field": "text",
                "answer_field": "code"
            }
        },
        "multilingual_mbpp_swift": {
            "hf_path": "allenai/multilingual_mbpp",
            "hf_config": "swift",
            "splits": [
                "test",
                "train",
                "validation"
            ],
            "transform": {
                "text_field": "text",
                "answer_field": "code"
            }
        },
        "multilingual_mbpp_go": {
            "hf_path": "allenai/multilingual_mbpp",
            "hf_config": "go",
            "splits": [
                "test",
                "train",
                "validation"
            ],
            "transform": {
                "text_field": "text",
                "answer_field": "code"
            }
        },
        "multilingual_mbpp_rust": {
            "hf_path": "allenai/multilingual_mbpp",
            "hf_config": "rust",
            "splits": [
                "test",
                "train",
                "validation"
            ],
            "transform": {
                "text_field": "text",
                "answer_field": "code"
            }
        },
        "multilingual_mbpp_ruby": {
            "hf_path": "allenai/multilingual_mbpp",
            "hf_config": "ruby",
            "splits": [
                "test",
                "train",
                "validation"
            ],
            "transform": {
                "text_field": "text",
                "answer_field": "code"
            }
        },
        "multilingual_mbpp_r": {
            "hf_path": "allenai/multilingual_mbpp",
            "hf_config": "r",
            "splits": [
                "test",
                "train",
                "validation"
            ],
            "transform": {
                "text_field": "text",
                "answer_field": "code"
            }
        },
        "multilingual_mbpp_matlab": {
            "hf_path": "allenai/multilingual_mbpp",
            "hf_config": "matlab",
            "splits": [
                "test",
                "train",
                "validation"
            ],
            "transform": {
                "text_field": "text",
                "answer_field": "code"
            }
        },
        "multilingual_mbpp_scala": {
            "hf_path": "allenai/multilingual_mbpp",
            "hf_config": "scala",
            "splits": [
                "test",
                "train",
                "validation"
            ],
            "transform": {
                "text_field": "text",
                "answer_field": "code"
            }
        },
        "multilingual_mbpp_haskell": {
            "hf_path": "allenai/multilingual_mbpp",
            "hf_config": "haskell",
            "splits": [
                "test",
                "train",
                "validation"
            ],
            "transform": {
                "text_field": "text",
                "answer_field": "code"
            }
        },

        "multipl_e_humaneval_cpp": {
            "hf_path": "nuprl/MultiPL-E",
            "hf_config": "humaneval-cpp",
            "splits": [
                "test"
            ],
            "transform": {
                "text_field": "prompt"
            },
            "no_answer_splits": [
                "test"
            ]
        },
        "multipl_e_humaneval_go": {
            "hf_path": "nuprl/MultiPL-E",
            "hf_config": "humaneval-go",
            "splits": [
                "test"
            ],
            "transform": {
                "text_field": "prompt"
            },
            "no_answer_splits": [
                "test"
            ]
        },
        "multipl_e_humaneval_java": {
            "hf_path": "nuprl/MultiPL-E",
            "hf_config": "humaneval-java",
            "splits": [
                "test"
            ],
            "transform": {
                "text_field": "prompt"
            },
            "no_answer_splits": [
                "test"
            ]
        },
        "multipl_e_humaneval_js": {
            "hf_path": "nuprl/MultiPL-E",
            "hf_config": "humaneval-js",
            "splits": [
                "test"
            ],
            "transform": {
                "text_field": "prompt"
            },
            "no_answer_splits": [
                "test"
            ]
        },
        "multipl_e_humaneval_python": {
            "hf_path": "nuprl/MultiPL-E",
            "hf_config": "humaneval-python",
            "splits": [
                "test"
            ],
            "transform": {
                "text_field": "prompt"
            }
        },
        # 'mrcr': {
        #     'hf_path': 'openai/mrcr',
        #     'splits': ['train'],
        #     'transform': {
        #         'text_field': 'prompt'
        #     }
        # },


        # TODO: multipl_e_mbpp_python has a ton of configs .... add them.
        "multipl_e_mbpp_python": {
            "hf_path": "nuprl/MultiPL-E",
            "hf_config": "mbpp-python",
            "splits": [
                "test"
            ],
            "transform": {
                "text_field": "prompt"
            }
        },
        "nq_open": {
            "hf_path": "google-research-datasets/nq_open",
            "splits": [
                "train",
                "validation"
            ],
            "transform": {
                "text_field": "question",
                "answer_field": "answer"
            }
        },
        "nq_open_mc": {
            "hf_path": "allenai/nq_open_mc",
            "splits": [
                "validation"
            ],
            "transform": {
                "text_field": "question",
                "answer_field": "answer_original"
            }
        },
        "omega_compositional_polynomial_gcd": {
            "hf_path": "allenai/omega-compositional",
            "hf_config": "comp_polynomial_gcd",
            "splits": [
                "train",
                "test"
            ],
            "transform": {
                "text_field": "messages.0.content",
                "answer_field": "ground_truth",
                "extra_fields": [
                    "setting_name",
                    "dataset"
                ]
            }
        },
        "omega_compositional_n_gon": {
            "hf_path": "allenai/omega-compositional",
            "hf_config": "comp_n_gon",
            "splits": [
                "train",
                "test"
            ],
            "transform": {
                "text_field": "messages.0.content",
                "answer_field": "ground_truth",
                "extra_fields": [
                    "setting_name",
                    "dataset"
                ]
            }
        },
        "omega_compositional_circles_algebra": {
            "hf_path": "allenai/omega-compositional",
            "hf_config": "comp_circles_algebra",
            "splits": [
                "train",
                "test"
            ],
            "transform": {
                "text_field": "messages.0.content",
                "answer_field": "ground_truth",
                "extra_fields": [
                    "setting_name",
                    "dataset"
                ]
            }
        },
        "omega_compositional_parametric_intersection": {
            "hf_path": "allenai/omega-compositional",
            "hf_config": "comp_parametric_intersection",
            "splits": [
                "train",
                "test"
            ],
            "transform": {
                "text_field": "messages.0.content",
                "answer_field": "ground_truth",
                "extra_fields": [
                    "setting_name",
                    "dataset"
                ]
            }
        },
        "omega_compositional_matrix_rank": {
            "hf_path": "allenai/omega-compositional",
            "hf_config": "comp_matrix_rank",
            "splits": [
                "train",
                "test"
            ],
            "transform": {
                "text_field": "messages.0.content",
                "answer_field": "ground_truth",
                "extra_fields": [
                    "setting_name",
                    "dataset"
                ]
            }
        },
        "omega_compositional_vertex_color": {
            "hf_path": "allenai/omega-compositional",
            "hf_config": "comp_vertex_color",
            "splits": [
                "train",
                "test"
            ],
            "transform": {
                "text_field": "messages.0.content",
                "answer_field": "ground_truth",
                "extra_fields": [
                    "setting_name",
                    "dataset"
                ]
            }
        },
        "omega_compositional_grid_chips": {
            "hf_path": "allenai/omega-compositional",
            "hf_config": "comp_grid_chips",
            "splits": [
                "train",
                "test"
            ],
            "transform": {
                "text_field": "messages.0.content",
                "answer_field": "ground_truth",
                "extra_fields": [
                    "setting_name",
                    "dataset"
                ]
            }
        },

        "omega_explorative_algebra_func_area": {
            "hf_path": "allenai/omega-explorative",
            "hf_config": "algebra_func_area",
            "splits": [
                "train",
                "test_in",
                "test_out"
            ],
            "transform": {
                "text_field": "messages.0.content",
                "answer_field": "ground_truth",
                "extra_fields": [
                    "setting_name",
                    "dataset"
                ]
            }
        },
        "omega_explorative_algebra_func_derivative_sign": {
            "hf_path": "allenai/omega-explorative",
            "hf_config": "algebra_func_derivative_sign",
            "splits": [
                "train",
                "test_in",
                "test_out"
            ],
            "transform": {
                "text_field": "messages.0.content",
                "answer_field": "ground_truth",
                "extra_fields": [
                    "setting_name",
                    "dataset"
                ]
            }
        },
        "omega_explorative_algebra_func_extrema": {
            "hf_path": "allenai/omega-explorative",
            "hf_config": "algebra_func_extrema",
            "splits": [
                "train",
                "test_in",
                "test_out"
            ],
            "transform": {
                "text_field": "messages.0.content",
                "answer_field": "ground_truth",
                "extra_fields": [
                    "setting_name",
                    "dataset"
                ]
            }
        },
        "omega_explorative_algebra_func_extrema_coords": {
            "hf_path": "allenai/omega-explorative",
            "hf_config": "algebra_func_extrema_coords",
            "splits": [
                "train",
                "test_in",
                "test_out"
            ],
            "transform": {
                "text_field": "messages.0.content",
                "answer_field": "ground_truth",
                "extra_fields": [
                    "setting_name",
                    "dataset"
                ]
            }
        },
        "omega_explorative_algebra_func_intersection": {
            "hf_path": "allenai/omega-explorative",
            "hf_config": "algebra_func_intersection",
            "splits": [
                "train",
                "test_in",
                "test_out"
            ],
            "transform": {
                "text_field": "messages.0.content",
                "answer_field": "ground_truth",
                "extra_fields": [
                    "setting_name",
                    "dataset"
                ]
            }
        },
        "omega_explorative_algebra_func_intersection_coords": {
            "hf_path": "allenai/omega-explorative",
            "hf_config": "algebra_func_intersection_coords",
            "splits": [
                "train",
                "test_in",
                "test_out"
            ],
            "transform": {
                "text_field": "messages.0.content",
                "answer_field": "ground_truth",
                "extra_fields": [
                    "setting_name",
                    "dataset"
                ]
            }
        },
        "omega_explorative_algebra_func_zeros": {
            "hf_path": "allenai/omega-explorative",
            "hf_config": "algebra_func_zeros",
            "splits": [
                "train",
                "test_in",
                "test_out"
            ],
            "transform": {
                "text_field": "messages.0.content",
                "answer_field": "ground_truth",
                "extra_fields": [
                    "setting_name",
                    "dataset"
                ]
            }
        },
        "omega_explorative_algebra_linear_equation": {
            "hf_path": "allenai/omega-explorative",
            "hf_config": "algebra_linear_equation",
            "splits": [
                "train",
                "test_in",
                "test_out"
            ],
            "transform": {
                "text_field": "messages.0.content",
                "answer_field": "ground_truth",
                "extra_fields": [
                    "setting_name",
                    "dataset"
                ]
            }
        },
        "omega_explorative_algebra_polynomial_roots": {
            "hf_path": "allenai/omega-explorative",
            "hf_config": "algebra_polynomial_roots",
            "splits": [
                "train",
                "test_in",
                "test_out"
            ],
            "transform": {
                "text_field": "messages.0.content",
                "answer_field": "ground_truth",
                "extra_fields": [
                    "setting_name",
                    "dataset"
                ]
            }
        },
        "omega_explorative_arithmetic_gcd": {
            "hf_path": "allenai/omega-explorative",
            "hf_config": "arithmetic_gcd",
            "splits": [
                "train",
                "test_in",
                "test_out"
            ],
            "transform": {
                "text_field": "messages.0.content",
                "answer_field": "ground_truth",
                "extra_fields": [
                    "setting_name",
                    "dataset"
                ]
            }
        },
        "omega_explorative_arithmetic_list_prime_factors": {
            "hf_path": "allenai/omega-explorative",
            "hf_config": "arithmetic_list_prime_factors",
            "splits": [
                "train",
                "test_in",
                "test_out"
            ],
            "transform": {
                "text_field": "messages.0.content",
                "answer_field": "ground_truth",
                "extra_fields": [
                    "setting_name",
                    "dataset"
                ]
            }
        },
        "omega_explorative_arithmetic_mixed": {
            "hf_path": "allenai/omega-explorative",
            "hf_config": "arithmetic_mixed",
            "splits": [
                "train",
                "test_in",
                "test_out"
            ],
            "transform": {
                "text_field": "messages.0.content",
                "answer_field": "ground_truth",
                "extra_fields": [
                    "setting_name",
                    "dataset"
                ]
            }
        },
        "omega_explorative_arithmetic_matrix_determinant": {
            "hf_path": "allenai/omega-explorative",
            "hf_config": "arithmetic_matrix_determinant",
            "splits": [
                "train",
                "test_in",
                "test_out"
            ],
            "transform": {
                "text_field": "messages.0.content",
                "answer_field": "ground_truth",
                "extra_fields": [
                    "setting_name",
                    "dataset"
                ]
            }
        },
        "omega_explorative_arithmetic_matrix_eigenvalues": {
            "hf_path": "allenai/omega-explorative",
            "hf_config": "arithmetic_matrix_eigenvalues",
            "splits": [
                "train",
                "test_in",
                "test_out"
            ],
            "transform": {
                "text_field": "messages.0.content",
                "answer_field": "ground_truth",
                "extra_fields": [
                    "setting_name",
                    "dataset"
                ]
            }
        },
        "omega_explorative_arithmetic_matrix_inverse": {
            "hf_path": "allenai/omega-explorative",
            "hf_config": "arithmetic_matrix_inverse",
            "splits": [
                "train",
                "test_in",
                "test_out"
            ],
            "transform": {
                "text_field": "messages.0.content",
                "answer_field": "ground_truth",
                "extra_fields": [
                    "setting_name",
                    "dataset"
                ]
            }
        },
        "omega_explorative_arithmetic_matrix_multiplication": {
            "hf_path": "allenai/omega-explorative",
            "hf_config": "arithmetic_matrix_multiplication",
            "splits": [
                "train",
                "test_in",
                "test_out"
            ],
            "transform": {
                "text_field": "messages.0.content",
                "answer_field": "ground_truth",
                "extra_fields": [
                    "setting_name",
                    "dataset"
                ]
            }
        },
        "omega_explorative_arithmetic_matrix_power": {
            "hf_path": "allenai/omega-explorative",
            "hf_config": "arithmetic_matrix_power",
            "splits": [
                "train",
                "test_in",
                "test_out"
            ],
            "transform": {
                "text_field": "messages.0.content",
                "answer_field": "ground_truth",
                "extra_fields": [
                    "setting_name",
                    "dataset"
                ]
            }
        },
        "omega_explorative_arithmetic_matrix_rank": {
            "hf_path": "allenai/omega-explorative",
            "hf_config": "arithmetic_matrix_rank",
            "splits": [
                "train",
                "test_in",
                "test_out"
            ],
            "transform": {
                "text_field": "messages.0.content",
                "answer_field": "ground_truth",
                "extra_fields": [
                    "setting_name",
                    "dataset"
                ]
            }
        },
        "omega_explorative_arithmetic_matrix_svd": {
            "hf_path": "allenai/omega-explorative",
            "hf_config": "arithmetic_matrix_svd",
            "splits": [
                "train",
                "test_in",
                "test_out"
            ],
            "transform": {
                "text_field": "messages.0.content",
                "answer_field": "ground_truth",
                "extra_fields": [
                    "setting_name",
                    "dataset"
                ]
            }
        },
        "omega_explorative_combinatory_distribution": {
            "hf_path": "allenai/omega-explorative",
            "hf_config": "combinatory_distribution",
            "splits": [
                "train",
                "test_in",
                "test_out"
            ],
            "transform": {
                "text_field": "messages.0.content",
                "answer_field": "ground_truth",
                "extra_fields": [
                    "setting_name",
                    "dataset"
                ]
            }
        },
        "omega_explorative_combinatory_pattern_matching": {
            "hf_path": "allenai/omega-explorative",
            "hf_config": "combinatory_pattern_matching",
            "splits": [
                "train",
                "test_in",
                "test_out"
            ],
            "transform": {
                "text_field": "messages.0.content",
                "answer_field": "ground_truth",
                "extra_fields": [
                    "setting_name",
                    "dataset"
                ]
            }
        },
        "omega_explorative_combinatory_probability_at_least_n_specific_fixed": {
            "hf_path": "allenai/omega-explorative",
            "hf_config": "combinatory_probability_at_least_n_specific_fixed",
            "splits": [
                "train",
                "test_in",
                "test_out"
            ],
            "transform": {
                "text_field": "messages.0.content",
                "answer_field": "ground_truth",
                "extra_fields": [
                    "setting_name",
                    "dataset"
                ]
            }
        },
        "omega_explorative_combinatory_probability_exactly_n_specific_fixed": {
            "hf_path": "allenai/omega-explorative",
            "hf_config": "combinatory_probability_exactly_n_specific_fixed",
            "splits": [
                "train",
                "test_in",
                "test_out"
            ],
            "transform": {
                "text_field": "messages.0.content",
                "answer_field": "ground_truth",
                "extra_fields": [
                    "setting_name",
                    "dataset"
                ]
            }
        },
        "omega_explorative_combinatory_probability_no_fixed_points": {
            "hf_path": "allenai/omega-explorative",
            "hf_config": "combinatory_probability_no_fixed_points",
            "splits": [
                "train",
                "test_in",
                "test_out"
            ],
            "transform": {
                "text_field": "messages.0.content",
                "answer_field": "ground_truth",
                "extra_fields": [
                    "setting_name",
                    "dataset"
                ]
            }
        },
        "omega_explorative_combinatory_probability_no_specific_letter_fixed": {
            "hf_path": "allenai/omega-explorative",
            "hf_config": "combinatory_probability_no_specific_letter_fixed",
            "splits": [
                "train",
                "test_in",
                "test_out"
            ],
            "transform": {
                "text_field": "messages.0.content",
                "answer_field": "ground_truth",
                "extra_fields": [
                    "setting_name",
                    "dataset"
                ]
            }
        },
        "omega_explorative_geometry_basic": {
            "hf_path": "allenai/omega-explorative",
            "hf_config": "geometry_basic",
            "splits": [
                "train",
                "test_in",
                "test_out"
            ],
            "transform": {
                "text_field": "messages.0.content",
                "answer_field": "ground_truth",
                "extra_fields": [
                    "setting_name",
                    "dataset"
                ]
            }
        },
        "omega_explorative_geometry_circle": {
            "hf_path": "allenai/omega-explorative",
            "hf_config": "geometry_circle",
            "splits": [
                "train",
                "test_in",
                "test_out"
            ],
            "transform": {
                "text_field": "messages.0.content",
                "answer_field": "ground_truth",
                "extra_fields": [
                    "setting_name",
                    "dataset"
                ]
            }
        },
        "omega_explorative_geometry_perpendicular_intersection": {
            "hf_path": "allenai/omega-explorative",
            "hf_config": "geometry_perpendicular_intersection",
            "splits": [
                "train",
                "test_in",
                "test_out"
            ],
            "transform": {
                "text_field": "messages.0.content",
                "answer_field": "ground_truth",
                "extra_fields": [
                    "setting_name",
                    "dataset"
                ]
            }
        },
        "omega_explorative_geometry_polygon": {
            "hf_path": "allenai/omega-explorative",
            "hf_config": "geometry_polygon",
            "splits": [
                "train",
                "test_in",
                "test_out"
            ],
            "transform": {
                "text_field": "messages.0.content",
                "answer_field": "ground_truth",
                "extra_fields": [
                    "setting_name",
                    "dataset"
                ]
            }
        },
        "omega_explorative_geometry_polygon_rotation": {
            "hf_path": "allenai/omega-explorative",
            "hf_config": "geometry_polygon_rotation",
            "splits": [
                "train",
                "test_in",
                "test_out"
            ],
            "transform": {
                "text_field": "messages.0.content",
                "answer_field": "ground_truth",
                "extra_fields": [
                    "setting_name",
                    "dataset"
                ]
            }
        },
        "omega_explorative_geometry_triangle": {
            "hf_path": "allenai/omega-explorative",
            "hf_config": "geometry_triangle",
            "splits": [
                "train",
                "test_in",
                "test_out"
            ],
            "transform": {
                "text_field": "messages.0.content",
                "answer_field": "ground_truth",
                "extra_fields": [
                    "setting_name",
                    "dataset"
                ]
            }
        },
        "omega_explorative_geometry_polygon_chords": {
            "hf_path": "allenai/omega-explorative",
            "hf_config": "geometry_polygon_chords",
            "splits": [
                "train",
                "test_in",
                "test_out"
            ],
            "transform": {
                "text_field": "messages.0.content",
                "answer_field": "ground_truth",
                "extra_fields": [
                    "setting_name",
                    "dataset"
                ]
            }
        },
        "omega_explorative_geometry_polygon_color": {
            "hf_path": "allenai/omega-explorative",
            "hf_config": "geometry_polygon_color",
            "splits": [
                "train",
                "test_in",
                "test_out"
            ],
            "transform": {
                "text_field": "messages.0.content",
                "answer_field": "ground_truth",
                "extra_fields": [
                    "setting_name",
                    "dataset"
                ]
            }
        },
        "omega_explorative_logic_gridworld_blocked": {
            "hf_path": "allenai/omega-explorative",
            "hf_config": "logic_gridworld_blocked",
            "splits": [
                "train",
                "test_in",
                "test_out"
            ],
            "transform": {
                "text_field": "messages.0.content",
                "answer_field": "ground_truth",
                "extra_fields": [
                    "setting_name",
                    "dataset"
                ]
            }
        },
        "omega_explorative_logic_gridworld_knight_move": {
            "hf_path": "allenai/omega-explorative",
            "hf_config": "logic_gridworld_knight_move",
            "splits": [
                "train",
                "test_in",
                "test_out"
            ],
            "transform": {
                "text_field": "messages.0.content",
                "answer_field": "ground_truth",
                "extra_fields": [
                    "setting_name",
                    "dataset"
                ]
            }
        },
        "omega_explorative_logic_gridworld_rookmove": {
            "hf_path": "allenai/omega-explorative",
            "hf_config": "logic_gridworld_rookmove",
            "splits": [
                "train",
                "test_in",
                "test_out"
            ],
            "transform": {
                "text_field": "messages.0.content",
                "answer_field": "ground_truth",
                "extra_fields": [
                    "setting_name",
                    "dataset"
                ]
            }
        },
        "omega_explorative_logic_zebralogic": {
            "hf_path": "allenai/omega-explorative",
            "hf_config": "logic_zebralogic",
            "splits": [
                "train",
                "test_in",
                "test_out"
            ],
            "transform": {
                "text_field": "messages.0.content",
                "answer_field": "ground_truth",
                "extra_fields": [
                    "setting_name",
                    "dataset"
                ]
            }
        },
        "omega_explorative_logic_puzzles_grid_chip": {
            "hf_path": "allenai/omega-explorative",
            "hf_config": "logic_puzzles_grid_chip",
            "splits": [
                "train",
                "test_in",
                "test_out"
            ],
            "transform": {
                "text_field": "messages.0.content",
                "answer_field": "ground_truth",
                "extra_fields": [
                    "setting_name",
                    "dataset"
                ]
            }
        },
        "omega_explorative_numbertheory_lte_qr": {
            "hf_path": "allenai/omega-explorative",
            "hf_config": "numbertheory_lte_qr",
            "splits": [
                "train",
                "test_in",
                "test_out"
            ],
            "transform": {
                "text_field": "messages.0.content",
                "answer_field": "ground_truth",
                "extra_fields": [
                    "setting_name",
                    "dataset"
                ]
            }
        },
        "omega_explorative_numbertheory_ordered_lte": {
            "hf_path": "allenai/omega-explorative",
            "hf_config": "numbertheory_ordered_lte",
            "splits": [
                "train",
                "test_in",
                "test_out"
            ],
            "transform": {
                "text_field": "messages.0.content",
                "answer_field": "ground_truth",
                "extra_fields": [
                    "setting_name",
                    "dataset"
                ]
            }
        },
        "omega_explorative_numbertheory_qr_sum": {
            "hf_path": "allenai/omega-explorative",
            "hf_config": "numbertheory_qr_sum",
            "splits": [
                "train",
                "test_in",
                "test_out"
            ],
            "transform": {
                "text_field": "messages.0.content",
                "answer_field": "ground_truth",
                "extra_fields": [
                    "setting_name",
                    "dataset"
                ]
            }
        },

        "omega_transformative_matrix_rank": {
            "hf_path": "allenai/omega-transformative",
            "hf_config": "trans_matrix_rank",
            "splits": [
                "train",
                "test"
            ],
            "transform": {
                "text_field": "messages.0.content",
                "answer_field": "ground_truth"
            }
        },
        "omega_transformative_func_intersection": {
            "hf_path": "allenai/omega-transformative",
            "hf_config": "trans_func_intersection",
            "splits": [
                "train",
                "test"
            ],
            "transform": {
                "text_field": "messages.0.content",
                "answer_field": "ground_truth"
            }
        },
        "omega_transformative_de_moivre": {
            "hf_path": "allenai/omega-transformative",
            "hf_config": "trans_de_moivre",
            "splits": [
                "train",
                "test"
            ],
            "transform": {
                "text_field": "messages.0.content",
                "answer_field": "ground_truth"
            }
        },
        "omega_transformative_prob_letter": {
            "hf_path": "allenai/omega-transformative",
            "hf_config": "trans_prob_letter",
            "splits": [
                "train",
                "test"
            ],
            "transform": {
                "text_field": "messages.0.content",
                "answer_field": "ground_truth"
            }
        },
        "omega_transformative_integrations": {
            "hf_path": "allenai/omega-transformative",
            "hf_config": "trans_integrations",
            "splits": [
                "train",
                "test"
            ],
            "transform": {
                "text_field": "messages.0.content",
                "answer_field": "ground_truth"
            }
        },
        "omega_transformative_gridworld": {
            "hf_path": "allenai/omega-transformative",
            "hf_config": "trans_gridworld",
            "splits": [
                "train",
                "test"
            ],
            "transform": {
                "text_field": "messages.0.content",
                "answer_field": "ground_truth"
            }
        },
        "omega_transformative_circles": {
            "hf_path": "allenai/omega-transformative",
            "hf_config": "trans_circles",
            "splits": [
                "train",
                "test"
            ],
            "transform": {
                "text_field": "messages.0.content",
                "answer_field": "ground_truth"
            }
        },
        "openbookqa": {
            "hf_path": "allenai/openbookqa",
            "splits": [
                "train",
                "validation",
                "test"
            ],
            "transform": {
                "text_field": "question_stem",
                "answer_field": "choices"
            }
        },
        "piqa": {
            "hf_path": "piqa",
            "splits": [
                "train",
                "test",
                "validation"
            ],
            "transform": {
                "text_field": "goal",
                "answer_field": "label"
            }
        },
        "popqa": {
            "hf_path": "akariasai/PopQA",
            "splits": [
                "test"
            ],
            "transform": {
                "text_field": "question",
                "answer_field": "possible_answers"
            }
        },
        "qasper_yesno": {
            "hf_path": "allenai/qasper-yesno",
            "splits": [
                "train",
                "validation",
                "test"
            ],
            "transform": {
                "text_field": "question",
                "last_paragraph_field": "full_text.paragraphs",
                "min_paragraph_words": 10,
                "answer_field": "answer"
            }
        },
        "repobench_python_cross_file_first": {
            "hf_path": "tianyang/repobench_python_v1.1",
            "splits": [
                "cross_file_first"
            ],
            "transform": {
                "text_field": "cropped_code",
                "answer_field": "next_line",
                "extra_fields": [
                    "repo_name",
                    "file_path",
                    "level"
                ],
                "strip_python_comments": True
            }
        },
        "repobench_python_cross_file_random": {
            "hf_path": "tianyang/repobench_python_v1.1",
            "splits": [
                "cross_file_random"
            ],
            "transform": {
                "text_field": "cropped_code",
                "answer_field": "next_line",
                "extra_fields": [
                    "repo_name",
                    "file_path",
                    "level"
                ],
                "strip_python_comments": True
            }
        },
        "repobench_python_in_file": {
            "hf_path": "tianyang/repobench_python_v1.1",
            "splits": [
                "in_file"
            ],
            "transform": {
                "text_field": "cropped_code",
                "answer_field": "next_line",
                "extra_fields": [
                    "repo_name",
                    "file_path",
                    "level"
                ],
                "strip_python_comments": True
            }
        },
        "sciq": {
            "hf_path": "allenai/sciq",
            "splits": [
                "train",
                "validation",
                "test"
            ],
            "transform": {
                "text_field": "question",
                "context_field": "support",
                "answer_field": "correct_answer"
            }
        },
        "sciriff_yesno": {
            "hf_path": "allenai/sciriff-yesno",
            "splits": [
                "train",
                "validation",
                "test"
            ],
            "transform": {
                "text_field": "question",
                "context_field": "context",
                "answer_field": "answer"
            }
        },
        "simple_qa": {
            "hf_path": "lighteval/SimpleQA",
            "splits": [
                "test"
            ],
            "transform": {
                "text_field": "problem",
                "answer_field": "answer"
            }
        },
        "simple_tom_mental_state": {
            "hf_path": "allenai/SimpleToM",
            "hf_config": "mental-state-qa",
            "splits": [
                "test"
            ],
            "transform": {
                "text_field": "question",
                "context_field": "story",
                "answer_field": "answerKey"
            }
        },
        "social_i_qa": {
            "hf_path": "allenai/social_i_qa",
            "splits": [
                "train",
                "validation"
            ],
            "transform": {
                "context_field": "context",
                "text_field": "question",
                "answer_key_field": "label",
                "answer_prefix": "answer",
                "answer_label_transform": "numbers_to_letters"
            }
        },
        "squad": {
            "hf_path": "squad",
            "splits": [
                "train",
                "validation"
            ],
            "transform": {
                "text_field": "question",
                "context_field": "context",
                "answer_field": "answers"
            }
        },
        "squad_mc": {
            "hf_path": "allenai/squad_mc",
            "splits": [
                "validation"
            ],
            "transform": {
                "text_field": "question_original",
                "context_field": "context_original",
                "answer_field": "answers_original"
            }
        },
        "squad_v2": {
            "hf_path": "rajpurkar/squad_v2",
            "splits": [
                "train",
                "validation"
            ],
            "transform": {
                "text_field": "question",
                "context_field": "context",
                "answer_field": "answers"
            }
        },
        "super_glue_boolq": {
            "hf_path": "aps/super_glue",
            "hf_config": "boolq",
            "splits": [
                "train",
                "validation",
                "test"
            ],
            "transform": {
                "text_field": "question",
                "context_field": "passage",
                "answer_field": "label"
            }
        },
        "super_glue_rte": {
            "hf_path": "aps/super_glue",
            "hf_config": "rte",
            "splits": [
                "train",
                "validation",
                "test"
            ],
            "transform": {
                "text_field": "hypothesis",
                "context_field": "premise",
                "answer_field": "label"
            }
        },
        "super_gpqa": {
            "hf_path": "m-a-p/SuperGPQA",
            "splits": [
                "train"
            ],
            "transform": {
                "text_field": "question",
                "answer_field": "answer",
                "choices_field": "options",
                "extra_fields": [
                    "discipline",
                    "field",
                    "subfield"
                ]
            }
        },
        # Massive with bad chars and unusual language. TODO Take a look at decontaminating these if easy plug it in, but not used explicitly
        # 'tydiqa_primary': {
        #     'hf_path': 'google-research-datasets/tydiqa',
        #     'hf_config': 'primary_task',
        #     'splits': ['train', 'validation'],
        #     'transform': {
        #         'text_field': 'question_text',
        #         'context_field': 'document_plaintext'
        #     }
        # },

        "trivia_qa": {
            "hf_path": "mandarjoshi/trivia_qa",
            "hf_config": "rc",
            "splits": [
                "train",
                "validation",
                "test"
            ],
            "transform": {
                "text_field": "question",
                "answer_field": "answer.value"
            },
            "no_answer_splits": [
                "test"
            ]
        },
        "truthful_qa": {
            "hf_path": "truthfulqa/truthful_qa",
            "hf_config": "generation",
            "splits": [
                "validation"
            ],
            "transform": {
                "text_field": "question",
                "answer_field": "best_answer"
            }
        },
        "tulu3_do_anything_now": {
            "hf_path": "allenai/tulu-3-do-anything-now-eval",
            "splits": [
                "test"
            ],
            "transform": {
                "text_field": "adversarial",
                "extra_fields": [
                    "vanilla",
                    "jailbreak",
                    "platform",
                    "source"
                ]
            },
            "no_answer_splits": [
                "test"
            ]
        },
        "tulu3_trustllm_jailbreak": {
            "hf_path": "allenai/tulu-3-trustllm-jailbreaktrigger-eval",
            "splits": [
                "test"
            ],
            "transform": {
                "text_field": "prompt",
                "extra_fields": [
                    "label",
                    "source"
                ]
            },
            "no_answer_splits": [
                "test"
            ]
        },
        "wildguardtest": {
            "hf_path": "walledai/WildGuardTest",
            "splits": [
                "train"
            ],
            "transform": {
                "text_field": "prompt",
                "answer_field": "label",
                "extra_fields": [
                    "adversarial"
                ]
            }
        },
        "wildjailbreak_eval": {
            "hf_path": "allenai/wildjailbreak",
            "hf_config": "eval",
            "splits": [
                "train"
            ],
            "transform": {
                "text_field": "adversarial",
                "extra_fields": [
                    "label",
                    "data_type"
                ]
            },
            "no_answer_splits": [
                "train"
            ]
        },
        "wildjailbreak_train": {
            "hf_path": "allenai/wildjailbreak",
            "hf_config": "train",
            "splits": [
                "train"
            ],
            "transform": {
                "text_field": "vanilla",
                "extra_fields": [
                    "data_type"
                ]
            }
        },
        "winogrande": {
            "hf_path": "allenai/winogrande",
            "hf_config": "winogrande_xl",
            "splits": [
                "train",
                "test",
                "validation"
            ],
            "transform": {
                "text_field": "sentence",
                "answer_key_field": "answer",
                "answer_prefix": "option"
            },
            "no_answer_splits": [
                "test"
            ]
        },
        "xstest": {
            "hf_path": "walledai/XSTest",
            "splits": [
                "test"
            ],
            "transform": {
                "text_field": "prompt",
                "answer_field": "label",
                "extra_fields": [
                    "focus",
                    "type"
                ]
            }
        },
        "zebra_logic_grid": {
            "hf_path": "allenai/ZebraLogicBench-private",
            "hf_config": "grid_mode",
            "splits": [
                "test"
            ],
            "transform": {
                "text_field": "puzzle",
                "answer_field": "solution"
            }
        },
        "zebra_logic_mc": {
            "hf_path": "allenai/ZebraLogicBench-private",
            "hf_config": "mc_mode",
            "splits": [
                "test"
            ],
            "transform": {
                "combine_context_and_question": True,
                "context_field": "puzzle",
                "text_field": "question",
                "answer_field": "answer"
            }
        },
        "zero_scrolls_qasper": {  # Optional to remove maybe
            "hf_path": "tau/zero_scrolls",
            "hf_config": "qasper",
            "splits": [
                "validation",
                "test"
            ],
            "transform": {
                "text_field": "input",
                "answer_field": "output"
            },
            "no_answer_splits": [
                "test"
            ]
        }
    }
}

#         # 'ultrachat_200k': {
#         #     'hf_path': 'HuggingFaceH4/ultrachat_200k',
#         #     'splits': ['train_sft', 'test_sft', 'train_gen', 'test_gen'],
#         #     'transform': 'auto'  # Will use auto extraction to handle the messages field
#         # },

#         # 'wildchat': {
#         #     'hf_path': 'allenai/WildChat',
#         #     'splits': ['train'],  # Only has train split
#         #     'transform': {
#         #         'text_field': 'text',
#         #     }
#         # },

def load_local_jsonl(file_path):
    """Load a local JSONL file as a HuggingFace Dataset."""
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return Dataset.from_list(data)


def auto_extract(example):
    """Automatically extract question and answer fields from complex dataset structures.

    Returns:
        tuple: (question, answer) - tries to find the longest text as question and second longest as answer
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

    # Return the longest as question and second longest as answer
    if len(text_candidates) == 0:
        return None, None
    elif len(text_candidates) == 1:
        return text_candidates[0][0], None
    else:
        return text_candidates[0][0], text_candidates[1][0]


def get_nested_field(obj, field_path):
    """Access nested fields using dot notation (e.g., 'answer.value' or 'answer_original.spans.0')"""
    fields = field_path.split('.')
    value = obj
    for field in fields:
        # Check if field is a numeric index
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


def transform_task_mapping(flat_mapping):
    """Transform flat task mapping into nested structure for efficient lookups.

    Args:
        flat_mapping: Original flat dictionary mapping task keys to metadata

    Returns:
        Nested dict: {dataset_path: {split: {config: [task_keys]}}}
        Also returns local_mapping for agi_eval datasets
    """
    hf_mapping = {}
    local_mapping = {}

    for task_key, task_info in flat_mapping.items():
        dataset_path = task_info.get("dataset_path", "")
        split = task_info.get("split", "")
        dataset_name = task_info.get("dataset_name", "")

        # Skip if missing required fields
        if not dataset_path or not split:
            continue

        # Check if this is a local dataset (has file path format)
        if "/" in dataset_path and dataset_path.startswith("/"):
            # This is a local path - extract dataset name for agi_eval
            # Store in local_mapping by dataset_name
            if dataset_name:
                if dataset_name not in local_mapping:
                    local_mapping[dataset_name] = {}
                if split not in local_mapping[dataset_name]:
                    local_mapping[dataset_name][split] = []
                local_mapping[dataset_name][split].append(task_key)
        else:
            # This is a HuggingFace dataset path
            if dataset_path not in hf_mapping:
                hf_mapping[dataset_path] = {}
            if split not in hf_mapping[dataset_path]:
                hf_mapping[dataset_path][split] = {}

            # Use dataset_name as config
            if dataset_name:
                # Store all matching tasks for each path/split/config combo
                if dataset_name not in hf_mapping[dataset_path][split]:
                    hf_mapping[dataset_path][split][dataset_name] = []
                hf_mapping[dataset_path][split][dataset_name].append(task_key)

    return hf_mapping, local_mapping


def find_oe_eval_task(hf_mapping, local_mapping, eval_name, hf_path, split, config, subject=None):
    """Find matching oe-eval-task keys using efficient lookup.

    Args:
        hf_mapping: Nested dict for HF datasets {path: {split: {config: [task_keys]}}}
        local_mapping: Nested dict for local datasets {name: {split: [task_keys]}}
        eval_name: Name of the eval dataset (e.g., "ai2_arc_challenge")
        hf_path: HuggingFace dataset path (e.g., "allenai/ai2_arc")
        split: Dataset split (e.g., "train", "test", "validation")
        config: HuggingFace config name (e.g., "ARC-Challenge")
        subject: Optional subject field for MMLU datasets

    Returns:
        List of matching task keys or None if no match found
    """
    # For local datasets (agi_eval_*)
    if eval_name.startswith("agi_eval_"):
        local_dataset_name = eval_name.replace("agi_eval_", "").replace("_", "-")
        if local_dataset_name in local_mapping:
            if split in local_mapping[local_dataset_name]:
                return local_mapping[local_dataset_name][split]
        return None

    # For HuggingFace datasets
    if not hf_path:
        return None

    # Try direct path match first
    paths_to_try = [hf_path]

    # Also try just the dataset name part (e.g., "ai2_arc" from "allenai/ai2_arc")
    if "/" in hf_path:
        paths_to_try.append(hf_path.split("/")[-1])

    for path in paths_to_try:
        if path in hf_mapping:
            if split in hf_mapping[path]:
                # For MMLU with config="all", use subject field for matching
                if eval_name == "mmlu" and config == "all" and subject:
                    # Try exact subject match
                    if subject in hf_mapping[path][split]:
                        return hf_mapping[path][split][subject]
                    
                    # Try normalized subject match
                    subject_normalized = subject.lower().replace("-", "").replace("_", "")
                    for stored_config, task_keys in hf_mapping[path][split].items():
                        stored_normalized = stored_config.lower().replace("-", "").replace("_", "")
                        if subject_normalized == stored_normalized:
                            return task_keys
                
                # Try exact config match first
                if config and config in hf_mapping[path][split]:
                    return hf_mapping[path][split][config]

                # Try normalized config match
                if config:
                    config_normalized = config.lower().replace("-", "").replace("_", "")
                    for stored_config, task_keys in hf_mapping[path][split].items():
                        stored_normalized = stored_config.lower().replace("-", "").replace("_", "")
                        if config_normalized == stored_normalized:
                            return task_keys

    return None

def is_answer_empty(answer):
    """Check if an answer is considered empty (None, blank string, or '<unk>')"""
    if answer is None:
        return True
    if isinstance(answer, str):
        return answer.strip() == '' or answer.strip() == '<unk>'
    return False


def strip_python_comments(text):
    """Remove lines that start with # from the text"""
    if not isinstance(text, str):
        return text

    lines = text.split('\n')
    filtered_lines = []

    for line in lines:
        # Check if line starts with # (after stripping whitespace)
        if not line.strip().startswith('#'):
            filtered_lines.append(line)

    return '\n'.join(filtered_lines)


def transform_answer_label(label, transform_type):
    """Transform answer labels based on the specified transform type"""
    if transform_type == 'numbers_to_letters':
        # Convert "1" -> "A", "2" -> "B", "3" -> "C", etc.
        try:
            # Handle both string and int labels
            label_int = int(label)
            # Convert 1-based to 0-based, then to letter
            if label_int >= 1:
                return chr(ord('A') + label_int - 1)
            else:
                # If label is 0, use 'A'
                return 'A'
        except (ValueError, TypeError):
            # If conversion fails, return original label
            return label

    # Add more transform types here as needed
    return label


def download_and_transform_eval(eval_name, eval_config, global_config, document_id_counter, stats, datasets_without_answers, hf_mapping, local_mapping, subject_index_counters=None):
    """Download HF dataset and transform to our JSONL format with separate question, context, and answer fields"""

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
        missing_fields_count = {"question": 0, "context": 0, "answer": 0}

        # Track character counts for this dataset
        dataset_stats = {
            "question_chars": 0,
            "context_chars": 0,
            "answer_chars": 0,
            "total_chars": 0,
            "records": 0,
            "records_with_answers": 0,
            "records_without_answers": 0
        }

        # Lazy ChunkedFileWriter - only created when needed
        writer_best_available = None

        try:
            for idx, example in enumerate(dataset[split]):
                # Check if we should use auto extraction
                if eval_config['transform'] == 'auto':
                    # Use auto extraction for question and answer
                    question, answer = auto_extract(example)

                    # Skip if no valid question found
                    if question is None:
                        continue

                    # Create record with separate fields
                    record = {
                        global_config['jsonl_format']['eval_field']: eval_name,
                        global_config['jsonl_format']['index_field']: idx,
                        global_config['jsonl_format']['split_field']: split,
                        "question": question,
                        "context": None,  # Auto extraction doesn't extract context
                        "answer": answer,
                        "config": eval_config.get('hf_config')
                    }

                    # Add oe-eval-task field
                    hf_path = eval_config.get('hf_path')
                    subject = record.get('subject')  # Get subject if it exists (e.g., for MMLU)
                    oe_task = find_oe_eval_task(hf_mapping, local_mapping, eval_name, hf_path, split, eval_config.get('hf_config'), subject)
                    record['oe-eval-task'] = oe_task

                    # Log missing fields
                    if question is None:
                        missing_fields_count["question"] += 1
                    if answer is None:
                        missing_fields_count["answer"] += 1
                    missing_fields_count["context"] += 1  # Always missing for auto extraction

                    # Add unique document ID
                    record['doc_id'] = document_id_counter[0]
                    document_id_counter[0] += 1

                    # Track character statistics
                    if question:
                        dataset_stats["question_chars"] += len(question)
                    if answer:
                        dataset_stats["answer_chars"] += len(answer)
                    dataset_stats["records"] += 1

                    # Create best-available writer if not already created
                    if writer_best_available is None:
                        writer_best_available = ChunkedFileWriter(output_file)

                    # For best-available, we keep answers if they exist, remove if empty
                    best_available_record = record.copy()
                    if is_answer_empty(answer):
                        best_available_record.pop('answer', None)
                        dataset_stats["records_without_answers"] += 1
                    else:
                        dataset_stats["records_with_answers"] += 1
                    writer_best_available.write(best_available_record)

                else:
                    # Check if we should use parallel arrays extraction
                    if 'parallel_arrays' in eval_config['transform']:
                        parallel_config = eval_config['transform']['parallel_arrays']

                        # Get the arrays
                        question_array = get_nested_field(example, parallel_config['question_array'])
                        answer_array = get_nested_field(example, parallel_config['answer_array'])

                        # Get context if specified
                        context = None
                        if 'context_field' in eval_config['transform']:
                            context = get_nested_field(example, eval_config['transform']['context_field'])
                            if context is not None and not isinstance(context, str):
                                context = str(context)

                        # Process each Q&A pair
                        if question_array and answer_array:
                            # Ensure both are lists
                            if not isinstance(question_array, list):
                                question_array = [question_array]
                            if not isinstance(answer_array, list):
                                answer_array = [answer_array]

                            # Process pairs
                            for q_idx, (q, a) in enumerate(zip(question_array, answer_array)):
                                # Skip if question is empty
                                if not q or (isinstance(q, str) and not q.strip()):
                                    continue

                                question = str(q) if q is not None else ""
                                answer = str(a) if a is not None else None

                                # Apply strip_python_comments if configured
                                if eval_config['transform'].get('strip_python_comments', False):
                                    if question:
                                        question = strip_python_comments(question)
                                    if answer:
                                        answer = strip_python_comments(answer)

                                # Create record
                                record = {
                                    global_config['jsonl_format']['eval_field']: eval_name,
                                    global_config['jsonl_format']['index_field']: idx,
                                    global_config['jsonl_format']['split_field']: split,
                                    "question": question,
                                    "context": context,
                                    "answer": answer,
                                    "sub_index": q_idx,  # Track which Q&A pair within the example
                                    "config": eval_config.get('hf_config')
                                }

                                # Add oe-eval-task field
                                hf_path = eval_config.get('hf_path')
                                subject = record.get('subject')  # Get subject if it exists (e.g., for MMLU)
                                oe_task = find_oe_eval_task(hf_mapping, local_mapping, eval_name, hf_path, split, eval_config.get('hf_config'), subject)
                                record['oe-eval-task'] = oe_task

                                # Add unique document ID
                                record['doc_id'] = document_id_counter[0]
                                document_id_counter[0] += 1

                                # Track character statistics
                                if question:
                                    dataset_stats["question_chars"] += len(question)
                                if context:
                                    dataset_stats["context_chars"] += len(context)
                                if answer:
                                    dataset_stats["answer_chars"] += len(answer)
                                dataset_stats["records"] += 1

                                # Write to best-available file
                                if writer_best_available is None:
                                    writer_best_available = ChunkedFileWriter(output_file)

                                best_available_record = record.copy()
                                if is_answer_empty(answer):
                                    best_available_record.pop('answer', None)
                                    dataset_stats["records_without_answers"] += 1
                                else:
                                    dataset_stats["records_with_answers"] += 1
                                writer_best_available.write(best_available_record)

                        # Skip the rest of the normal processing
                        continue

                    # Use the existing manual extraction logic, but extract fields separately
                    # Extract question field (previously text_field)
                    text_field = eval_config['transform']['text_field']
                    question = get_nested_field(example, text_field)

                    # Handle cases where question might be a list
                    if isinstance(question, list):
                        # Take the first element if it's a list
                        if question:  # Check if list is not empty
                            question = str(question[0])
                        else:
                            question = ""
                    elif question is not None and not isinstance(question, str):
                        # Convert to string if it's not already
                        question = str(question)

                    # Extract context field if configured
                    context = None
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

                            # Check for duplicate contexts
                            if context and context.strip():
                                # Check if we've seen this context before
                                context_hash = hash(context)
                                if context_hash in seen_contexts:
                                    skipped_duplicates += 1
                                    continue  # Skip this record
                                else:
                                    seen_contexts.add(context_hash)

                    # Extract answer field if configured
                    answer = None

                    # Check if we should use answer_prefix + answer_key_field directly (without answer_field)
                    if 'answer_key_field' in eval_config['transform'] and 'answer_prefix' in eval_config['transform']:
                        answer_key_field = eval_config['transform']['answer_key_field']
                        answer_key = get_nested_field(example, answer_key_field)

                        if answer_key is not None:
                            # Apply label transformation if configured
                            transformed_key = answer_key
                            if 'answer_label_transform' in eval_config['transform']:
                                transform_type = eval_config['transform']['answer_label_transform']
                                transformed_key = transform_answer_label(answer_key, transform_type)

                            # Construct field name using prefix + transformed key value
                            answer_prefix = eval_config['transform']['answer_prefix']
                            answer_field_name = f"{answer_prefix}{transformed_key}"
                            # Get the answer from the constructed field name
                            answer_value = get_nested_field(example, answer_field_name)
                            if answer_value is not None:
                                answer = str(answer_value)
                            else:
                                # No answer at that field, skip answer
                                answer = None

                    elif 'answer_field' in eval_config['transform']:
                        answer_field = eval_config['transform']['answer_field']
                        answer_value = get_nested_field(example, answer_field)

                        # Check if we need to use answer_key to index into the answer value
                        if 'answer_key_field' in eval_config['transform'] and answer_value is not None:
                            answer_key_field = eval_config['transform']['answer_key_field']
                            answer_key = get_nested_field(example, answer_key_field)

                            if answer_key is not None:
                                # Check if we need to look up the answer key in a lookup field
                                if 'answer_lookup_field' in eval_config['transform']:
                                    lookup_field = eval_config['transform']['answer_lookup_field']
                                    lookup_values = get_nested_field(example, lookup_field)

                                    if isinstance(lookup_values, list) and isinstance(answer_value, list):
                                        # Find the index of answer_key in lookup_values
                                        try:
                                            # Convert answer_key to string for comparison
                                            answer_key_str = str(answer_key)
                                            # Find matching index in lookup values
                                            for idx, lookup_val in enumerate(lookup_values):
                                                if str(lookup_val) == answer_key_str:
                                                    if 0 <= idx < len(answer_value):
                                                        answer = str(answer_value[idx])
                                                        break
                                            else:
                                                # No match found
                                                answer = None
                                        except (ValueError, IndexError):
                                            answer = None
                                    else:
                                        # Fallback to original logic
                                        answer = str(answer_value) if answer_value is not None else None
                                # Original logic: answer_key is an integer index into answer_value
                                elif isinstance(answer_value, list) and isinstance(answer_key, int):
                                    if 0 <= answer_key < len(answer_value):
                                        answer = str(answer_value[answer_key])
                                    else:
                                        # Invalid index, skip this answer
                                        answer = None
                                else:
                                    # If answer_value is not a list or answer_key is not int,
                                    # try to use it as-is
                                    answer = str(answer_value) if answer_value is not None else None
                        elif answer_value is not None:
                            # Handle different answer field types (original logic)
                            if isinstance(answer_value, list):
                                # Array of answers - take the first one
                                if answer_value and answer_value[0] is not None:
                                    answer = str(answer_value[0])
                            else:
                                answer = str(answer_value)

                    # Skip if no question found
                    if not question or question.strip() == '':
                        continue

                    # Handle last_paragraph_field if configured
                    if 'last_paragraph_field' in eval_config['transform']:
                        paragraph_field = eval_config['transform']['last_paragraph_field']
                        min_words = eval_config['transform'].get('min_paragraph_words', 10)
                        paragraphs = get_nested_field(example, paragraph_field)

                        if paragraphs and isinstance(paragraphs, list):
                            # Find the last non-empty paragraph with enough words
                            last_paragraph = None
                            # Iterate through paragraphs in reverse order
                            for para_group in reversed(paragraphs):
                                if isinstance(para_group, list):
                                    # Iterate through inner paragraphs in reverse
                                    for para in reversed(para_group):
                                        if para and isinstance(para, str):
                                            # Count words
                                            word_count = len(para.split())
                                            if word_count >= min_words:
                                                last_paragraph = para
                                                break
                                elif isinstance(para_group, str) and para_group:
                                    # Handle case where it's a flat list of strings
                                    word_count = len(para_group.split())
                                    if word_count >= min_words:
                                        last_paragraph = para_group
                                        break
                                if last_paragraph:
                                    break

                            # Concatenate with question if we found a suitable paragraph
                            if last_paragraph:
                                question = f"{last_paragraph} {question}"
                    # Combine context and question if configured
                    elif eval_config['transform'].get('combine_context_and_question', False):
                        if context and context.strip():
                            # Combine context and question with a space separator
                            question = f"{context} {question}"

                    # Apply strip_python_comments if configured
                    if eval_config['transform'].get('strip_python_comments', False):
                        if question:
                            question = strip_python_comments(question)
                        if answer:
                            answer = strip_python_comments(answer)

                    # Create record with separate fields
                    record = {
                        global_config['jsonl_format']['eval_field']: eval_name,
                        global_config['jsonl_format']['index_field']: idx,
                        global_config['jsonl_format']['split_field']: split,
                        "question": question,
                        "context": context,
                        "answer": answer,
                        "config": eval_config.get('hf_config')
                    }

                    # Add any extra fields
                    if 'extra_fields' in eval_config['transform']:
                        for field in eval_config['transform']['extra_fields']:
                            if field in example:
                                record[field] = example[field]

                    # For MMLU, use subject-specific index counters
                    if eval_name == "mmlu" and subject_index_counters is not None:
                        subject = record.get('subject')
                        if subject:
                            # Create key for this dataset+subject combination
                            counter_key = f"{eval_name}:{subject}"
                            if counter_key not in subject_index_counters:
                                subject_index_counters[counter_key] = 0
                            # Use and increment the subject-specific counter
                            record[global_config['jsonl_format']['index_field']] = subject_index_counters[counter_key]
                            subject_index_counters[counter_key] += 1

                    # Add oe-eval-task field
                    hf_path = eval_config.get('hf_path')
                    subject = record.get('subject')  # Get subject if it exists (e.g., for MMLU)
                    oe_task = find_oe_eval_task(hf_mapping, local_mapping, eval_name, hf_path, split, eval_config.get('hf_config'), subject)
                    record['oe-eval-task'] = oe_task

                    # Log missing fields
                    if question is None:
                        missing_fields_count["question"] += 1
                    if context is None:
                        missing_fields_count["context"] += 1
                    if answer is None:
                        missing_fields_count["answer"] += 1

                    # Add unique document ID
                    record['doc_id'] = document_id_counter[0]
                    document_id_counter[0] += 1

                    # Track character statistics
                    if question:
                        dataset_stats["question_chars"] += len(question)
                    if context:
                        dataset_stats["context_chars"] += len(context)
                    if answer:
                        dataset_stats["answer_chars"] += len(answer)
                    dataset_stats["records"] += 1

                    # Create best-available writer if not already created
                    if writer_best_available is None:
                        writer_best_available = ChunkedFileWriter(output_file)

                    # For best-available, we keep answers if they exist, remove if empty
                    best_available_record = record.copy()
                    if is_answer_empty(answer):
                        best_available_record.pop('answer', None)
                        dataset_stats["records_without_answers"] += 1
                    else:
                        dataset_stats["records_with_answers"] += 1
                    writer_best_available.write(best_available_record)

                    # Handle choices field if configured (write as separate records)
                    if 'choices_field' in eval_config['transform']:
                        choices_field = eval_config['transform']['choices_field']
                        choices = get_nested_field(example, choices_field)
                        if choices is not None:
                            # Handle choices structure: {'text': [...], 'label': [...]}
                            if isinstance(choices, dict) and 'text' in choices:
                                for choice_text in choices['text']:
                                    choice_answer = str(choice_text)
                                    # Apply strip_python_comments to choice if configured
                                    if eval_config['transform'].get('strip_python_comments', False):
                                        choice_answer = strip_python_comments(choice_answer)

                                    choice_record = record.copy()
                                    choice_record["answer"] = choice_answer  # Use choice as answer
                                    choice_record['doc_id'] = document_id_counter[0]
                                    document_id_counter[0] += 1

                                    # Track character statistics for choice
                                    if question:
                                        dataset_stats["question_chars"] += len(question)
                                    if context:
                                        dataset_stats["context_chars"] += len(context)
                                    dataset_stats["answer_chars"] += len(choice_answer)
                                    dataset_stats["records"] += 1

                                    # Write to best-available file (choices always have answers)
                                    if writer_best_available is None:
                                        writer_best_available = ChunkedFileWriter(output_file)

                                    writer_best_available.write(choice_record)
                                    dataset_stats["records_with_answers"] += 1
                            elif isinstance(choices, list):
                                # Handle simple list of choices
                                for choice in choices:
                                    choice_answer = str(choice)
                                    # Apply strip_python_comments to choice if configured
                                    if eval_config['transform'].get('strip_python_comments', False):
                                        choice_answer = strip_python_comments(choice_answer)

                                    choice_record = record.copy()
                                    choice_record["answer"] = choice_answer  # Use choice as answer
                                    choice_record['doc_id'] = document_id_counter[0]
                                    document_id_counter[0] += 1

                                    # Track character statistics for choice
                                    if question:
                                        dataset_stats["question_chars"] += len(question)
                                    if context:
                                        dataset_stats["context_chars"] += len(context)
                                    dataset_stats["answer_chars"] += len(choice_answer)
                                    dataset_stats["records"] += 1

                                    # Write to best-available file (choices always have answers)
                                    if writer_best_available is None:
                                        writer_best_available = ChunkedFileWriter(output_file)

                                    writer_best_available.write(choice_record)
                                    dataset_stats["records_with_answers"] += 1

        finally:
            # Close the writer
            if writer_best_available is not None:
                writer_best_available.close()

        # Log missing fields loudly (only for question and answer)
        # Check if this split is expected to have no answers
        no_answer_splits = eval_config.get('no_answer_splits', [])
        is_no_answer_split = split in no_answer_splits

        # Determine which fields to check based on no_answer_splits
        if is_no_answer_split:
            important_fields = ["question"]  # Only check question for no_answer_splits
        else:
            important_fields = ["question", "answer"]

        important_missing = sum(missing_fields_count[field] for field in important_fields if field in missing_fields_count)
        if important_missing > 0:
            print(f"\n⚠️  WARNING: Missing fields detected in {eval_name} {split}:")
            for field in important_fields:
                if field in missing_fields_count and missing_fields_count[field] > 0:
                    count = missing_fields_count[field]
                    print(f"   - {field}: {count} missing out of {len(dataset[split])} examples ({count/len(dataset[split])*100:.1f}%)")
            print()

        # Calculate total characters
        dataset_stats["total_chars"] = (dataset_stats["question_chars"] +
                                       dataset_stats["context_chars"] +
                                       dataset_stats["answer_chars"])

        # Update global stats
        stats["total_question_chars"] += dataset_stats["question_chars"]
        stats["total_context_chars"] += dataset_stats["context_chars"]
        stats["total_answer_chars"] += dataset_stats["answer_chars"]
        stats["total_records"] += dataset_stats["records"]
        stats["datasets_processed"] += 1

        # Track datasets with no answers (unless split is in no_answer_splits)
        if dataset_stats["records"] > 0 and dataset_stats["records_with_answers"] == 0:
            # Check if this split is expected to have no answers
            no_answer_splits = eval_config.get('no_answer_splits', [])
            if split not in no_answer_splits:
                if eval_name not in datasets_without_answers:
                    datasets_without_answers[eval_name] = []
                datasets_without_answers[eval_name].append(split)

        if skipped_duplicates > 0:
            print(f"Saved {dataset_stats['records']} records total (skipped {skipped_duplicates} duplicates)")
        else:
            print(f"Saved {dataset_stats['records']} records total")

        if dataset_stats['records'] > 0:
            # Print chunk information
            if writer_best_available:
                chunk_files = writer_best_available.get_chunk_files()
                if len(chunk_files) > 1:
                    print(f"  - Saved {dataset_stats['records']} records → {len(chunk_files)} chunks")
                    for chunk_file in chunk_files:
                        print(f"    - {chunk_file.name}")
                else:
                    print(f"  - Saved {dataset_stats['records']} records → {output_file}")

        # Print dataset statistics
        print(f"  Character counts for {eval_name} {split}:")
        print(f"    - Questions: {dataset_stats['question_chars']:,} chars")
        print(f"    - Context: {dataset_stats['context_chars']:,} chars")
        print(f"    - Answers: {dataset_stats['answer_chars']:,} chars")
        print(f"    - Total: {dataset_stats['total_chars']:,} chars")

def download_all_evals():
    """Download all eval datasets"""
    print(f"Processing {len(EVAL_CONFIG['evals'])} eval datasets...")

    # Load task mapping
    project_root = Path(__file__).parent.parent
    mapping_file = project_root / "fixtures/oe-eval-map/task_dataset_mapping.json"
    hf_mapping = {}
    local_mapping = {}
    if mapping_file.exists():
        print(f"Loading task mapping from {mapping_file}")
        with open(mapping_file, 'r') as f:
            flat_mapping = json.load(f)
        print(f"Loaded {len(flat_mapping)} task mappings")

        # Transform into efficient nested structure
        hf_mapping, local_mapping = transform_task_mapping(flat_mapping)
        print(f"Transformed into {len(hf_mapping)} HF paths and {len(local_mapping)} local datasets")
    else:
        print(f"Warning: Task mapping file not found at {mapping_file}")

    # Clear output directory before starting
    output_dir = project_root / EVAL_CONFIG['output_dir']

    # Remove existing directory if it exists
    if output_dir.exists():
        print(f"Clearing existing directory: {output_dir}")
        shutil.rmtree(output_dir)

    # Create fresh directory
    output_dir.mkdir(parents=True, exist_ok=True)
    print("Created fresh output directory")

    # Initialize global document ID counter (using list for mutable reference)
    document_id_counter = [1]

    # Initialize global statistics
    stats = {
        "total_question_chars": 0,
        "total_context_chars": 0,
        "total_answer_chars": 0,
        "total_records": 0,
        "datasets_processed": 0
    }

    # Track datasets that have no answers
    datasets_without_answers = {}
    
    # Initialize subject-specific index counters for MMLU
    subject_index_counters = {}

    # Process each eval dataset
    for eval_name, eval_config in EVAL_CONFIG['evals'].items():
        download_and_transform_eval(eval_name, eval_config, EVAL_CONFIG, document_id_counter, stats, datasets_without_answers, hf_mapping, local_mapping, subject_index_counters)

    print(f"\n{'='*80}")
    print("FINAL STATISTICS")
    print(f"{'='*80}")
    print(f"Datasets processed: {stats['datasets_processed']}")
    print(f"Total records: {stats['total_records']:,}")
    print(f"Total document IDs: {document_id_counter[0] - 1:,}")
    print(f"\nCharacter counts:")
    print(f"  - Questions: {stats['total_question_chars']:,} chars")
    print(f"  - Context: {stats['total_context_chars']:,} chars")
    print(f"  - Answers: {stats['total_answer_chars']:,} chars")
    total_chars = stats['total_question_chars'] + stats['total_context_chars'] + stats['total_answer_chars']
    print(f"  - TOTAL: {total_chars:,} chars")

    # Show percentages
    if total_chars > 0:
        print(f"\nBreakdown by category:")
        print(f"  - Questions: {stats['total_question_chars']/total_chars*100:.1f}%")
        print(f"  - Context: {stats['total_context_chars']/total_chars*100:.1f}%")
        print(f"  - Answers: {stats['total_answer_chars']/total_chars*100:.1f}%")

    # Show average lengths
    if stats['total_records'] > 0:
        print(f"\nAverage lengths per record:")
        print(f"  - Question: {stats['total_question_chars']/stats['total_records']:.1f} chars")
        print(f"  - Context: {stats['total_context_chars']/stats['total_records']:.1f} chars")
        print(f"  - Answer: {stats['total_answer_chars']/stats['total_records']:.1f} chars")

    print(f"{'='*80}\n")

    # Print datasets with no answers
    if datasets_without_answers:
        print(f"\n{'='*80}")
        print("DATASETS WITH NO ANSWERS")
        print(f"{'='*80}")
        print(f"Found {len(datasets_without_answers)} datasets where all entries have empty or no answers:\n")

        for dataset_name, splits in sorted(datasets_without_answers.items()):
            if len(splits) == 1:
                print(f"  - {dataset_name} (split: {splits[0]})")
            else:
                print(f"  - {dataset_name} (splits: {', '.join(splits)})")

        print(f"\nThese datasets are saved to: {EVAL_CONFIG['output_dir']}/")
        print(f"{'='*80}\n")


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
