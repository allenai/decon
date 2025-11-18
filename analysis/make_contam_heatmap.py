#!/usr/bin/env python3
"""
Build a heatmap of contamination by data source × canonical benchmark.

Inputs (read-only; all under analysis/ by default):
- name maps in analysis/name_maps/*.json
- evaluation set sizes in analysis/eval_stats/{test,validation,train}.csv
- per-source contamination stats in analysis/per_source_stats/<source>/{all,test,validation,train}/*.csv
- performance CSVs in analysis/decon_perf_results/*.csv (used to filter canonicals)
- midtrain mapping in analysis/midtrain_decon_map.csv (optional, for source name normalization)

Outputs (under --output-dir):
- matrix.csv: sources × canonical benchmarks
-,margins.csv: row/column totals, non-eval per benchmark, percent-unique-eval-contam
-,metadata.json: configuration, diagnostics
-,heatmap.png/.svg: annotated heatmap
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

import itertools
import numpy as np

try:
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
except Exception:  # pragma: no cover
    plt = None
    LogNorm = None

def _is_base_name(name: str) -> bool:
    n = (name or "").lower()
    return any(tok in n for tok in ["non-decon", "non_decon", "nondecon", "non decon"])

def _is_decon_name(name: str) -> bool:
    n = (name or "").lower()
    if _is_base_name(n):
        return False
    return "decon" in n or "sparkle-motion" in n or "sparkle_motion" in n


# ----------------------------
# CLI and configuration
# ----------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate contamination heatmap for sources × canonical benchmarks."
    )
    repo_root = Path(__file__).resolve().parents[1]
    parser.add_argument(
        "--name-maps-dir",
        type=Path,
        default=repo_root / "analysis" / "name_maps",
        help="Directory containing name mapping JSONs.",
    )
    parser.add_argument(
        "--per-source-stats-dir",
        type=Path,
        default=repo_root / "analysis" / "per_source_stats",
        help="Directory containing per-source contamination stats.",
    )
    parser.add_argument(
        "--eval-stats-dir",
        type=Path,
        default=repo_root / "analysis" / "eval_stats",
        help="Directory containing eval stats CSVs.",
    )
    parser.add_argument(
        "--perf-results-dir",
        type=Path,
        default=repo_root / "analysis" / "decon_perf_results",
        help="Directory containing performance CSVs to select canonicals.",
    )
    parser.add_argument(
        "--midtrain-csv",
        type=Path,
        default=repo_root / "analysis" / "midtrain_decon_map.csv",
        help="Optional midtrain mapping CSV to augment source normalization.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo_root / "analysis" / "figures" / "contam_heatmap",
        help="Output directory for matrix CSV and plots.",
    )
    parser.add_argument(
        "--value",
        choices=["occurrences", "unique"],
        default="occurrences",
        help="Cell metric: total occurrences (default) or unique eval instances.",
    )
    parser.add_argument(
        "--include-non-eval",
        choices=["true", "false"],
        default="true",
        help="Include a column for non-evaluation contamination (all − eval-split).",
    )
    parser.add_argument(
        "--log-scale",
        action="store_true",
        help="Use log color scale for the heatmap.",
    )
    parser.add_argument(
        "--topk-cols",
        type=int,
        default=None,
        help="If set, only include top-K benchmarks by total contamination.",
    )
    parser.add_argument(
        "--topk-rows",
        type=int,
        default=None,
        help="If set, only include top-K sources by total contamination.",
    )
    parser.add_argument(
        "--save-svg",
        action="store_true",
        help="Save SVG in addition to PNG.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging.",
    )
    return parser


def configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(levelname)s %(message)s",
    )


# ----------------------------
# Data loading helpers
# ----------------------------


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def slugify_name(name: str) -> str:
    import re

    slug = re.sub(r"[^\w.-]+", "_", name.strip())
    slug = re.sub(r"_+", "_", slug).strip("_")
    return slug or "unknown"


@dataclass(frozen=True)
class NameMaps:
    eval_to_canonical: Mapping[str, List[str]]
    perf_to_canonical: Mapping[str, str]
    canonical_to_contam: Mapping[str, List[str]]
    canonical_benchmarks: List[str]
    canonical_to_eval_splits: Mapping[str, str]


def load_name_maps(root: Path) -> NameMaps:
    paths = {
        "eval_to_canonical": root / "eval_to_canonical.json",
        "perf_to_canonical": root / "perf_to_canonical.json",
        "canonical_to_contam": root / "canonical_to_contam.json",
        "canonical_benchmarks": root / "canonical_benchmarks.json",
        "canonical_to_eval_splits": root / "canonical_to_eval_splits.json",
    }
    missing = [k for k, p in paths.items() if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing name map(s): {missing}")
    maps = NameMaps(
        eval_to_canonical=read_json(paths["eval_to_canonical"]),
        perf_to_canonical=read_json(paths["perf_to_canonical"]),
        canonical_to_contam=read_json(paths["canonical_to_contam"]),
        canonical_benchmarks=read_json(paths["canonical_benchmarks"]),
        canonical_to_eval_splits=read_json(paths["canonical_to_eval_splits"]),
    )
    return maps


def load_paper_name_to_display(root: Path) -> Dict[str, str]:
    """Load mapping from CSV paper names to display names."""
    path = root / "paper_name_to_display.json"
    if not path.exists():
        return {}
    return read_json(path)


def canonical_base(name: str) -> str:
    """Return base benchmark identifier (left of first colon)."""
    return name.split(":", 1)[0] if ":" in name else name


def scan_perf_canonicals(perf_dir: Path, maps: NameMaps) -> Set[str]:
    """Return the set of canonical names referenced by perf CSV columns."""
    canonicals: Set[str] = set()
    if not perf_dir.exists():
        logging.warning("perf_results_dir %s does not exist", perf_dir)
        return canonicals
    for path in sorted(perf_dir.glob("*.csv")):
        try:
            with path.open("r", encoding="utf-8") as f:
                reader = csv.reader(f)
                header = next(reader, None)
                if not header:
                    continue
            for col in header:
                if col == "name":
                    continue
                mapped = maps.perf_to_canonical.get(col)
                if mapped:
                    canonicals.add(mapped)
                elif col in maps.canonical_benchmarks:
                    canonicals.add(col)
        except Exception as exc:
            logging.warning("Failed to read %s: %s", path, exc)
    # Expand set to include paired canonicals with the same base (e.g., xlarge ↔ mc variants)
    bases = {canonical_base(c) for c in canonicals}
    for c in maps.canonical_benchmarks:
        if canonical_base(c) in bases:
            canonicals.add(c)
    return canonicals


def load_midtrain_source_aliases(midtrain_csv: Path) -> Dict[str, str]:
    """
    Build a mapping of friendly Paper name → normalized folder-ish slug to help
    relate midtrain rows to per_source_stats directory names.
    """
    if not midtrain_csv.exists():
        return {}
    aliases: Dict[str, str] = {}
    with midtrain_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        # Detect header on arbitrary row containing 'paper name'
        if reader.fieldnames and "Paper name" not in reader.fieldnames:
            # Try to find the header row
            f.seek(0)
            lines = f.readlines()
            header_idx = None
            for i, line in enumerate(lines):
                if "paper name" in line.lower():
                    header_idx = i
                    break
            if header_idx is None:
                return {}
            hdr_reader = csv.DictReader(lines[header_idx:])
            reader = hdr_reader
        for row in reader:
            paper = (row.get("Paper name") or "").strip()
            if not paper:
                continue
            aliases[paper] = slugify_name(paper)
    return aliases


def build_source_display_names(per_source_root: Path, midtrain_csv: Path, name_maps_dir: Path) -> Dict[str, str]:
    """
    Return mapping of per_source_stats directory name -> display name.
    Uses CSV paper names matched via slugified directory names, then translates
    to display names via paper_name_to_display.json mapping.
    """
    # Load paper name to display name mapping
    paper_to_display = load_paper_name_to_display(name_maps_dir)
    
    # Build mapping: slugified directory name -> CSV paper name
    forward = load_midtrain_source_aliases(midtrain_csv)  # Paper -> slug
    slug_to_paper: Dict[str, str] = {slugify_name(v): k for k, v in forward.items()}

    def pretty_dir(name: str) -> str:
        # Replace underscores with spaces and normalize parentheses-like suffixes
        disp = name.replace("_", " ").strip()
        return disp

    mapping: Dict[str, str] = {}
    for d in list_sources(per_source_root):
        key = d
        slug = slugify_name(d)
        # Match directory to CSV paper name via slug
        paper_name = slug_to_paper.get(slug)
        if paper_name:
            # Translate paper name to display name
            display_name = paper_to_display.get(paper_name, paper_name)
            mapping[key] = display_name
        else:
            # Fall back to prettified directory name
            mapping[key] = pretty_dir(d)
    return mapping


def shorten_canonical_label(canonical: str) -> str:
    """
    Produce a compact x-axis label:
    - base benchmark name (left of first colon)
    - remove underscores and apply proper capitalization
    - add ' (MC)' if ':mc' present anywhere
    - add ' (@16)' if 'pass_at_16' present (drop 'p')
    - Condense names: deepseek leetcode -> leetcode, multipl-e-humaneval -> M-E-HumEval, codex humaneval -> HumEval
    """
    base = canonical.split(":", 1)[0]
    
    # Handle special condensed names first
    base_lower = base.lower()
    if "deepseek" in base_lower and "leetcode" in base_lower:
        # deepseek leetcode -> leetcode
        base_formatted = "LeetCode"
    elif "multipl" in base_lower and "humaneval" in base_lower:
        # multipl-e-humaneval -> M-E-HumEval
        base_formatted = "M-E-HumEval"
    elif "codex" in base_lower and "humaneval" in base_lower:
        # codex humaneval -> HumEval
        base_formatted = "HumEval"
    else:
        # Remove underscores and apply title case
        # Handle special cases: keep acronyms uppercase, handle multi-word names
        words = base.replace("_", " ").split()
        # Apply title case but handle special acronyms
        formatted_words = []
        for word in words:
            # Keep common acronyms uppercase
            if word.upper() in ["MC", "OCR", "FIM", "LLM", "AI"]:
                formatted_words.append(word.upper())
            # Handle special cases like "humaneval" -> "HumanEval", "mmlu" -> "MMLU"
            elif word.lower() == "humaneval":
                formatted_words.append("HumanEval")
            elif word.lower() == "mmlu":
                formatted_words.append("MMLU")
            elif word.lower() == "gsm8k":
                formatted_words.append("GSM8K")
            elif word.lower() == "sciq":
                formatted_words.append("SciQ")
            elif word.lower() == "csqa":
                formatted_words.append("CSQA")
            elif word.lower() == "arc":
                formatted_words.append("ARC")
            elif word.lower() == "piqa":
                formatted_words.append("PIQA")
            elif word.lower() == "squad":
                formatted_words.append("SQuAD")
            elif word.lower() == "coqa":
                formatted_words.append("CoQA")
            elif word.lower() == "drop":
                formatted_words.append("DROP")
            elif word.lower() == "lambada":
                formatted_words.append("LAMBADA")
            elif word.lower() == "winogrande":
                formatted_words.append("Winogrande")
            elif word.lower() == "socialiqa":
                formatted_words.append("SocialIQA")
            elif word.lower() == "hellaswag":
                formatted_words.append("HellaSwag")
            elif word.lower() == "medmcqa":
                formatted_words.append("MedMCQA")
            elif word.lower() == "medqa":
                formatted_words.append("MedQA")
            elif word.lower() == "deepseek":
                formatted_words.append("DeepSeek")
            elif word.lower() == "leetcode":
                formatted_words.append("LeetCode")
            elif word.lower() == "multipl":
                formatted_words.append("MultiPL")
            elif word.lower() == "codex":
                formatted_words.append("Codex")
            elif word.lower() == "minerva":
                formatted_words.append("Minerva")
            elif word.lower() == "jeopardy":
                formatted_words.append("Jeopardy")
            else:
                # Default: title case
                formatted_words.append(word.capitalize())
        base_formatted = " ".join(formatted_words)
    
    parts: List[str] = []
    if ":mc" in canonical:
        parts.append("MC")
    if "pass_at_16" in canonical:
        parts.append("@16")  # Drop 'p' from p@16
    suffix = f" ({', '.join(parts)})" if parts else ""
    return f"{base_formatted}{suffix}"


def format_canonical_display(canonical: str, maps: NameMaps) -> str:
    """Short label without eval split tag (split info shown separately)."""
    base = shorten_canonical_label(canonical)
    return base


def suite_to_canonicals(suite: str, maps: NameMaps) -> List[str]:
    """
    Resolve an eval suite to one or more canonical benchmarks.
    - Prefer exact mapping in eval_to_canonical.json
    - Special-case: any 'hendrycks_math_*' maps to Minerva canonical ('minerva:n4:v2')
    """
    canonicals = maps.eval_to_canonical.get(suite)
    if canonicals:
        return list(canonicals)
    if suite.startswith("hendrycks_math_"):
        return ["minerva:n4:v2"]
    return []


# ----------------------------
# Per-source stats parsing
# ----------------------------


@dataclass
class SplitStats:
    total_occurrences: float
    total_unique_eval_instances: float
    suite_unique: Dict[str, float]  # eval_suite -> unique_eval_instances
    suite_docs: Dict[str, float]  # eval_suite -> training_docs_contaminated


def read_summary_csv(path: Path) -> Tuple[float, float]:
    total_occurrences = 0.0
    total_unique = 0.0
    if not path.exists():
        return total_occurrences, total_unique
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            metric = (row.get("metric") or "").strip()
            value_str = (row.get("value") or "").strip()
            try:
                value = float(value_str)
            except Exception:
                continue
            if metric == "total_contamination_instances":
                total_occurrences = value
            elif metric == "unique_eval_instances":
                total_unique = value
    return total_occurrences, total_unique


def read_suite_kv_csv(path: Path, key_col: str, value_col: str) -> Dict[str, float]:
    out: Dict[str, float] = {}
    if not path.exists():
        return out
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (row.get(key_col) or "").strip()
            val_str = (row.get(value_col) or "").strip()
            if not key:
                continue
            try:
                out[key] = float(val_str)
            except Exception:
                # tolerate partial/malformed rows
                continue
    return out


def load_split_stats(split_dir: Path) -> SplitStats:
    summary_path = split_dir / "summary.csv"
    eval_inst_path = split_dir / "eval_instances_by_suite.csv"
    train_docs_path = split_dir / "training_docs_by_suite.csv"
    total_occ, total_unique = read_summary_csv(summary_path)
    suite_unique = read_suite_kv_csv(eval_inst_path, "eval_suite", "unique_eval_instances")
    suite_docs = read_suite_kv_csv(train_docs_path, "eval_suite", "training_docs_contaminated")
    return SplitStats(
        total_occurrences=total_occ,
        total_unique_eval_instances=total_unique,
        suite_unique=suite_unique,
        suite_docs=suite_docs,
    )


def list_sources(per_source_root: Path) -> List[str]:
    if not per_source_root.exists():
        return []
    return sorted([p.name for p in per_source_root.iterdir() if p.is_dir()])


# ----------------------------
# Aggregation logic
# ----------------------------


def safe_sum(values: Iterable[float]) -> float:
    s = 0.0
    for v in values:
        if math.isfinite(v):
            s += v
    return s


def allocate_occurrences_per_suite(total_occurrences: float, suite_weights: Dict[str, float]) -> Dict[str, float]:
    total_weight = safe_sum(suite_weights.values())
    if total_weight <= 0:
        return {k: 0.0 for k in suite_weights}
    return {k: (total_occurrences * (w / total_weight)) for k, w in suite_weights.items()}


def aggregate_for_split(
    split_stats: SplitStats,
    eval_to_canonical: Mapping[str, List[str]],
    canonical_to_contam: Mapping[str, List[str]],
    included_canonicals: Set[str],
    metric: str,
) -> Dict[str, float]:
    """
    Return canonical -> value for a single split.
    metric: "occurrences" (allocated by training_docs; fallback to unique weights) or "unique"
    """
    # Determine observed suites and keep only those that map to an included canonical
    # Build reverse mapping from canonical_to_contam: suite -> list of canonicals
    suite_to_canonicals_map: Dict[str, List[str]] = defaultdict(list)
    for canonical, suites_list in canonical_to_contam.items():
        for suite in suites_list:
            suite_to_canonicals_map[suite].append(canonical)
    
    observed_suites = set(split_stats.suite_unique.keys()) | set(split_stats.suite_docs.keys())
    suites = []
    for s in sorted(observed_suites):
        # First try reverse mapping from canonical_to_contam (most authoritative)
        mapped = suite_to_canonicals_map.get(s, [])
        
        # Fallback to eval_to_canonical if not found in canonical_to_contam
        if not mapped:
            mapped = suite_to_canonicals(s, NameMaps(
                eval_to_canonical=eval_to_canonical,
                perf_to_canonical={},  # unused here
                canonical_to_contam={},  # unused here
                canonical_benchmarks=[],
                canonical_to_eval_splits={},  # unused here
            ))
        
        if any(c in included_canonicals for c in mapped):
            suites.append(s)
    if not suites:
        return {}

    # Values per suite
    suite_values: Dict[str, float] = {}
    if metric == "unique":
        for s in suites:
            suite_values[s] = float(split_stats.suite_unique.get(s, 0.0))
    else:
        # occurrences: Use training_docs_contaminated directly (not allocated from total_contamination_instances)
        # Fallback to unique_eval_instances if training_docs_contaminated is not available
        for s in suites:
            docs_val = float(split_stats.suite_docs.get(s, 0.0))
            if docs_val > 0:
                suite_values[s] = docs_val
            else:
                suite_values[s] = float(split_stats.suite_unique.get(s, 0.0))

    # Map to canonical (using the reverse mapping already built above)
    canonical_values: Dict[str, float] = defaultdict(float)
    for suite, value in suite_values.items():
        # Use reverse mapping from canonical_to_contam (most authoritative)
        # This ensures suites like "squad_mc" map ONLY to "squad:mc::gen2mc", not to "squad::xlarge"
        canonicals = suite_to_canonicals_map.get(suite, [])
        
        # Only fallback to eval_to_canonical if suite is NOT in canonical_to_contam at all
        # This prevents suites like "squad" from mapping to both canonicals when they should only map to one
        if not canonicals:
            # Check if this suite appears in ANY canonical's suite list
            suite_in_any_canonical = any(suite in suites_list for suites_list in canonical_to_contam.values())
            if not suite_in_any_canonical:
                # Suite not in canonical_to_contam, use eval_to_canonical fallback
                canonicals = suite_to_canonicals(suite, NameMaps(
                    eval_to_canonical=eval_to_canonical,
                    perf_to_canonical={},
                    canonical_to_contam={},
                    canonical_benchmarks=[],
                    canonical_to_eval_splits={},
                ))
        
        for c in canonicals:
            if c in included_canonicals:
                canonical_values[c] += float(value)
    return dict(canonical_values)


def compute_denominators_for_percent_unique(
    eval_stats_dir: Path,
    maps: NameMaps,
    included_canonicals: Set[str],
) -> Tuple[Dict[str, float], Dict[str, Dict[str, float]]]:
    """
    For canonicals whose eval split is in {test,validation,train}, sum the questions counts
    across the contributing eval suites for that canonical from the respective eval_stats CSV.
    Also return raw per-suite denominators per split for later union-capped numerator computation.
    """
    denominators: Dict[str, float] = {}
    # Load eval stats CSVs
    split_to_rows: Dict[str, Dict[str, Dict[str, str]]] = {}
    for split in ("test", "validation", "train"):
        path = eval_stats_dir / f"{split}.csv"
        table: Dict[str, Dict[str, str]] = {}
        if path.exists():
            with path.open("r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    name = (row.get("eval_name") or "").strip()
                    if not name:
                        continue
                    table[name] = row
        split_to_rows[split] = table

    # Per-split suite denominators
    suite_questions_by_split: Dict[str, Dict[str, float]] = {k: {} for k in ("test", "validation", "train")}
    for split, rows in split_to_rows.items():
        for suite, row in rows.items():
            try:
                suite_questions_by_split[split][suite] = float(row.get("questions", 0) or 0)
            except Exception:
                suite_questions_by_split[split][suite] = 0.0

    # Compute denominators for all canonicals in included_canonicals
    for canonical in included_canonicals:
        split = maps.canonical_to_eval_splits.get(canonical)
        suites = maps.canonical_to_contam.get(canonical, [])
        denom = 0.0
        
        if split in ("test", "validation", "train"):
            # For specific splits: sum questions from that split's eval_stats
            for suite in suites:
                denom += float(suite_questions_by_split.get(split, {}).get(suite, 0.0))
            # Fallback: prefix/normalized-name matching if denom is still zero (handles social_i_qa etc.)
            if denom == 0.0:
                rows = suite_questions_by_split.get(split, {})
                base = canonical_base(canonical)
                base_norm = base.replace("_", "").lower()
                for name, val in rows.items():
                    name_norm = name.replace("_", "").lower()
                    if base_norm and base_norm in name_norm:
                        denom += float(val or 0.0)
                # Special families
                if "minerva" in canonical and denom == 0.0:
                    for name, val in rows.items():
                        if name.startswith("hendrycks_math_"):
                            denom += float(val or 0.0)
                if "humaneval" in canonical and denom == 0.0:
                    for name, val in rows.items():
                        if "humaneval" in name:
                            denom += float(val or 0.0)
                if "multipl-e-humaneval" in canonical and denom == 0.0:
                    for name, val in rows.items():
                        if name.startswith("multipl_e_humaneval_"):
                            denom += float(val or 0.0)
        elif split == "all":
            # For "all" splits: sum questions across test, validation, train from eval_stats
            for suite in suites:
                suite_denom = sum(float(suite_questions_by_split.get(sp, {}).get(suite, 0.0)) for sp in ("test", "validation", "train"))
                denom += suite_denom
        
        if denom > 0:
            denominators[canonical] = denom
    return denominators, suite_questions_by_split


def load_overall_suite_uniques(overall_root: Path) -> Dict[str, Dict[str, float]]:
    """
    Load overall unique eval instance counts per suite from top-level decon stats:
    analysis/decon_stats_{test,validation,train,all}/eval_instances_by_suite.csv
    Returns: split -> { suite -> unique_count }
    """
    split_map = {
        "test": overall_root.parent / "decon_stats_test",
        "validation": overall_root.parent / "decon_stats_validation",
        "train": overall_root.parent / "decon_stats_train",
        "all": overall_root.parent / "decon_stats_all",
    }
    result: Dict[str, Dict[str, float]] = {k: {} for k in split_map}
    for split, path in split_map.items():
        try:
            csv_path = path / "eval_instances_by_suite.csv"
            if not csv_path.exists():
                continue
            with csv_path.open("r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                # Try common column names
                for row in reader:
                    suite = (row.get("eval_suite") or row.get("suite") or "").strip()
                    if not suite:
                        continue
                    val_s = (row.get("unique_eval_instances") or row.get("unique") or "").strip()
                    try:
                        val = float(val_s)
                    except Exception:
                        continue
                    result[split][suite] = val
        except Exception:
            continue
    return result


# ----------------------------
# Orchestration
# ----------------------------


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    configure_logging(args.verbose)

    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    maps = load_name_maps(args.name_maps_dir)
    included_canonicals = scan_perf_canonicals(args.perf_results_dir, maps)
    if not included_canonicals:
        logging.warning("No canonical benchmarks discovered from perf results; using all canonicals as fallback.")
        included_canonicals = set(maps.canonical_benchmarks)

    # Filter included to those with a split mapping we can evaluate
    allowed_splits = {"all", "test", "validation", "train"}
    included_canonicals = {
        c
        for c in included_canonicals
        if c in maps.canonical_to_eval_splits
        and maps.canonical_to_eval_splits.get(c) in allowed_splits
    }
    missing_split = sorted(set(maps.canonical_benchmarks) & included_canonicals - set(maps.canonical_to_eval_splits.keys()))
    if missing_split:
        logging.warning("Some canonicals omitted due to missing eval split mapping: %s", missing_split)

    sources = list_sources(args.per_source_stats_dir)
    if not sources:
        logging.error("No per-source stats found in %s", args.per_source_stats_dir)
        return 2

    # Prepare denominators for percent-unique computation
    percent_unique_denoms, suite_questions_by_split = compute_denominators_for_percent_unique(
        args.eval_stats_dir, maps, included_canonicals
    )
    # Load overall unique counts by suite per split to avoid double counting overlaps across sources
    overall_suite_uniques = load_overall_suite_uniques(args.eval_stats_dir)

    # Accumulators
    source_by_canonical: Dict[str, Dict[str, float]] = {
        s: {c: 0.0 for c in included_canonicals} for s in sources
    }
    non_eval_by_canonical: Dict[str, float] = {c: 0.0 for c in included_canonicals}
    # For percent unique numerator: use overall per-split suite uniques from top-level stats
    suite_unique_by_split: Dict[str, Dict[str, float]] = {
        "test": defaultdict(float, overall_suite_uniques.get("test", {})),
        "validation": defaultdict(float, overall_suite_uniques.get("validation", {})),
        "train": defaultdict(float, overall_suite_uniques.get("train", {})),
    }

    # Compute per source
    for source in sources:
        base_dir = args.per_source_stats_dir / source
        # Load splits
        split_dirs = {
            "all": base_dir / "all",
            "test": base_dir / "test",
            "validation": base_dir / "validation",
            "train": base_dir / "train",
        }
        split_stats: Dict[str, SplitStats] = {
            k: load_split_stats(p) for k, p in split_dirs.items()
        }

        # For each canonical, select its eval split and aggregate
        eval_values_for_source: Dict[str, float] = {c: 0.0 for c in included_canonicals}
        all_values_for_source: Dict[str, float] = {c: 0.0 for c in included_canonicals}

        # Aggregate 'all' split (used for non-eval)
        all_split_data = aggregate_for_split(
            split_stats.get("all", SplitStats(0, 0, {}, {})),
            maps.eval_to_canonical,
            maps.canonical_to_contam,
            included_canonicals,
            args.value,
        )
        for c, v in all_split_data.items():
            all_values_for_source[c] = float(v)

        for canonical in included_canonicals:
            split = maps.canonical_to_eval_splits.get(canonical, "")
            if split not in ("all", "test", "validation", "train"):
                continue
            agg = aggregate_for_split(
                split_stats.get(split, SplitStats(0, 0, {}, {})),
                maps.eval_to_canonical,
                maps.canonical_to_contam,
                {canonical},
                args.value,
            )
            eval_values_for_source[canonical] = float(agg.get(canonical, 0.0))
        # Do not sum per-source uniques here to avoid overlap; rely on overall_suite_uniques instead.

        # Save eval split values into matrix
        for canonical, value in eval_values_for_source.items():
            source_by_canonical[source][canonical] = value

        # Compute non-eval (all − eval)
        if args.include_non_eval == "true":
            for canonical in included_canonicals:
                delta = float(all_values_for_source.get(canonical, 0.0)) - float(
                    eval_values_for_source.get(canonical, 0.0)
                )
                if delta < 0:
                    delta = 0.0
                non_eval_by_canonical[canonical] += delta

    # Row/column totals
    row_totals: Dict[str, float] = {
        s: safe_sum(source_by_canonical[s].values()) for s in sources
    }
    col_totals: Dict[str, float] = {
        c: safe_sum(source_by_canonical[s][c] for s in sources) for c in included_canonicals
    }

    # Sorting: sources by row totals
    sorted_sources = sorted(sources, key=lambda s: (row_totals.get(s, 0.0), s), reverse=True)
    
    # Sorting: canonicals grouped by eval split (val/test first, then all), then by column totals
    def canonical_sort_key(c: str) -> tuple:
        split = maps.canonical_to_eval_splits.get(c, "")
        # Primary sort: val/test come first (group 0), all comes second (group 1)
        if split in ("validation", "test"):
            group = 0
        elif split == "all":
            group = 1
        else:
            group = 2  # Other splits go last
        # Secondary sort: by column total (descending)
        total = col_totals.get(c, 0.0)
        return (group, -total, c)  # Negative total for descending order
    
    sorted_canonicals = sorted(included_canonicals, key=canonical_sort_key)
    # Limit to top 10 rows
    sorted_sources = sorted_sources[:10]
    if args.topk_cols is not None:
        sorted_canonicals = sorted_canonicals[: args.topk_cols]

    # Write matrix.csv
    matrix_csv = output_dir / "matrix.csv"
    with matrix_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["source", *sorted_canonicals])
        for s in sorted_sources:
            row = [s] + [source_by_canonical[s].get(c, 0.0) for c in sorted_canonicals]
            writer.writerow(row)
    logging.info("Wrote %s", matrix_csv)

    # Percent unique eval contaminated per canonical
    # Numerator: unique counts from decon_stats_{split}/eval_instances_by_suite.csv
    # Denominator: total questions from eval_stats/{split}.csv
    unique_eval_numerators: Dict[str, float] = {}
    for c in sorted_canonicals:
        split = maps.canonical_to_eval_splits.get(c, "")
        suites = list(maps.canonical_to_contam.get(c, []))
        
        if split in ("test", "validation", "train"):
            # For specific splits: use unique counts from that split's decon_stats
            split_uniques = overall_suite_uniques.get(split, {})
            # Add family-based suites if needed
            if "minerva" in c:
                suites += [s for s in split_uniques.keys() if s.startswith("hendrycks_math_")]
            if "codex_humaneval" in c:
                suites += [s for s in split_uniques.keys() if "humaneval" in s]
            if "multipl-e-humaneval" in c:
                suites += [s for s in split_uniques.keys() if s.startswith("multipl_e_humaneval_")]
            # Deduplicate
            suites = list(dict.fromkeys(suites))
            num_sum = 0.0
            for suite in suites:
                # Use unique count directly from decon_stats for this split
                num_sum += float(split_uniques.get(suite, 0.0))
            unique_eval_numerators[c] = num_sum
        elif split == "all":
            # For "all" splits: use unique counts from decon_stats_all
            all_uniques = overall_suite_uniques.get("all", {})
            num_sum = 0.0
            for suite in suites:
                # Use unique count directly from decon_stats_all
                num_sum += float(all_uniques.get(suite, 0.0))
            unique_eval_numerators[c] = num_sum
        else:
            unique_eval_numerators[c] = 0.0

    percent_unique: Dict[str, Optional[float]] = {}
    for c in sorted_canonicals:
        split = maps.canonical_to_eval_splits.get(c, "")
        suites = maps.canonical_to_contam.get(c, [])
        
        # Get numerator (already computed above)
        num = unique_eval_numerators.get(c, 0.0)
        
        # Compute denominator: total questions from eval_stats
        denom = 0.0
        if c in percent_unique_denoms:
            denom = percent_unique_denoms[c]
        else:
                if split in ("test", "validation", "train"):
                    # For specific splits: sum questions from that split's eval_stats
                    denom = sum(float(suite_questions_by_split.get(split, {}).get(s, 0.0)) for s in suites)
                    # family-based fallback when denom is zero
                    if denom == 0.0:
                        rows = suite_questions_by_split.get(split, {})
                        if "minerva" in c:
                            denom = sum(float(rows.get(name, 0.0)) for name in rows.keys() if name.startswith("hendrycks_math_"))
                        if "codex_humaneval" in c and denom == 0.0:
                            denom = sum(float(rows.get(name, 0.0)) for name in rows.keys() if "humaneval" in name)
                        if "multipl-e-humaneval" in c and denom == 0.0:
                            denom = sum(float(rows.get(name, 0.0)) for name in rows.keys() if name.startswith("multipl_e_humaneval_"))
                elif split == "all":
                    # For "all" splits: sum questions across test, validation, train from eval_stats
                    denom = 0.0
                    for s in suites:
                        s_denom = sum(float(suite_questions_by_split.get(sp, {}).get(s, 0.0)) for sp in ("test", "validation", "train"))
                        denom += s_denom
        
        # Calculate percentage: unique / total * 100
        if denom > 0:
            pct = (num / denom) * 100.0
            percent_unique[c] = min(max(pct, 0.0), 100.0)
        else:
            percent_unique[c] = None

    # Write margins.csv
    margins_csv = output_dir / "margins.csv"
    with margins_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["type", "name", "value"])
        for s in sorted_sources:
            writer.writerow(["row_total", s, row_totals.get(s, 0.0)])
        # Choose per-canonical totals based on metric
        if args.value == "unique":
            col_total_values = {c: unique_eval_numerators.get(c, 0.0) for c in sorted_canonicals}
        else:
            col_total_values = {c: col_totals.get(c, 0.0) for c in sorted_canonicals}
        for c in sorted_canonicals:
            writer.writerow(["col_total", c, col_total_values.get(c, 0.0)])
        if args.include_non_eval == "true":
            for c in sorted_canonicals:
                writer.writerow(["non_eval_total", c, non_eval_by_canonical.get(c, 0.0)])
        for c in sorted_canonicals:
            val = percent_unique.get(c)
            writer.writerow(["percent_unique_eval_contam", c, "" if val is None else val])
        # Add performance metrics to margins: contaminated, decontaminated, and drop
        def parse_perf_files_for_margins() -> Dict[str, Tuple[Optional[float], Optional[float], Optional[float]]]:
            results: Dict[str, Tuple[Optional[float], Optional[float], Optional[float]]] = {}
            for p in sorted(args.perf_results_dir.glob("*.csv")):
                try:
                    with p.open("r", encoding="utf-8") as fh:
                        data = fh.read()
                        fh.seek(0)
                        rdr = csv.DictReader(data.splitlines())
                        rows = list(rdr)
                        if not rows:
                            continue
                        fields = rdr.fieldnames or []
                        decon = None
                        base = None
                        if "name" in fields:
                            base_cand = [r for r in rows if _is_base_name(r.get("name") or "")]
                            decon_cand = [r for r in rows if _is_decon_name(r.get("name") or "")]
                            if base_cand:
                                base = base_cand[0]
                            if decon_cand:
                                decon = decon_cand[0]
                        if decon is None or base is None:
                            if len(rows) >= 2:
                                if base is None and decon is None:
                                    base, decon = rows[0], rows[1]
                                elif base is None:
                                    other = rows[0] if rows[0] is not decon else (rows[1] if len(rows) > 1 else None)
                                    base = other or decon
                                elif decon is None:
                                    other = rows[0] if rows[0] is not base else (rows[1] if len(rows) > 1 else None)
                                    decon = other or base
                            else:
                                continue
                        if decon is None:
                            continue
                        cols = [c for c in fields if c != "name"]
                        for col in cols:
                            canonical = maps.perf_to_canonical.get(col)
                            if not canonical:
                                canonical = col if col in maps.canonical_benchmarks else None
                            if not canonical:
                                continue
                            try:
                                base_v = float((base or {}).get(col, "nan"))
                                decon_v = float(decon.get(col, "nan"))
                            except Exception:
                                continue
                            drop_v = base_v - decon_v if (np.isfinite(base_v) and np.isfinite(decon_v)) else np.nan
                            if canonical not in results or (not np.isfinite(results[canonical][0]) and np.isfinite(base_v)):
                                results[canonical] = (base_v, decon_v, drop_v)
                except Exception:
                    continue
            return results
        perf_for_margins = parse_perf_files_for_margins()
        for c in sorted_canonicals:
            base_v, decon_v, drop_v = perf_for_margins.get(c, (None, None, None))
            if base_v is not None and np.isfinite(base_v):
                writer.writerow(["perf_contam", c, base_v])
            if decon_v is not None and np.isfinite(decon_v):
                writer.writerow(["perf_decon", c, decon_v])
            if drop_v is not None and np.isfinite(drop_v):
                writer.writerow(["perf_drop", c, drop_v])
    logging.info("Wrote %s", margins_csv)

    # Write metadata.json
    metadata_json = output_dir / "metadata.json"
    metadata = {
        "args": {
            "value": args.value,
            "include_non_eval": args.include_non_eval,
            "log_scale": bool(args.log_scale),
            "topk_cols": args.topk_cols,
            "topk_rows": args.topk_rows,
        },
        "included_canonicals": sorted(sorted_canonicals),
        "sources": sorted(sorted_sources),
        "notes": [
            "Occurrences use training_docs_contaminated directly (not allocated from total_contamination_instances); fallback to unique_eval_instances if docs are unavailable.",
            "Percent unique uses 'unique' metric over eval split as numerator; denominator from eval_stats for canonical's split only when available.",
        ],
    }
    with metadata_json.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, sort_keys=True)
    logging.info("Wrote %s", metadata_json)

    # Plot heatmap
    if plt is None:
        logging.warning("matplotlib is not available; skipping plot.")
        return 0

    # Build matrix in sorted order
    values_2d = np.array(
        [[float(source_by_canonical[s].get(c, 0.0)) for c in sorted_canonicals] for s in sorted_sources],
        dtype=float,
    )

    # Drop all-zero rows/cols for the plot aesthetics (do not change CSV outputs)
    row_mask = values_2d.sum(axis=1) > 0
    col_mask = values_2d.sum(axis=0) > 0
    if not np.any(row_mask) or not np.any(col_mask):
        logging.warning("All entries are zero; skipping plot rendering.")
        return 0
    values_2d = values_2d[row_mask][:, col_mask]
    plot_sources = [s for s, keep in zip(sorted_sources, row_mask) if keep]
    plot_canonicals = [c for c, keep in zip(sorted_canonicals, col_mask) if keep]
    
    # Find the split point between val/test and all groups
    # Count how many val/test benchmarks are in the filtered list
    val_test_count = 0
    for c in plot_canonicals:
        split = maps.canonical_to_eval_splits.get(c, "")
        if split in ("validation", "test"):
            val_test_count += 1
        elif split == "all":
            break  # Once we hit "all", we're done counting val/test
    # val_test_count is the index where "all" group starts (0-indexed)

    # Build display labels (filtered)
    source_display = build_source_display_names(args.per_source_stats_dir, args.midtrain_csv, args.name_maps_dir)
    y_labels = [source_display.get(s, s) for s in plot_sources]
    x_labels = [format_canonical_display(c, maps) for c in plot_canonicals]

    # Figure sizing tuned for readability with dynamic margins
    cell_in = 0.45
    width_in = max(8.0, len(plot_canonicals) * cell_in + 4.0)
    height_in = max(6.0, len(plot_sources) * cell_in + 3.5)
    # Add extra space for vertical x-axis labels and y-axis labels
    width_in += 1.0  # Extra width for vertical x-axis labels
    height_in += 1.5  # Extra height for y-axis labels
    fig = plt.figure(figsize=(width_in, height_in))
    # Determine whether to use log scale automatically if dynamic range is extreme
    use_log = bool(args.log_scale)
    flat_vals = values_2d[values_2d > 0].ravel().tolist()
    if not use_log and flat_vals:
        vmin_pos = min(flat_vals)
        vmax_pos = max(flat_vals)
        if vmax_pos > 0 and vmin_pos > 0 and (vmax_pos / max(vmin_pos, 1e-12) >= 100.0):
            use_log = True

    # Build marginal totals for the filtered matrix
    row_totals_plot = values_2d.sum(axis=1)
    col_totals_plot = values_2d.sum(axis=0)
    n_rows, n_cols = values_2d.shape

    # Mask arrays (zeros/NaNs are transparent)
    masked_main = np.ma.masked_where(~np.isfinite(values_2d) | (values_2d <= 0), values_2d)
    left_arr = row_totals_plot.reshape(-1, 1)
    left_cond = (~np.isfinite(left_arr)) | (left_arr <= 0)
    masked_left = np.ma.masked_where(left_cond, left_arr)
    bottom_arr = col_totals_plot.reshape(1, -1)
    bottom_cond = (~np.isfinite(bottom_arr)) | (bottom_arr <= 0)
    masked_bottom = np.ma.masked_where(bottom_cond, bottom_arr)
    # Use white-to-hot-pink colormap (white at zero, hot pink at high values)
    from matplotlib.colors import LinearSegmentedColormap
    colors = ['white', '#FF69B4']  # white to hot pink (lighter than deep pink)
    n_bins = 256
    cmap = LinearSegmentedColormap.from_list('white_to_pink', colors, N=n_bins)
    cmap.set_bad(color="white", alpha=0.0)

    # Determine color norm for main and left (instance counts). Marginal rows may use separate norms.
    data_for_range = np.concatenate(
        [
            masked_main.compressed() if masked_main.count() > 0 else np.array([]),
            masked_left.compressed() if masked_left.count() > 0 else np.array([]),
        ]
    )
    if data_for_range.size == 0:
        vmin, vmax = 1e-6, 1.0
    else:
        vmin, vmax = float(np.min(data_for_range)), float(np.max(data_for_range))
        vmin = max(vmin, 1e-6)

    # Grid layout: left totals column, main heatmap, and 2 marginal rows
    # Use gridspec with spacing to create gaps for manual positioning
    gs = fig.add_gridspec(2, 2, width_ratios=[1, n_cols], height_ratios=[n_rows, 4], 
                          wspace=0.1, hspace=0.1)
    ax_left = fig.add_subplot(gs[0, 0])
    ax_main = fig.add_subplot(gs[0, 1], sharey=ax_left)
    ax_bottom = fig.add_subplot(gs[1, 1], sharex=ax_main)
    ax_corner = fig.add_subplot(gs[1, 0]); ax_corner.axis("off")

    # Draw heatmaps with explicit extent to ensure alignment
    # Define extent explicitly: [left, right, bottom, top] in data coordinates
    main_extent = [-0.5, n_cols - 0.5, n_rows - 0.5, -0.5]  # Note: top > bottom for imshow
    left_extent = [-0.5, 0.5, n_rows - 0.5, -0.5]
    
    if use_log and LogNorm is not None:
        norm = LogNorm(vmin=vmin, vmax=max(vmax, vmin * 10))
        im_main = ax_main.imshow(masked_main, aspect="equal", cmap=cmap, norm=norm, extent=main_extent)
        # Use aspect="equal" for left margin to maintain square cells matching main heatmap
        im_left = ax_left.imshow(masked_left, aspect="equal", cmap=cmap, norm=norm, extent=left_extent)
        im_total = None
    else:
        norm = None
        im_main = ax_main.imshow(masked_main, aspect="equal", cmap=cmap, extent=main_extent)
        # Use aspect="equal" for left margin to maintain square cells matching main heatmap
        im_left = ax_left.imshow(masked_left, aspect="equal", cmap=cmap, extent=left_extent)
        im_total = None

    # Ensure exact x alignment between main and bottom strips
    ax_main.set_xlim(-0.5, n_cols - 0.5)
    ax_main.set_ylim(n_rows - 0.5, -0.5)  # Match imshow extent (top > bottom)
    # Ticks and labels
    ax_main.set_xticks(range(n_cols))
    ax_main.set_xticklabels([], rotation=45, ha="right", fontsize=14)
    ax_main.set_yticks(range(n_rows))
    # Show row labels only on the left totals panel to reduce clutter
    ax_main.set_yticklabels([])
    ax_main.tick_params(axis="y", labelleft=False, left=False)
    # Hide all x ticks/labels on main; keep only on the bottom totals row
    ax_main.tick_params(axis="x", which="both", labelbottom=False, bottom=False, top=False, length=0)
    ax_main.set_xlabel("")
    ax_main.set_ylabel("")
    # No plot title - title is on colorbar instead

    ax_left.set_xticks([])
    ax_left.set_xlabel("Total\ncontam", fontsize=14, labelpad=5)
    ax_left.set_yticks(range(n_rows))
    ax_left.set_yticklabels(y_labels, fontsize=14, rotation=0, ha="right", va="center")
    ax_left.set_ylabel("Midtraining Data Sources", fontsize=18, fontweight='bold', labelpad=10)
    # CRITICAL: Set y-limits to match main heatmap exactly for same cell size
    ax_left.set_ylim(n_rows - 0.5, -0.5)  # Match main heatmap y-limits exactly
    ax_left.set_xlim(-0.5, 0.5)  # Match left extent
    # Thicker separator line on the right edge of totals column
    for spine in ("right",):
        ax_left.spines[spine].set_linewidth(2.0)
    
    # Ensure all sections have the same cell size and are properly aligned
    # CRITICAL: Set limits FIRST before positioning to ensure aspect="equal" calculates correctly
    ax_left.set_ylim(n_rows - 0.5, -0.5)  # Must match main exactly
    ax_left.set_xlim(-0.5, 0.5)
    
    # Draw to get positions after limits are set
    fig.canvas.draw()
    pos_main = ax_main.get_position()
    pos_left_initial = ax_left.get_position()
    pos_bottom_initial = ax_bottom.get_position()
    
    # <-- ADJUST SPACING HERE: Change these values to tighten/loosen gaps
    # Left margin spacing: increase to move left column further right (more gap)
    # Bottom margin spacing: increase to move bottom rows further up (more gap)
    fig_width_inches = fig.get_figwidth()
    fig_height_inches = fig.get_figheight()
    left_spacing_cm = 0.5  # <-- ADJUST THIS: spacing between left margin and main (in cm)
    bottom_spacing_cm = 1.0  # <-- ADJUST THIS: spacing between main and bottom (in cm)
    
    left_spacing = (left_spacing_cm * 0.3937) / fig_width_inches  # Convert cm to figure coords
    bottom_spacing = (bottom_spacing_cm * 0.3937) / fig_height_inches  # Convert cm to figure coords
    
    # Position left margin: use EXACT same height and y-position as main heatmap
    # Move right by spacing amount, but keep exact height match
    ax_left.set_position([pos_left_initial.x0 + left_spacing, pos_main.y0, pos_left_initial.width, pos_main.height])
    # Re-apply limits to ensure they're maintained
    ax_left.set_ylim(n_rows - 0.5, -0.5)
    ax_left.set_xlim(-0.5, 0.5)
    
    # Force a final draw and re-check/force alignment
    fig.canvas.draw()
    pos_main_final = ax_main.get_position()
    pos_left_final = ax_left.get_position()
    
    # Calculate the correct width for left margin to maintain square cells
    # With aspect="equal", width should be: height * (x_extent / y_extent)
    # Left extent: x from -0.5 to 0.5 (width=1), y from n_rows-0.5 to -0.5 (height=n_rows)
    # So width should be: height * (1 / n_rows)
    left_target_height = pos_main_final.height
    left_target_width = left_target_height * (1.0 / n_rows)  # Maintain square cells
    
    # Force left margin to match main height exactly and calculate correct width
    ax_left.set_position([pos_left_final.x0, pos_main_final.y0, left_target_width, left_target_height])
    ax_left.set_ylim(n_rows - 0.5, -0.5)  # Re-apply limits
    ax_left.set_xlim(-0.5, 0.5)
    
    # Position bottom margin: use EXACT same width and x-position as main heatmap
    bot_rows = 2  # Only % contam and Perf Δ
    main_cell_height = pos_main.height / n_rows
    # Increase height to accommodate vertical x-axis labels
    # Calculate extra space needed: estimate label height + padding
    fig_height_inches = fig.get_figheight()
    extra_height_fraction = 0.25 / fig_height_inches  # ~0.25 inches for vertical labels
    bottom_height = main_cell_height * bot_rows + extra_height_fraction
    bottom_y0 = pos_bottom_initial.y0 + bottom_spacing
    ax_bottom.set_position([pos_main.x0, bottom_y0, pos_main.width, bottom_height])
    # Ensure limits match exactly
    ax_bottom.set_xlim(-0.5, n_cols - 0.5)
    ax_bottom.set_ylim(-0.5, bot_rows - 0.5)

    # Non-eval data no longer displayed in bottom rows (simplified figure)

    # Percent unique (0..100) row
    percent_vec = np.array(
        [float(percent_unique.get(c, np.nan)) if percent_unique.get(c) is not None else np.nan for c in plot_canonicals],
        dtype=float,
    )
    percent_arr = percent_vec.reshape(1, -1)
    percent_cond = ~np.isfinite(percent_arr)
    masked_percent = np.ma.masked_where(percent_cond, percent_arr)
    # (Will draw as part of composite bottom)

    # Performance drop row
    def parse_perf_files() -> Dict[str, Tuple[Optional[float], Optional[float], Optional[float]]]:
        out: Dict[str, Tuple[Optional[float], Optional[float], Optional[float]]] = {}
        for p in sorted(args.perf_results_dir.glob("*.csv")):
            try:
                with p.open("r", encoding="utf-8") as f:
                    text = f.read()
                    f.seek(0)
                    reader = csv.DictReader(text.splitlines())
                    rows = list(reader)
                    if not rows:
                        continue
                    fieldnames = reader.fieldnames or []
                    # Determine base/decon rows
                    decon_row = None
                    base_row = None
                    if "name" in fieldnames:
                        base_candidates = [r for r in rows if _is_base_name(r.get("name") or "")]
                        decon_candidates = [r for r in rows if _is_decon_name(r.get("name") or "")]
                        if base_candidates:
                            base_row = base_candidates[0]
                        if decon_candidates:
                            decon_row = decon_candidates[0]
                    # Fallback: assume first is base, second is decon if two rows exist
                    if base_row is None and decon_row is None:
                        if len(rows) >= 2:
                            base_row, decon_row = rows[0], rows[1]
                        else:
                            continue
                    elif base_row is None and decon_row is not None and len(rows) >= 2:
                        # pick the other row as base if possible
                        other = rows[0] if rows[0] is not decon_row else (rows[1] if len(rows) > 1 else None)
                        base_row = other or decon_row
                    elif decon_row is None and base_row is not None and len(rows) >= 2:
                        other = rows[0] if rows[0] is not base_row else (rows[1] if len(rows) > 1 else None)
                        decon_row = other or base_row
                    if decon_row is None:
                        continue
                    cols = [c for c in fieldnames if c != "name"]
                    for col in cols:
                        # Map column to canonical
                        canonical = maps.perf_to_canonical.get(col)
                        if not canonical:
                            # Fallback to identity if column already canonical
                            canonical = col if col in maps.canonical_benchmarks else None
                        if not canonical:
                            continue
                        try:
                            base_v = float((base_row or {}).get(col, "nan"))
                            decon_v = float(decon_row.get(col, "nan"))
                        except Exception:
                            continue
                        drop_v = base_v - decon_v if (np.isfinite(base_v) and np.isfinite(decon_v)) else np.nan
                        # Prefer keeping an existing entry if already filled; otherwise set
                        if canonical not in out or (not np.isfinite(out[canonical][0]) and np.isfinite(base_v)):
                            out[canonical] = (base_v, decon_v, drop_v)
            except Exception:
                continue
        return out
    perf_metrics = parse_perf_files()
    drop_vec = np.array([float(perf_metrics.get(c, (np.nan, np.nan, np.nan))[2]) for c in plot_canonicals], dtype=float)
    drop_arr = drop_vec.reshape(1, -1)
    drop_cond = ~np.isfinite(drop_arr)
    masked_drop = np.ma.masked_where(drop_cond, drop_arr)
    # (Will draw as part of composite bottom)

    # Build bottom rows: only % unique and Perf Δ (no heatmap coloring)
    # Row 0: % unique, Row 1: Perf Δ
    bottom_data = np.vstack([
        percent_arr,  # % unique (row 0)
        drop_arr,      # Perf Δ (row 1)
    ])
    # Use explicit extent matching main heatmap x-extent exactly
    # y-extent: [-0.5, 1.5] so row 0 is at bottom (closer to main), row 1 at top
    bottom_extent = [-0.5, n_cols - 0.5, -0.5, bot_rows - 0.5]  # [left, right, bottom, top]
    # Draw bottom rows with a light gray background (no heatmap coloring)
    ax_bottom.imshow(np.ones_like(bottom_data), aspect="equal", cmap='gray', vmin=0.9, vmax=1.0, extent=bottom_extent, origin='lower', alpha=0.1)
    ax_bottom.set_xlim(-0.5, n_cols - 0.5)
    ax_bottom.set_ylim(-0.5, bot_rows - 0.5)
    
    # Final alignment check - ensure bottom matches main exactly
    fig.canvas.draw()
    pos_m_final = ax_main.get_position()
    pos_b_final = ax_bottom.get_position()
    # Fine-tune x alignment if needed (should already be aligned, but check)
    # <-- FINE-TUNE X ALIGNMENT HERE if bottom rows are slightly off:
    x_fine_tune = 0.0 # <-- ADJUST THIS: positive moves right, negative moves left (in figure coords)
    y_fine_tune = 0.03  # <-- ADJUST THIS: positive moves up, negative moves down (in figure coords)
    ax_bottom.set_position([pos_m_final.x0 + x_fine_tune, pos_b_final.y0 + y_fine_tune, pos_m_final.width, pos_b_final.height])
    # Redraw after position change
    fig.canvas.draw()
    ax_bottom.set_yticks([0, 1])
    ax_bottom.set_yticklabels(["% contam", "Perf Δ"], fontsize=14, rotation=0, va="center")
    # Remove default xlabel, we'll position it manually
    ax_bottom.set_xlabel("")
    # Keep y labels on the left; ensure they do not overflow by slightly reducing font size
    ax_bottom.tick_params(axis="y", labelleft=True, labelright=False, labelsize=13, pad=5, left=True)
    ax_bottom.set_xticks(range(n_cols))
    ax_bottom.set_xticklabels(x_labels, rotation=90, ha="center", va="top", fontsize=12)
    ax_bottom.tick_params(axis="x", labelbottom=True, bottom=True, top=False)
    
    # Manually position the x-axis label: right and up
    pos_bottom = ax_bottom.get_position()
    # Position label: right side of the plot, slightly above the bottom
    label_x = pos_bottom.x1 - 0.35  # Move right (subtract from right edge)
    label_y = pos_bottom.y0 - 0.2  # Move up (add to bottom)
    fig.text(label_x, label_y, "Benchmark (Metric)", 
            ha="right", va="bottom", fontsize=18, fontweight='bold', 
            transform=fig.transFigure)
    for spine in ("top",):
        ax_bottom.spines[spine].set_linewidth(2.0)

    # Draw vertical separator line between val/test and all groups
    if val_test_count > 0 and val_test_count < n_cols:
        # Draw line at the boundary between groups (between last val/test and first all)
        # x position is at val_test_count - 0.5 (between columns)
        split_x = val_test_count - 0.5
        # Draw on main heatmap (full height)
        ax_main.axvline(x=split_x, color='black', linewidth=3, linestyle='-', zorder=10)
        # Draw on bottom marginal rows (full height of bottom section)
        ax_bottom.axvline(x=split_x, color='black', linewidth=3, linestyle='-', zorder=10)
        
        # Add label above bottom margin rows: "Evaluated splits: Val/Test       All"
        # Position it between main heatmap and bottom margin rows
        # Calculate positions: "Val/Test" should align with left side, "All" with right side
        pos_main = ax_main.get_position()
        pos_bottom = ax_bottom.get_position()
        # Label goes above bottom margin, centered vertically between main and bottom
        label_y = (pos_main.y0 + pos_bottom.y1) / 2
        
        # Calculate x positions for alignment
        # Left side: align with start of val/test group (x=0 in data coordinates)
        # Right side: align with start of all group (x=val_test_count in data coordinates)
        # Convert data coordinates to figure coordinates
        x_left_data = -0.5  # Start of first column
        x_right_data = val_test_count - 0.5  # Start of all group
        
        # Get transform to convert from data to figure coordinates
        trans_main = ax_main.transData + fig.transFigure.inverted()
        x_left_fig, _ = trans_main.transform((x_left_data, 0))
        x_right_fig, _ = trans_main.transform((x_right_data, 0))
        
        # Add label with manual spacing to align text
        # "Evaluated splits: Val/Test" on left, "All" on right
        # Use figure coordinates for positioning
        fig.text(x_left_fig, label_y, "Evaluated splits:              Val/Test", 
                ha="left", va="center", fontsize=14, fontweight='bold', 
                transform=fig.transFigure)
        fig.text(x_right_fig, label_y, "                            All", 
                ha="left", va="center", fontsize=14, fontweight='bold', 
                transform=fig.transFigure)
    
    # Remove colorbar - use right spine label instead
    # cbar = plt.colorbar(im_main, ax=[ax_main, ax_left], fraction=0.046, pad=0.04)
    # cbar.set_label("Occurrences of benchmark contamination", fontsize=15)
    
    # Add label to right spine
    ax_main.spines['right'].set_visible(True)
    ax_main.yaxis.set_label_position("right")
    ax_main.set_ylabel("Occurrences Of Contamination", fontsize=18, fontweight='bold', rotation=-90, va="bottom")

    # Optionally annotate cells for small matrices
    def _format_val(v: float) -> str:
        if v >= 1e3:  # Switch to scientific notation at 1000 (3 digits)
            s = f"{v:.0e}"  # 1 significant digit, no decimal
            # Normalize exponent style: 1e+06 -> 1e6, 2e+03 -> 2e3
            s = s.replace("e+0", "e").replace("e+", "e").replace("e-0", "e-")
            return s
        return f"{int(round(v))}"

    def _contrast_color(v: float, norm_obj, cmap_obj) -> str:
        try:
            t = float(norm_obj(v)) if norm_obj is not None else (v - vmin) / (max(vmax - vmin, 1e-12))
            r, g, b, _ = cmap_obj(t)
            # Relative luminance approximation
            luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
            return "black" if luminance > 0.6 else "white"
        except Exception:
            return "white"

    if (n_rows) <= 30 and (n_cols) <= 40:
        # Main matrix annotations - show all values including zeros
        for i in range(n_rows):
            for j in range(n_cols):
                val = values_2d[i, j]
                if np.isfinite(val):
                    # Show zero values as "0", positive values as formatted
                    if val == 0:
                        text_val = "0"
                    else:
                        text_val = _format_val(val)
                    ax_main.text(
                        j,
                        i,
                        text_val,
                        ha="center",
                        va="center",
                        color=_contrast_color(val, norm if use_log else (lambda x: (x - vmin) / (max(vmax - vmin, 1e-12))), cmap),
                        fontsize=12,
                        clip_on=True,
                    )
        # Left totals annotations
        for i in range(n_rows):
            val = row_totals_plot[i]
            if np.isfinite(val):
                # Show zero values as "0", positive values as formatted
                if val == 0:
                    text_val = "0"
                else:
                    text_val = _format_val(val)
                ax_left.text(
                    0,
                    i,
                    text_val,
                    ha="center",
                    va="center",
                    color=_contrast_color(val, norm if use_log else (lambda x: (x - vmin) / (max(vmax - vmin, 1e-12))), cmap),
                    fontsize=12,
                    clip_on=False,  # Don't clip to allow text to be visible
                )
        # Bottom row annotations: only % contam and Perf Δ
        # Format helpers to fit 3 characters max
        def format_percent(v: float) -> str:
            """Format percentage to fit 3 chars: '12%' or '-1%'"""
            rounded = round(v)
            if rounded < 0:
                # Negative: max 1 digit before %, e.g., "-1%"
                return f"{rounded}%"
            else:
                # Positive: max 2 digits before %, e.g., "12%"
                return f"{rounded}%"
        
        def format_perf_delta(v: float) -> str:
            """Format performance delta to always use 1 decimal place for all values"""
            # Always format with one decimal place: 14 -> 14.0, -1 -> -1.0, 0 -> 0.0
            # For values >= 10, round down slightly to fit better (e.g., 14.0 -> 13.9)
            abs_v = abs(v)
            
            if abs_v >= 10:
                # Round down by 0.1 to fit: 14.0 -> 13.9, 15.0 -> 14.9, etc.
                rounded_val = round(v, 1) - 0.1 if v > 0 else round(v, 1) + 0.1
                return f"{rounded_val:.1f}"
            else:
                # Always show one decimal place
                return f"{v:.1f}"
        
        # Row 0: % contam - use medium pink for values > 10%
        dark_pink = '#C71585'  # Medium violet red (lighter than dark magenta)
        for j in range(n_cols):
            val = percent_vec[j]
            if np.isfinite(val):
                # Use dark pink for values > 10%, black otherwise
                text_color = dark_pink if val > 10.0 else "black"
                ax_bottom.text(
                    j,
                    0,
                    format_percent(val),
                    ha="center",
                    va="center",
                    color=text_color,
                    fontsize=12,
                    clip_on=True,
                )
        # Row 1: Perf Δ - use dark pink for values > 1.0
        for j, c in enumerate(plot_canonicals):
            base_v, decon_v, _ = perf_metrics.get(c, (np.nan, np.nan, np.nan))
            if np.isfinite(base_v) and np.isfinite(decon_v):
                perf_delta = base_v - decon_v
                # Use dark pink for values > 1.0, black otherwise
                text_color = dark_pink if perf_delta > 1.0 else "black"
                ax_bottom.text(
                    j,
                    1,
                    format_perf_delta(perf_delta),
                    ha="center",
                    va="center",
                    color=text_color,
                    fontsize=12,
                    clip_on=True,
                )

    # Final draw before saving
    fig.canvas.draw()
    
    # Save with bbox_inches='tight' to ensure all labels are visible
    # Increased padding to accommodate larger bold labels
    plot_png = output_dir / "heatmap.png"
    fig.savefig(plot_png, dpi=200, bbox_inches='tight', pad_inches=0.2)
    logging.info("Wrote %s", plot_png)
    
    # Save PDF version
    plot_pdf = output_dir / "heatmap.pdf"
    fig.savefig(plot_pdf, bbox_inches='tight', pad_inches=0.2)
    logging.info("Wrote %s", plot_pdf)
    
    if args.save_svg:
        plot_svg = output_dir / "heatmap.svg"
        fig.savefig(plot_svg, bbox_inches='tight', pad_inches=0.2)
        logging.info("Wrote %s", plot_svg)
    plt.close(fig)

    return 0


if __name__ == "__main__":
    sys.exit(main())


