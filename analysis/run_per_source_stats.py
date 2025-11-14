#!/usr/bin/env python3
"""
Run contamination stats review for each midtrain data source listed in
`midtrain_decon_map.csv`.

For every source we invoke:
    cargo run --release -- review --stats [--split SPLIT] DATA_PATH --output-dir OUT_DIR

We run once per split (test, validation, train) and once with no split to capture
the overall totals.
"""

from __future__ import annotations

import argparse
import csv
import re
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List, Optional, Tuple
from urllib.parse import urlparse


DEFAULT_SPLITS: Tuple[str, ...] = ("test", "validation", "train")
S3_URI_PATTERN = re.compile(r"s3://[^\s,]+")


def slugify_name(name: str) -> str:
    """Convert a human-friendly name into a filesystem-safe slug."""
    slug = re.sub(r"[^\w.-]+", "_", name.strip())
    slug = re.sub(r"_+", "_", slug).strip("_")
    return slug or "unknown"


def candidates_from_path(value: str) -> List[str]:
    """Generate possible dataset directory names from an S3/GS path string."""
    results: List[str] = []
    if not value:
        return results

    # Handle multi-line CSV fields (commas/newlines inside quotes).
    for raw in value.splitlines():
        piece = raw.strip().strip(",")
        if not piece:
            continue

        piece = piece.rstrip("/")
        piece = piece.replace("\\", "/")
        piece = piece.removeprefix("s3://").removeprefix("gs://")

        segments = [seg for seg in piece.split("/") if seg]
        if not segments:
            continue

        last = segments[-1]
        if "*" in last:
            last = last.replace("*", "")
        if last and last != ".":
            results.append(last)

        if "sources" in segments:
            tail = segments[segments.index("sources") + 1 :]
            if tail:
                joined = "-".join(tail)
                results.append(joined)
                results.append(f"ai2-llm-pretraining-data-sources-{joined}")

    # Deduplicate while preserving order
    seen = set()
    ordered: List[str] = []
    for candidate in results:
        if candidate not in seen:
            seen.add(candidate)
            ordered.append(candidate)
    return ordered


def uri_to_local_path(uri: str, root: Path) -> Optional[Path]:
    """Map an S3 URI to a local path rooted under the reports directory."""
    if not uri:
        return None

    cleaned = uri.strip().strip(",")
    if not cleaned:
        return None

    cleaned = cleaned.replace("*", "")
    parsed = urlparse(cleaned)
    if not parsed.scheme or not parsed.netloc or not parsed.path:
        return None

    candidate = (root / parsed.netloc / parsed.path.lstrip("/")).resolve()

    try:
        candidate.relative_to(root.resolve())
    except ValueError:
        return None

    while candidate != root and not candidate.exists():
        candidate = candidate.parent

    if candidate == root:
        return None
    return candidate


def lookup_directory(candidate: str, mapping: dict[str, Path]) -> Optional[Path]:
    """Resolve a candidate string using common normalized variants."""
    if not candidate:
        return None

    keys = {
        candidate,
        candidate.lower(),
        slugify_name(candidate),
        slugify_name(candidate).lower(),
    }
    for key in keys:
        match = mapping.get(key)
        if match:
            return match
    return None


def resolve_dataset_dir(
    row: dict, data_root: Path, available_dirs: dict[str, Path]
) -> Optional[Path]:
    """Resolve the local dataset directory corresponding to a CSV row."""
    def try_uris(raw: str) -> Optional[Path]:
        for uri in S3_URI_PATTERN.findall(raw or ""):
            local = uri_to_local_path(uri, data_root)
            if local:
                return local
        return None

    direct = try_uris(row.get("Decon Reports", ""))
    if direct:
        return direct

    for field in ("data path", "npy"):
        fallback = try_uris(row.get(field, ""))
        if fallback:
            return fallback

    candidates: List[str] = []

    for field in ("Decon Reports", "data path", "npy"):
        value = row.get(field, "")
        if value:
            candidates.extend(candidates_from_path(value))

    # Fall back to slugified paper name if nothing else matches.
    if not candidates and row.get("Paper name"):
        candidates.append(slugify_name(row["Paper name"]))

    for candidate in candidates:
        match = lookup_directory(candidate, available_dirs)
        if match:
            return match

    return None


def load_available_dirs(data_root: Path) -> dict[str, Path]:
    """Return a mapping of directory identifiers to Path objects (recursive)."""
    mapping: dict[str, Path] = {}
    root_resolved = data_root.resolve()
    for path in root_resolved.rglob("*"):
        if not path.is_dir():
            continue
        rel = path.relative_to(root_resolved).as_posix()
        keys = {
            path.name,
            rel,
            slugify_name(path.name),
            slugify_name(rel),
        }
        for key in keys:
            key = key.strip()
            if not key:
                continue
            mapping.setdefault(key, path)
            mapping.setdefault(key.lower(), path)
    return mapping


def build_command(
    dataset_path: Path,
    output_dir: Path,
    split: Optional[str],
) -> List[str]:
    cmd = ["cargo", "run", "--release", "--", "review", "--stats"]
    if split:
        cmd += ["--split", split]
    cmd += [str(dataset_path), "--output-dir", str(output_dir)]
    return cmd


def run_commands(
    csv_path: Path,
    data_root: Path,
    output_root: Path,
    splits: Iterable[Optional[str]],
    dry_run: bool = False,
) -> None:
    available_dirs = load_available_dirs(data_root)

    with csv_path.open(newline="", encoding="utf-8") as handle:
        header_fields: Optional[List[str]] = None

        # Seek to the line containing the header.
        while True:
            line = handle.readline()
            if not line:
                break

            if "paper name" in line.lower():
                header_fields = next(csv.reader([line]))
                break

        if not header_fields:
            print(
                f"[WARN] Unable to locate header row in {csv_path}. Nothing to do.",
                file=sys.stderr,
            )
            return

        reader = csv.DictReader(handle, fieldnames=header_fields)
        for row in reader:
            # Skip the remainder of the file once we hit an empty first column and no data.
            if all(not (value or "").strip() for value in row.values()):
                continue

            paper_name = (row.get("Paper name") or "").strip()
            if not paper_name:
                continue

            dataset_dir = resolve_dataset_dir(row, data_root, available_dirs)
            if not dataset_dir:
                print(
                    f"[WARN] Skipping '{paper_name}' – could not map to a local dataset directory.",
                    file=sys.stderr,
                )
                continue

            paper_slug = slugify_name(paper_name)
            for split in splits:
                split_label = split if split else "all"
                output_dir = output_root / paper_slug / split_label
                output_dir.mkdir(parents=True, exist_ok=True)

                cmd = build_command(dataset_dir, output_dir, split)
                print(f"[INFO] ({paper_name}) split={split_label}: {' '.join(cmd)}")

                if dry_run:
                    continue

                subprocess.run(cmd, cwd=data_root.parent, check=True)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run per-source contamination stats for all midtrain datasets."
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "midtrain_decon_map.csv",
        help="Path to midtrain_decon_map.csv (default: repository root version).",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "reports",
        help="Directory containing mirrored decon report folders.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "per_source_stats",
        help="Directory to write per-source stats into.",
    )
    parser.add_argument(
        "--splits",
        nargs="*",
        default=list(DEFAULT_SPLITS),
        help="Dataset splits to evaluate (default: test validation train).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without running them.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)

    splits: List[Optional[str]] = [*args.splits, None]

    try:
        run_commands(
            csv_path=args.csv,
            data_root=args.data_root,
            output_root=args.output_root,
            splits=splits,
            dry_run=args.dry_run,
        )
    except subprocess.CalledProcessError as exc:
        print(f"[ERROR] Command failed with exit code {exc.returncode}", file=sys.stderr)
        return exc.returncode

    return 0


if __name__ == "__main__":
    sys.exit(main())

