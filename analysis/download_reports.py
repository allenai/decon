#!/usr/bin/env python3
"""
Download all decontamination report directories referenced in the midtrain map.

This script gathers every S3 URI present in `midtrain_decon_map.csv`
(`Decon Reports` column) and mirrors them locally using `s5cmd run`.
"""

from __future__ import annotations

import argparse
import csv
import re
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Set
from urllib.parse import urlparse


def normalize_header(name: str) -> str:
    """Normalize a column name to snake_case alphanumerics."""
    return re.sub(r"[^a-z0-9]+", "_", name.strip().lower()).strip("_")


def find_header(handle, required_token: str) -> Optional[List[str]]:
    """
    Advance the file handle until a row containing `required_token`
    (case-insensitive) appears, then return that row as the header.
    """
    while True:
        pos = handle.tell()
        line = handle.readline()
        if not line:
            return None
        if required_token.lower() in line.lower():
            handle.seek(pos)
            reader = csv.reader(handle)
            try:
                header = next(reader)
            except StopIteration:
                return None
            return header


def extract_s3_uri(value: str) -> Optional[str]:
    """Return the S3 URI if present, otherwise None."""
    if not value:
        return None
    value = value.strip()
    if not value:
        return None
    if not value.startswith("s3://"):
        return None
    # Drop any trailing commas or whitespace/newlines that slipped through.
    value = value.rstrip(",")
    return value


def parse_midtrain_decon_map(csv_path: Path) -> Set[str]:
    """Collect report URIs from `midtrain_decon_map.csv`."""
    uris: Set[str] = set()
    with csv_path.open(newline="", encoding="utf-8") as handle:
        header = find_header(handle, "paper name")
        if not header:
            print(
                f"[WARN] {csv_path} does not contain a 'Paper name' header; skipping.",
                file=sys.stderr,
            )
            return uris

        header_norm = [normalize_header(col) for col in header]
        try:
            idx = header_norm.index("decon_reports")
        except ValueError:
            print(
                f"[WARN] {csv_path} missing 'Decon Reports' column; skipping.",
                file=sys.stderr,
            )
            return uris

        reader = csv.reader(handle)
        for row in reader:
            if not row:
                continue
            if len(row) <= idx:
                continue
            uri = extract_s3_uri(row[idx])
            if uri:
                uris.add(uri)

    return uris


def build_s5cmd_commands(
    uris: Iterable[str],
    destination_root: Path,
) -> List[Sequence[str]]:
    """Return s5cmd command lines to mirror each URI into destination_root."""
    destination_root.mkdir(parents=True, exist_ok=True)
    commands: List[Sequence[str]] = []
    for uri in sorted(set(uris)):
        parsed = urlparse(uri)
        if parsed.scheme != "s3" or not parsed.netloc:
            continue

        bucket = parsed.netloc
        key = parsed.path.lstrip("/")

        key_path = Path(*[part for part in key.split("/") if part])
        dest_path = (destination_root / bucket / key_path).resolve()
        dest_path.mkdir(parents=True, exist_ok=True)

        # Ensure trailing slash so s5cmd treats it as a directory.
        dest_str = str(dest_path) + ("/" if not str(dest_path).endswith("/") else "")
        # s5cmd sync requires a wildcard in the source argument.
        source_str = (uri.rstrip("/") + "/*")

        commands.append(["sync", source_str, dest_str])
    return commands


def run_s5cmd(commands: List[Sequence[str]], dry_run: bool) -> None:
    """Execute the commands via `s5cmd`, one at a time. If dry_run, just print them."""
    if not commands:
        print("[INFO] No report URIs discovered; nothing to download.")
        return

    print("[INFO] Prepared s5cmd command list:")
    for args in commands:
        print(f"  {' '.join(args)}")

    if dry_run:
        print("[INFO] Dry run requested; not invoking s5cmd.")
        return

    for args in commands:
        try:
            subprocess.run(["s5cmd", *args], check=True)
        except FileNotFoundError:
            print("[ERROR] s5cmd executable not found in PATH.", file=sys.stderr)
            raise
        except subprocess.CalledProcessError as exc:
            print(
                f"[ERROR] Command failed ({' '.join(exc.cmd)}): {exc.stderr or exc.stdout}",
                file=sys.stderr,
            )
            raise


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download all decon reports referenced in midtrain_decon_map.csv."
    )
    repo_root = Path(__file__).resolve().parents[1]
    parser.add_argument(
        "--midtrain-csv",
        type=Path,
        default=repo_root / "midtrain_decon_map.csv",
        help="Path to midtrain_decon_map.csv.",
    )
    parser.add_argument(
        "--dest",
        type=Path,
        default=repo_root / "reports",
        help="Destination directory to mirror reports into.",
    )
    parser.add_argument(
        "--skip-file",
        type=Path,
        default=repo_root / "config" / "skipped_reports.txt",
        help="File containing newline-delimited S3 prefixes to skip.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the s5cmd commands without downloading.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)

    all_uris: Set[str] = set()

    if args.midtrain_csv.exists():
        all_uris.update(parse_midtrain_decon_map(args.midtrain_csv))
    else:
        print(f"[WARN] {args.midtrain_csv} not found; skipping.", file=sys.stderr)

    skipped_prefixes: Set[str] = set()
    if args.skip_file and args.skip_file.exists():
        with args.skip_file.open(encoding="utf-8") as handle:
            for line in handle:
                stripped = line.strip()
                if not stripped or stripped.startswith("#"):
                    continue
                skipped_prefixes.add(stripped.rstrip("/"))

    if skipped_prefixes:
        before = len(all_uris)
        all_uris = {
            uri
            for uri in all_uris
            if not any(uri.rstrip("/").startswith(prefix) for prefix in skipped_prefixes)
        }
        skipped_count = before - len(all_uris)
        if skipped_count:
            print(f"[INFO] Skipping {skipped_count} report prefixes listed in {args.skip_file}.")

    commands = build_s5cmd_commands(all_uris, args.dest)
    run_s5cmd(commands, args.dry_run)
    return 0


if __name__ == "__main__":
    sys.exit(main())

