#!/usr/bin/env python3
"""Summarize metadata.article_type distribution from JSONL files."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Iterable


def iter_jsonl_records(paths: Iterable[Path]):
    """Yield JSON objects from the given .jsonl file paths."""
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    print(f"Skipping {path}:{line_no} (invalid JSON)")
                    continue


def resolve_input_paths(input_path: Path) -> list[Path]:
    """Return a list of .jsonl files from a file or directory path."""
    if input_path.is_file():
        return [input_path]
    if input_path.is_dir():
        files = sorted(input_path.glob("*.jsonl"))
        if not files:
            raise SystemExit(f"No .jsonl files found in {input_path}")
        return files
    raise SystemExit(f"Path not found: {input_path}")


def collect_article_type_counts(paths: Iterable[Path]):
    """Return counts plus bookkeeping for metadata.article_type values."""
    counts: Counter[str] = Counter()
    total_records = 0
    missing_article_type = 0

    for rec in iter_jsonl_records(paths):
        total_records += 1
        meta = rec.get("metadata")
        if not isinstance(meta, dict):
            missing_article_type += 1
            continue
        value = meta.get("article_type")
        if isinstance(value, str):
            cleaned = value.strip()
            if cleaned:
                counts[cleaned] += 1
            else:
                missing_article_type += 1
        elif isinstance(value, list):
            seen = False
            for item in value:
                if isinstance(item, str) and item.strip():
                    counts[item.strip()] += 1
                    seen = True
            if not seen:
                missing_article_type += 1
        else:
            missing_article_type += 1
    return counts, total_records, missing_article_type


def format_percentage(count: int, total: int) -> str:
    if total <= 0:
        return "0.00%"
    return f"{(count / total) * 100:.2f}%"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Print distribution stats for metadata.article_type values in JSONL files.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("__dataset__/jsonl"),
        help="Path to a .jsonl file or a directory containing .jsonl files.",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=None,
        help="Limit printed rows to the top N article types.",
    )
    parser.add_argument(
        "--save-json",
        type=Path,
        default=None,
        help="Optional path to write counts as JSON.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    files = resolve_input_paths(args.input)

    counts, total, missing = collect_article_type_counts(files)
    present = total - missing

    print(f"Processed {total} records across {len(files)} file(s).")
    print(f"With article_type: {present} ({format_percentage(present, total)})")
    print(f"Missing article_type: {missing} ({format_percentage(missing, total)})\n")

    if not counts:
        print("No article_type values found.")
        return

    print("Article type counts (sorted by frequency):")
    for rank, (name, count) in enumerate(counts.most_common(), 1):
        if args.top is not None and rank > args.top:
            break
        pct = format_percentage(count, total)
        print(f"{rank:>3}. {name}: {count} ({pct})")

    if args.save_json:
        args.save_json.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "total_records": total,
            "with_article_type": present,
            "missing_article_type": missing,
            "counts": dict(counts),
        }
        with args.save_json.open("w", encoding="utf-8") as out:
            json.dump(payload, out, ensure_ascii=False, indent=2)
        print(f"\nSaved counts to {args.save_json}")


if __name__ == "__main__":
    main()
