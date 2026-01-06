#!/usr/bin/env python3
"""Merge all JSONL files from a directory into a single JSONL file."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Iterator


def iter_jsonl_lines(paths: Iterable[Path]) -> Iterator[str]:
    """Yield non-empty JSONL lines from each path."""
    for path in paths:
        with path.open("r", encoding="utf-8") as fh:
            for line_no, line in enumerate(fh, 1):
                cleaned = line.rstrip("\n")
                if cleaned.strip():
                    yield cleaned
                else:
                    print(f"Skipping empty line in {path.name}:{line_no}")


def merge_jsonl(input_dir: Path, output_path: Path) -> tuple[int, int]:
    """Merge all .jsonl files in input_dir into output_path.

    Returns:
        (records_written, files_merged)
    """
    files = sorted(p for p in input_dir.glob("*.jsonl") if p.resolve() != output_path.resolve())
    if not files:
        raise SystemExit(f"No .jsonl files found in {input_dir}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    record_count = 0
    with output_path.open("w", encoding="utf-8") as out_fh:
        for path in files:
            for line in iter_jsonl_lines([path]):
                out_fh.write(line + "\n")
                record_count += 1
    return record_count, len(files)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge all JSONL files in a directory into a single JSONL output file."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("__dataset__/jsonl_filtered"),
        help="Directory containing the source .jsonl files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("__dataset__/filtered_merged.jsonl"),
        help="Path where the merged .jsonl will be written.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records, files = merge_jsonl(args.input_dir, args.output)
    print(f"Wrote {records} records from {files} files to {args.output}")


if __name__ == "__main__":
    main()
