#!/usr/bin/env python3
"""Sample random records from a JSONL file into a new JSONL file."""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Iterator


def iter_jsonl_lines(path: Path) -> Iterator[str]:
    """Yield non-empty JSONL lines from a file."""
    with path.open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, 1):
            cleaned = line.rstrip("\n")
            if cleaned.strip():
                yield cleaned
            else:
                print(f"Skipping empty line in {path.name}:{line_no}")


def reservoir_sample(
    path: Path, sample_size: int, seed: int | None = None
) -> list[str]:
    """Reservoir sampling to pick sample_size lines without loading entire file."""
    if sample_size <= 0:
        raise SystemExit("sample_size must be a positive integer")

    rng = random.Random(seed)
    reservoir: list[str] = []
    count = 0
    for line in iter_jsonl_lines(path):
        count += 1
        if len(reservoir) < sample_size:
            reservoir.append(line)
        else:
            j = rng.randint(0, count - 1)
            if j < sample_size:
                reservoir[j] = line

    if count == 0:
        raise SystemExit(f"No records found in {path}")

    if count < sample_size:
        print(f"Requested {sample_size} samples but only {count} records available; returning all.")
    rng.shuffle(reservoir)
    return reservoir[: min(sample_size, count)]


def write_lines(lines: list[str], out_path: Path) -> int:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as out_fh:
        for line in lines:
            out_fh.write(line + "\n")
    return len(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Take N random samples from a JSONL file and write them to a new JSONL file."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("__dataset__/filtered_merged.jsonl"),
        help="Path to the source JSONL file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Path to the sampled JSONL output file.",
    )
    parser.add_argument(
        "--samples",
        type=int,
        required=True,
        help="Number of random samples to take.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional RNG seed for reproducible sampling.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sampled = reservoir_sample(args.input, args.samples, args.seed)
    written = write_lines(sampled, args.output)
    print(f"Wrote {written} sampled records to {args.output}")


if __name__ == "__main__":
    main()
