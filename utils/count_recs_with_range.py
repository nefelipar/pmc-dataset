#!/usr/bin/env python3
from pathlib import Path
import argparse
import json


def iter_records(stats_folder):
    for path in Path(stats_folder).glob("*.jsonl"):
        with path.open("r", encoding="utf-8") as fh:
            for line_no, line in enumerate(fh, 1):
                text = line.strip()
                if not text:
                    continue
                try:
                    yield json.loads(text)
                except json.JSONDecodeError:
                    print(f"Skipping {path.name}:{line_no} (invalid JSON)")
                    continue


def count_in_range(stats_folder, text_min, text_max, abs_min, abs_max):
    total = 0
    matched = 0
    for rec in iter_records(stats_folder):
        total += 1
        text_tokens = rec.get("text_tokens")
        abstract_tokens = rec.get("abstract_tokens")
        if text_tokens is None or abstract_tokens is None:
            continue
        if text_min <= text_tokens <= text_max and abs_min <= abstract_tokens <= abs_max:
            matched += 1
    return matched, total


def main():
    parser = argparse.ArgumentParser(
        description="Count records whose text_tokens and abstract_tokens fall inside given ranges."
    )
    parser.add_argument(
        "--stats-folder",
        default="__stats__/Qwen_Qwen1.5-1.8B/stats_per_record",
        help="Directory containing stats_per_record *.jsonl files.",
    )
    parser.add_argument(
        "--text-min", 
        type=int, 
        required=True, 
        help="Inclusive min for text_tokens.")
    parser.add_argument(
        "--text-max", 
        type=int, 
        required=True, 
        help="Inclusive max for text_tokens.")
    parser.add_argument(
        "--abstract-min", 
        type=int, 
        required=True, 
        help="Inclusive min for abstract_tokens.")
    parser.add_argument(
        "--abstract-max", 
        type=int, 
        required=True, 
        help="Inclusive max for abstract_tokens.")
    args = parser.parse_args()

    matched, total = count_in_range(
        args.stats_folder, args.text_min, args.text_max, args.abstract_min, args.abstract_max
    )
    print(
        f"Matched {matched} of {total} records in {args.stats_folder} "
        f"(text_tokens {args.text_min}-{args.text_max}, abstract_tokens {args.abstract_min}-{args.abstract_max})."
    )


if __name__ == "__main__":
    main()
