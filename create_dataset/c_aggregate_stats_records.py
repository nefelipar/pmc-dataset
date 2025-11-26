#!/usr/bin/env python3
"""Aggregate per-record token stats from a stats_per_record folder to calculate overall statistics."""

from __future__ import annotations
import argparse
import json
import logging
import math
from pathlib import Path
from statistics import median
from typing import Dict, Iterable, List, Optional

PLACEHOLDER_TOKENS = [
    "[MATH]",
    "[CODE]",
    "[CIT_REF]",
    "[FIG_REF]",
    "[TAB_REF]",
    "[APP_REF]",
    "[SUP_REF]",
    "[BOX_REF]",
]


def safe_mean(values: Iterable[int]) -> Optional[float]:
    """Compute the mean of a list of integers safely, returning None for empty input."""
    values = list(values)
    if not values:
        return None
    return sum(values) / float(len(values))


def percentile(sorted_values: List[int], q: float) -> float:
    """Compute the q-th percentile of a sorted list of integers."""
    if not sorted_values:
        return math.nan
    if q <= 0:
        return float(sorted_values[0])
    if q >= 1:
        return float(sorted_values[-1])
    position = (len(sorted_values) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(sorted_values[lower])
    weight = position - lower
    return float(sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight)


def describe(values: List[int]) -> Dict[str, Optional[float]]:
    """Compute descriptive statistics for a list of integers."""
    if not values:
        return {
            "count": 0,
            "min": None,
            "max": None,
            "mean": None,
            "median": None,
            "p90": None,
            "p95": None,
            "p99": None,
        }

    sorted_vals = sorted(values)
    return {
        "count": len(sorted_vals),
        "min": float(sorted_vals[0]),
        "max": float(sorted_vals[-1]),
        "mean": safe_mean(sorted_vals),
        "median": float(median(sorted_vals)),
        "p90": percentile(sorted_vals, 0.90),
        "p95": percentile(sorted_vals, 0.95),
        "p99": percentile(sorted_vals, 0.99),
    }


def aggregate_records(stats_dir: Path) -> Dict[str, object]:
    """Aggregate per-record token stats from a stats_per_record folder."""
    if not stats_dir.exists() or not stats_dir.is_dir():
        raise SystemExit(f"Stats per-record folder not found: {stats_dir}")

    files = sorted(stats_dir.glob("*.jsonl"))
    if not files:
        raise SystemExit(f"No .jsonl files found in {stats_dir}")

    text_counts: List[int] = []
    abstract_counts: List[int] = []
    placeholder_totals: Dict[str, int] = {token: 0 for token in PLACEHOLDER_TOKENS}
    record_count = 0

    for path in files:
        logging.info("Reading %s", path)
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError as exc:
                    logging.warning("Bad JSON in %s: %s", path, exc)
                    continue

                record_count += 1

                t_tokens = rec.get("text_tokens")
                if isinstance(t_tokens, int):
                    text_counts.append(t_tokens)

                a_tokens = rec.get("abstract_tokens")
                if isinstance(a_tokens, int):
                    abstract_counts.append(a_tokens)

                placeholders = rec.get("placeholders") or {}
                if isinstance(placeholders, dict):
                    for key, val in placeholders.items():
                        if isinstance(val, int):
                            placeholder_totals[key] = placeholder_totals.get(key, 0) + val

    return {
        "records": record_count,
        "text_tokens": describe(text_counts),
        "abstract_tokens": describe(abstract_counts),
        "placeholder_counts": placeholder_totals,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate per-record token stats from stats_per_record/*.jsonl",
    )
    parser.add_argument(
        "--tokenizer",
        default="Qwen/Qwen1.5-1.8B",
        help="Tokenizer name; used to locate __stats__/<tokenizer>/stats_per_record and write overall.json",
    )
    parser.add_argument(
        "--stats-per-record",
        type=Path,
        default=None,
        help="Path to stats_per_record directory (default: __stats__/<tokenizer>/stats_per_record)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Path to output JSON file (default: __stats__/<tokenizer>/overall.json)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Logging level (DEBUG, INFO, ...)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(levelname)s: %(message)s",
    )

    tok = args.tokenizer.replace("/", "_")
    stats_root = Path("__stats__") / tok
    stats_dir = args.stats_per_record or (stats_root / "stats_per_record")
    output_json = args.output or (stats_root / "overall.json")

    result = aggregate_records(stats_dir)

    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as fp:
        json.dump(result, fp, indent=2, ensure_ascii=False)
        fp.write("\n")
    logging.info("Saved aggregated stats: %s", output_json)


if __name__ == "__main__":
    main()
