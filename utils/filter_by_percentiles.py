#!/usr/bin/env python3
"""Filter dataset JSONL files by article_type and token percentile range."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple


def load_overall_percentiles(overall_path: Path, p_low_key: str, p_high_key: str) -> Tuple[float, float]:
    """Read overall.json and return (low, high) percentile values for text_tokens."""
    try:
        overall = json.loads(overall_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise SystemExit(f"Failed to read overall stats from {overall_path}: {exc}") from exc

    text_stats = overall.get("text_tokens") or {}
    if p_low_key not in text_stats or p_high_key not in text_stats:
        raise SystemExit(
            f"Missing percentiles {p_low_key}/{p_high_key} in text_tokens at {overall_path}"
        )
    return float(text_stats[p_low_key]), float(text_stats[p_high_key])


def load_text_tokens(stats_path: Path) -> Dict[str, float]:
    """Return {pmcid: text_tokens} from a stats_per_record JSONL file."""
    tokens: Dict[str, float] = {}
    with stats_path.open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                print(f"Skipping bad JSON in {stats_path.name}:{line_no}")
                continue
            pmcid = obj.get("pmcid")
            t_tok = obj.get("text_tokens")
            if isinstance(pmcid, str) and isinstance(t_tok, (int, float)):
                tokens[pmcid] = float(t_tok)
    return tokens


def article_type_matches(meta: dict | None, target: str) -> bool:
    """Check if metadata.article_type matches target (handles string or list)."""
    if not meta:
        return False
    value = meta.get("article_type")
    if isinstance(value, str):
        return value == target
    if isinstance(value, list):
        return any(isinstance(v, str) and v == target for v in value)
    return False


def filter_file(
    data_path: Path,
    stats_path: Path,
    out_path: Path,
    token_low: float,
    token_high: float,
    article_type: str,
) -> Tuple[int, int]:
    """Filter a single dataset JSONL file and write the kept records."""
    token_map = load_text_tokens(stats_path)
    kept = 0
    total = 0

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with data_path.open("r", encoding="utf-8") as data_fh, out_path.open(
        "w", encoding="utf-8"
    ) as out_fh:
        for line_no, line in enumerate(data_fh, 1):
            line = line.strip()
            if not line:
                continue
            total += 1
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                print(f"Skipping bad JSON in {data_path.name}:{line_no}")
                continue

            pmcid = rec.get("pmcid")
            tokens = token_map.get(pmcid)
            if tokens is None:
                continue
            if not article_type_matches(rec.get("metadata"), article_type):
                continue
            if token_low <= tokens <= token_high:
                out_fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                kept += 1
    return kept, total


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Keep only records with article_type=='research-article' whose text_tokens "
            "lie between the chosen percentiles from overall.json."
        )
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("__dataset__/jsonl"),
        help="Directory containing input dataset .jsonl files.",
    )
    parser.add_argument(
        "--stats-dir",
        type=Path,
        default=Path("__stats__/Qwen_Qwen1.5-1.8B/stats_per_record"),
        help="Directory containing matching stats_per_record .jsonl files.",
    )
    parser.add_argument(
        "--overall",
        type=Path,
        default=Path("__stats__/Qwen_Qwen1.5-1.8B/overall.json"),
        help="Path to overall.json with percentile values.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("__dataset__/jsonl_filtered"),
        help="Directory where filtered .jsonl files will be written.",
    )
    parser.add_argument(
        "--p-low",
        default="p10",
        help="Lower percentile key from overall.json text_tokens (e.g., p10).",
    )
    parser.add_argument(
        "--p-high",
        default="p80",
        help="Upper percentile key from overall.json text_tokens (e.g., p80).",
    )
    parser.add_argument(
        "--article-type",
        default="research-article",
        help="Metadata article_type value to retain.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    token_low, token_high = load_overall_percentiles(args.overall, args.p_low, args.p_high)

    dataset_files = sorted(args.dataset_dir.glob("*.jsonl"))
    if not dataset_files:
        raise SystemExit(f"No .jsonl files found in {args.dataset_dir}")

    total_records = 0
    total_kept = 0
    for data_path in dataset_files:
        stats_path = args.stats_dir / data_path.name
        if not stats_path.exists():
            print(f"Skipping {data_path.name} (missing stats file {stats_path.name})")
            continue
        out_path = args.out_dir / data_path.name
        kept, total = filter_file(
            data_path,
            stats_path,
            out_path,
            token_low,
            token_high,
            args.article_type,
        )
        total_records += total
        total_kept += kept
        print(
            f"{data_path.name}: kept {kept}/{total} records "
            f"(text_tokens in [{token_low}, {token_high}])"
        )

    print(f"\nDone. Kept {total_kept} of {total_records} records across {len(dataset_files)} files.")


if __name__ == "__main__":
    main()
