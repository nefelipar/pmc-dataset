#!/usr/bin/env python3
"""
Find and report token statistics of the .jsonl files of the jsonl folder (dataset). More specifically, it computes
token counts for the "text" and "abstract" fields of each record, and generates summary statistics
and histograms. It has as a default tokenizer the Qwen1.5-1.8B model, but any HuggingFace tokenizer can be used.
"""

from __future__ import annotations
import argparse
import json
import logging
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Dict, Iterable, List, Optional
from tqdm import tqdm


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


def load_tokenizer(name: str, trust_remote_code: bool = True, placeholder_tokens: Optional[List[str]] = None):
    """ Load a HuggingFace tokenizer by name safely, with optional placeholder tokens added."""
    try:
        from transformers import AutoTokenizer  
    except ImportError as exc: 
        raise SystemExit("The 'transformers' library is not installed."
        ) from exc

    try:
        logging.info("Loading tokenizer: %s", name)
        tokenizer = AutoTokenizer.from_pretrained(name, trust_remote_code=trust_remote_code)
        
        # We are not doing inference, so disable token limit.
        tokenizer.model_max_length = sys.maxsize
        tokenizer.init_kwargs["model_max_length"] = sys.maxsize
        tokenizer.deprecation_warnings["sequence_length_is_longer_than_the_maximum_length"] = "ignore" 
        if placeholder_tokens:
            new_tokens = [tok for tok in placeholder_tokens if tok not in tokenizer.get_vocab()]
            if new_tokens:
                tokenizer.add_special_tokens({"additional_special_tokens": new_tokens})
                logging.info("Added %d placeholder tokens", len(new_tokens))
        return tokenizer
    except Exception as exc:
        raise SystemExit(f"Failed to load tokenizer '{name}'. Make sure it is available (e.g., `huggingface-cli download`).\nDetails: {exc}"
        ) from exc


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
            "p99": None
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
        "p99": percentile(sorted_vals, 0.99)
    }

def clean_text(value) -> Optional[str]:
    if isinstance(value, str):
        cleaned = value.strip()
        if cleaned:
            return cleaned
    return None


"""Container for statistics of a single file."""
@dataclass
class FileStats:
    file_name: str
    text_tokens: List[int]
    abstract_tokens: List[int]
    placeholder_counts: Dict[str, int]


def update_placeholder_counts(container: Dict[str, int], text: str) -> None:
    """Update the counts of placeholder tokens found in the given text."""
    for token in PLACEHOLDER_TOKENS:
        container[token] += text.count(token)


def count_lines(path: Path) -> int:
    """Return the number of lines in a file so tqdm can show progress."""
    with path.open("rb") as handle:
        return sum(1 for _ in handle)


def compute_file_stats(path: Path, tokenizer, add_special_tokens: bool) -> FileStats:
    """Compute token statistics for a single .jsonl file."""
    total_lines = count_lines(path)
    text_counts: List[int] = []
    abstract_counts: List[int] = []
    placeholder_totals: Dict[str, int] = {token: 0 for token in PLACEHOLDER_TOKENS}
    with path.open("r", encoding="utf-8") as handle:
        for idx, raw_line in enumerate(tqdm(handle, desc=path.name, unit="line", leave=False, total=total_lines),1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                logging.warning("Skip of record %s:%d - bad JSON (%s)", path.name, idx, exc)
                continue

            text_value = clean_text(record.get("text"))
            if text_value is not None:
                text_counts.append(len(tokenizer.encode(text_value, add_special_tokens=add_special_tokens)))
                update_placeholder_counts(placeholder_totals, text_value)

            abstract_value = clean_text(record.get("abstract"))
            if abstract_value is not None:
                abstract_counts.append(len(tokenizer.encode(abstract_value, add_special_tokens=add_special_tokens)))
                update_placeholder_counts(placeholder_totals, abstract_value)

    return FileStats(
        file_name=path.name,
        text_tokens=text_counts,
        abstract_tokens=abstract_counts,
        placeholder_counts=placeholder_totals,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Statistics for token counts in .jsonl dataset files.",
    )
    parser.add_argument(
        "--input-dir", 
        default=Path("__dataset__/jsonl"),
        type=Path, 
        help="Folder with *.jsonl files (default: __dataset__/jsonl)."
    )
    parser.add_argument(
        "--tokenizer", 
        default="Qwen/Qwen1.5-1.8B",
        help="Tokenizer name from HuggingFace (e.g., 'Qwen/Qwen1.5-1.8B')"
        )
    parser.add_argument(
        "--output-json",
        default=None,
        type=Path,
        help="Final JSON with the statistics (default: __stats__/<tokenizer>/stats.json)",
    )
    parser.add_argument(
        "--add-special-tokens", 
        action="store_true", 
        help="Include CLS/SEP tokens during encoding"
    )
    parser.add_argument(
        "--trust-remote-code", 
        action="store_true", 
        help="Pass to AutoTokenizer (default: False)"
    )
    parser.add_argument(
        "--log-level", 
        default="INFO", 
        help="Logging level (DEBUG, INFO, ...)"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    tok = args.tokenizer.replace("/", "_")
    tokenizer_dir = Path("__stats__") / tok
    if args.output_json is None:
        args.output_json = tokenizer_dir / "stats.json"

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(levelname)s: %(message)s")

    input_dir: Path = args.input_dir

    if not input_dir.exists() or not input_dir.is_dir():
        raise SystemExit(f"The folder '{input_dir}' does not exist or is not a directory.")

    jsonl_files = sorted(input_dir.glob("*.jsonl"))
    if not jsonl_files:
        raise SystemExit(f"No .jsonl files found in {input_dir}")
    tokenizer = load_tokenizer(
        args.tokenizer,
        trust_remote_code=args.trust_remote_code,
        placeholder_tokens=PLACEHOLDER_TOKENS,
    )

    per_file_entries = []
    all_text_tokens: List[int] = []
    all_abstract_tokens: List[int] = []
    overall_placeholder_counts: Dict[str, int] = {token: 0 for token in PLACEHOLDER_TOKENS}

    for file_path in tqdm(jsonl_files, desc="Files", unit="file"):
        logging.info("Processing %s", file_path)
        stats = compute_file_stats(file_path, tokenizer, args.add_special_tokens)
        per_file_entries.append(
            {
                "file_name": stats.file_name,
                "text_tokens": describe(stats.text_tokens),
                "abstract_tokens": describe(stats.abstract_tokens),
                "placeholder_counts": stats.placeholder_counts,
            }
        )
        all_text_tokens.extend(stats.text_tokens)
        all_abstract_tokens.extend(stats.abstract_tokens)
        for token, count in stats.placeholder_counts.items():
            overall_placeholder_counts[token] += count

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "tokenizer": args.tokenizer,
        "add_special_tokens": args.add_special_tokens,
        "file_count": len(per_file_entries),
        "files": per_file_entries,
        "overall": {
            "text_tokens": describe(all_text_tokens),
            "abstract_tokens": describe(all_abstract_tokens),
            "placeholder_counts": overall_placeholder_counts,
        },
    }

    with args.output_json.open("w", encoding="utf-8") as fp:
        json.dump(result, fp, indent=2, ensure_ascii=False)
        fp.write("\n")
    logging.info("Saved JSON: %s", args.output_json)

if __name__ == "__main__":
    main()
