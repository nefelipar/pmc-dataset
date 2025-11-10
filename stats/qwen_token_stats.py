#!/usr/bin/env python3
"""
Compute dataset statistics using a Qwen tokenizer.

For each `*.jsonl.gz` file under the target directory the script reports:
- total records, missing text, missing abstract, missing PMCID
- token statistics (min/mean/median/max/percentiles) for input text and abstracts
- character-length statistics (helpful for sanity checks)
- ratios between abstract and text lengths (tokens & characters)
- αποθήκευση αποτελεσμάτων σε JSON αρχείο (προεπιλογή: stats/qwen_token_stats.json)
- προαιρετικά γραφήματα token counts & αριθμού εγγραφών (σε φάκελο plots) για εγγραφές με text+abstract

Usage:
    python stats/qwen_token_stats.py --data-dir data/jsonl
"""

from __future__ import annotations

import argparse
import gzip
import json
import logging
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Any, Dict, Iterable, List, Optional, Tuple


# --------------------------------------------------------------------------- #
# Data structures
# --------------------------------------------------------------------------- #

@dataclass
class LengthBuckets:
    text_tokens: List[int]
    abstract_tokens: List[int]
    text_chars: List[int]
    abstract_chars: List[int]
    text_with_abstract_tokens: List[int]
    abstract_with_text_tokens: List[int]
    token_ratio: List[float]
    char_ratio: List[float]


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def load_tokenizer(model_name: str, trust_remote_code: bool = True):
    """Load a HuggingFace tokenizer for a given model name."""
    try:
        from transformers import AutoTokenizer  # type: ignore
    except ImportError as exc:  # pragma: no cover - dependency hint
        raise SystemExit(
            "Η βιβλιοθήκη transformers δεν βρέθηκε. Εγκατάστησέ την με "
            "`pip install transformers` πριν τρέξεις το script."
        ) from exc

    try:
        logging.info("Φόρτωση tokenizer: %s", model_name)
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=trust_remote_code,
        )
        # Δεν τρέχουμε το μοντέλο, μόνο μετράμε tokens, οπότε αφαιρούμε το όριο.
        tokenizer.model_max_length = sys.maxsize
        tokenizer.init_kwargs["model_max_length"] = sys.maxsize
        tokenizer.deprecation_warnings["sequence_length_is_longer_than_the_maximum_length"] = "ignore"  # type: ignore[attr-defined]
    except Exception as exc:  # pragma: no cover - runtime environment specific
        raise SystemExit(
            f"Αποτυχία φόρτωσης του tokenizer '{model_name}'. "
            "Σιγουρέψου ότι το έχεις κατεβάσει (π.χ. με `huggingface-cli download`). "
            f"Λεπτομέρειες σφάλματος: {exc}"
        ) from exc
    return tokenizer


def is_missing(value: Optional[str]) -> bool:
    """Return True when a field should be counted as 'missing'."""
    if value is None:
        return True
    if isinstance(value, str) and not value.strip():
        return True
    return False


def safe_mean(values: Iterable[float]) -> Optional[float]:
    values = list(values)
    if not values:
        return None
    return sum(values) / float(len(values))


def percentile(sorted_values: List[int], q: float) -> float:
    """Interpolate percentile on pre-sorted values."""
    if not sorted_values:
        return math.nan
    if q <= 0:
        return float(sorted_values[0])
    if q >= 1:
        return float(sorted_values[-1])
    k = (len(sorted_values) - 1) * q
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return float(sorted_values[int(k)])
    d0 = sorted_values[f] * (c - k)
    d1 = sorted_values[c] * (k - f)
    return float(d0 + d1)


def summarise(values: List[int]) -> Dict[str, Optional[float]]:
    """Return descriptive statistics for a list of numeric values."""
    if not values:
        return {
            "count": 0,
            "min": None,
            "mean": None,
            "median": None,
            "p90": None,
            "p95": None,
            "p99": None,
            "max": None,
        }
    sorted_vals = sorted(values)
    return {
        "count": len(sorted_vals),
        "min": float(sorted_vals[0]),
        "mean": safe_mean(sorted_vals),
        "median": float(median(sorted_vals)),
        "p90": percentile(sorted_vals, 0.90),
        "p95": percentile(sorted_vals, 0.95),
        "p99": percentile(sorted_vals, 0.99),
        "max": float(sorted_vals[-1]),
    }


def summarise_ratio(values: List[float]) -> Dict[str, Optional[float]]:
    if not values:
        return {
            "count": 0,
            "min": None,
            "mean": None,
            "median": None,
            "max": None,
        }
    sorted_vals = sorted(values)
    return {
        "count": len(sorted_vals),
        "min": float(sorted_vals[0]),
        "mean": safe_mean(sorted_vals),
        "median": float(median(sorted_vals)),
        "max": float(sorted_vals[-1]),
    }


def collect_file_stats(
    path: Path,
    tokenizer,
    add_special_tokens: bool = False,
    limit: Optional[int] = None,
) -> Tuple[Dict[str, object], LengthBuckets]:
    total = 0
    missing_text = 0
    missing_abstract = 0
    missing_pmcid = 0
    json_errors = 0

    text_tokens: List[int] = []
    abstract_tokens: List[int] = []
    text_chars: List[int] = []
    abstract_chars: List[int] = []
    token_ratio: List[float] = []
    char_ratio: List[float] = []
    text_with_abstract_tokens: List[int] = []
    abstract_with_text_tokens: List[int] = []

    with gzip.open(path, "rt", encoding="utf-8") as fh:
        for idx, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            total += 1
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                json_errors += 1
                logging.warning("Σφάλμα JSON στη γραμμή %d του %s", idx, path.name)
                continue

            text_val = record.get("text")
            abstract_val = record.get("abstract")
            pmcid_val = record.get("pmcid")

            if is_missing(text_val):
                missing_text += 1
            else:
                assert isinstance(text_val, str)
                tokenised = tokenizer.encode(
                    text_val,
                    add_special_tokens=add_special_tokens,
                )
                text_tokens.append(len(tokenised))
                text_chars.append(len(text_val))

            if is_missing(abstract_val):
                missing_abstract += 1
            else:
                assert isinstance(abstract_val, str)
                tokenised_abs = tokenizer.encode(
                    abstract_val,
                    add_special_tokens=add_special_tokens,
                )
                abstract_tokens.append(len(tokenised_abs))
                abstract_chars.append(len(abstract_val))

            if is_missing(pmcid_val):
                missing_pmcid += 1

            if (
                not is_missing(text_val)
                and not is_missing(abstract_val)
                and text_tokens
                and abstract_tokens
            ):
                text_with_abstract_tokens.append(text_tokens[-1])
                abstract_with_text_tokens.append(abstract_tokens[-1])
                if text_tokens[-1] > 0:
                    token_ratio.append(
                        abstract_tokens[-1] / float(text_tokens[-1])
                    )
                if text_chars[-1] > 0:
                    char_ratio.append(
                        abstract_chars[-1] / float(text_chars[-1])
                    )

            if limit is not None and total >= limit:
                break

    summary = {
        "file": path.name,
        "total_records": total,
        "missing_text": missing_text,
        "missing_abstract": missing_abstract,
        "missing_pmcid": missing_pmcid,
        "json_errors": json_errors,
        "text_token_stats": summarise(text_tokens),
        "abstract_token_stats": summarise(abstract_tokens),
        "text_char_stats": summarise(text_chars),
        "abstract_char_stats": summarise(abstract_chars),
        "paired_records": len(text_with_abstract_tokens),
        "paired_text_token_stats": summarise(text_with_abstract_tokens),
        "paired_abstract_token_stats": summarise(abstract_with_text_tokens),
        "token_ratio_stats": summarise_ratio(token_ratio),
        "char_ratio_stats": summarise_ratio(char_ratio),
    }

    buckets = LengthBuckets(
        text_tokens=text_tokens,
        abstract_tokens=abstract_tokens,
        text_chars=text_chars,
        abstract_chars=abstract_chars,
        text_with_abstract_tokens=text_with_abstract_tokens,
        abstract_with_text_tokens=abstract_with_text_tokens,
        token_ratio=token_ratio,
        char_ratio=char_ratio,
    )
    return summary, buckets


def merge_buckets(into: LengthBuckets, other: LengthBuckets) -> None:
    into.text_tokens.extend(other.text_tokens)
    into.abstract_tokens.extend(other.abstract_tokens)
    into.text_chars.extend(other.text_chars)
    into.abstract_chars.extend(other.abstract_chars)
    into.text_with_abstract_tokens.extend(other.text_with_abstract_tokens)
    into.abstract_with_text_tokens.extend(other.abstract_with_text_tokens)
    into.token_ratio.extend(other.token_ratio)
    into.char_ratio.extend(other.char_ratio)


def run_analysis(
    data_dir: Path,
    glob_pattern: str,
    tokenizer_model: str,
    trust_remote_code: bool,
    add_special_tokens: bool,
    limit: Optional[int],
) -> Tuple[Dict[str, object], LengthBuckets]:
    tokenizer = load_tokenizer(
        tokenizer_model,
        trust_remote_code=trust_remote_code,
    )

    paths = sorted(data_dir.glob(glob_pattern))
    if not paths:
        raise SystemExit(
            f"Δεν βρέθηκαν αρχεία με pattern '{glob_pattern}' στον φάκελο {data_dir}."
        )

    overall_buckets = LengthBuckets(
        text_tokens=[],
        abstract_tokens=[],
        text_chars=[],
        abstract_chars=[],
        text_with_abstract_tokens=[],
        abstract_with_text_tokens=[],
        token_ratio=[],
        char_ratio=[],
    )

    per_file: List[Dict[str, object]] = []
    total_records = 0
    missing_text = 0
    missing_abstract = 0
    missing_pmcid = 0
    json_errors = 0

    for path in paths:
        logging.info("Ανάλυση αρχείου: %s", path.name)
        file_summary, buckets = collect_file_stats(
            path,
            tokenizer=tokenizer,
            add_special_tokens=add_special_tokens,
            limit=limit,
        )
        per_file.append(file_summary)
        merge_buckets(overall_buckets, buckets)

        total_records += file_summary["total_records"]  # type: ignore[arg-type]
        missing_text += file_summary["missing_text"]  # type: ignore[arg-type]
        missing_abstract += file_summary["missing_abstract"]  # type: ignore[arg-type]
        missing_pmcid += file_summary["missing_pmcid"]  # type: ignore[arg-type]
        json_errors += file_summary["json_errors"]  # type: ignore[arg-type]

    overall = {
        "total_records": total_records,
        "missing_text": missing_text,
        "missing_abstract": missing_abstract,
        "missing_pmcid": missing_pmcid,
        "json_errors": json_errors,
        "text_token_stats": summarise(overall_buckets.text_tokens),
        "abstract_token_stats": summarise(overall_buckets.abstract_tokens),
        "text_char_stats": summarise(overall_buckets.text_chars),
        "abstract_char_stats": summarise(overall_buckets.abstract_chars),
        "paired_records": len(overall_buckets.text_with_abstract_tokens),
        "paired_text_token_stats": summarise(overall_buckets.text_with_abstract_tokens),
        "paired_abstract_token_stats": summarise(
            overall_buckets.abstract_with_text_tokens
        ),
        "token_ratio_stats": summarise_ratio(overall_buckets.token_ratio),
        "char_ratio_stats": summarise_ratio(overall_buckets.char_ratio),
    }

    return {
        "tokenizer": tokenizer_model,
        "add_special_tokens": add_special_tokens,
        "data_dir": str(data_dir),
        "files": per_file,
        "overall": overall,
    }, overall_buckets


def generate_plots(
    plots_dir: Path,
    result: Dict[str, Any],
    buckets: LengthBuckets,
    hist_bins: int,
) -> None:
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise SystemExit(
            "Για να δημιουργηθούν γραφήματα χρειάζεται το matplotlib "
            "(π.χ. `pip install matplotlib`)."
        ) from exc

    plots_dir.mkdir(parents=True, exist_ok=True)

    files_info = result.get("files", [])
    if files_info:
        labels = [str(f["file"]) for f in files_info]
        counts = [int(f["total_records"]) for f in files_info]
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(range(len(labels)), counts, color="#4e79a7")
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_ylabel("Αριθμός εγγραφών")
        ax.set_title("Εγγραφές ανά αρχείο")
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        fig.tight_layout()
        fig.savefig(plots_dir / "records_per_file.png", dpi=200)
        plt.close(fig)
    else:
        logging.warning("Δεν υπάρχουν πληροφορίες αρχείων για γράφημα εγγραφών.")

    if buckets.text_with_abstract_tokens:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.hist(
            buckets.text_with_abstract_tokens,
            bins=hist_bins,
            color="#59a14f",
            edgecolor="black",
        )
        ax.set_xlabel("Μήκος input (tokens) σε εγγραφές με abstract")
        ax.set_ylabel("Συχνότητα")
        ax.set_title("Κατανομή tokens για text")
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        fig.tight_layout()
        fig.savefig(plots_dir / "text_tokens_hist.png", dpi=200)
        plt.close(fig)
    else:
        logging.warning(
            "Δεν βρέθηκαν ζευγάρια text+abstract για histogram κειμένου."
        )

    if buckets.abstract_with_text_tokens:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.hist(
            buckets.abstract_with_text_tokens,
            bins=hist_bins,
            color="#f28e2b",
            edgecolor="black",
        )
        ax.set_xlabel("Μήκος abstract (tokens) σε εγγραφές με text")
        ax.set_ylabel("Συχνότητα")
        ax.set_title("Κατανομή tokens για abstract")
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        fig.tight_layout()
        fig.savefig(plots_dir / "abstract_tokens_hist.png", dpi=200)
        plt.close(fig)
    else:
        logging.warning(
            "Δεν βρέθηκαν ζευγάρια text+abstract για histogram abstract."
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Υπολογίζει στατιστικά μήκους tokens/χαρακτήρων με tokenizer Qwen."
    )
    parser.add_argument(
        "--data-dir",
        default="data/jsonl",
        type=Path,
        help="Φάκελος με τα αρχεία *.jsonl.gz.",
    )
    parser.add_argument(
        "--glob",
        default="*.jsonl.gz",
        help="Pattern αρχείων (προεπιλογή: *.jsonl.gz).",
    )
    parser.add_argument(
        "--tokenizer",
        default="Qwen/Qwen2.5-7B",
        help="Όνομα ή τοπικό path tokenizer από HuggingFace.",
    )
    parser.add_argument(
        "--no-trust-remote-code",
        action="store_true",
        help="Απενεργοποιεί το trust_remote_code (αν δεν χρειάζεται).",
    )
    parser.add_argument(
        "--add-special-tokens",
        action="store_true",
        help="Υπολογίζει μήκη συμπεριλαμβάνοντας ειδικά tokens BOS/EOS.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Περιορισμός πλήθους εγγραφών ανά αρχείο για γρήγορο sampling.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("stats/qwen_token_stats.json"),
        help="Που θα αποθηκευτεί το JSON με τα συνοπτικά αποτελέσματα.",
    )
    parser.add_argument(
        "--plots-dir",
        type=Path,
        default=Path("stats/plots"),
        help="Φάκελος για αποθήκευση PNG γραφημάτων (default stats/plots).",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Να μην δημιουργηθούν καθόλου γραφήματα.",
    )
    parser.add_argument(
        "--hist-bins",
        type=int,
        default=50,
        help="Πλήθος bins για τα histograms tokens (default 50).",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Ελάχιστο επίπεδο logging.",
    )
    return parser.parse_args()


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False)
    logging.info("Αποθήκευση αποτελεσμάτων στο %s", path)


def main() -> None:
    args = parse_args()
    configure_logging(args.log_level)

    logging.info("Εκκίνηση ανάλυσης στον φάκελο %s", args.data_dir)
    result, buckets = run_analysis(
        data_dir=args.data_dir,
        glob_pattern=args.glob,
        tokenizer_model=args.tokenizer,
        trust_remote_code=not args.no_trust_remote_code,
        add_special_tokens=args.add_special_tokens,
        limit=args.limit,
    )

    write_json(args.output_json, result)

    if not args.no_plots and args.plots_dir is not None:
        logging.info("Δημιουργία γραφημάτων στο %s", args.plots_dir)
        generate_plots(
            plots_dir=args.plots_dir,
            result=result,
            buckets=buckets,
            hist_bins=max(1, args.hist_bins),
        )

if __name__ == "__main__":
    main()
