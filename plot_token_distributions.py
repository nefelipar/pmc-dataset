#!/usr/bin/env python3
"""Plot histograms for text and abstract token counts from stats_per_record JSONL files."""

from __future__ import annotations
import argparse
import json
import logging
from pathlib import Path
from typing import Iterable, List, Tuple
import matplotlib.pyplot as plt 
import numpy as np 
from tqdm import tqdm  


def iter_jsonl_records(paths: Iterable[Path]):
    """Yield JSON objects from a list of .jsonl file paths."""
    for path in paths:
        logging.info("Reading %s", path)
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as exc:
                    logging.warning("Skipping bad JSON in %s: %s", path.name, exc)
                    continue
                yield obj


def collect_token_counts(stats_dir: Path) -> Tuple[List[int], List[int]]:
    """Return token counts for text and abstract fields."""
    files = sorted(stats_dir.glob("*.jsonl"))
    if not files:
        raise SystemExit(f"No .jsonl files found in {stats_dir}")

    text_tokens: List[int] = []
    abstract_tokens: List[int] = []

    for rec in tqdm(iter_jsonl_records(files), desc="Records", unit="rec"):
        t_tok = rec.get("text_tokens")
        if isinstance(t_tok, int):
            text_tokens.append(t_tok)
        a_tok = rec.get("abstract_tokens")
        if isinstance(a_tok, int):
            abstract_tokens.append(a_tok)
    return text_tokens, abstract_tokens


def plot_histogram(
    values: List[int],
    title: str,
    xlabel: str,
    output_path: Path,
    bins: int,
    x_min: float | None = None,
    x_max: float | None = None,
    stats_lines: dict | None = None,
) -> None:
    """Plot and save a histogram for the given values."""
    if not values:
        logging.warning("No data to plot for %s", title)
        return

    arr = np.array(values, dtype=np.int64)

    plt.figure(figsize=(10, 6))
    lower_edge = float(x_min) if x_min is not None else float(arr.min())
    upper_edge = float(x_max) if x_max is not None else float(arr.max())
    if lower_edge <= 0:
        lower_edge = 1.0
    if upper_edge <= lower_edge:
        upper_edge = lower_edge + 1.0

    # Auto-select log x when range is very wide to avoid squashing.
    use_log_x = (upper_edge / lower_edge) > 1e3
    if use_log_x:
        edges = np.logspace(np.log10(lower_edge), np.log10(upper_edge), bins + 1)
        plt.xscale("log")
    else:
        edges = np.linspace(lower_edge, upper_edge, bins + 1)

    plt.hist(arr, bins=edges, color="#4D9FE2", edgecolor="black", alpha=0.9)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Records")

    plt.yscale("log")
    ymin, ymax = plt.ylim()
    plt.ylim(bottom=max(1, ymin))

    if stats_lines:
        mean = stats_lines.get("mean")
        median = stats_lines.get("median")
        p95 = stats_lines.get("p95")
        if mean is not None:
            plt.axvline(mean, color="#F70000", linestyle="--", linewidth=1.5, label=f"Mean: {mean:.1f}")
        if median is not None:
            plt.axvline(median, color="#0C042E", linestyle="-.", linewidth=1.5, label=f"Median: {median:.1f}")
        if p95 is not None:
            plt.axvline(p95, color="#026C1B", linestyle=":", linewidth=1.5, label=f"P95: {p95:.1f}")
    plt.legend()
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()
    logging.info("Saved plot: %s", output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create token count plots from stats_per_record/*.jsonl files.",
    )
    parser.add_argument(
        "--tokenizer",
        default="Qwen/Qwen1.5-1.8B",
        help="Tokenizer name; used to infer default paths under __stats__/",
    )
    parser.add_argument(
        "--stats-per-record",
        type=Path,
        default=None,
        help="Path to stats_per_record directory (default: __stats__/<tokenizer>/stats_per_record)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to write plots (default: __stats__/<tokenizer>/plots)",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=100,
        help="Number of histogram bins.",
    )
    parser.add_argument(
        "--overall-json",
        type=Path,
        default=None,
        help="Optional overall.json to read aggregate stats (used for logging/summary lines).",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Logging level (DEBUG, INFO, ...).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(levelname)s: %(message)s",
    )

    tok = args.tokenizer.replace("/", "_")
    stats_dir = args.stats_per_record or (Path("__stats__") / tok / "stats_per_record")
    output_dir = args.output_dir or (Path("__stats__") / tok / "plots")
    overall_path = args.overall_json or (Path("__stats__") / tok / "overall.json")

    if not stats_dir.exists() or not stats_dir.is_dir():
        raise SystemExit(f"Stats folder not found: {stats_dir}")
    if not overall_path.exists():
        raise SystemExit(f"overall.json not found: {overall_path}")

    text_tokens, abstract_tokens = collect_token_counts(stats_dir)

    try:
        overall = json.loads(overall_path.read_text(encoding="utf-8"))
        text_stats = overall.get("text_tokens") or {}
        abstract_stats = overall.get("abstract_tokens") or {}
        logging.info(
            "Text tokens (from overall.json): count=%s min=%s max=%s mean=%.1f p99=%.1f",
            text_stats.get("count"),
            text_stats.get("min"),
            text_stats.get("max"),
            text_stats.get("mean", 0.0),
            text_stats.get("p99"),
        )
        logging.info(
            "Abstract tokens (from overall.json): count=%s min=%s max=%s mean=%.1f p99=%.1f",
            abstract_stats.get("count"),
            abstract_stats.get("min"),
            abstract_stats.get("max"),
            abstract_stats.get("mean", 0.0),
            abstract_stats.get("p99"),
        )
    except Exception as exc:
        raise SystemExit(f"Failed to read overall stats from {overall_path}: {exc}") from exc

    plot_histogram(
        text_tokens,
        title="Text token counts per record",
        xlabel="text_tokens",
        output_path=output_dir / "text_tokens_hist.png",
        bins=args.bins,
        x_min=text_stats.get("min"),
        x_max=text_stats.get("max"),
        stats_lines={
            "mean": text_stats.get("mean"),
            "median": text_stats.get("median"),
            "p95": text_stats.get("p95"),
        },
    )
    plot_histogram(
        abstract_tokens,
        title="Abstract token counts per record",
        xlabel="abstract_tokens",
        output_path=output_dir / "abstract_tokens_hist.png",
        bins=args.bins,
        x_min=abstract_stats.get("min"),
        x_max=abstract_stats.get("max"),
        stats_lines={
            "mean": abstract_stats.get("mean"),
            "median": abstract_stats.get("median"),
            "p95": abstract_stats.get("p95"),
        },
    )


if __name__ == "__main__":
    main()
