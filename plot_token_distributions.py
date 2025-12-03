#!/usr/bin/env python3
"""Plot token distributions with a zoomed percentile view plus stat markers."""

from __future__ import annotations
import argparse
import json
import logging
from pathlib import Path
from typing import Iterable, Tuple
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


def collect_token_counts(stats_dir: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Return token counts for text and abstract fields."""
    files = sorted(stats_dir.glob("*.jsonl"))
    if not files:
        raise SystemExit(f"No .jsonl files found in {stats_dir}")

    text_tokens: list[int] = []
    abstract_tokens: list[int] = []

    for rec in tqdm(iter_jsonl_records(files), desc="Records", unit="rec"):
        t_tok = rec.get("text_tokens")
        if isinstance(t_tok, int):
            text_tokens.append(t_tok)
        a_tok = rec.get("abstract_tokens")
        if isinstance(a_tok, int):
            abstract_tokens.append(a_tok)
    return np.asarray(text_tokens, dtype=np.int64), np.asarray(abstract_tokens, dtype=np.int64)


def _add_stat_lines(ax, stats_lines: dict | None, x_min: float, x_max: float, with_labels: bool = True) -> None:
    """Draw mean/median/p95 lines within the visible x-range."""
    if not stats_lines:
        return
    line_styles = {
        "mean": ("#F70000", "--", "Mean"),
        "median": ("#004BFA", "-.", "Median"),
        "p95": ("#008332", ":", "P95"),
    }
    for key, (color, style, label_text) in line_styles.items():
        val = stats_lines.get(key)
        try:
            x_val = float(val)
        except (TypeError, ValueError):
            continue
        if not (x_min <= x_val <= x_max):
            continue
        label = f"{label_text}: {x_val:.1f}" if with_labels else None
        ax.axvline(x_val, color=color, linestyle=style, linewidth=1.5, label=label)


def plot_distribution(
    values: Iterable[int],
    title: str,
    xlabel: str,
    output_path: Path,
    bins: int,
    focus_percentile: float,
    full_output_path: Path | None = None,
    x_min: float | None = None,
    x_max: float | None = None,
    stats_lines: dict | None = None,
) -> None:
    """Plot and save a zoomed linear histogram plus a separate full-range view."""
    arr = np.asarray(values, dtype=np.int64)
    if arr.size == 0:
        logging.warning("No data to plot for %s", title)
        return

    lower_edge = float(x_min) if x_min is not None else float(arr.min())
    upper_edge = float(x_max) if x_max is not None else float(arr.max())
    if upper_edge <= lower_edge:
        upper_edge = lower_edge + 1.0

    focus_percentile = float(np.clip(focus_percentile, 0.0, 100.0))
    focus_edge = min(upper_edge, float(np.percentile(arr, focus_percentile)))
    if focus_edge <= lower_edge:
        focus_edge = lower_edge + (upper_edge - lower_edge) * 0.05

    # Bulk histogram (linear) limited to the chosen percentile so the main mass is visible.
    bulk_bins = np.linspace(lower_edge, focus_edge, bins + 1)
    bulk_counts, _ = np.histogram(arr, bins=bulk_bins)
    bulk_fraction = bulk_counts.sum() / arr.size
    tail_fraction = 1.0 - bulk_fraction

    full_use_log_x = (upper_edge / max(lower_edge, 1.0)) > 1e3
    full_bins = (
        np.logspace(np.log10(max(lower_edge, 1.0)), np.log10(upper_edge), bins + 1)
        if full_use_log_x
        else np.linspace(lower_edge, upper_edge, bins + 1)
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    full_output_path = full_output_path or output_path.with_name(
        f"{output_path.stem}__full{output_path.suffix}"
    )
    full_output_path.parent.mkdir(parents=True, exist_ok=True)

    fig_bulk, ax_bulk = plt.subplots(figsize=(10.5, 5.5))
    ax_bulk.bar(
        bulk_bins[:-1],
        bulk_counts,
        width=np.diff(bulk_bins),
        align="edge",
        color="#4D9FE2",
        edgecolor="black",
        alpha=0.9,
    )
    ax_bulk.set_xlim(lower_edge, focus_edge)
    ax_bulk.set_title(f"{title} (<= p{focus_percentile:.1f} = {focus_edge:.1f} tokens)")
    ax_bulk.set_ylabel("Records (linear)")
    _add_stat_lines(ax_bulk, stats_lines, lower_edge, focus_edge, with_labels=True)
    if tail_fraction > 0:
        ax_bulk.text(
            0.99,
            0.92,
            f"{tail_fraction * 100:.2f}% of data above this range",
            transform=ax_bulk.transAxes,
            ha="right",
            va="top",
            fontsize=9,
            bbox={"facecolor": "white", "edgecolor": "lightgray", "alpha": 0.8},
        )
    ax_bulk.legend()
    ax_bulk.grid(alpha=0.2)
    fig_bulk.tight_layout()
    fig_bulk.savefig(output_path, dpi=200)
    plt.close(fig_bulk)
    logging.info("Saved plot: %s", output_path)

    fig_full, ax_full = plt.subplots(figsize=(10.5, 5.0))
    ax_full.hist(
        arr,
        bins=full_bins,
        color="#4D9FE2",
        edgecolor="black",
        alpha=0.9,
    )
    if full_use_log_x:
        ax_full.set_xscale("log")
    ax_full.set_yscale("log")
    ymin, _ = ax_full.get_ylim()
    ax_full.set_ylim(bottom=max(1, ymin))
    ax_full.set_xlim(lower_edge, upper_edge)
    ax_full.set_ylabel("Records (log)")
    ax_full.set_xlabel(xlabel)
    ax_full.set_title(f"{title} - Full distribution (log-y, auto log-x for wide ranges)")
    _add_stat_lines(ax_full, stats_lines, lower_edge, upper_edge, with_labels=False)
    ax_full.grid(alpha=0.25)
    fig_full.tight_layout()
    fig_full.savefig(full_output_path, dpi=200)
    plt.close(fig_full)
    logging.info("Saved plot: %s", full_output_path)


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
        "--focus-percentile",
        type=float,
        default=99.5,
        help="Upper percentile to show on the linear histogram (keeps the long tail off-plot).",
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

    plot_distribution(
        text_tokens,
        title="Text token counts per record",
        xlabel="text_tokens",
        output_path=output_dir / "text_tokens_counts_per_record.png",
        full_output_path=output_dir / "text_tokens_full_distribution.png",
        bins=args.bins,
        focus_percentile=args.focus_percentile,
        x_min=text_stats.get("min"),
        x_max=text_stats.get("max"),
        stats_lines={
            "mean": text_stats.get("mean"),
            "median": text_stats.get("median"),
            "p95": text_stats.get("p95"),
        },
    )
    plot_distribution(
        abstract_tokens,
        title="Abstract token counts per record",
        xlabel="abstract_tokens",
        output_path=output_dir / "abstract_tokens_counts_per_record.png",
        full_output_path=output_dir / "abstract_tokens_full_distribution.png",
        bins=args.bins,
        focus_percentile=args.focus_percentile,
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
