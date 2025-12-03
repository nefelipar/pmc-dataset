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

LINE_STYLE_DEFAULTS = {
    "mean": ("#F70000", "--", "Mean"),
    "median": ("#0C042E", "-.", "Median"),
    "p01": ("#8A1C7C", ":", "P01"),
    "p05": ("#C12D6B", "--", "P05"),
    "p10": ("#7B3F00", ":", "P10"),
    "p15": ("#5D4037", "--", "P15"),
    "p20": ("#0B7285", "-.", "P20"),
    "p80": ("#0B7285", "--", "P80"),
    "p85": ("#1B5E20", "-.", "P85"),
    "p90": ("#026C1B", "--", "P90"),
    "p95": ("#026C1B", ":", "P95"),
    "p99": ("#AA2C20", "-", "P99"),
}


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


def _format_percentile_key(pct: int) -> str:
    """Format an integer percentile (0-100) as a stat key (e.g., 1 -> p01)."""
    pct_clamped = max(0, min(100, int(pct)))
    return f"p{pct_clamped:02d}"


def _parse_percentile_value(raw: str) -> int | None:
    """Parse user input like 'p01', '1', '99%' into an integer percentile."""
    cleaned = raw.strip().lower().replace("%", "")
    if cleaned.startswith("p"):
        cleaned = cleaned[1:]
    try:
        pct = float(cleaned)
    except ValueError:
        logging.warning("Could not parse percentile value: %s", raw)
        return None
    if not (0 <= pct <= 100):
        logging.warning("Percentile out of range [0, 100]: %s", raw)
        return None
    if not pct.is_integer():
        logging.warning("Non-integer percentile provided (%s); rounding to nearest integer.", raw)
    return int(round(pct))


def parse_percentile_pairs(raw_pairs: Iterable[str]) -> list[tuple[int, int]]:
    """Parse CLI percentile pairs like 'p01,p99' or '5,95'."""
    pairs: list[tuple[int, int]] = []
    for raw in raw_pairs:
        parts = [p for p in raw.split(",") if p.strip()]
        if len(parts) != 2:
            logging.warning("Ignoring percentile pair (needs two values): %s", raw)
            continue
        low = _parse_percentile_value(parts[0])
        high = _parse_percentile_value(parts[1])
        if low is None or high is None:
            continue
        pairs.append((low, high))
    return pairs


def percentile_lines_from_stats(
    percentiles: Iterable[int],
    overall_stats: dict,
    data: np.ndarray,
) -> dict[str, float]:
    """Build a {key: value} map for requested percentiles using overall.json or fallback."""
    lines: dict[str, float] = {}
    for pct in percentiles:
        key = _format_percentile_key(pct)
        val = overall_stats.get(key)
        if val is None:
            # Fallback to computing directly if overall.json does not include it.
            try:
                val = float(np.percentile(data, pct))
                logging.warning("Percentile %s missing in overall stats; computed from data.", key)
            except Exception as exc:  # pragma: no cover - defensive
                logging.warning("Failed to compute percentile %s: %s", key, exc)
                continue
        try:
            lines[key] = float(val)
        except (TypeError, ValueError):
            logging.warning("Ignoring non-numeric percentile %s value: %s", key, val)
    return lines


def _add_stat_lines(
    ax,
    stats_lines: dict | None,
    x_min: float,
    x_max: float,
    with_labels: bool = True,
    line_styles: dict[str, tuple[str, str, str]] | None = None,
) -> None:
    """Draw stat lines (mean/median/percentiles) within the visible x-range."""
    if not stats_lines:
        return
    styles = line_styles or LINE_STYLE_DEFAULTS
    fallback_color_cycle = ["#444444", "#666666", "#888888", "#AAAAAA"]
    fallback_styles = ["--", "-.", ":", "-"]
    fallback_count = 0

    for key in stats_lines:
        color, style, label_text = styles.get(
            key,
            (
                fallback_color_cycle[fallback_count % len(fallback_color_cycle)],
                fallback_styles[fallback_count % len(fallback_styles)],
                key.upper(),
            ),
        )
        if key not in styles:
            fallback_count += 1
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
    line_styles: dict[str, tuple[str, str, str]] | None = None,
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
    _add_stat_lines(ax_bulk, stats_lines, lower_edge, focus_edge, with_labels=True, line_styles=line_styles)
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
    handles, labels = ax_bulk.get_legend_handles_labels()
    if labels:
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
    _add_stat_lines(ax_full, stats_lines, lower_edge, upper_edge, with_labels=True, line_styles=line_styles)
    ax_full.grid(alpha=0.25)
    handles, labels = ax_full.get_legend_handles_labels()
    if labels:
        ax_full.legend()
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
        "--percentile-pairs",
        nargs="*",
        default=[],
        help=(
            "Optional percentile pairs to plot as additional figures, e.g. 'p01,p99 p05,p95'. "
            "Each pair draws labeled vertical lines at those percentiles."
        ),
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
    percentile_pairs = parse_percentile_pairs(args.percentile_pairs)

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
        line_styles=LINE_STYLE_DEFAULTS,
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
        line_styles=LINE_STYLE_DEFAULTS,
    )

    for low, high in percentile_pairs:
        lo_pct, hi_pct = sorted((low, high))
        suffix = f"p{lo_pct:02d}_p{hi_pct:02d}"
        pair_dir = output_dir / suffix
        text_lines = percentile_lines_from_stats((lo_pct, hi_pct), text_stats, text_tokens)
        if text_lines:
            plot_distribution(
                text_tokens,
                title=f"Text tokens (lines at p{lo_pct:02d} & p{hi_pct:02d})",
                xlabel="text_tokens",
                output_path=pair_dir / "text_tokens.png",
                full_output_path=pair_dir / "text_tokens__full.png",
                bins=args.bins,
                focus_percentile=args.focus_percentile,
                x_min=text_stats.get("min"),
                x_max=text_stats.get("max"),
                stats_lines=text_lines,
                line_styles=LINE_STYLE_DEFAULTS,
            )
        abstract_lines = percentile_lines_from_stats((lo_pct, hi_pct), abstract_stats, abstract_tokens)
        if abstract_lines:
            plot_distribution(
                abstract_tokens,
                title=f"Abstract tokens (lines at p{lo_pct:02d} & p{hi_pct:02d})",
                xlabel="abstract_tokens",
                output_path=pair_dir / "abstract_tokens.png",
                full_output_path=pair_dir / "abstract_tokens__full.png",
                bins=args.bins,
                focus_percentile=args.focus_percentile,
                x_min=abstract_stats.get("min"),
                x_max=abstract_stats.get("max"),
                stats_lines=abstract_lines,
                line_styles=LINE_STYLE_DEFAULTS,
            )


if __name__ == "__main__":
    main()
