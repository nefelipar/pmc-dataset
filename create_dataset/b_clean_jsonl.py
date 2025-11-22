#!/usr/bin/env python3
"""Keep only records that include both a non-empty text and abstract.
The kept records are saved to a new file with a specified suffix (default: .clean), and the
original file is deleted. A log of removed PMCIDs is also created per file.
Also generates a summary report JSON file with statistics per file and overall totals.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# Define the keys used for statistics
STAT_KEYS = ("kept", "removed", "text_only", "abstract_only", "missing_both")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Keep only records that contain both a non-empty text and abstract.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("__dataset__/jsonl"),
        help="Directory containing the *.jsonl files (default: __dataset__/jsonl).",
    )
    parser.add_argument(
        "--suffix",
        default=".clean",
        help="Suffix inserted before .jsonl for the cleaned file (default: .clean).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("__dataset__/cleaned"),
        help="Optional directory for cleaned files (default: __dataset__/cleaned).",
    )
    parser.add_argument(
        "--removed-log-dir",
        type=Path,
        default=Path("__dataset__/cleaned/removed_pmcs"),
        help="Directory where per-file lists of removed PMCIDs are stored (default: __dataset__/cleaned/removed_pmcs).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing cleaned files instead of skipping them.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show extra progress information.",
    )
    return parser.parse_args()


def has_text(value: object) -> bool:
    """Check if the given value is a non-empty string."""
    return isinstance(value, str) and bool(value.strip())


def build_output_path(path: Path, suffix: str, output_dir: Path | None) -> Path:
    """Build the output file path with the given suffix and optional output directory."""
    if not path.name.endswith(".jsonl"):
        raise SystemExit(f"{path} is not a .jsonl file")

    suffix = suffix or ""
    if suffix and not suffix.startswith("."):
        suffix = f".{suffix}"

    base_name = path.name[:-len(".jsonl")]
    target_dir = output_dir if output_dir else path.parent
    target_dir.mkdir(parents=True, exist_ok=True)
    return target_dir / f"{base_name}{suffix}.jsonl"


def clean_file(path: Path, args: argparse.Namespace) -> tuple[dict[str, int], bool]:
    """Clean the given .jsonl file by keeping only records with both text and abstract."""
    stats = {key: 0 for key in STAT_KEYS}
    output_path = build_output_path(path, args.suffix, args.output_dir)

    if output_path.exists() and not args.overwrite:
        if args.verbose:
            print(f"[skip] Skipping {path.name} because the cleaned file already exists")
        return stats, True

    if args.removed_log_dir:
        removed_dir = args.removed_log_dir
    else:
        base_dir = args.output_dir if args.output_dir else path.parent
        removed_dir = base_dir / "removed_pmcs"
    removed_dir.mkdir(parents=True, exist_ok=True)
    log_path = removed_dir / f"{path.name[:-len('.jsonl')]}.removed.log"
    print(f"[LOG] Removed PMCIDs stored in {log_path}")

    with (
        path.open("r", encoding="utf-8") as src,
        output_path.open("w", encoding="utf-8") as dst,
        log_path.open("w", encoding="utf-8") as removed_log,
    ):
        for line_no, line in enumerate(src, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                if args.verbose:
                    print(f"[skip] Invalid JSON in {path.name} line {line_no}, skipping it")
                continue

            text_ok = has_text(record.get("text"))
            abstract_ok = has_text(record.get("abstract"))

            if text_ok and abstract_ok:
                json.dump(record, dst, ensure_ascii=False)
                dst.write("\n")
                stats["kept"] += 1
                continue

            stats["removed"] += 1
            if text_ok and not abstract_ok:
                stats["text_only"] += 1
            elif abstract_ok and not text_ok:
                stats["abstract_only"] += 1
            else:
                stats["missing_both"] += 1

            pmcid = record.get("pmcid")
            if pmcid not in (None, ""):
                removed_log.write(f"{pmcid}\n")

    print(f"[OUT] Cleaned file written to {output_path}")
    try:
        path.unlink()
        print(f"[cleanup] Removed original file {path}")
    except OSError as exc:
        if args.verbose:
            print(f"[error] Could not delete {path}: {exc}")

    return stats, False


def main() -> None:
    args = parse_args()
    print("[info] Step 2: Keep only the records with both non-empty text and abstract")
    data_dir = args.data_dir
    if not data_dir.is_dir():
        raise SystemExit(f"Data directory not found: {data_dir}")

    files = sorted(data_dir.glob("*.jsonl"))
    if not files:
        raise SystemExit(f"No .jsonl files found in {data_dir}")

    report_path = args.output_dir / "cleaning_report.json"
    totals = {key: 0 for key in STAT_KEYS}
    report_entries: list[dict[str, int | str]] = []

    for file_path in files:
        stats, skipped = clean_file(file_path, args)
        for key in STAT_KEYS:
            totals[key] += stats[key]

        if skipped:
            print(f"[skip] {file_path.name}: already cleaned")
        else:
            print(
                f"[OK] {file_path.name}: kept={stats['kept']} | only text={stats['text_only']} | "
                f"only abstract={stats['abstract_only']} | missing both={stats['missing_both']}"
            )

        report_entries.append(
            {
                "file": file_path.name,
                "text_and_abstract": stats["kept"],
                "only_abstract": stats["abstract_only"],
                "only_text": stats["text_only"],
                "missing_both": stats["missing_both"],
            }
        )

    report_entries.append(
        {
            "file": "TOTAL",
            "text_and_abstract": totals["kept"],
            "only_abstract": totals["abstract_only"],
            "only_text": totals["text_only"],
            "missing_both": totals["missing_both"],
        }
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as report_file:
        json.dump(report_entries, report_file, ensure_ascii=False, indent=2)
    print(f"[OUT] Stats saved to {report_path}")

    try:
        if not any(data_dir.iterdir()):
            data_dir.rmdir()
            print(f"[total cleanup] Removed empty directory {data_dir}")
    except OSError:
        pass
    print("----------------------------------------------")


if __name__ == "__main__":
    main()
