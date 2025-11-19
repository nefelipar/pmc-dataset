#!/usr/bin/env python3
"""
Filter jsonl.gz files keeping only records that include both text and abstract.
For every `*.jsonl.gz` file inside the target directory the script emits a new
gzipped JSONL file that only contains records where both the `text` and
`abstract` fields exist and are non-empty (after stripping whitespace). By
default the cleaned file sits next to the source one and uses the suffix
`.clean.jsonl.gz`, e.g. `foo.jsonl.gz -> foo.clean.jsonl.gz`. After cleaning,
the original `*.jsonl.gz` file is deleted to save space.
"""

from __future__ import annotations
import argparse
import gzip
import json
import logging
from pathlib import Path
from typing import Iterable, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Keep only records that contain both a non-empty text and abstract.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/jsonl"),
        help="Directory containing the *.jsonl.gz files (default: data/jsonl).",
    )
    parser.add_argument(
        "--suffix",
        default=".clean",
        help="Suffix inserted before .jsonl.gz for the cleaned file (default: .clean).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/cleaned"),
        help="Optional directory for cleaned files (default: data/cleaned).",
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        default=Path("data/cleaned/cleaning_report.json"),
        help=(
            "Path to a JSON log file that will store per-file stats (default: data/cleaned/cleaning_report.json)."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing cleaned files instead of skipping them.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable INFO logging.",
    )
    return parser.parse_args()


def iter_jsonl_records(path: Path) -> Iterable[Tuple[int, dict]]:
    """Yield (line_number, json_record) pairs for a gzipped JSONL file."""
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        for idx, line in enumerate(fh, 1):
            if not line.strip():
                continue
            try:
                yield idx, json.loads(line)
            except json.JSONDecodeError:
                logging.warning("Skipping invalid JSON at %s line %d", path, idx)


def has_text(value) -> bool:
    """Return True when value is a non-empty string."""
    return isinstance(value, str) and bool(value.strip())


def output_path_for(input_path: Path, suffix: str, output_dir: Path | None) -> Path:
    """Return the output path for a cleaned file."""
    name = input_path.name
    if not name.endswith(".jsonl.gz"):
        raise ValueError(f"{input_path} is not a .jsonl.gz file")
    base = name[: -len(".jsonl.gz")]
    if suffix and not suffix.startswith("."):
        suffix = f".{suffix}"
    cleaned_name = f"{base}{suffix}.jsonl.gz"
    target_dir = output_dir if output_dir else input_path.parent
    return target_dir / cleaned_name


def remove_source_tar(input_path: Path) -> None:
    """Delete the original *.jsonl.gz (if present) once cleaning finishes."""
    gz_path = input_path.with_suffix(".gz")
    if not gz_path.exists():
        return
    try:
        gz_path.unlink()
        logging.info("Deleted original archive %s", gz_path)
    except OSError as exc:
        logging.warning("Failed to delete %s: %s", gz_path, exc)


def clean_file(path: Path,suffix: str, output_dir: Path | None, overwrite: bool) -> Tuple[int, int, int, int, int]:
    """Return (kept, skipped, text_only, abstract_only, missing_both) for one file."""
    output_path = output_path_for(path, suffix, output_dir)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists() and not overwrite:
        logging.info("Skipping %s (already exists)", output_path)
        return 0, 0, 0, 0, 0

    kept = 0
    skipped = 0
    text_only = 0
    abstract_only = 0
    missing_both = 0

    with gzip.open(output_path, "wt", encoding="utf-8") as out_fh:
        for line_no, record in iter_jsonl_records(path):
            text_ok = has_text(record.get("text"))
            abstract_ok = has_text(record.get("abstract"))

            if text_ok and abstract_ok:
                json.dump(record, out_fh, ensure_ascii=False)
                out_fh.write("\n")
                kept += 1
                continue

            skipped += 1
            if text_ok and not abstract_ok:
                text_only += 1
            elif abstract_ok and not text_ok:
                abstract_only += 1
            else:
                missing_both += 1
            logging.debug("Removed record without full text+abstract from %s line %d", path, line_no)

    logging.info("Cleaned %s -> %s (kept %d, removed %d)", path.name, output_path.name, kept, skipped)
    
    remove_source_tar(path)
    return kept, skipped, text_only, abstract_only, missing_both


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s: %(message)s",
    )

    data_dir: Path = args.data_dir
    if not data_dir.is_dir():
        raise SystemExit(f"Data directory not found: {data_dir}")

    log_path: Path | None = args.log_file
    log_entries: list[dict[str, int | str]] = []
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)

    files = sorted(data_dir.glob("*.jsonl.gz"))
    if not files:
        raise SystemExit(f"No .jsonl.gz files found in {data_dir}")

    total_kept = 0
    total_removed = 0
    total_text_only = 0
    total_abstract_only = 0
    total_missing_both = 0

    for file_path in files:
        kept, removed, text_only, abstract_only, missing_both = clean_file(
            file_path,
            args.suffix,
            args.output_dir,
            args.overwrite,
        )
        total_kept += kept
        total_removed += removed
        total_text_only += text_only
        total_abstract_only += abstract_only
        total_missing_both += missing_both

        missing_text = abstract_only + missing_both
        missing_abstract = text_only + missing_both

        if log_path is not None:
            log_entries.append(
                {
                    "file": file_path.name,
                    "text_and_abstract": kept,
                    "only_abstract": abstract_only,
                    "only_text": text_only,
                    "missing_text": missing_text,
                    "missing_abstract": missing_abstract,
                    "missing_both": missing_both,
                }
            )

    total_missing_text = total_abstract_only + total_missing_both
    total_missing_abstract = total_text_only + total_missing_both

    if log_path is not None:
        log_entries.append(
            {
                "file": "TOTAL",
                "text_and_abstract": total_kept,
                "only_abstract": total_abstract_only,
                "only_text": total_text_only,
                "missing_text": total_missing_text,
                "missing_abstract": total_missing_abstract,
                "missing_both": total_missing_both,
            }
        )
        with log_path.open("w", encoding="utf-8") as log_file:
            json.dump(log_entries, log_file, ensure_ascii=False, indent=2)
        print(f"Stats saved to {log_path}")


if __name__ == "__main__":
    main()
