#!/usr/bin/env python3

"""Scan PMC tar archives and report element usage inside abstract/body sections."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path
from typing import DefaultDict, Iterable, Optional
import tarfile
import xml.etree.ElementTree as ET


TARGET_SECTIONS = {"abstract", "body"}
DEFAULT_SOURCE = Path("data/tar")


def strip_namespace(tag: str) -> str:
    """Remove namespace prefix from an XML tag."""
    if "}" in tag:
        return tag.rsplit("}", 1)[-1]
    return tag


def iter_tar_paths(source: Path) -> Iterable[Path]:
    """Yield tar archive paths found directly under `source`."""
    for candidate in sorted(source.iterdir()):
        if not candidate.is_file():
            continue
        try:
            with tarfile.open(candidate, mode="r:*"):
                yield candidate
        except tarfile.TarError:
            continue


def collect_section_tags(root: ET.Element, counters: DefaultDict[str, Counter[str]]) -> None:
    """Walk the tree and count element usage inside the target sections."""

    def traverse(node: ET.Element, context: Optional[str]) -> None:
        tag_name = strip_namespace(node.tag)
        new_context = context
        if tag_name in TARGET_SECTIONS:
            new_context = tag_name
        elif context in TARGET_SECTIONS:
            counters[context][tag_name] += 1

        for child in list(node):
            traverse(child, new_context)

    traverse(root, None)


def process_tar_archive(path: Path, counters: DefaultDict[str, Counter[str]]) -> tuple[int, list[str]]:
    """Process a single tar archive and update counters."""
    processed = 0
    errors: list[str] = []

    with tarfile.open(path, mode="r:*") as archive:
        for member in archive.getmembers():
            if not member.isfile() or not member.name.lower().endswith(".xml"):
                continue
            extracted = archive.extractfile(member)
            if extracted is None:
                errors.append(f"{path.name}:{member.name}: unable to extract member")
                continue
            try:
                try:
                    tree = ET.parse(extracted)
                except ET.ParseError as exc:
                    errors.append(f"{path.name}:{member.name}: parse error: {exc}")
                    continue
                collect_section_tags(tree.getroot(), counters)
                processed += 1
            finally:
                extracted.close()

    return processed, errors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Inspect tar archives containing PMC XML files and report element usage "
            "within abstract and body sections."
        )
    )
    parser.add_argument(
        "source",
        nargs="?",
        default=str(DEFAULT_SOURCE),
        help=f"Directory containing tar archives (default: {DEFAULT_SOURCE})",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on number of archives to process (for quick sampling).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source = Path(args.source)
    if not source.exists():
        print(f"Source directory does not exist: {source}")
        return 1
    if not source.is_dir():
        print(f"Source is not a directory: {source}")
        return 1

    counters: DefaultDict[str, Counter[str]] = defaultdict(Counter)
    total_archives = 0
    total_xml = 0
    all_errors: list[str] = []

    for archive_path in iter_tar_paths(source):
        if args.limit is not None and total_archives >= args.limit:
            break
        processed, errors = process_tar_archive(archive_path, counters)
        total_archives += 1
        total_xml += processed
        all_errors.extend(errors)

    print(f"Archives processed: {total_archives}")
    print(f"XML files inspected: {total_xml}")
    if all_errors:
        print("Warnings:")
        for message in all_errors:
            print(f"  - {message}")

    for section in sorted(TARGET_SECTIONS):
        counter = counters.get(section)
        if not counter:
            print(f"\nNo elements found inside <{section}>.")
            continue
        print(f"\nElements inside <{section}>:")
        width = max(len(tag) for tag in counter)
        for tag, count in counter.most_common():
            print(f"  {tag.ljust(width)}  {count}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

