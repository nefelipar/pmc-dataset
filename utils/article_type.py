#!/usr/bin/env python3

"""Report article-type frequencies in PMC XML files."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import sys
import tarfile
from typing import IO, Optional, Tuple
import xml.etree.ElementTree as ET


DEFAULT_SOURCE = Path("data/tar/PMC000xxxxxx")
UNTYPED_LABEL = "untyped"


def _strip_namespace(tag: str) -> str:
    if "}" in tag:
        return tag.rsplit("}", 1)[-1]
    return tag


def _normalise(value: Optional[str]) -> str:
    if value is None:
        return UNTYPED_LABEL
    value = value.strip()
    if not value:
        return UNTYPED_LABEL
    return value


def _article_type_from_stream(stream: IO[bytes], label: str) -> Tuple[Optional[str], Optional[str]]:
    try:
        for event, elem in ET.iterparse(stream, events=("start",)):
            if _strip_namespace(elem.tag) == "article":
                article_type = (
                    elem.attrib.get("article-type")
                    or elem.attrib.get("articleType")
                    or elem.attrib.get("type")
                )
                return _normalise(article_type), None
        return None, None
    except ET.ParseError as exc:
        return None, f"{label}: {exc}"


def _article_type_from_path(path: Path) -> Tuple[Optional[str], Optional[str]]:
    with path.open("rb") as handle:
        return _article_type_from_stream(handle, str(path))


def _process_directory(path: Path, counter: Counter[str]) -> Tuple[int, list[str]]:
    processed = 0
    errors: list[str] = []
    for xml_path in sorted(path.rglob("*.xml")):
        article_type, error = _article_type_from_path(xml_path)
        if error:
            errors.append(error)
            continue
        if article_type is None:
            continue
        counter[article_type] += 1
        processed += 1
    return processed, errors


def _process_tar(path: Path, counter: Counter[str]) -> Tuple[int, list[str]]:
    processed = 0
    errors: list[str] = []
    with tarfile.open(path, mode="r:*") as archive:
        for member in archive.getmembers():
            if not member.isfile() or not member.name.lower().endswith(".xml"):
                continue
            extracted = archive.extractfile(member)
            if extracted is None:
                errors.append(f"{member.name}: unable to read from archive")
                continue
            try:
                article_type, error = _article_type_from_stream(extracted, member.name)
            finally:
                extracted.close()
            if error:
                errors.append(error)
                continue
            if article_type is None:
                continue
            counter[article_type] += 1
            processed += 1
    return processed, errors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect article-type values used in PMC XML files."
    )
    parser.add_argument(
        "source",
        nargs="?",
        default=str(DEFAULT_SOURCE),
        help=(
            "Directory, tar, or tar.gz archive containing PMC XML files "
            f"(default: {DEFAULT_SOURCE})"
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source = Path(args.source)
    counter: Counter[str] = Counter()

    if source.is_dir():
        processed, errors = _process_directory(source, counter)
    elif source.is_file() and tarfile.is_tarfile(source):
        processed, errors = _process_tar(source, counter)
    elif source.is_file() and source.suffix.lower() == ".xml":
        article_type, error = _article_type_from_path(source)
        processed = 0 if error else 1
        errors = [error] if error else []
        if article_type is not None:
            counter[article_type] += 1
    else:
        print(f"Source '{source}' is not a directory, tar archive, or XML file.", file=sys.stderr)
        return 1

    if errors:
        print("Some files could not be processed:", file=sys.stderr)
        for message in errors:
            print(f"  - {message}", file=sys.stderr)

    total_articles = sum(counter.values())
    print(f"XML files with article-type: {processed}")
    print(f"Unique article-type values: {len(counter)}")

    if not counter:
        return 0

    width = max(len(label) for label in counter)
    print("\narticle-type".ljust(width + 2) + "count    share")
    for article_type, count in counter.most_common():
        share = (count / total_articles * 100) if total_articles else 0.0
        print(f"{article_type.ljust(width + 2)}{count:>5}  {share:6.2f}%")

    return 0


if __name__ == "__main__":
    sys.exit(main())

