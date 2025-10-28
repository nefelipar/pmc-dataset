#!/usr/bin/env python3

"""Summarise abstract-type usage across PMC XML files.

The script scans a directory of PMC articles (or a tar/tar.gz archive) and
aggregates the values of the ``abstract-type`` attribute found on ``<abstract>``
elements. Untyped abstracts are reported under the ``untyped`` bucket.
"""

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
    """Return the local name of an XML tag."""
    if "}" in tag:
        return tag.rsplit("}", 1)[-1]
    return tag


def _normalise_type(value: Optional[str]) -> str:
    """Normalise the abstract-type attribute."""
    if value is None:
        return UNTYPED_LABEL
    value = value.strip()
    if not value:
        return UNTYPED_LABEL
    return value


def _collect_from_stream(stream: IO[bytes], label: str, counter: Counter[str]) -> Tuple[int, Optional[str]]:
    """Collect abstract types from an XML byte stream.

    Returns:
        A tuple (abstract_count, error_message). The error message is None when
        parsing succeeded.
    """
    abstracts_in_file = 0
    try:
        # Only listening to "start" keeps memory usage low for large files.
        for _, elem in ET.iterparse(stream, events=("start",)):
            if _strip_namespace(elem.tag) == "abstract":
                abstracts_in_file += 1
                abstract_type = (
                    elem.attrib.get("abstract-type")
                    or elem.attrib.get("abstractType")
                    or elem.attrib.get("type")
                )
                counter[_normalise_type(abstract_type)] += 1
        return abstracts_in_file, None
    except ET.ParseError as exc:
        return 0, f"{label}: {exc}"


def _collect_from_path(path: Path, counter: Counter[str]) -> Tuple[int, Optional[str]]:
    """Collect abstract types from a filesystem XML path."""
    with path.open("rb") as handle:
        return _collect_from_stream(handle, str(path), counter)


def _process_directory(path: Path, counter: Counter[str]) -> Tuple[int, int, int, list[str]]:
    """Scan every XML file within a directory tree."""
    parsed_files = 0
    total_abstracts = 0
    errors: list[str] = []
    files_without_abstract = 0
    for xml_path in sorted(path.rglob("*.xml")):
        abstracts_found, error = _collect_from_path(xml_path, counter)
        if error:
            errors.append(error)
            continue
        parsed_files += 1
        total_abstracts += abstracts_found
        if abstracts_found == 0:
            files_without_abstract += 1
    return parsed_files, total_abstracts, files_without_abstract, errors


def _process_tar(path: Path, counter: Counter[str]) -> Tuple[int, int, int, list[str]]:
    """Scan XML files inside a tar/tar.gz archive."""
    parsed_files = 0
    total_abstracts = 0
    errors: list[str] = []
    files_without_abstract = 0
    with tarfile.open(path, mode="r:*") as archive:
        for member in archive.getmembers():
            if not member.isfile() or not member.name.lower().endswith(".xml"):
                continue
            extracted = archive.extractfile(member)
            if extracted is None:
                errors.append(f"{member.name}: unable to read from archive")
                continue
            try:
                abstracts_found, error = _collect_from_stream(extracted, member.name, counter)
            finally:
                extracted.close()
            if error:
                errors.append(error)
                continue
            parsed_files += 1
            total_abstracts += abstracts_found
            if abstracts_found == 0:
                files_without_abstract += 1
    return parsed_files, total_abstracts, files_without_abstract, errors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Report which abstract-type values appear in PMC XML files."
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
        parsed_files, total_abstracts, files_without_abstract, errors = _process_directory(source, counter)
    elif source.is_file() and tarfile.is_tarfile(source):
        parsed_files, total_abstracts, files_without_abstract, errors = _process_tar(source, counter)
    elif source.is_file() and source.suffix.lower() == ".xml":
        total_abstracts, error = _collect_from_path(source, counter)
        parsed_files = 0 if error else 1
        files_without_abstract = 1 if (error is None and total_abstracts == 0) else 0
        errors = [error] if error else []
    else:
        print(f"Source '{source}' is not a directory, tar archive, or XML file.", file=sys.stderr)
        return 1

    if errors:
        print("Some files could not be processed:", file=sys.stderr)
        for message in errors:
            print(f"  - {message}", file=sys.stderr)

    print(f"XML files processed: {parsed_files}")
    print(f"Abstract elements found: {total_abstracts}")
    print(f"Files without abstract: {files_without_abstract}")
    print(f"Unique abstract-type values: {len(counter)}")

    if not counter:
        return 0

    width = max(len(label) for label in counter)
    print("\nabstract-type".ljust(width + 2) + "count    share")
    for abstract_type, count in counter.most_common():
        share = (count / total_abstracts * 100) if total_abstracts else 0.0
        print(f"{abstract_type.ljust(width + 2)}{count:>5}  {share:6.2f}%")

    return 0


if __name__ == "__main__":
    sys.exit(main())
