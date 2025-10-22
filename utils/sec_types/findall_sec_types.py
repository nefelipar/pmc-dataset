#!/usr/bin/env python3
"""Collect unique <sec sec-type="..."> values from PMC tarballs.

The script scans one or more `.tar` archives (optionally compressed, e.g.
`.tar.gz`) and prints the distinct `sec-type` attribute values found inside
XML files. By default it looks into `data/tar`, but you can point it to any
individual tarball or directory containing tarballs.
"""

from __future__ import annotations

import argparse
import sys
import tarfile
from collections import Counter
from pathlib import Path
from typing import Iterable, Iterator, Tuple
import xml.etree.ElementTree as ET


def find_tar_paths(target: Path) -> Tuple[Path, ...]:
    """Resolve tarball paths from a file or directory target."""
    if target.is_file():
        return (target,)

    if not target.exists() or not target.is_dir():
        raise FileNotFoundError(f"{target} is not a file or directory")

    tar_paths = [
        path
        for path in sorted(target.iterdir())
        if path.is_file() and path.name.lower().endswith(".tar.gz")
    ]

    if not tar_paths:
        raise FileNotFoundError(f"No tar archives found under {target}")

    return tuple(tar_paths)


def iter_sec_types(fileobj) -> Iterator[str]:
    """Yield sec-type attribute values from a single XML file."""
    # iterparse consumes the file stream; ElementTree handles namespaces by
    # returning tags like '{namespace}sec', so we only keep the suffix.
    try:
        for event, elem in ET.iterparse(fileobj, events=("start",)):
            if elem.tag.rsplit("}", 1)[-1] != "sec":
                continue
            sec_type = elem.attrib.get("sec-type")
            if sec_type:
                cleaned = sec_type.strip()
                if cleaned:
                    yield cleaned
    except ET.ParseError as exc:
        raise ValueError(f"Failed to parse XML: {exc}") from exc


def collect_from_tar(path: Path) -> Counter[str]:
    """Collect sec-type counts from a tar archive."""
    counts: Counter[str] = Counter()

    try:
        with tarfile.open(path, mode="r:*") as tar:
            for member in tar:
                if not member.isfile() or not member.name.lower().endswith(".xml"):
                    continue
                extracted = tar.extractfile(member)
                if extracted is None:
                    continue
                try:
                    for sec_type in iter_sec_types(extracted):
                        counts[sec_type] += 1
                except ValueError as err:
                    print(
                        f"Warning: skipping {member.name} in {path.name}: {err}",
                        file=sys.stderr,
                    )
                finally:
                    extracted.close()
    except (tarfile.TarError) as exc:
        raise RuntimeError(f"Error while processing {path}: {exc}") from exc

    return counts


def parse_args(argv: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "target",
        nargs="?",
        default=Path("data") / "tar",
        type=Path,
        help="Tarball path or directory containing tarballs (default: data/tar)",
    )
    parser.add_argument(
        "--counts",
        action="store_true",
        help="Print counts for each sec-type in addition to listing them.",
    )
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)

    try:
        tar_paths = find_tar_paths(args.target)
    except FileNotFoundError as exc:
        print(exc, file=sys.stderr)
        return 1

    grand_total = Counter()
    for tar_path in tar_paths:
        counts = collect_from_tar(tar_path)
        grand_total.update(counts)

    if not grand_total:
        output_path = Path(__file__).resolve().parent / "sec_types.txt"
        output_path.write_text("No sec-type attributes found.\n", encoding="utf-8")
        return 0

    unique_values = sorted(grand_total)
    output_lines = []

    for value in unique_values:
        if args.counts:
            output_lines.append(f"{value}\t{grand_total[value]}")
        else:
            output_lines.append(value)

    output_path = Path(__file__).resolve().parent / "sec_types.txt"
    output_path.write_text("\n".join(output_lines) + "\n", encoding="utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
