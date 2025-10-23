#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Set
import xml.etree.ElementTree as ET


def local_name(tag: str | None) -> str:
    """Return the local part of an XML tag, ignoring any namespace."""
    if not tag:
        return ""
    if tag.startswith("{"):
        return tag.partition("}")[2]
    return tag


TARGET_TAGS = {"body", "abstract"}


def collect_ref_types(xml_path: Path) -> Set[str]:
    """Collect ref-type values from xref elements inside body/abstract sections."""
    ref_types: Set[str] = set()
    try:
        context = ET.iterparse(xml_path, events=("start", "end"))
    except ET.ParseError as exc:
        print(f"[WARN] Failed to parse {xml_path}: {exc}", file=sys.stderr)
        return ref_types
    except OSError as exc:
        print(f"[WARN] Could not read {xml_path}: {exc}", file=sys.stderr)
        return ref_types

    depth: Dict[str, int] = {tag: 0 for tag in TARGET_TAGS}

    for event, elem in context:
        tag = local_name(elem.tag)
        if event == "start":
            if tag in TARGET_TAGS:
                depth[tag] += 1
            elif tag == "xref" and any(depth.values()):
                ref_type = elem.attrib.get("ref-type")
                if ref_type:
                    cleaned = ref_type.strip()
                    if cleaned:
                        ref_types.add(cleaned)
        elif event == "end":
            if tag in TARGET_TAGS and depth[tag] > 0:
                depth[tag] -= 1
            elem.clear()

    return ref_types


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Find unique xref ref-type attribute values inside <body>/<abstract> sections "
            "across XML files under data/tar (or a custom directory)."
        )
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        help="Root directory containing XML files (defaults to <project>/data/tar).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Path to the output text file (defaults to utils/ref_types/ref_types.txt).",
    )
    return parser.parse_args()


def main() -> int:
    script_path = Path(__file__).resolve()
    project_root = script_path.parents[2]

    args = parse_args()
    data_dir = (args.data_dir or (project_root / "data" / "tar")).resolve()
    output_path = (args.output or (script_path.parent / "ref_types.txt")).resolve()

    if not data_dir.exists():
        print(f"[ERROR] Data directory not found: {data_dir}", file=sys.stderr)
        return 1

    all_ref_types: Set[str] = set()
    xml_file_count = 0

    for xml_path in sorted(data_dir.rglob("*.xml")):
        xml_file_count += 1
        all_ref_types.update(collect_ref_types(xml_path))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        for value in sorted(all_ref_types):
            fh.write(f"{value}\n")

    try:
        relative_output = output_path.relative_to(project_root)
    except ValueError:
        relative_output = output_path

    print(
        f"Found {len(all_ref_types)} unique ref-type values "
        f"across {xml_file_count} XML files."
    )
    print(f"Wrote results to {relative_output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
