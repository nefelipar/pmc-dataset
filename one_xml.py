#!/usr/bin/env python3

# Simple script to download one PMC oa_comm tarball, extract a single XML,
# and print the unique XML tag names found in that file.

import os
import re
import sys
import tarfile
import urllib.request
import xml.etree.ElementTree as ET
import csv


BASE_URL = "https://ftp.ncbi.nlm.nih.gov/pub/pmc/oa_bulk/oa_comm/xml/"
RAW_DIR = "data/raw"
XML_DIR = "data/xml"


def get_first_tarball_url(base_url):
    resp = urllib.request.urlopen(base_url)
    html = resp.read().decode("utf-8", errors="ignore")
    links = re.findall(r'href="([^"]+\.tar\.gz)"', html)
    if not links:
        print("No .tar.gz links found at:", base_url)
        sys.exit(1)
    name = links[0]
    if not base_url.endswith("/"):
        base_url += "/"
    return base_url + name


def download(url, dest_path):
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    if os.path.exists(dest_path):
        return dest_path
    # Basic download; no progress bar to keep it simple
    urllib.request.urlretrieve(url, dest_path)
    return dest_path


def extract_first_xml(tar_path, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    with tarfile.open(tar_path, mode="r:gz") as tar:
        # Find the first member that looks like an XML file
        member = None
        for m in tar.getmembers():
            if m.isfile() and m.name.lower().endswith(".xml"):
                member = m
                break
        if member is None:
            print("No XML file found inside:", tar_path)
            sys.exit(1)
        # Write to a flat path using only the basename
        xml_name = os.path.basename(member.name)
        xml_path = os.path.join(out_dir, xml_name)
        with tar.extractfile(member) as src, open(xml_path, "wb") as dst:
            dst.write(src.read())
    return xml_path


def list_unique_tags(xml_path):
    # Parse and collect tag names without namespaces
    tree = ET.parse(xml_path)
    root = tree.getroot()

    def strip_ns(tag):
        if "}" in tag:
            return tag.split("}", 1)[1]
        return tag

    tags = set()
    for elem in root.iter():
        tags.add(strip_ns(elem.tag))
    return sorted(tags)


def count_articles(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    def local(tag):
        if "}" in tag:
            return tag.split("}", 1)[1]
        return tag

    articles = 0
    sub_articles = 0
    for elem in root.iter():
        t = local(elem.tag)
        if t == "article":
            articles += 1
        elif t == "sub-article":
            sub_articles += 1
    return articles, sub_articles


def main():
    tar_url = get_first_tarball_url(BASE_URL)
    tar_name = tar_url.rsplit("/", 1)[-1]
    tar_path = os.path.join(RAW_DIR, tar_name)

    print("Selected tarball:", tar_url)
    download(tar_url, tar_path)
    print("Saved:", tar_path)

    xml_path = extract_first_xml(tar_path, XML_DIR)
    print("Extracted XML:", xml_path)

    tags = list_unique_tags(xml_path)
    print("Unique tags (", len(tags), "):", sep="")
    for t in tags:
        print(t)

    # Also save to CSV (one column: tag)
    os.makedirs("data", exist_ok=True)
    csv_path = os.path.join("data", "tags.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["tag"])  # header
        for t in tags:
            writer.writerow([t])
    print("Saved CSV:", csv_path)

    # Count how many articles and sub-articles are in this XML
    a, sa = count_articles(xml_path)
    print("Article count:", a)
    print("Sub-article count:", sa)


if __name__ == "__main__":
    main()
