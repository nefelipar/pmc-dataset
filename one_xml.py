#!/usr/bin/env python3

# Script to download one PMC oa_comm tarball, extract all XML files,
# and then analyze the first extracted XML for tag names and counts.

import os
import re
import sys
import tarfile
import urllib.request
from bs4 import BeautifulSoup
import csv


BASE_URL = "https://ftp.ncbi.nlm.nih.gov/pub/pmc/oa_bulk/oa_comm/xml/"
RAW_DIR = "data/tar"
XML_DIR = "data/xml"

# Get list of tarball URLs from the base URL from the baselines (oa_comm_xml.PMC...tar.gz)
def get_tarball_urls(base_url):
    resp = urllib.request.urlopen(base_url)
    html = resp.read().decode("utf-8", errors="ignore")
    links = re.findall(r'href="(oa_comm_xml\.PMC[^"]+\.tar\.gz)"', html)

    if not links:
        print("No .tar.gz links found at:", base_url)
        sys.exit(1)

    if not base_url.endswith("/"):
        base_url += "/"

    print("Using tarballs:")
    # for link in links:
    #     print(link)
    return [base_url + name for name in links]


def download(url, dest_path):
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    if os.path.exists(dest_path):
        return dest_path
    # Basic download
    urllib.request.urlretrieve(url, dest_path)
    return dest_path


def extract_all_xmls(tar_path, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    extracted = []
    with tarfile.open(tar_path, mode="r:gz") as tar:
        for m in tar.getmembers():
            if m.isfile() and m.name.lower().endswith(".xml"):
                xml_name = os.path.basename(m.name)
                xml_path = os.path.join(out_dir, xml_name)
                with tar.extractfile(m) as src, open(xml_path, "wb") as dst:
                    dst.write(src.read())
                extracted.append(xml_path)
    if not extracted:
        print("No XML files found inside:", tar_path)
        sys.exit(1)
    return extracted


# def list_unique_tags(xml_path):
#     # Parse and collect tag names using BeautifulSoup (XML parser)
#     with open(xml_path, "rb") as f:
#         soup = BeautifulSoup(f, "xml")

#     tags = set()
#     for tag in soup.find_all(True):  # all tags
#         # Strip namespace prefixes if present (e.g., mml:math -> math)
#         name = tag.name.split(":", 1)[1] if ":" in tag.name else tag.name
#         tags.add(name)
#     return sorted(tags)


# def count_articles(xml_path):
#     with open(xml_path, "rb") as f:
#         soup = BeautifulSoup(f, "xml")

#     # In JATS, 'article' and 'sub-article' are distinct tag names
#     articles = len(soup.find_all("article"))
#     sub_articles = len(soup.find_all("sub-article"))
#     return articles, sub_articles


def main():
    tar_urls = get_tarball_urls(BASE_URL)
    tar_urls = tar_urls[:2]  # Limit to first tarball for testing

    for tar_url in tar_urls:
        tar_name = tar_url.rsplit("/", 1)[-1] #keep only the file name
        print("Processing tarball:", tar_name)    
        tar_path = os.path.join(RAW_DIR, tar_name)
        download(tar_url, tar_path)
        print("Saved:", tar_path)

        xml_dir = XML_DIR + "/" + tar_name.replace(".tar.gz", "")
        xml_paths = extract_all_xmls(tar_path, xml_dir)
        print("Extracted", len(xml_paths), "XML files into:", xml_dir)

    # # Analyze only the first XML to keep runtime and output manageable
    # xml_path = xml_paths[0]
    # print("Analyzing first XML:", xml_path)

    # tags = list_unique_tags(xml_path)
    # print("Unique tags (", len(tags), "):", sep="")
    # for t in tags:
    #     print(t)

    # # Also save to CSV (one column: tag)
    # os.makedirs("data", exist_ok=True)
    # csv_path = os.path.join("data", "tags.csv")
    # with open(csv_path, "w", newline="", encoding="utf-8") as f:
    #     writer = csv.writer(f)
    #     writer.writerow(["tag"])  # header
    #     for t in tags:
    #         writer.writerow([t])
    # print("Saved CSV:", csv_path)

    # # Count how many articles and sub-articles are in this XML
    # a, sa = count_articles(xml_path)
    # print("Article count:", a)
    # print("Sub-article count:", sa)


if __name__ == "__main__":
    main()
