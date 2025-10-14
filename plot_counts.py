#!/usr/bin/env python3
"""
ΑΠΛΟ script: Διαβάζει counts (text/abstract) από τα JSONL στο `data/jsonl`
και φτιάχνει 2 ιστογράμματα σε λευκό φόντο με ονομασμένους άξονες.

Υποστηριζόμενα κλειδιά με σειρά προτίμησης:
- κείμενο:  metadata.text_count, count_text
- περίληψη: metadata.abstract_count, count_abstract

Χρήση:
  python plot_counts.py                # σαρώνει data/jsonl και γράφει PNG
  python plot_counts.py -o out.png     # ορισμός ονόματος αρχείου εξόδου
"""

import argparse
import gzip
import json
import os
from typing import List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def iter_jsonl_files(folder: str):
    # Επέλεξε ΜΙΑ εκδοχή ανά αρχείο: αν συνυπάρχουν .jsonl και .jsonl.gz,
    # προτίμησε το .jsonl.gz για να μην διπλομετράμε εγγραφές.
    selected = {}
    for root, _, files in os.walk(folder):
        for f in files:
            if not (f.endswith(".jsonl") or f.endswith(".jsonl.gz")):
                continue
            path = os.path.join(root, f)
            # canonical key χωρίς το .gz ώστε το pair (.jsonl, .jsonl.gz) να έχει ίδιο key
            key = f[:-3] if f.endswith(".gz") else f
            prev = selected.get(key)
            if prev is None:
                selected[key] = path
            else:
                # Αν υπάρχει και τα δύο, κράτα το .gz
                if path.endswith(".jsonl.gz"):
                    selected[key] = path
    # Απόδωσε τα επιλεγμένα paths
    for path in sorted(selected.values()):
        yield path


def read_counts(path: str) -> Tuple[List[int], List[int]]:
    text_counts: List[int] = []
    abstract_counts: List[int] = []

    if path.endswith(".gz"):
        fh = lambda p: gzip.open(p, "rt", encoding="utf-8")
    else:
        fh = lambda p: open(p, "r", encoding="utf-8")

    with fh(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            md = rec.get("metadata") or {}

            t = md.get("text_count")
            a = md.get("abstract_count")

            if isinstance(t, int) and t >= 0:
                text_counts.append(t)
            if isinstance(a, int) and a >= 0:
                abstract_counts.append(a)

    return text_counts, abstract_counts


def collect_counts(folder: str) -> Tuple[List[int], List[int]]:
    all_t: List[int] = []
    all_a: List[int] = []
    for p in iter_jsonl_files(folder):
        t, a = read_counts(p)
        all_t.extend(t)
        all_a.extend(a)
    return all_t, all_a


def _apply_bin_range_xticklabels(ax, bins):
    # Βάλε ετικέτες με range για ΚΑΘΕ μπάρα (bin)
    centers = [(bins[i] + bins[i + 1]) / 2 for i in range(len(bins) - 1)]
    labels = []
    for i in range(len(bins) - 1):
        start = int(bins[i])
        end = int(bins[i + 1]) - 1
        if end < start:
            end = start
        labels.append(f"{start}–{end}")
    ax.set_xticks(centers)
    ax.set_xticklabels(labels, rotation=90, fontsize=8)


def plot_histograms(text_counts: List[int], abstract_counts: List[int], out_path: str):
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.patch.set_facecolor("white")
    for ax in axes:
        ax.set_facecolor("white")

    # Ιστόγραμμα text_count – πάντα περιλαμβάνει το μηδέν στο εύρος
    def _annotate_bar_counts(ax, patches):
        for rect in patches:
            try:
                h = rect.get_height()
                if h <= 0:
                    continue
                ax.annotate(
                    f"{int(h)}",
                    xy=(rect.get_x() + rect.get_width() / 2, h),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    color="#222",
                )
            except Exception:
                pass

    if text_counts:
        tmax = max(text_counts)
        n, bins, patches = axes[0].hist(text_counts, bins=30, range=(0, max(1, tmax)), color="#1f77b4", edgecolor="white")
        _apply_bin_range_xticklabels(axes[0], bins)
        _annotate_bar_counts(axes[0], patches)
    axes[0].set_title("Σχήμα 1: Κατανομή text_count")
    axes[0].set_xlabel("Αριθμός λέξεων στο πλήρες κείμενο (text_count)")
    axes[0].set_ylabel("Συχνότητα (πλήθος άρθρων)")

    # Ιστόγραμμα abstract_count – πάντα περιλαμβάνει το μηδέν στο εύρος
    if abstract_counts:
        amax = max(abstract_counts)
        n, bins, patches = axes[1].hist(abstract_counts, bins=30, range=(0, max(1, amax)), color="#2ca02c", edgecolor="white")
        _apply_bin_range_xticklabels(axes[1], bins)
        _annotate_bar_counts(axes[1], patches)
    axes[1].set_title("Σχήμα 2: Κατανομή abstract_count")
    axes[1].set_xlabel("Αριθμός λέξεων στην περίληψη (abstract_count)")
    axes[1].set_ylabel("Συχνότητα (πλήθος άρθρων)")

    plt.tight_layout()
    fig.savefig(out_path, dpi=160, facecolor="white")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", default="data/jsonl", help="Φάκελος JSONL (προεπιλογή: data/jsonl)")
    ap.add_argument("-o", "--output", default="data/count_distributions.png", help="Αρχείο PNG εξόδου")
    args = ap.parse_args()

    t, a = collect_counts(args.input)
    print(f"[info] Βρέθηκαν {len(t)} text_count και {len(a)} abstract_count.")
    if not t and not a:
        print("[warn] Δεν βρέθηκαν counts στα JSONL.")
    plot_histograms(t, a, args.output)
    print(f"[ok] Αποθηκεύτηκε: {args.output}")


if __name__ == "__main__":
    main()
