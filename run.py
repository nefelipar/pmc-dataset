#!/usr/bin/env python3
# pip install beautifulsoup4 lxml

import os
import re
import sys
import tarfile
import urllib.request
import gzip
import json
from bs4 import BeautifulSoup
from unicodedata import normalize

BASE_URL = "https://ftp.ncbi.nlm.nih.gov/pub/pmc/oa_bulk/oa_comm/xml/"
RAW_DIR  = "data/tar"     # save .tar.gz
OUT_DIR  = "data/jsonl"   # save .jsonl.gz

os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

# ---------- Βοηθοί κειμένου ----------
import re, html
from unicodedata import normalize

def clean(s: str) -> str:
    if s is None:
        return ""
    s = html.unescape(s)  # Αντιμετωπίζει HTML entities όπως &#x02013; (en dash)
    s = normalize("NFC", s)  # Κανονικοποιεί το unicode

    # Συμπίεση whitespace (αφαιρεί όλα τα επιπλέον κενά ή νέα γραμμές)
    s = re.sub(r"[ \t\r\f\v]+", " ", s).strip()

    # 1) Αφαιρεί κενά πριν από στίξη (π.χ. gene , name → gene, name)
    s = re.sub(r"\s+([,.;:!?%])", r"\1", s)

    # 2) Μην αφήνεις κενό πριν από κλείσιμο παρενθέσεων/εισαγωγικών κλπ
    s = re.sub(r"\s+([)\]\}»”’])", r"\1", s)

    # 3) Μην αφήνεις κενό αμέσως μετά από άνοιγμα παρενθέσεων/εισαγωγικών/αγκύλων
    s = re.sub(r"([(\[\{«“‘])\s+", r"\1", s)

    # 4) Αριθμοί με κόμμα (π.χ. 5,400) να μην σπάνε σε 5, 400
    #    (απλώς διασφαλίζουμε ότι δεν υπάρχει κενό γύρω από κόμμα όταν είναι μεταξύ ψηφίων)
    s = re.sub(r"(?<=\d)\s*,\s*(?=\d)", ",", s)

    # 5) Ενδο-λέξης παύλα ή en dash να παραμένει κολλητή (nef-par, nef–par)
    s = re.sub(r"(?<=\w)\s*([\-–])\s*(?=\w)", r"\1", s)
    #    και εύρος αριθμών με en dash χωρίς κενά (π.χ. 5 – 10 → 5–10)
    s = re.sub(r"(?<=\d)\s*–\s*(?=\d)", "–", s)

    # 6) Όχι κενό γύρω από '=' (x = 5 → x=5)
    s = re.sub(r"\s*=\s*", "=", s)

    # 7) Τελική καθαριότητα διπλών κενών
    s = re.sub(r"\s{2,}", " ", s).strip()

    return s




def text_from(node) -> str:
    # ΧΡΗΣΙΜΟ: separator=" " για να κρατάει το κενό σε inline tags,
    # το clean() μετά διορθώνει τα κενά πριν από στίξη
    return clean(node.get_text(" ", strip=True))



def in_boundary_without_tables(tag, boundary_name: str) -> bool:
    p = getattr(tag, "parent", None)
    boundary_name = (boundary_name or "").lower()
    while p is not None:
        name = (getattr(p, "name", None) or "").lower()
        if not name:
            p = getattr(p, "parent", None)
            continue
        if name in {"table", "table-wrap"}:
            return False
        if name == boundary_name:
            return True
        p = getattr(p, "parent", None)
    return False



# ---------- JATS pickers ----------
def extract_text(soup):
    """Μόνο παράγραφοι κάτω από <body>, χωρίς tables (boundary-aware)."""
    body = soup.find("body")
    if not body:
        return ""
    paras = []
    for p in body.find_all("p"):
        if in_boundary_without_tables(p, "body"):
            t = text_from(p)
            if t:
                paras.append(t)
    return "\n\n".join(paras)



# --- ΔΙΟΡΘΩΣΗ extract_abstract ---
def extract_abstract(art):
    """Μόνο οι παράγραφοι του επιλεγμένου *κανονικού* abstract."""
    def pick_normal_abstract(art):
        BAD = {"graphical","teaser","author-summary","editor-summary","lay-summary"}
        if not art: return None
        cands = art.find_all("abstract", recursive=False)
        for a in cands:
            a_type = (a.get("abstract-type") or "").lower()
            if a_type and a_type in BAD:
                continue
            return a
        return cands[0] if cands else None

    abs_el = pick_normal_abstract(art)
    if not abs_el:
        return ""

    paras = []
    for p in abs_el.find_all("p"):
        # νέο boundary-aware check
        if in_boundary_without_tables(p, "abstract"):
            t = text_from(p)
            if t:
                paras.append(t)
    return "\n\n".join(paras)



def extract_authors(art):
    """Authors ως 'Given Surname' strings, με τη σειρά."""
    out, seen = [], set()
    if not art: return out
    for c in art.select("contrib-group > contrib[contrib-type=author]"):
        collab = c.find("collab")
        if collab and collab.get_text(strip=True):
            name = clean(collab.get_text(" ", strip=True))
        else:
            sur = c.find("surname"); giv = c.find("given-names")
            name = clean(" ".join(x for x in [
                (giv.get_text(strip=True) if giv else ""),
                (sur.get_text(strip=True) if sur else "")
            ] if x))
        if name and name not in seen:
            seen.add(name); out.append(name)
    return out


def get_article_meta(soup):
    front = soup.find("front")
    return front.find("article-meta") if front else None

def get_journal_meta(soup):
    front = soup.find("front")
    return front.find("journal-meta") if front else None

def article_id(art, kind: str):
    """ Take article-id of given kind (pmcid, pmid, doi) """
    if not art: return None
    el = art.find("article-id", attrs={"pub-id-type": kind})
    return clean(el.get_text()) if el else None


def extract_metadata(soup):
    """metadata: pmid, doi, article_title, journal_title, epub, authors[]"""
    art = get_article_meta(soup); jmeta = get_journal_meta(soup)
    pmc  = article_id(art, "pmcid") or article_id(art, "pmc")
    pmid = article_id(art, "pmid")
    doi  = article_id(art, "doi")

    atitle = art.select_one("title-group > article-title") if art else None
    jtitle = jmeta.select_one("journal-title-group > journal-title") if jmeta else None

    # epub σε ISO-ish YYYY[-MM[-DD]]
    epub = None
    if art:
        for pd in art.find_all("pub-date"):
            if (pd.get("pub-type") or "").lower() == "epub":
                y = pd.find("year").get_text(strip=True) if pd.find("year") else None
                m = pd.find("month").get_text(strip=True) if pd.find("month") else None
                d = pd.find("day").get_text(strip=True)  if pd.find("day")  else None
                if y:
                    epub = y + (f"-{m.zfill(2)}" if m else "") + (f"-{d.zfill(2)}" if d else "")
                break

    authors = extract_authors(art)
    meta = {
        "pmid": pmid,
        "doi": doi,
        "article_title": clean(atitle.get_text(" ", strip=True)) if atitle else "",
        "journal_title": clean(jtitle.get_text(" ", strip=True)) if jtitle else "",
        "epub": epub,
        "authors": authors
    }
    return pmc, meta



def build_record(xml_bytes: bytes):
    """Makes a JSON line for JSONL: {pmc, text, abstract, metadata{...}}"""
    soup = BeautifulSoup(xml_bytes, "lxml-xml")
    pmc, metadata = extract_metadata(soup)
    if not pmc:
        return None  # we need pmc for primary ID
    abstract = extract_abstract(get_article_meta(soup))
    text = extract_text(soup)
    return {"pmc": pmc, "text": text, "abstract": abstract, "metadata": metadata}



def tar_to_jsonl(tar_path: str) -> str:
    """ Convert a .tar.gz of JATS XML files to a single .jsonl.gz file.
    parsed = number of XML files parsed
    kept   = number of JSON records written
    skipped= number of XML files skipped (no pmc or errors)"""

    base   = os.path.basename(tar_path)
    prefix = os.path.splitext(os.path.splitext(base)[0])[0]  # drop .tar(.gz)
    out_path = os.path.join(OUT_DIR, f"{prefix}.jsonl.gz") # e.g. data/jsonl/oa_comm_xml.PMC012xxxxxx.baseline.2025-06-26.jsonl.gz

    parsed = kept = skipped = 0
    with tarfile.open(tar_path, "r:gz") as tf, gzip.open(out_path, "wt", encoding="utf-8") as fout:
        for m in tf.getmembers():
            if not m.isfile() or not (m.name or "").lower().endswith(".xml"):
                continue
            f = tf.extractfile(m)
            if not f: 
                continue
            xml_bytes = f.read()
            parsed += 1
            try:
                rec = build_record(xml_bytes)
                if rec is None:
                    skipped += 1
                    continue
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                kept += 1
            except Exception:
                skipped += 1  # skip invalid XML or other errors
    print(f"[OK] {base}: parsed={parsed}, written={kept}, skipped={skipped}")
    print(f"[OUT] {out_path}")
    return out_path



# ---------- Download tarball & make JSONL ----------
def download_tar(tar_name: str) -> str:
    url = BASE_URL + tar_name
    out = os.path.join(RAW_DIR, tar_name)
    if not os.path.exists(out):
        print(f"[download] {url}")
        urllib.request.urlretrieve(url, out)
    else:
        print(f"[skip] already exists: {out}")
    return out



def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print("python run.py <tar_name_or_url>  # π.χ. oa_comm_xml.PMC012xxxxxx.baseline.2025-06-26.tar.gz")
        sys.exit(1)

    arg = sys.argv[1] # tar file name or full URL
    # assume it's a URL
    if arg.startswith("http"): 
        tar_path = os.path.join(RAW_DIR, os.path.basename(arg))  # e.g. data/tar/oa_comm_xml.PMC012xxxxxx.baseline.2025-06-26.tar.gz
        if not os.path.exists(tar_path):
            print(f"[download] {arg}")
            urllib.request.urlretrieve(arg, tar_path)
    # assume it's a local path or just a tar file name
    elif arg.endswith(".tar") or arg.endswith(".tar.gz"): 
        tar_path = download_tar(os.path.basename(arg)) if not os.path.isabs(arg) else arg
    else:
        tar_path = download_tar(arg) 

    tar_to_jsonl(tar_path)




if __name__ == "__main__":
    main()
