#!/usr/bin/env python3
# pip install beautifulsoup4 lxml

import os
import re
import sys
import tarfile
import urllib.request
import gzip
import json
import re, html
from bs4 import BeautifulSoup, NavigableString
from unicodedata import normalize

BASE_URL = "https://ftp.ncbi.nlm.nih.gov/pub/pmc/oa_bulk/oa_comm/xml/"
RAW_DIR  = "data/tar"     # save .tar.gz
OUT_DIR  = "data/jsonl"   # save .jsonl.gz
LOG_DIR  = "data/logs"    # logs (e.g., missing abstracts)

os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)


# ---------- Text cleaning & extraction ----------
def clean(s: str) -> str:
    if s is None:
        return ""
    s = html.unescape(s)  # Handling HTML entities  like &#x02013;
    s = normalize("NFC", s)  # Normalize to unicode


    # Remove parenthetical references to figures/tables and citations, replacing
    # the entire parenthesis with "" per requirement, e.g.:
    # (Figure 1a) → "", (Table 2) → "", (Bozdech et al. 2003) → "",
    # (see Fig. 3; Smith 2004) → ""
    def _paren_replacer(match: re.Match) -> str:
        inner = match.group(1)
        inner_lc = inner.lower()
        # figure/table hints
        if re.search(r"\b(fig(?:\.|ures?)?|figure|figs?|table|tables)\b", inner_lc):
            return ""
        # citation with 'et al.' and year
        if re.search(r"\bet\s+al\.?\b", inner_lc) and re.search(r"\b(19|20)\d{2}[a-z]?\b", inner):
            return ""
        # surname + year pattern (optionally two surnames joined by and/&)
        if re.search(r"\b[A-Z][A-Za-z-]+(?:\s+(?:and|&)\s+[A-Z][A-Za-z-]+)?\s+(?:\d{4}[a-z]?)\b", inner):
            return ""
        # multiple years separated by ; or , typical for grouped citations
        if re.search(r"\b(19|20)\d{2}\b", inner) and (";" in inner or "," in inner):
            return ""
        return match.group(0)

    # Replace non-nested parentheses of the above forms
    s = re.sub(r"\(([^()]*)\)", _paren_replacer, s)

    # Also remove figure/table references even when not in parentheses
    # e.g., "see Figure 1a", "Fig. 2B", "Tables 3–4"
    figtab_pattern = re.compile(
        r"\b(?:see(?:\s+also)?\s+)?(?:fig(?:\.|s\.?|ures?)|figure|figs?|table|tables)\s*"
        r"(?:"  # at least one token like 1, 1A, S1, II, 2B, 1A–C
        r"[A-Za-z]?\s*(?:[IVX]+|\d+)[A-Za-z]*"  # allow zero or more letters after number
        r"(?:\s*[–-]\s*[A-Za-z]?\s*(?:[IVX]+|\d+)[A-Za-z]*)?"
        r")"
        r"(?:\s*(?:,|;|and)\s*"  # optionally more tokens joined by connectors
        r"(?:[A-Za-z]?\s*(?:[IVX]+|\d+)[A-Za-z]*"
        r"(?:\s*[–-]\s*[A-Za-z]?\s*(?:[IVX]+|\d+)[A-Za-z]*)?" 
        r"))*"
        r"[A-Za-z]{0,2}",  # swallow any trailing panel letters glued to </xref>
        flags=re.IGNORECASE,
    )
    s = figtab_pattern.sub(" ", s)

    # Remove inline author-year style citations not in parentheses
    # e.g., "Bozdech et al. 2003", "Smith and Doe 2014", "Smith 2014a", "Smith et al., 2003"
    author_year_pattern = re.compile(
        r"\b"                                 # start at word boundary
        r"[A-Z][A-Za-z-]+"                     # Surname
        r"(?:\s+(?:and|&)\s+[A-Z][A-Za-z-]+|\s+et\s+al\.?)*"  # and/& second or et al.
        r",?\s+"                              # optional comma then space
        r"\(?\d{4}[a-z]?\)?"                 # year with optional letter and parentheses
        r"\b"
    )
    s = author_year_pattern.sub(" ", s)

    # Compress whitespace (spaces, tabs, newlines) to single space
    s = re.sub(r"[ \t\r\f\v]+", " ", s).strip()

    # Remove spaces before punctuation (e.g. gene , name → gene, name)
    s = re.sub(r"\s+([,.;:!?%])", r"\1", s)

    # Don't leave space before closing brackets/quotes
    s = re.sub(r"\s+([)\]\}»”’])", r"\1", s)

    # Don't leave space after opening brackets/quotes
    s = re.sub(r"([(\[\{«“‘])\s+", r"\1", s)

    # Digit grouping commas: "1, 000" → "1,000"
    # Only if between digits (to avoid messing with lists)
    s = re.sub(r"(?<=\d)\s*,\s*(?=\d)", ",", s)

    # Remove spaces around hyphens in words (e.g. "state - of - the - art" → "state-of-the-art")
    # Do not touch cases like "17- and"
    s = re.sub(r"(?<=[A-Za-z])\s*-\s*(?=[A-Za-z])", "-", s)

    # Range of numbers with en dash: "2000 – 2021" → "2000–2021"
    s = re.sub(r"(?<=\d)\s*–\s*(?=\d)", "–", s)

    # No spaces around '=' (x = 5 → x=5)
    s = re.sub(r"\s*=\s*", "=", s)

    # whitespace cleanup again
    s = re.sub(r"\s{2,}", " ", s).strip()

    return s



def text_from(node) -> str:
    # Extract clean text from a BeautifulSoup node
    return clean(node.get_text(" ", strip=True))


def _word_count_with_nltk(text: str):
    try:
        from nltk.tokenize import wordpunct_tokenize  # type: ignore
        return len([t for t in wordpunct_tokenize(text) if t.strip()])
    except Exception:
        return None


def count_words(text: str) -> int:
    """Count tokens in text.
    Prefer NLTK wordpunct if available; otherwise whitespace split.
    """
    if not text:
        return 0
    n = _word_count_with_nltk(text)
    if n is not None:
        return n
    # Fallback: Unicode-aware whitespace split
    return len(re.findall(r"\S+", text, flags=re.UNICODE))



def in_boundary_without_forbidden(tag, boundary_name: str) -> bool:
    """ Check if `tag` is within a boundary (e.g. "body", "abstract") 
        but not inside tables, figures/images, captions, or table-wraps.
        If boundary_name is None or empty, just check not in table/table-wrap.
    """
    p = getattr(tag, "parent", None)
    boundary_name = (boundary_name or "").lower()
    while p is not None:
        name = (getattr(p, "name", None) or "").lower()
        if not name:
            p = getattr(p, "parent", None)
            continue
        if name in {"table", "table-wrap", "fig", "figure", "graphic", "inline-graphic", "media", "caption"}:
            return False
        if name == boundary_name:
            return True
        p = getattr(p, "parent", None)
    return False


def is_in_ignored_section(tag) -> bool:
    """Detect if a tag is inside sections we want to exclude from body text
    (acknowledgments, references, bibliography, appendix, supplementary, etc.).
    """
    IGNORE_SECTION_TITLES = re.compile(r"^(acknowledg(e)?ments?|references?|bibliograph(y|ies)|appendix|supplementar(y|ies|y materia(l|ls))|supplemental|conflict(s)? of interest|author contribution(s)?|funding)$", re.IGNORECASE)
    IGNORE_SEC_TYPE = {"supplementary-material", "display-objects"}
    
    p = getattr(tag, "parent", None)
    while p is not None:
        name = (getattr(p, "name", None) or "").lower()
        if name in {"ack", "ref-list", "glossary", "fn-group"}:  # explicit back-matter containers
            return True
        
        if name == "sec":
            # 1. Check for ignored section types
            sec_type = (p.get("sec-type") or "").lower()
            if sec_type in IGNORE_SEC_TYPE:
                return True

            # 2. Look for a title child to decide
            t = p.find("title")
            if t:
                title_text = clean(t.get_text(" ", strip=True))
                if title_text and IGNORE_SECTION_TITLES.match(title_text):
                    return True
        p = getattr(p, "parent", None)
    return False


# def replace_math_with_placeholder(soup_or_tag):
#     """Replace all math nodes with the literal placeholder [MATH].
#     Covers inline and display math in common JATS forms.
#     """
#     math_like = []
#     # Collect various math representations
#     math_like.extend(soup_or_tag.find_all(["inline-formula", "disp-formula", "tex-math"]))
#     # Namespaced MathML can appear as mml:math or math
#     math_like.extend([t for t in soup_or_tag.find_all(True) if (t.name or "").lower().endswith(":math") or (t.name or "").lower() == "math"])
#     for m in math_like:
#         m.replace_with(soup_or_tag.new_string("[MATH]"))


def remove_images_and_captions(soup_or_tag):
    """Remove figures/images and their captions completely from the tree."""
    for tname in ["fig", "figure", "graphic", "inline-graphic", "media", "caption"]:
        for el in soup_or_tag.find_all(tname):
            el.decompose()


def remove_figure_table_xrefs_and_glued(soup_or_tag):
    """Remove <xref ref-type="fig|figure|table">…</xref> and any glued panel letters right after it.
    Example: <xref ref-type="fig">Figure 1</xref>F → remove entire xref content and the trailing 'F'.
    Only removes the trailing letters if they are immediately adjacent (no leading space).
    """
    for xr in soup_or_tag.find_all("xref"):
        rt = (xr.get("ref-type") or "").lower()
        if rt in {"fig", "figure", "table", "tables"}:
            ns = xr.next_sibling
            if isinstance(ns, NavigableString):
                s = str(ns)
                # Remove 1–2 leading letters (panel labels), optionally followed by range like A–C
                m = re.match(r"^([A-Za-z]{1,2}(?:\s*[–-]\s*[A-Za-z]{1,2})?)", s)
                if m:
                    ns.replace_with(s[len(m.group(1)):])
            xr.decompose()



# ---------- JATS pickers ----------
def _process_section_markdown(section_tag, level):
    """
    Recursive helper function to process a section (<sec>) and its children.
    Returns a list of Markdown-formatted text blocks.
    """
    content_blocks = []
    # Process direct children to maintain order
    for child in section_tag.children:
        if isinstance(child, NavigableString):
            continue  # Ignore strings that are just whitespace between tags

        tag_name = (child.name or "").lower()
        
        # Filter out unwanted sections before processing them
        if is_in_ignored_section(child):
            continue

        if tag_name == 'title':
            title_text = text_from(child)
            if title_text:
                # Use the level to set the Markdown hashes (e.g., #, ##, ###)
                content_blocks.append(f"{'#' * level} {title_text}")

        elif tag_name == 'p':
            if in_boundary_without_forbidden(child, "body"):
                p_text = text_from(child)
                if p_text:
                    content_blocks.append(p_text)

        elif tag_name == 'sec':
            # Recursive call for the subsection, increasing the hierarchy level
            content_blocks.extend(_process_section_markdown(child, level + 1))

    return content_blocks


def extract_body_with_markdown(soup):
    """
    Extracts the main body of the article, preserving section titles and subtitles
    and formatting them as Markdown, starting from Level 1 (#).
    """
    body = soup.find("body")
    if not body:
        return ""
        
    # First, sanitize the structure within the body
    remove_images_and_captions(body)
    remove_figure_table_xrefs_and_glued(body)
    
    # Start the recursive processing from the body.
    # Main sections will start with level 0 (#).
    all_blocks = _process_section_markdown(body, 0)
    
    return "\n\n".join(all_blocks)



def extract_abstract(art):
    """Only normal abstracts, no graphical/teaser/etc; math → [MATH]; no tables/images."""
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
    # sanitize abstract structure
    remove_images_and_captions(abs_el)
    remove_figure_table_xrefs_and_glued(abs_el)
    # replace_math_with_placeholder(abs_el)
    paras = []
    for p in abs_el.find_all("p"):
        # boundary-aware check
        if in_boundary_without_forbidden(p, "abstract"):
            t = text_from(p)
            if t:
                paras.append(t)
    return "\n\n".join(paras)



def extract_authors(art):
    """Authors with format "Given Names Surname" or "Collaboration Name"."""
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

    # epub YYYY[-MM[-DD]]
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



def build_record(xml_bytes: bytes, count_error_log_path: str = None):
    """
    Creates a JSON record where:
    - "abstract": contains the abstract in plain text.
    - "text": contains ONLY the main body with its titles in Markdown.
    - "metadata": contains all metadata.
    """
    soup = BeautifulSoup(xml_bytes, "lxml-xml")
    pmc, metadata = extract_metadata(soup)

    if pmc:
        metadata["pmc"] = pmc
    
    # 1. Extract abstract.
    abstract = extract_abstract(get_article_meta(soup))

    # 2. Extract the main body (as Markdown).
    text = extract_body_with_markdown(soup)

    # 3. Add word counts to metadata
    if count_error_log_path:
        try:
            os.makedirs(os.path.dirname(count_error_log_path), exist_ok=True)
        except Exception:
            pass

    # abstract_count
    try:
        metadata["abstract_count"] = count_words(abstract)
    except Exception as e:
        if count_error_log_path and pmc:
            try:
                with open(count_error_log_path, "a", encoding="utf-8") as lf:
                    lf.write(f"{pmc}\tabstract_count\t{type(e).__name__}: {str(e).replace('\n',' ')}\n")
            except Exception:
                pass
                
    # text_count
    try:
        metadata["text_count"] = count_words(text)
    except Exception as e:
        if count_error_log_path and pmc:
            try:
                with open(count_error_log_path, "a", encoding="utf-8") as lf:
                    lf.write(f"{pmc}\ttext_count\t{type(e).__name__}: {str(e).replace('\n',' ')}\n")
            except Exception:
                pass
            
    return {"abstract": abstract, "text": text, "metadata": metadata}



def tar_to_jsonl(tar_path: str) -> str:
    """ Convert a .tar.gz of JATS XML files to a single .jsonl.gz file.
    parsed = number of XML files parsed
    kept   = number of JSON records written
    skipped= number of XML files skipped (no pmc or errors)
    """
    base = os.path.basename(tar_path)
    prefix = os.path.splitext(os.path.splitext(base)[0])[0]  # drop .tar(.gz)
    out_path = os.path.join(OUT_DIR, f"{prefix}.jsonl.gz") # e.g. data/jsonl/oa_comm_xml.PMC012xxxxxx.baseline.2025-06-26.jsonl.gz
    log_path = os.path.join(LOG_DIR, f"{prefix}/missing_abstracts.log")
    count_error_log_path = os.path.join(LOG_DIR, f"{prefix}/error_count.log")
    try:
        os.makedirs(os.path.dirname(count_error_log_path), exist_ok=True)
    except Exception:
        pass

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
                rec = build_record(xml_bytes, count_error_log_path=count_error_log_path)
                # Log PMC IDs that lack an abstract
                if not (rec.get("abstract") or "").strip():
                    try:
                        os.makedirs(os.path.dirname(log_path), exist_ok=True)
                        with open(log_path, "a", encoding="utf-8") as lf:
                            title = (rec.get("metadata", {}).get("article_title") or "").replace("\n", " ")
                            lf.write(f"{rec.get('metadata',{}).get('pmc','')}\t{title}\n")
                    except Exception:
                        # Logging must not break the pipeline; ignore log I/O errors
                        pass
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                kept += 1
            except Exception as e:
                skipped += 1  # skip invalid XML or other errors
                print(f"[ERROR] Skipped {m.name} due to error: {e}", file=sys.stderr)

    print(f"[OK] {base}: parsed={parsed}, written={kept}, skipped={skipped}")
    print(f"[OUT] {out_path}")
    return out_path


# ---------- Download tarball & make JSONL ----------
def download_tar(tar_name: str) -> str:
    url = BASE_URL + tar_name
    out = os.path.join(RAW_DIR, tar_name)
    if not os.path.exists(out):
        print(f"[download] {url}")
        # Download to a temporary file to avoid corrupted files on interruption
        tmp_out = out + ".tmp"
        try:
            urllib.request.urlretrieve(url, tmp_out)
            os.rename(tmp_out, out)
        except Exception as e:
            print(f"[ERROR] Download failed: {e}", file=sys.stderr)
            if os.path.exists(tmp_out):
                os.remove(tmp_out)
            sys.exit(1)
    else:
        print(f"[skip] already exists: {out}")
    return out



def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print("python run.py <tar_name_or_url>  # e.g., oa_comm_xml.PMC012xxxxxx.baseline.2025-06-26.tar.gz")
        sys.exit(1)

    arg = sys.argv[1] # tar file name or full URL
    tar_path = None
    
    if arg.startswith("http"):
        tar_name = os.path.basename(arg)
        tar_path = os.path.join(RAW_DIR, tar_name)
        if not os.path.exists(tar_path):
            print(f"[download] {arg}")
            tmp_path = tar_path + ".tmp"
            try:
                urllib.request.urlretrieve(arg, tmp_path)
                os.rename(tmp_path, tar_path)
            except Exception as e:
                print(f"[ERROR] Download failed: {e}", file=sys.stderr)
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
                sys.exit(1)
        else:
            print(f"[skip] already exists: {tar_path}")
    else:
        tar_path = download_tar(os.path.basename(arg))

    if tar_path and os.path.exists(tar_path):
        tar_to_jsonl(tar_path)
    else:
        print(f"[ERROR] Tar file not found at: {tar_path}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()