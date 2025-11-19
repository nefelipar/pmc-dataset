#!/usr/bin/env python3
# pip install beautifulsoup4 lxml nltk

import os
import re
import sys
import tarfile
import urllib.request
import gzip
import json
import html
import argparse
from bs4 import BeautifulSoup, NavigableString
from unicodedata import normalize

BASE_URL = "https://ftp.ncbi.nlm.nih.gov/pub/pmc/oa_bulk/oa_comm/xml/"
RAW_DIR  = "data/tar"     # save .tar.gz
OUT_DIR  = "data/jsonl"   # save .jsonl.gz
LOG_DIR  = "data/logs"    # logs (e.g., missing abstracts)

# Containers we don't want any text from (drop completely)
FORBIDDEN_CONTAINER_TAGS = {
    "fig", "figure", "fig-group",
    "graphic", "inline-graphic", "media",
    "table", "table-wrap", "table-wrap-foot",
    "chem-struct-wrap",
    "array", "supplementary-material"}

os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)


# ---------- Text cleaning & extraction ----------
def clean(s: str) -> str:
    if s is None:
        return ""
    s = html.unescape(s) # Handling HTML entities like &#x02013;
    s = normalize("NFC", s) # Normalize to unicode

    # Compress whitespace (spaces, tabs, newlines) to single space
    s = re.sub(r"[\s]+", " ", s).strip()

    # Remove spaces before punctuation (e.g. gene , name → gene, name)
    s = re.sub(r"\s+([,.;:!?%])", r"\1", s)

    # Don't leave space before closing brackets/quotes (e.g. "(text )" → "(text)")
    s = re.sub(r"\s+([)\]\}»”’])", r"\1", s)

    # Don't leave space after opening brackets/quotes (e.g. "( text)" → "(text)")
    s = re.sub(r"([(\[\{«“‘])\s+", r"\1", s)
    

    # --- Normalization placeholders ---
    PLACEHOLDERS = ["CIT_REF","FIG_REF","TAB_REF","APP_REF","BOX_REF","SUP_REF"]
    SEP_WORDS = r"(?i:and\/or|and|or|&|\+)"
    SEP = rf"(?:[-–—,;]|{SEP_WORDS})"

    # This loop repeatedly applies the rules until the string
    # stops changing, fixing nested cases like "([[CIT_REF]])".
    s_before = None
    while s_before != s:
        s_before = s

        for PH in PLACEHOLDERS:
            # 1) Any number of surrounding () or [] around a single placeholder -> same placeholder
            # e.g. "([CIT_REF])", "(([CIT_REF]))",  -> "[CIT_REF]"
            # e.g. "[[CIT_REF]]", "[[[CIT_REF]]]"-> "[CIT_REF]"
            s = re.sub(rf'(?:\(\s*|\[\s*)+\[{PH}\](?:\s*\)|\s*\])+', f'[{PH}]', s)

            # 2) Lists of same placeholders inside () separated by - – — , ;  -> one placeholder
            # e.g.  "( [CIT_REF] - [CIT_REF] )", "( [CIT_REF], [CIT_REF] )", "( [CIT_REF] ; [CIT_REF] )" → [CIT_REF]
            s = re.sub(rf'(?:\(\s*)+\[{PH}\](?:\s*{SEP}\s*\[{PH}\])+(?:\s*\))+', f'[{PH}]', s)

            # 3) Lists of same placeholders inside [] separated by - – — , ;  -> one placeholder
            #  e.g.  "[ [CIT_REF] - [CIT_REF] ]", "[ [CIT_REF], [CIT_REF] ]", "[ [CIT_REF] ; [CIT_REF] ]" → [CIT_REF]
            s = re.sub(rf'(?:\[\s*)+\[{PH}\](?:\s*{SEP}\s*\[{PH}\])+(?:\s*\])+', f'[{PH}]', s)

            # 4) Lists of same placeholders inside () separated only by spaces -> one placeholder
            # e.g. "( [CIT_REF]  [CIT_REF]  [CIT_REF] )" → [CIT_REF]
            s = re.sub(rf'(?:\(\s*)+(?:\[{PH}\]\s*){{2,}}(?:\s*\))+', f'[{PH}]', s)

            # 5) Lists of same placeholders inside [] separated only by spaces -> one placeholder
            #  e.g.  "[ [CIT_REF]  [CIT_REF] ]" → [CIT_REF]
            s = re.sub(rf'(?:\[\s*)+(?:\[{PH}\]\s*){{2,}}(?:\s*\])+', f'[{PH}]', s)

        # 6) Failsafes OUTSIDE brackets/parentheses:
        # e.g. "[CIT_REF] - [CIT_REF]", "[CIT_REF]; [CIT_REF]", "[CIT_REF] [CIT_REF]" → [CIT_REF]
        for PH in PLACEHOLDERS:
            s = re.sub(rf'\[{PH}\](?:\s*{SEP}\s*\[{PH}\])+', f'[{PH}]', s)
            s = re.sub(rf'\[{PH}\](?:\s+\[{PH}\])+', f'[{PH}]', s)

    # Digit grouping commas: "1, 000" → "1,000"
    # Only if between digits (to avoid messing with lists)
    s = re.sub(r"(?<=\d)\s*,\s*(?=\d)", ",", s)

    # Remove spaces around hyphens in words (e.g. "state - of - the - art" → "state-of-the-art")
    # Do not touch cases like "17- and"
    s = re.sub(r"(?<=[A-Za-z])\s*-\s*(?=[A-Za-z])", "-", s)

    # Remove spaces around en dash when it appears between digits: "2000 – 2021" → "2000–2021"
    s = re.sub(r"(?<=\d)\s*–\s*(?=\d)", "–", s)

    # No spaces around '=' ("x = 5" → "x=5")
    s = re.sub(r"\s*=\s*", "=", s)

    # --- Subscript/Superscript cleanup ---
    # 1. Remove whitespace inside { ... }
    # e.g. "^{ 3 }" -> "^{3}", "_{ max }" -> "_{max}"
    s = re.sub(r'\{\s*([^}]*)\s*\}', r'{\1}', s)

    # 2. Attach preceding letter/digit to following subscript/superscript tokens (_{...} or ^{...})
    # e.g. "D _{3} KO" -> "D_{3} KO", "x ^{2}" -> "x^{2}"
    s = re.sub(r'(?<=\w)\s+([_^]\{[^}]+\})', r'\1', s)

    # 3. inside brackets/parentheses: don't leave spaces immediately after opening
    # or before closing, so "[ ^{3} H ]" -> "[^{3}H]"
    s = re.sub(r'\[\s+(?=[\w(_^])', '[', s)
    s = re.sub(r'(?<=[\w})])\s+\]', ']', s)
    s = re.sub(r'\(\s+(?=[\w(_^])', '(', s)
    s = re.sub(r'(?<=[\w})])\s+\)', ')', s)

    # 4. if followed by uppercase/punctuation, glue the token
    # e.g. "D_{3} KO" (without space), "[^{3} H]" -> "[^{3}H]"
    s = re.sub(r'([_^]\{[^}]+\})\s+(?=[A-Z\)\]\},\.;:!?])', r'\1', s)
    # --- End Subscript/Superscript cleanup ---

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
    """ 
    If boundary_name is provided, ensure the tag is inside that boundary and not
    inside a forbidden container. If boundary_name is None/empty, only enforce
    that it's not inside a forbidden container.
    """
    p = getattr(tag, "parent", None)
    boundary_name = (boundary_name or "").lower()
    while p is not None:
        name = (getattr(p, "name", None) or "").lower()
        if not name:
            p = getattr(p, "parent", None)
            continue
        if name in FORBIDDEN_CONTAINER_TAGS:
            return False
        if name == boundary_name:
            return True
        p = getattr(p, "parent", None)
    return False


def is_in_ignored_section(tag) -> bool:
    """Detect if a tag is inside sections we want to exclude from body text
    (acknowledgments, references, bibliography, appendix, supplementary, etc.).
    """
    IGNORE_SECTION_TITLES = re.compile(
    r"^(?:"
    # Core sections
    r"acknowledg(e)?ments?|"
    r"references?|"
    r"bibliograph(?:y|ies)|"

    # Author Contributions
    r"(?:authours|autors|authors?)'?(?:s|'s)? ?(?:contributions?|contribtions?|conributions?|constributions?|protections|rights(?: and users' rights)?|voice|competing interest(?:s)?|conflict of interest)|"
    r"contributions?(?: of .*)?|"
    r"contributor(?:s)?(?: information)?|"
    r"credit authorship contribution statement|"

    # Collaboration
    r"col{1,2}ab(?:o|a)?rations?|"

    # List of people/committees
    r"(?:clinical centres and )?investigators|committee members|working group|"

    # Competing Interests & Ethics
    r"conflict of interests?|"
    r"(?:declaration of )?competing interest(?:s)?|"
    r"ethics statement|"
    r"(?:informed )?consent(?: (?:for publication|to participate|statement|and assent|and ethics))?(?:[:\s].*)?|"
    r"ethical consideration(?:s)?(?:[:\s].*)?|"

    # History & Notes
    r"pre-publication history|"
    r"note added in proof|"
    r"endnotes?|"
    r"notes?|"
    r"footnotes?|"

    # Supplementary & Data
    r"supplementar(?:y|ies|y material(?:s)?)?|"
    r"supplemental|"
    r"supporting information|"
    r"additional (?:data )?files?|"
    r"associated data|"

    # Lists & Terminology
    r"(?:lists? of )?(?:non-standard )?abb?re?vi?ations?(?: .*)?|"
    r"nomenclature|"
    r"list of symbols|"

    # Availability
    r"data availability(?: statement)?|"
    r"availability(?: and requirements)?|"
    r"free(?:, full-text)?(?: access)? (?:versus open access|for all articles?)|"

    # Meta, Legal & Publishing
    r"(?:source of )?funding(?: sources?| acknowledg[em]+nts?| and ethics| for .* research| statement| support|/support)?|"
    r"disclaimer|"
    r"open access|"
    r"copyright(?: .*)?|"
    r"license|"
    r"permissions|"
    r"intellectual property rights|"
    r"peer review policy|journal scope|update|"
    r"how to cite|"

    # Summary-like sections (excluding main abstract)
    r"highlights|"
    r"key points|"
    r"keywords|"
    r"graphical abstract|"
    r"glossary|term definitions|"

    # Other common meta
    # r"limitation(?:s)?|"
    r"append(?:ix|ices)(?:[:\s].*)?"
    r")$",
    re.IGNORECASE
)

    IGNORE_SEC_TYPES = {"supplementary-material", "display-objects", "data-availability", "COI-statement"}

    node_to_check = tag
    while node_to_check is not None:
        name = (getattr(node_to_check, "name", None) or "").lower()
        if not name:
            node_to_check = getattr(node_to_check, "parent", None)
            continue

        if name in {"ack", "app-group", "app", "bio", "bios" ,"author-notes", "notes", "trans-abstract" ,"ref-list", "glossary", "fn-group", "fn"}:  
            return True
        if name == "sec":
            # 1. Check for ignored section types
            sec_type = (node_to_check.get("sec-type") or "").lower()
            if sec_type in IGNORE_SEC_TYPES:
                return True

            # 2. Look for a title child to decide
            t = node_to_check.find("title", recursive=False)
            if t:
                title_text = clean(t.get_text(" ", strip=True))
                if title_text and IGNORE_SECTION_TITLES.match(title_text):
                    return True
        node_to_check = getattr(node_to_check, "parent", None)
    return False


def remove_figures_and_tables(soup_or_tag):
    """Remove figures and tables and their captions completely from the tree."""
    for tname in FORBIDDEN_CONTAINER_TAGS:
        for el in soup_or_tag.find_all(tname):
            el.decompose()


def replace_citations_with_placeholder(soup_or_tag):
    """
    Find all <xref> tags that are citations (ref-type=bibr or citation)
    and replace them with the placeholder "[CIT_REF]".
    This is more robust than regex-based removal.
    """
    CITATION_REFS = {"bibr", "citation", "ref"}
 
    for xr in soup_or_tag.find_all("xref"):
        rt = (xr.get("ref-type") or "").lower()
        if rt in CITATION_REFS:
            xr.replace_with("[CIT_REF]")

def replace_math_with_placeholder(soup_or_tag):
    """
    Find all <inline-formula> and <disp-formula> tags
    and replace them with the placeholder "[MATH]".
    """
    MATH_TAGS = ["inline-formula", "disp-formula", "tex-math", "math"]
    for tag_name in MATH_TAGS:
        for math_el in soup_or_tag.find_all(tag_name):
            math_el.replace_with("[MATH]")

def replace_code_with_placeholder(soup_or_tag):
    """
    Replace any <code> or <preformat> blocks with the placeholder "[CODE]".
    This runs before inline markup conversion so inner content isn't processed.
    """
    for tag_name in ["code", "preformat"]:
        for el in soup_or_tag.find_all(tag_name):
            el.replace_with("[CODE]")

def convert_inline_markup(soup_or_tag):
    """
    Converts <sup>, <sub> to caret/underscore and unwraps <italic>/<bold> etc.,
    so that spaces are not inserted by get_text.
    """
    # superscripts: ^3
    for sup in soup_or_tag.find_all('sup'):
        txt = sup.get_text("", strip=True)
        sup.replace_with(f"^{{{txt}}}")

    # subscripts: _max
    for sub in soup_or_tag.find_all('sub'):
        txt = sub.get_text("", strip=True)
        sub.replace_with(f"_{{{txt}}}")

    # italics/bold/smallcaps -> plain text (or optionally add *…*)
    for tag in soup_or_tag.find_all(['italic','i','em','bold','b','strong']):
        tag.unwrap()

    # smallcaps <sc> -> uppercase text
    for sc in soup_or_tag.find_all('sc'):
        sc.string = (sc.get_text("", strip=True) or "").upper()
        sc.unwrap()

    # monospace -> `txt`
    for mono in soup_or_tag.find_all('monospace'):
        txt = mono.get_text("", strip=True)
        mono.replace_with(f"`{txt}`")

def fix_glued_xrefs(soup_or_tag):
    """
    - KEEPS content <xref> tags (e.g., fig, table).
    - For kept tags, it fixes "glued panel letters" by merging them
      (e.g., <xref>Fig 1</xref>A becomes <xref>Fig 1A</xref>)
      to prevent "Fig 1 A" during text extraction.
    """
    CONTENT_REFS = {"fig", "table", "table-fn", "app", "boxed-text", "supplementary-material"}

    for xr in soup_or_tag.find_all("xref"):
        rt = (xr.get("ref-type") or "").lower()
        if rt not in CONTENT_REFS:
            continue
        
        # Look ahead to find panel letters or ranges
        ns = xr.next_sibling
        while ns and not isinstance(ns, NavigableString):
            ns = ns.next_sibling

        if isinstance(ns, NavigableString):
            s = str(ns)
            # Regex to find glued panel letters (e.g., "A", "B", "A–C")
            m = re.match(r"^([A-Za-z]{1,2}\d{0,2}(?:\s*[–-]\s*[A-Za-z]{1,2}\d{0,2})?)", s)
            if m:
                panel_text = m.group(1)
                remaining_text = s[len(panel_text):]

                    # Merge panel_text into the end of the xref
                if xr.string:
                    xr.string.replace_with(xr.string + panel_text)
                else:
                    # Try to find last NavigableString in contents
                    last_string = None
                    for content in reversed(xr.contents):
                        if isinstance(content, NavigableString):
                            last_string = content
                            break
                    if last_string:
                        last_string.replace_with(last_string + panel_text)
                    else:
                        xr.append(NavigableString(panel_text))

                # Remove merged part from the following string
                ns.replace_with(remaining_text)

def replace_content_refs_with_placeholders(soup_or_tag):
    """
    Replace inline content xref tags (figures, tables, etc) with [FIG], [TAB], [APP], [BOX].
    Assumes fix_glued_xrefs has already been applied to glue panel letters.
    """
    placeholder_map = {
        "fig": "[FIG_REF]",
        "table": "[TAB_REF]",
        "table-fn": "[TAB_REF]",
        "app": "[APP_REF]",
        "boxed-text": "[BOX_REF]",
        "supplementary-material": "[SUP_REF]"
        }

    for xr in soup_or_tag.find_all("xref"):
        rt = (xr.get("ref-type") or "").lower()
        if rt not in placeholder_map:
            continue

        placeholder = placeholder_map[rt]

        # Check if previous sibling is a Fig./Table prefix and remove it
        prev = xr.previous_sibling
        if isinstance(prev, NavigableString):
            prev_text = str(prev)
            new_text = re.sub(
                r'(\s*[\(\[]?\s*)(?:Fig(?:\.|ure)?|Table|Appendix|Box)\.?\s*$',
                '',
                prev_text,
                flags=re.I,
            )
            if new_text != prev_text:
                if new_text:
                    prev.replace_with(new_text)
                else:
                    prev.extract()

        # Check for glued suffix (e.g., "-C") — though usually handled in fix_glued_xrefs
        next_ = xr.next_sibling
        if isinstance(next_, NavigableString):
            m = re.match(r'^\s*[-–—]\s*\w{1,3}', next_)
            if m:
                # Remove suffix from sibling
                next_.replace_with(next_[m.end():])

        xr.replace_with(placeholder)

# ---------- JATS pickers ----------
def _process_section_markdown(section_tag, level, kept_titles_set, blockquote_prefix: str = ""):
    """
    Recursive helper function to process a section (<sec>) and its children.
    Handles nested <sec>, <p>, <list>, <caption> and <boxed-text>.
    Applies a `blockquote_prefix` (e.g., "> ") if inside a <boxed-text>.
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
                if level == 1:
                    kept_titles_set.add(title_text)
                content_blocks.append(f"{blockquote_prefix}{'#' * level} {title_text}")

        # --- HANDLE paragraphs ---
        elif tag_name == 'p':
            # Check if this <p> tag contains other block-level elements (like <list>)
            # This is technically invalid JATS, but common in real-world XML.
            block_children = child.find_all(['list', 'sec', 'boxed-text', 'p'], recursive=False)
            
            if not block_children:
                # --- Simple Case ---
                # This <p> only contains text and inline tags (<b>, <i>, etc.)
                # We can safely get its text.
                if in_boundary_without_forbidden(child, "body") or \
                   in_boundary_without_forbidden(child, "abstract") or \
                   in_boundary_without_forbidden(child, "boxed-text"):
                    
                    p_text = text_from(child)
                    if p_text:
                        content_blocks.append(f"{blockquote_prefix}{p_text}")
            else:
                # --- Complex Case ---
                # This <p> is acting as a "wrapper" for other blocks (like <list>).
                # Do NOT call text_from(). Instead, recurse into this <p>
                # as if it were a <sec>, but without changing the header level.
                content_blocks.extend(
                    _process_section_markdown(child, level, kept_titles_set, blockquote_prefix)
                )

        # --- Handle lists ---
        elif tag_name == 'list':
            list_items = []
            list_type = (child.get("list-type") or "bullet").lower()
            is_ordered = "order" in list_type or "decimal" in list_type
            
            item_counter = 1
            # A <list-item> can also contain <p> tags, so we must recurse!
            for item in child.find_all("list-item", recursive=False):
                # We get the text by processing the *children* of the list-item
                # This correctly handles <list-item><p>...</p></list-item>
                
                # We use a helper function to get text *without* calling _process_section_markdown
                # to avoid nested list detection issues. text_from() is correct here.
                item_text = text_from(item)
                
                if item_text:
                    if is_ordered:
                        prefix = f"{item_counter}."
                        item_counter += 1
                    else:
                        prefix = "*" # Use asterisk for bullets
                    
                    list_items.append(f"{blockquote_prefix}{prefix} {item_text}")
            
            if list_items:
                content_blocks.append("\n".join(list_items))
        
        # --- Handle captions ---
        elif tag_name == 'caption':
            title_tag = child.find("title", recursive=False)
            if title_tag:
                title_text = text_from(title_tag)
                if title_text:
                    content_blocks.append(f"{blockquote_prefix}### {title_text}")
            
            for p in child.find_all("p", recursive=False):
                p_text = text_from(p)
                if p_text:
                    content_blocks.append(f"{blockquote_prefix}{p_text}")

        # --- Handle boxed-text ---
        elif tag_name == 'boxed-text':
            inner_blocks = _process_section_markdown(child, level + 1, kept_titles_set, blockquote_prefix + "> ")
            if inner_blocks:
                content_blocks.append(f"{blockquote_prefix}> ---")   # start rule at current depth
                content_blocks.extend(inner_blocks)                   # box content 
                content_blocks.append(f"{blockquote_prefix}> ---")   # end rule at current depth

        # --- Handle display quotes ---
        elif tag_name == 'disp-quote':
            quote_blocks = _process_section_markdown(child, level, kept_titles_set, blockquote_prefix + "> ")
            content_blocks.extend(quote_blocks)

        # --- Handle statements ---
        elif tag_name == 'statement':
            new_prefix = blockquote_prefix + "> "
            
            label_tag = child.find("label", recursive=False)
            if not label_tag:
                label_tag = child.find("title", recursive=False)
            
            label_str = ""
            if label_tag:
                label_text = text_from(label_tag)
                if label_text:
                    label_str = f"**{label_text}**"
                label_tag.decompose()

            # find the first <p> child (if any)
            first_p = child.find("p", recursive=False)
            
            if label_str and first_p:
                # Stick label and first paragraph together
                p_text = text_from(first_p)
                content_blocks.append(f"{new_prefix}{label_str} {p_text}")
                first_p.decompose()
            elif label_str:
                # No <p> found, just add the label on its own
                content_blocks.append(f"{new_prefix}{label_str}")

            # Process remaining content
            inner_blocks = _process_section_markdown(child, level, kept_titles_set, new_prefix)
            content_blocks.extend(inner_blocks)

        # --- Handle speeches ---
        elif tag_name == 'speech':
            new_prefix = blockquote_prefix + "> "

            speaker_tag = child.find("speaker", recursive=False)
            speaker_str = ""
            if speaker_tag:
                speaker_text = text_from(speaker_tag)
                if speaker_text.endswith(':'):
                    speaker_text = speaker_text[:-1].strip()
                if speaker_text:
                    speaker_str = f"**{speaker_text}:**"
                speaker_tag.decompose()

            # find the first <p> child (if any)
            first_p = child.find("p", recursive=False)
            
            if speaker_str and first_p:
                # Stick speaker and first paragraph together
                p_text = text_from(first_p)
                content_blocks.append(f"{new_prefix}{speaker_str} {p_text}")
                first_p.decompose() 
            elif speaker_str:
                # No <p> found, just add the speaker on its own
                content_blocks.append(f"{new_prefix}{speaker_str}")

            # Process remaining content
            inner_blocks = _process_section_markdown(child, level, kept_titles_set, new_prefix)
            content_blocks.extend(inner_blocks)


        # --- Handle nested sections ---
        elif tag_name == 'sec':
            content_blocks.extend(_process_section_markdown(child, level + 1, kept_titles_set, blockquote_prefix))
    
    # Return only blocks that actually have content
    # We join blocks with \n\n in the parent function (extract_body_with_markdown)
    return [block for block in content_blocks if block.strip()]



def extract_body_with_markdown(soup, kept_titles_set):
    """
    Extracts the main body of the article, preserving section titles and subtitles
    and formatting them as Markdown, starting from Level 1 (#).
    """
    body = soup.find("body")
    if not body:
        return ""

    # First, sanitize the structure within the body
    remove_figures_and_tables(body)
    replace_citations_with_placeholder(body)
    replace_math_with_placeholder(body)
    replace_code_with_placeholder(body)
    fix_glued_xrefs(body)
    replace_content_refs_with_placeholders(body)
    convert_inline_markup(body)
    # Start the recursive processing from the body.
    # Initial call with level=0 makes main sections start with '#'.
    all_blocks = _process_section_markdown(body, 0, kept_titles_set)

    # Join all collected blocks
    return "\n\n".join(all_blocks)


def extract_abstract(art):
    """Only normal abstracts, no graphical/teaser/etc; handles structured abstracts."""
    
    def pick_normal_abstract(art):
        """
        Picks the main technical abstract using a strict "whitelist" approach.
        It will ONLY return an abstract if its type is known to be
        a main technical abstract.
        """
        WANTED = {"", "abstract"} # normal abstract
        
        if not art: return None
        
        cands = art.find_all("abstract", recursive=False)
        if not cands: 
            return None

        for a in cands:
            a_type = (a.get("abstract-type") or "").lower()
            if a_type in WANTED:
                return a # Found the best case, return it

        return None  # No suitable abstract found
    
    abs_el = pick_normal_abstract(art)
    if not abs_el:
        return ""
    
    # Find the top-level <title> of the abstract (if it exists)
    top_title = abs_el.find("title", recursive=False)
    if top_title:
        title_text = text_from(top_title)
        # If the title is just "ABSTRACT", remove it before processing.
        if title_text.strip().upper() == "ABSTRACT":
            top_title.decompose()

    # sanitize abstract structure
    remove_figures_and_tables(abs_el)
    replace_citations_with_placeholder(abs_el)
    replace_math_with_placeholder(abs_el)
    replace_code_with_placeholder(abs_el)
    fix_glued_xrefs(abs_el)
    replace_content_refs_with_placeholders(abs_el)
    convert_inline_markup(abs_el)
    dummy_title_set = set()
    all_blocks = _process_section_markdown(abs_el, 1, dummy_title_set)
    return "\n\n".join(all_blocks)


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

def get_article_type(soup) -> str:
    art = soup.find("article")
    return ((art.get("article-type") or "") if art else "").strip().lower()

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
    art = get_article_meta(soup); 
    jmeta = get_journal_meta(soup)
    
    pmc  = article_id(art, "pmcid") or article_id(art, "pmc") 
    pmid = article_id(art, "pmid")
    doi  = article_id(art, "doi")

    atitle = art.select_one("title-group > article-title") if art else None
    jtitle = jmeta.select_one("journal-title-group > journal-title") if jmeta else None
    article_type = get_article_type(soup)

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
        "authors": authors,
        "article_type": article_type
    }
    return pmc, meta


def build_record(xml_bytes: bytes, kept_titles_set: set, count_error_log_path: str = None):
    """
    Creates a JSON record where:
    - "abstract": contains the abstract in plain text.
    - "text": contains ONLY the main body with its titles in Markdown.
    - "metadata": contains all metadata.
    """
    soup = BeautifulSoup(xml_bytes, "lxml-xml")
    pmc, metadata = extract_metadata(soup)

    # 1. Extract abstract.
    abstract = extract_abstract(get_article_meta(soup))

    # 2. Extract the main body (as Markdown).
    text = extract_body_with_markdown(soup, kept_titles_set)

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

    return {"pmcid": pmc ,"abstract": abstract, "text": text, "metadata": metadata}


def tar_to_jsonl(tar_path: str, out_dir: str, log_dir: str) -> str:
    """ Convert a .tar.gz of JATS XML files to a single .jsonl.gz file.
    parsed = number of XML files parsed
    kept   = number of JSON records written
    skipped= number of XML files skipped (no pmc or errors)
    """
    base = os.path.basename(tar_path)
    prefix = os.path.splitext(os.path.splitext(base)[0])[0]  # drop .tar(.gz)
    
    # Use specified output directory and log directory 
    out_path = os.path.join(out_dir, f"{prefix}.jsonl.gz") 
    
    # Create log subdirectory for this tarball
    tar_log_dir = os.path.join(log_dir, prefix)
    os.makedirs(tar_log_dir, exist_ok=True)
    
    log_path = os.path.join(tar_log_dir, "missing_abstracts.log")
    count_error_log_path = os.path.join(tar_log_dir, "error_count.log")
    titles_log_path = os.path.join(tar_log_dir, "kept_section_titles.log")

    parsed = kept = skipped = 0
    all_kept_titles = set()

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
                rec = build_record(xml_bytes, all_kept_titles, count_error_log_path=count_error_log_path)
                # Log PMC IDs that lack an abstract
                if not (rec.get("abstract") or "").strip():
                    try:
                        with open(log_path, "a", encoding="utf-8") as lf:
                            pmcid = rec.get("pmcid") or rec.get("metadata", {}).get("pmc", "")
                            title = (rec.get("metadata", {}).get("article_title") or "").replace("\n", " ")
                            lf.write(f"{pmcid}\t{title}\n")
                    except Exception:
                        # Logging must not break the pipeline; ignore log I/O errors
                        pass
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                kept += 1
            except Exception as e:
                skipped += 1  # skip invalid XML or other errors
                print(f"[ERROR] Skipped {m.name} due to error: {e}", file=sys.stderr)

    ## After processing is complete, write the collected titles to the log file.
    if all_kept_titles:
        print(f"[LOG] Writing {len(all_kept_titles)} unique kept titles to {titles_log_path}")
        try:
            # Sort titles alphabetically for easier review
            sorted_titles = sorted(list(all_kept_titles))
            with open(titles_log_path, "w", encoding="utf-8") as logf:
                for title in sorted_titles:
                    logf.write(title + "\n")
        except Exception as e:
            print(f"[ERROR] Could not write titles log file: {e}", file=sys.stderr)


    print(f"[OK] {base}: parsed={parsed}, written={kept}, skipped={skipped}")
    print(f"[OUT] {out_path}")
    return out_path


def main():
    parser = argparse.ArgumentParser(
        description="Convert JATS XML tarball to JSONL.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # --- Required Arguments ---
    parser.add_argument(
        "source",
        type=str,
        help="The .tar.gz file name (e.g., oa_comm_xml.PMC012xxxxxx.tar.gz) or the full URL to download it."
    )

    # --- Optional Arguments for Directories---
    parser.add_argument(
        "--raw_dir",
        type=str,
        default=RAW_DIR,
        help="Directory to save downloaded .tar.gz files."
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=OUT_DIR,
        help="Directory to save output .jsonl.gz files."
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default=LOG_DIR,
        help="Directory to save log files."
    )

    # --- Optional Flags ---
    parser.add_argument(
        "-f", "--force-download",
        action="store_true",
        help="Force re-download of the tarball even if it exists locally."
    )

    args = parser.parse_args()

    # --- Setup Directories ---
    os.makedirs(args.raw_dir, exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    # --- Main Logic ---
    arg = args.source
    tar_name = os.path.basename(arg)
    tar_path = os.path.join(args.raw_dir, tar_name)

    should_download = True
    if os.path.exists(tar_path) and not args.force_download:
        print(f"[skip] already exists: {tar_path}")
        should_download = False

    if should_download:
        if arg.startswith("http"):
            url = arg
        else:
            url = BASE_URL + tar_name

        print(f"[download] {url}")
        tmp_path = tar_path + ".tmp"
        try:
            urllib.request.urlretrieve(url, tmp_path)
            os.rename(tmp_path, tar_path)
        except Exception as e:
            print(f"[ERROR] Download failed: {e}", file=sys.stderr)
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            sys.exit(1)
    
    # --- Process Logic ---
    if os.path.exists(tar_path):
        tar_to_jsonl(tar_path, args.out_dir, args.log_dir)
    else:
        print(f"[ERROR] Tar file not found at: {tar_path}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
    
