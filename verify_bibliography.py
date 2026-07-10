#!/usr/bin/env python3
"""Machine-verify a REVTeX thebibliography block against the arXiv API.

Usage:
    python3 verify_bibliography.py <path-to-tex-file> [output.json]

Stdlib only. Parses \\bibitem{key} ... entries, extracts:
  - key
  - first-author surname (as asserted in the tex)
  - asserted journal abbreviation, volume, page/article-id, year
  - arXiv id (if present)
Then queries http://export.arxiv.org/api/query?id_list=... in batches
(comma-separated ids, one HTTP request per batch, ~3s delay between
requests) and cross-checks:
  - id resolves on arXiv?
  - title returned (for human eyeballing / record)
  - first-author surname match (asserted vs arXiv metadata)
  - published journal ref (arXiv journal_ref / doi field) vs the asserted
    journal/volume/page string (fuzzy substring/number match, since arXiv's
    journal_ref field is free text supplied by the submitter and not always
    populated)

Entries with no arXiv id are marked NO_ARXIV_ID; the caller is expected to
sanity-check those by hand (e.g. one web search each) since there's no API
to hit.
"""
import json
import re
import sys
import time
import urllib.request
import urllib.error
import xml.etree.ElementTree as ET

ARXIV_API = "https://export.arxiv.org/api/query"
ATOM_NS = "{http://www.w3.org/2005/Atom}"
ARXIV_NS = "{http://arxiv.org/schemas/atom}"
BATCH_SIZE = 10
DELAY_SECONDS = 3


def parse_bibitems(tex_path):
    """Return list of dicts: key, raw_text, authors_raw, first_author_surname,
    journal, volume, page, year, arxiv_id."""
    with open(tex_path, "r", encoding="utf-8") as f:
        text = f.read()

    # Restrict to the thebibliography environment if present.
    m = re.search(r"\\begin\{thebibliography\}(.*?)\\end\{thebibliography\}", text, re.S)
    body = m.group(1) if m else text

    # Split on \bibitem{...}; keep the key.
    items = re.split(r"\\bibitem\{([^}]+)\}", body)
    # items[0] is preamble (junk); then alternating key, entry_text
    entries = []
    for i in range(1, len(items), 2):
        key = items[i].strip()
        raw = items[i + 1] if i + 1 < len(items) else ""
        # Entry text runs until the next \bibitem or \end{thebibliography};
        # the split already handles that. Trim trailing whitespace/newlines,
        # collapse internal newlines to spaces.
        raw = raw.strip()
        raw = re.sub(r"\s+", " ", raw)
        entries.append(parse_entry(key, raw))
    return entries


def latex_clean(s):
    """Strip common LaTeX markup/escapes for readable comparison text."""
    s = re.sub(r"\\emph\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\\textbf\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\\url\{([^}]*)\}", r"\1", s)
    s = s.replace(r"\ ", " ").replace("~", " ")
    s = re.sub(r"\\['\"^`.]?(\w)", r"\1", s)  # accents: \'e, \"o, \^i, etc.
    s = re.sub(r"\\[a-zA-Z]+", "", s)  # remaining control words
    s = s.replace("{", "").replace("}", "")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def parse_entry(key, raw):
    entry = {"key": key, "raw": raw}

    # Matches both new-style (YYMM.NNNNN) and old-style
    # (category/YYMMNNN, e.g. astro-ph/0311260 or gr-qc/9906015) arXiv ids.
    arxiv_match = re.search(
        r"arXiv:([a-zA-Z\-\.]+/\d{7}|\d{4}\.\d{4,5})(v\d+)?", raw
    )
    entry["arxiv_id"] = arxiv_match.group(1) if arxiv_match else None

    year_match = re.search(r"\((\d{4})\)", raw)
    entry["year"] = year_match.group(1) if year_match else None

    # Journal / volume / page: look for "\textbf{VOL}, PAGE (YEAR)" pattern,
    # with the journal name being the text before \textbf.
    jvp_match = re.search(
        r"([A-Za-z][A-Za-z.\\ ~]*?)\\textbf\{(\d+)\},\s*([A-Za-z0-9]+)\s*\((\d{4})\)",
        raw,
    )
    if jvp_match:
        entry["journal"] = latex_clean(jvp_match.group(1)).rstrip(",").strip()
        entry["volume"] = jvp_match.group(2)
        entry["page"] = jvp_match.group(3)
    else:
        entry["journal"] = None
        entry["volume"] = None
        entry["page"] = None

    # First author surname: text up to the earliest of a comma, "\emph{et
    # al.}", a "(YYYY)" year parenthesis, or a "\textbf{" (volume marker).
    # Authors are typically "A.~B.~Surname, C.~D.~Other, ..." with ~ as
    # non-breaking space, or "Collaboration Name,".
    markers = [
        m.start()
        for m in [
            re.search(r",", raw),
            re.search(r"\\emph\{et al\.\}", raw),
            re.search(r"\(\d{4}\)", raw),
            re.search(r"\\textbf\{", raw),
        ]
        if m
    ]
    cutoff = min(markers) if markers else len(raw)
    authors_part = raw[:cutoff]
    authors_clean = latex_clean(authors_part)
    tokens = authors_clean.split()
    if tokens:
        # Surname is the last token of the first author's name segment.
        entry["first_author_surname"] = tokens[-1]
    else:
        entry["first_author_surname"] = None
    entry["authors_raw"] = authors_clean

    return entry


def fetch_arxiv_batch(ids):
    """ids: list of arXiv ids (no version tag needed for lookup, but include
    if present). Returns dict id -> metadata dict (or None if not found)."""
    query_ids = ",".join(ids)
    url = f"{ARXIV_API}?id_list={urllib.request.quote(query_ids, safe=',')}&max_results={len(ids)}"
    req = urllib.request.Request(url, headers={"User-Agent": "bibliography-verify-script/1.0"})
    data = None
    last_err = None
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = resp.read()
            break
        except (urllib.error.URLError, ConnectionResetError, TimeoutError) as e:
            last_err = e
            if attempt < 2:
                time.sleep(DELAY_SECONDS * (attempt + 1))
    if data is None:
        return {i: {"error": str(last_err)} for i in ids}

    # Note on XXE: response is from a fixed trusted host (export.arxiv.org)
    # over a request we constructed ourselves; task constrains this script to
    # stdlib only (no defusedxml). Python 3's xml.etree.ElementTree (backed by
    # expat) does not resolve external entities/DTDs by default, so this is
    # not exploitable via the arXiv API's own Atom response.
    root = ET.fromstring(data)
    results = {}
    entries = root.findall(f"{ATOM_NS}entry")
    for entry_el in entries:
        id_url = entry_el.findtext(f"{ATOM_NS}id", default="")
        # id_url like http://arxiv.org/abs/2112.04510v2 or .../abs/gr-qc/9906015v1
        m = re.search(r"abs/([a-zA-Z\-\.]+/\d{7}|\d{4}\.\d{4,5})", id_url)
        found_id = m.group(1) if m else id_url
        title = entry_el.findtext(f"{ATOM_NS}title", default="").strip()
        title = re.sub(r"\s+", " ", title)
        authors = [
            a.findtext(f"{ATOM_NS}name", default="").strip()
            for a in entry_el.findall(f"{ATOM_NS}author")
        ]
        journal_ref = entry_el.findtext(f"{ARXIV_NS}journal_ref", default=None)
        doi = entry_el.findtext(f"{ARXIV_NS}doi", default=None)
        published = entry_el.findtext(f"{ATOM_NS}published", default="")
        results[found_id] = {
            "title": title,
            "authors": authors,
            "journal_ref": journal_ref,
            "doi": doi,
            "published": published,
            "found": True,
        }
    return results


def match_result_for_id(all_results, arxiv_id):
    """arXiv ids returned by the API sometimes drop a leading category
    (e.g. asked gr-qc/9906015, or asked with vN). Try a few lookups."""
    if arxiv_id in all_results:
        return all_results[arxiv_id]
    # Try stripping version suffix
    base = re.sub(r"v\d+$", "", arxiv_id)
    if base in all_results:
        return all_results[base]
    # Try matching by suffix (old-style ids sometimes normalized)
    for k, v in all_results.items():
        if k.endswith(base) or base.endswith(k):
            return v
    return None


def strip_diacritics(s):
    import unicodedata

    nfkd = unicodedata.normalize("NFKD", s)
    return "".join(c for c in nfkd if not unicodedata.combining(c))


def surname_matches(asserted, arxiv_first_author_name):
    if not asserted or not arxiv_first_author_name:
        return False
    asserted_norm = strip_diacritics(asserted).lower().strip(".,")
    arxiv_norm = strip_diacritics(arxiv_first_author_name).lower()
    return asserted_norm in arxiv_norm


def journal_matches(entry, meta):
    """Loose check: does the arXiv journal_ref (if any) contain the asserted
    volume and page/article-id numbers? journal_ref is free text and often
    absent even for published papers, so ABSENT is reported distinctly from
    MISMATCH."""
    jref = meta.get("journal_ref") if meta else None
    if not entry.get("volume"):
        return "NO_ASSERTED_JOURNAL_INFO"
    if not jref:
        return "ARXIV_JOURNAL_REF_ABSENT"
    vol_ok = entry["volume"] in jref
    page_ok = entry["page"] in jref if entry.get("page") else True
    if vol_ok and page_ok:
        return "MATCH"
    return f"POSSIBLE_MISMATCH (asserted vol {entry.get('volume')} page {entry.get('page')} vs journal_ref={jref!r})"


def verify(tex_path):
    entries = parse_bibitems(tex_path)
    with_id = [e for e in entries if e["arxiv_id"]]
    without_id = [e for e in entries if not e["arxiv_id"]]

    all_results = {}
    ids = [e["arxiv_id"] for e in with_id]
    for i in range(0, len(ids), BATCH_SIZE):
        batch = ids[i : i + BATCH_SIZE]
        batch_results = fetch_arxiv_batch(batch)
        all_results.update(batch_results)
        if i + BATCH_SIZE < len(ids):
            time.sleep(DELAY_SECONDS)

    report = []
    for e in entries:
        row = {
            "key": e["key"],
            "raw": e["raw"],
            "asserted_first_author_surname": e["first_author_surname"],
            "asserted_journal": e["journal"],
            "asserted_volume": e["volume"],
            "asserted_page": e["page"],
            "asserted_year": e["year"],
            "arxiv_id": e["arxiv_id"],
        }
        if not e["arxiv_id"]:
            row["status"] = "NO_ARXIV_ID"
            report.append(row)
            continue

        meta = match_result_for_id(all_results, e["arxiv_id"])
        if not meta or meta.get("error"):
            row["status"] = "UNRESOLVED"
            row["error"] = meta.get("error") if meta else "id not found in API response"
            report.append(row)
            continue

        arxiv_first_author = meta["authors"][0] if meta.get("authors") else None
        author_match = surname_matches(e["first_author_surname"], arxiv_first_author)
        journal_check = journal_matches(e, meta)

        row["arxiv_title"] = meta.get("title")
        row["arxiv_first_author"] = arxiv_first_author
        row["arxiv_journal_ref"] = meta.get("journal_ref")
        row["arxiv_doi"] = meta.get("doi")
        row["author_match"] = author_match
        row["journal_check"] = journal_check

        if not author_match:
            row["status"] = "METADATA_MISMATCH"
            row["mismatch_reason"] = "first-author surname mismatch"
        elif journal_check.startswith("POSSIBLE_MISMATCH"):
            row["status"] = "METADATA_MISMATCH"
            row["mismatch_reason"] = journal_check
        else:
            row["status"] = "OK"
        report.append(row)

    return report


def main():
    if len(sys.argv) < 2:
        print("usage: verify_bibliography.py <tex-file> [output.json]", file=sys.stderr)
        sys.exit(1)
    tex_path = sys.argv[1]
    out_path = sys.argv[2] if len(sys.argv) > 2 else None

    report = verify(tex_path)

    ok = sum(1 for r in report if r["status"] == "OK")
    mismatch = sum(1 for r in report if r["status"] == "METADATA_MISMATCH")
    unresolved = sum(1 for r in report if r["status"] == "UNRESOLVED")
    no_id = sum(1 for r in report if r["status"] == "NO_ARXIV_ID")

    summary = {
        "tex_file": tex_path,
        "total_entries": len(report),
        "ok": ok,
        "metadata_mismatch": mismatch,
        "unresolved": unresolved,
        "no_arxiv_id": no_id,
    }
    output = {"summary": summary, "entries": report}

    text = json.dumps(output, indent=2, ensure_ascii=False)
    if out_path:
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(text)
        print(json.dumps(summary, indent=2))
    else:
        print(text)


if __name__ == "__main__":
    main()
