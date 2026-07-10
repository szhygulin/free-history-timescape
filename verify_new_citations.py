#!/usr/bin/env python3
"""Fetch title + abstract for candidate NEW citations from the arXiv API so the
in-text sentence added for each can be matched against what the paper actually
claims (its abstract). Stdlib only; one batched HTTP request. Companion to
verify_bibliography.py (which cross-checks the already-cited thebibliography
block); this one is the live-verification pass for citations added in the
paper2-litreview-polish edit.

Usage: python3 verify_new_citations.py [output.json]
"""
import json
import sys
import time
import urllib.request
import xml.etree.ElementTree as ET

ARXIV_API = "https://export.arxiv.org/api/query"
ATOM = "{http://www.w3.org/2005/Atom}"

# id -> short label of the role it plays in the edit
CANDIDATES = {
    "2404.02129": "B1 Wiltshire (revived question)",
    "2504.01669": "B2 CosmoVerse white paper 2025",
    "2311.13305": "B2 Verde-Schoneberg-Gil-Marin ARA&A",
    "2211.04492": "B2 Kamionkowski-Riess",
    "2404.08038": "B3 Breuval 2024 SH0ES SMC anchor",
    "2408.03474": "B3 Lee JAGB H0",
    "2510.23823": "B3 H0DN consensus local-H0",
    "2506.03023": "B4 TDCOSMO lensing H0",
    "2001.09213": "B4 megamasers H0",
    "2111.03604": "B4 GWTC-3 standard sirens",
    "2601.20633": "B5 Ginat-Ferreira inhomogeneities mimic w",
    "2411.00148": "B6 DESIVAST DR1 void catalog",
    "2007.09013": "B6 Aubert eBOSS voids 0.6<z<2.2",
}


def fetch(ids):
    q = "?id_list=" + ",".join(ids) + "&max_results=" + str(len(ids))
    req = urllib.request.Request(ARXIV_API + q, headers={"User-Agent": "cite-verify/1.0"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read().decode("utf-8")


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else None
    xml = fetch(list(CANDIDATES))
    root = ET.fromstring(xml)
    results = []
    for e in root.findall(ATOM + "entry"):
        idurl = e.findtext(ATOM + "id") or ""
        aid = idurl.rsplit("/abs/", 1)[-1]
        aid_base = aid.split("v")[0] if "v" in aid.split("/")[-1] else aid
        title = " ".join((e.findtext(ATOM + "title") or "").split())
        summ = " ".join((e.findtext(ATOM + "summary") or "").split())
        authors = [a.findtext(ATOM + "name") for a in e.findall(ATOM + "author")]
        published = e.findtext(ATOM + "published") or ""
        results.append({
            "arxiv_id": aid,
            "title": title,
            "first_author": authors[0] if authors else None,
            "n_authors": len(authors),
            "published": published,
            "abstract": summ,
        })
    # Report which requested ids resolved.
    got = {r["arxiv_id"].split("v")[0] for r in results}
    missing = [i for i in CANDIDATES if i not in got]
    report = {"requested": CANDIDATES, "missing": missing, "entries": results}
    txt = json.dumps(report, indent=1, ensure_ascii=False)
    if out:
        with open(out, "w", encoding="utf-8") as f:
            f.write(txt)
    print(txt)


if __name__ == "__main__":
    main()
