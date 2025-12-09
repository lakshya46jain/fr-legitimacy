#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
website_cleaner.py
------------------
Normalize and identify the canonical homepage URL for each facial recognition
vendor in the dataset.

This script is used early in the SOC 4994 research pipeline and supports
downstream steps such as:
    • Wayback snapshot collection
    • HTML text extraction
    • Company-level NLP modeling (BERTopic)
    • Dashboard preparation

Features
--------
• Cleans URLs into standardized "homepage" forms
• Uses multiple HTML signals to find the best homepage:
      - <link rel="home">
      - <link rel="canonical">
      - <meta property="og:url">
      - Navbar links such as “Home”, “Inicio”, etc.
      - Logo anchors (<a class="logo">)
• Handles redirects, missing schemes, broken links
• Adds detailed notes explaining why a homepage was selected
• Supports both batch mode and single-URL testing

Usage
-----
Batch cleaning:
    python3 website_cleaner.py

Test a specific URL:
    python3 website_cleaner.py --url "http://www.example.com/about/"
"""

import csv
import re
import time
import argparse
from urllib.parse import urlparse, urlunparse, urljoin
import requests
from bs4 import BeautifulSoup


# -------------------------------------------------------
# SETTINGS
# -------------------------------------------------------
INPUT_FILE = "facial_recognition_vendors.csv"
OUTPUT_FILE = "facial_recognition_vendors.csv"

URL_COLUMN = "link"              # raw URL column
CLEAN_COLUMN = "clean_link"      # new canonical homepage column
NOTES_COLUMN = "link_clean_notes"  # metadata / reason column

TIMEOUT = 10
SLEEP_BETWEEN = 0.2
HEADERS = {"User-Agent": "WebsiteCleaner/1.0"}


# Prepare a requests session shared across all calls
session = requests.Session()
session.headers.update(HEADERS)


# -------------------------------------------------------
# NETWORK HELPERS
# -------------------------------------------------------
def safe_get(url):
    """
    Perform a GET request with error handling.

    Returns
    -------
    Response or None
        None indicates unreachable / timed out / invalid response.
    """
    try:
        return session.get(url, timeout=TIMEOUT, allow_redirects=True)
    except requests.RequestException:
        return None


# -------------------------------------------------------
# URL NORMALIZATION
# -------------------------------------------------------
def normalize_url(url):
    """
    Normalize a URL before attempting to access it.

    Steps:
        1. Ensure scheme exists (default http://)
        2. Lowercase hostname
        3. Strip :80 or :443
        4. Guarantee a path exists

    Returns
    -------
    str or None
    """
    if not url:
        return None

    # Add http:// if no scheme is present
    if not re.match(r'^https?://', url):
        url = "http://" + url

    parsed = urlparse(url)
    netloc = parsed.netloc.lower()

    # Remove default ports
    netloc = re.sub(r':(80|443)$', '', netloc)

    return urlunparse((
        parsed.scheme.lower(),
        netloc,
        parsed.path or "/",
        parsed.params,
        parsed.query,
        parsed.fragment
    ))


def path_depth(url):
    """
    Measure path depth to favor shorter homepage-like URLs.
    /         depth 0
    /about/   depth 1
    /a/b/c/   depth 3
    """
    return sum(1 for p in urlparse(url).path.split("/") if p)


# -------------------------------------------------------
# HOMEPAGE CANDIDATE EXTRACTION
# -------------------------------------------------------
def extract_candidates(html, base_url, base_netloc):
    """
    Parse homepage HTML to identify strong candidate homepage URLs.

    Signals used:
        • <link rel="home">
        • <link rel="canonical">
        • og:url (OpenGraph)
        • Navbar links containing “home”, “start”, “inicio”, etc.
        • Logo or branding links (<a class="logo">)

    Returns
    -------
    dict
        Mapping candidate_url -> [score, [reasons]]
    """
    soup = BeautifulSoup(html, "html.parser")
    candidates = {}

    def add(u, score, reason):
        """Helper to register a candidate homepage."""
        if not u:
            return

        p = urlparse(u)
        if not p.scheme.startswith("http"):
            return

        # Only keep URLs belonging to same domain
        if p.netloc and p.netloc.lower() != base_netloc:
            return

        # If path is bare or missing "/", ensure trailing slash consistency
        if not p.path or not p.path.endswith("/"):
            if not re.search(r'\.[a-zA-Z0-9]{1,6}$', p.path or ""):
                u = urlunparse((p.scheme, p.netloc, (p.path or "/") + "/", p.params, p.query, p.fragment))

        # Score accumulation
        candidates[u] = candidates.get(u, [0, []])
        candidates[u][0] += score
        candidates[u][1].append(reason)

    # --- <link> tags (strong HTML signals) ---
    for tag in soup.find_all("link", rel=True):
        rels = (
            tag.get("rel") if isinstance(tag.get("rel"), list)
            else [tag.get("rel")]
        )
        rels = [r.lower() for r in rels]

        if "home" in rels:
            add(urljoin(base_url, tag.get("href")), 30, "rel=home")
        if "canonical" in rels:
            add(urljoin(base_url, tag.get("href")), 20, "canonical")

    # --- og:url (common modern SEO) ---
    og = soup.find("meta", attrs={"property": "og:url"})
    if og and og.get("content"):
        add(urljoin(base_url, og["content"]), 15, "og:url")

    # --- Navigation links (medium signal) ---
    for a in soup.find_all("a", href=True):
        text = (a.get_text(strip=True) or "").lower()
        href = urljoin(base_url, a["href"])

        # Navbar / local language home words
        if text in ("home", "homepage", "start", "accueil", "inicio"):
            add(href, 15, f'nav "{text}"')

        # Logo anchors (common homepage links)
        classes = " ".join(a.get("class", [])).lower()
        if "logo" in classes or "brand" in classes:
            add(href, 10, "logo link")

    return candidates


# -------------------------------------------------------
# SELECT THE BEST HOMEPAGE CANDIDATE
# -------------------------------------------------------
def pick_best(candidates):
    """
    Pick the strongest homepage candidate based on score and path depth.

    Returns
    -------
    (best_url, reason_string)
    """
    if not candidates:
        return None, "no candidates"

    ranked = sorted(
        [(score, -path_depth(url), url, reasons)
         for url, (score, reasons) in candidates.items()],
        reverse=True
    )

    best = ranked[0]
    return best[2], ", ".join(best[3])


# -------------------------------------------------------
# CLEANING A SINGLE HOMEPAGE
# -------------------------------------------------------
def clean_homepage(url):
    """
    Determine the canonical homepage for a given raw URL.

    Steps:
        1. Normalize URL
        2. Attempt HTTPS → HTTP fallback
        3. Extract candidate homepages from HTML
        4. Score and rank candidates
        5. Add a “needs manual review” flag for ambiguous cases
    """
    url = normalize_url(url)
    if not url:
        return "", "empty or invalid url"

    parsed = urlparse(url)
    base_netloc = parsed.netloc

    # Prefer HTTPS if available
    for scheme in ["https", "http"]:
        root = f"{scheme}://{base_netloc}/"
        r = safe_get(root)
        if r and r.ok:
            break
    else:
        return f"https://{base_netloc}/", "site unreachable (used root)"

    # Parse homepage HTML
    candidates = extract_candidates(r.text, r.url, base_netloc)

    if not candidates:
        return r.url, "no strong candidates; using root landing"

    best, reasons = pick_best(candidates)
    if not best:
        return r.url, "no strong candidates; using root landing"

    # Confidence check
    low_conf = any(term in reasons.lower() for term in ["no strong", "fallback"])
    manual_flag = "needs manual review" if low_conf or len(candidates) <= 1 else ""

    # If canonical/home not detected explicitly, flag for manual check
    if not manual_flag and "home" not in reasons.lower() and "canonical" not in reasons.lower():
        manual_flag = "needs manual review"

    notes = reasons
    if manual_flag:
        notes += f" | {manual_flag}"

    return best, notes


# -------------------------------------------------------
# CSV PROCESSING
# -------------------------------------------------------
def process_csv():
    """
    Clean every URL in the vendor CSV file and write results back.

    Adds two new columns:
        clean_link       → canonical homepage
        link_clean_notes → metadata explaining selection
    """
    with open(INPUT_FILE, newline='', encoding='utf-8') as infile, \
         open(OUTPUT_FILE, 'w', newline='', encoding='utf-8') as outfile:

        reader = csv.DictReader(infile)
        fieldnames = list(reader.fieldnames or [])

        # Add new columns if missing
        if CLEAN_COLUMN not in fieldnames:
            fieldnames.append(CLEAN_COLUMN)
        if NOTES_COLUMN not in fieldnames:
            fieldnames.append(NOTES_COLUMN)

        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()

        # Row-by-row cleaning
        for row in reader:
            raw = (row.get(URL_COLUMN) or "").strip()

            if not raw:
                row[CLEAN_COLUMN] = ""
                row[NOTES_COLUMN] = "empty url"
                writer.writerow(row)
                continue

            try:
                clean, notes = clean_homepage(raw)
            except Exception as e:
                clean, notes = "", f"error: {e}"

            row[CLEAN_COLUMN] = clean
            row[NOTES_COLUMN] = notes
            writer.writerow(row)

            time.sleep(SLEEP_BETWEEN)

    print(f"Finished! Results saved to {OUTPUT_FILE}")


# -------------------------------------------------------
# CLI ENTRY POINT
# -------------------------------------------------------
def main():
    """
    Handle command-line arguments.

    Modes:
        • --url <url> → test cleaning for one URL
        • default     → batch clean entire CSV
    """
    parser = argparse.ArgumentParser(description="Clean and normalize website URLs to their homepage.")
    parser.add_argument("--url", help="Test a single URL (prints cleaned homepage)")
    args = parser.parse_args()

    if args.url:
        clean, notes = clean_homepage(args.url)
        print(f"\nInput URL: {args.url}\nCleaned: {clean}\nNotes: {notes}\n")
    else:
        process_csv()


# Script entry
if __name__ == "__main__":
    main()