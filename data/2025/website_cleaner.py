#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Website Homepage Cleaner
------------------------
Usage examples:
  • Batch mode:
      python3 website_cleaner.py
  • Single URL test:
      python3 website_cleaner.py --url "http://www.advanced-biometrics.com/solutions/"
"""

import csv
import re
import time
import argparse
from urllib.parse import urlparse, urlunparse, urljoin
import requests
from bs4 import BeautifulSoup

# ---------------- SETTINGS ----------------
INPUT_FILE = "facial_recognition_vendors.csv"
OUTPUT_FILE = "facial_recognition_vendors.csv"
URL_COLUMN = "link"
CLEAN_COLUMN = "clean_link"
NOTES_COLUMN = "link_clean_notes"
TIMEOUT = 10
SLEEP_BETWEEN = 0.2
HEADERS = {"User-Agent": "WebsiteCleaner/1.0"}
# ------------------------------------------

session = requests.Session()
session.headers.update(HEADERS)

def safe_get(url):
    try:
        return session.get(url, timeout=TIMEOUT, allow_redirects=True)
    except requests.RequestException:
        return None

def normalize_url(url):
    if not url:
        return None
    if not re.match(r'^https?://', url):
        url = "http://" + url
    parsed = urlparse(url)
    netloc = parsed.netloc.lower()
    netloc = re.sub(r':(80|443)$', '', netloc)
    return urlunparse((parsed.scheme.lower(), netloc, parsed.path or "/", parsed.params, parsed.query, parsed.fragment))

def path_depth(url):
    return sum(1 for p in urlparse(url).path.split("/") if p)

def extract_candidates(html, base_url, base_netloc):
    soup = BeautifulSoup(html, "html.parser")
    candidates = {}
    def add(u, score, reason):
        if not u:
            return
        p = urlparse(u)
        if not p.scheme.startswith("http"):
            return
        if p.netloc and p.netloc.lower() != base_netloc:
            return
        if not p.path or not p.path.endswith("/"):
            if not re.search(r'\.[a-zA-Z0-9]{1,6}$', p.path or ""):
                u = urlunparse((p.scheme, p.netloc, (p.path or "/") + "/", p.params, p.query, p.fragment))
        candidates[u] = candidates.get(u, [0, []])
        candidates[u][0] += score
        candidates[u][1].append(reason)

    # <link rel="home">, <link rel="canonical">, og:url
    for tag in soup.find_all("link", rel=True):
        rels = [r.lower() for r in (tag.get("rel") if isinstance(tag.get("rel"), list) else [tag.get("rel")])]
        if "home" in rels:
            add(urljoin(base_url, tag.get("href")), 30, "rel=home")
        if "canonical" in rels:
            add(urljoin(base_url, tag.get("href")), 20, "canonical")
    og = soup.find("meta", attrs={"property": "og:url"})
    if og and og.get("content"):
        add(urljoin(base_url, og["content"]), 15, "og:url")

    # Navbar or logo links
    for a in soup.find_all("a", href=True):
        text = (a.get_text(strip=True) or "").lower()
        href = urljoin(base_url, a["href"])
        if text in ("home", "homepage", "start", "accueil", "inicio"):
            add(href, 15, f'nav "{text}"')
        classes = " ".join(a.get("class", [])).lower()
        if "logo" in classes or "brand" in classes:
            add(href, 10, "logo link")

    return candidates

def pick_best(candidates):
    if not candidates:
        return None, "no candidates"
    ranked = sorted(
        [(score, -path_depth(url), url, reasons) for url, (score, reasons) in candidates.items()],
        reverse=True
    )
    best = ranked[0]
    return best[2], ", ".join(best[3])

def clean_homepage(url):
    url = normalize_url(url)
    if not url:
        return "", "empty or invalid url"

    parsed = urlparse(url)
    base_netloc = parsed.netloc

    # Try HTTPS first, fallback to HTTP
    for scheme in ["https", "http"]:
        root = f"{scheme}://{base_netloc}/"
        r = safe_get(root)
        if r and r.ok:
            break
    else:
        return f"https://{base_netloc}/", "site unreachable (used root)"

    # Parse homepage
    candidates = extract_candidates(r.text, r.url, base_netloc)
    if not candidates:
        return r.url, "no strong candidates; using root landing"

    best, reasons = pick_best(candidates)
    if not best:
        return r.url, "no strong candidates; using root landing"

    # Add manual review if low confidence
    low_conf = any(term in reasons.lower() for term in ["no strong", "fallback"])
    manual_flag = "needs manual review" if low_conf or len(candidates) <= 1 else ""

    if not manual_flag and "home" not in reasons.lower() and "canonical" not in reasons.lower():
        manual_flag = "needs manual review"

    notes = reasons
    if manual_flag:
        notes += f" | {manual_flag}"

    return best, notes

def process_csv():
    with open(INPUT_FILE, newline='', encoding='utf-8') as infile, \
         open(OUTPUT_FILE, 'w', newline='', encoding='utf-8') as outfile:
        reader = csv.DictReader(infile)
        fieldnames = list(reader.fieldnames or [])
        if CLEAN_COLUMN not in fieldnames:
            fieldnames.append(CLEAN_COLUMN)
        if NOTES_COLUMN not in fieldnames:
            fieldnames.append(NOTES_COLUMN)
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()

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
    print(f"✅ Finished! Results saved to {OUTPUT_FILE}")

def main():
    parser = argparse.ArgumentParser(description="Clean and normalize website URLs to their homepage.")
    parser.add_argument("--url", help="Test a single URL (prints cleaned homepage)")
    args = parser.parse_args()

    if args.url:
        clean, notes = clean_homepage(args.url)
        print(f"\nInput URL: {args.url}\nCleaned: {clean}\nNotes: {notes}\n")
    else:
        process_csv()

if __name__ == "__main__":
    main()