#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
snapshot_downloader_yearly.py
-----------------------------
Download archived website snapshots (homepage + HTML subpages) from the
Internet Archive **year-by-year** for each facial recognition vendor website.

This script is part of the SOC 4994 research pipeline and sits between:

    (1) website_cleaner.py     → produces clean homepage URLs
    (2) snapshot_info.py        → computes snapshot_count, first_snapshot, last_snapshot
    (3) THIS SCRIPT             → downloads HTML snapshots year-by-year
    (4) text_extractor.py       → extracts cleaned English text for NLP
    (5) bertopic_pipeline.py    → topic modeling

Purpose
-------
For each vendor:
    • Identify first → last snapshot year
    • For each year, find latest available snapshot
    • Download:
        - homepage HTML
        - up to MAX_SUBPAGES internal HTML pages
    • Save results in:
          <out_dir>/<domain>/<YYYY>/
    • Generate `_captures.json` for reproducibility
    • Write number of downloaded files back to CSV

Features
--------
- Resilient to errors, retries, rate-limits, and slow responses  
- Writes all operations to `<out_dir>/download_log.txt`  
- Safe filesystem paths for all pages  
- Strictly avoids over-requesting by using polite delays + backoff  
- Ready for large-scale archival collection (~1100 vendors)

Usage
-----
    python3 snapshot_downloader_yearly.py \
        --input fr_vendors_snapshot_info.csv \
        --output fr_vendors_snapshot_yearly.csv \
        --out-dir data/wayback/yearly/

"""

import argparse
import csv
import datetime as dt
import hashlib
import json
import os
import random
import re
import sys
import time
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any
from urllib.parse import urlparse, unquote

import pandas as pd
import requests


# -------------------------------------------------------
# CONFIGURATION
# -------------------------------------------------------
UA = "SOC4994-Wayback-Downloader/2.0 (email=lakshyajain@vt.edu)"
CDX_BASE = "https://web.archive.org/cdx/search/cdx"
RAW_FMT = "https://web.archive.org/web/{timestamp}id_/{original}"
MAX_SUBPAGES = 300  # maximum allowable HTML subpages per year


# -------------------------------------------------------
# LOGGING UTILITIES
# -------------------------------------------------------
def log(logfile: Path, msg: str):
    """
    Write a log message both to console and to <logfile>.

    Example:
        [2025-01-12 13:20:11] [YEAR] Processing https://abc.com for 2014
    """
    timestamp = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {msg}"
    print(line)
    with open(logfile, "a", encoding="utf-8") as f:
        f.write(line + "\n")


# -------------------------------------------------------
# BASIC HELPERS
# -------------------------------------------------------
def parse_mmddyy(d: str) -> dt.date:
    """Convert MM/DD/YY → datetime.date"""
    return dt.datetime.strptime(d.strip(), "%m/%d/%y").date()


def humanize_ts(ts: str) -> str:
    """Convert Wayback timestamp → readable string; fallback on raw."""
    try:
        return dt.datetime.strptime(ts, "%Y%m%d%H%M%S").strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return ts


def domain_from_url(url: str) -> str:
    """Extract domain from URL: https://abc.com → abc.com"""
    m = re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://([^/]+)/?", url)
    return m.group(1) if m else "unknown-domain"


def normalize_original_for_compare(url: str) -> str:
    """Normalize URL for consistent duplicate checking."""
    return url.rstrip("/")


def _safe_part(s: str) -> str:
    """
    Clean any string into a filesystem-safe path segment:
    replaces special characters with underscores.
    """
    s = unquote(s)
    s = re.sub(r"[^\w\-.]+", "_", s)
    return s.strip("_") or "_"


def path_from_original(original: str) -> str:
    """
    Convert URL path into a safe filesystem path.

    Examples:
        "/"                     → index.html
        "/products/"           → products/index.html
        "/about?id=12"         → about__q13dfaf9e.html
    """
    u = urlparse(original)
    p, q = u.path or "/", u.query

    # Choose fallback filename
    if p.endswith("/"):
        rel = p[1:] + "index"
    else:
        rel = p[1:] or "index"

    if not os.path.splitext(rel)[1]:
        rel += ".html"

    # Append query hash if needed
    if q:
        h = hashlib.sha1(q.encode("utf-8")).hexdigest()[:8]
        base, ext = os.path.splitext(rel)
        rel = f"{base}__q{h}{ext}"

    return "/".join(_safe_part(seg) for seg in rel.split("/"))


# -------------------------------------------------------
# POLITE REQUEST WRAPPER
# -------------------------------------------------------
class PoliteRequester:
    """
    A wrapper around requests.Session that applies:
        • polite delays
        • retry logic
        • exponential backoff
        • rate-limit handling
    """

    def __init__(self, session: requests.Session,
                 base_sleep=2.0, jitter=0.5,
                 max_retries=5, backoff=2.0,
                 timeout=40.0, verbose=False,
                 logfile: Optional[Path] = None):

        self.s = session
        self.base_sleep = base_sleep
        self.jitter = jitter
        self.max_retries = max_retries
        self.backoff = backoff
        self.timeout = timeout
        self.verbose = verbose
        self.logfile = logfile

    def _sleep_polite(self):
        """Sleep 2 ± jitter seconds before any request."""
        delta = random.uniform(-self.jitter, self.jitter)
        time.sleep(max(0, self.base_sleep + delta))

    def _sleep_backoff(self, attempt: int):
        """Exponential sleep on retry."""
        dur = (self.backoff ** attempt) + random.uniform(0, self.jitter)
        time.sleep(dur)

    def get(self, url: str, **kwargs) -> requests.Response:
        """GET request with retry + backoff handling."""
        kwargs.setdefault("timeout", self.timeout)
        attempt = 0

        while True:
            self._sleep_polite()
            try:
                resp = self.s.get(url, **kwargs)
            except Exception as e:
                if attempt >= self.max_retries:
                    if self.logfile:
                        log(self.logfile, f"[ERROR] Network error: {e}")
                    raise
                attempt += 1
                self._sleep_backoff(attempt)
                continue

            # Handle rate-limiting or server issues
            if resp.status_code in (429, 503) or (500 <= resp.status_code < 600):
                if attempt >= self.max_retries:
                    return resp
                attempt += 1
                self._sleep_backoff(attempt)
                continue

            return resp


# -------------------------------------------------------
# WAYBACK SNAPSHOT QUERIES
# -------------------------------------------------------
def latest_snapshot_for_range(preq, url, start_date, end_date, logfile):
    """
    Return latest snapshot in specified date range.

    Returns:
        (timestamp, original_url) OR None
    """
    params = {
        "url": url,
        "from": start_date.strftime("%Y%m%d"),
        "to": end_date.strftime("%Y%m%d"),
        "output": "json",
        "filter": "statuscode:200",
        "fl": "timestamp,original,mimetype",
        "limit": "1",
        "sort": "reverse",
    }

    try:
        r = preq.get(CDX_BASE, params=params, allow_redirects=True)
        data = r.json()

        if len(data) < 2:
            log(logfile, f"[CDX] No snapshot in range {start_date}..{end_date}")
            return None

        ts, original = data[1][0], data[1][1]
        return ts, original

    except Exception as e:
        log(logfile, f"[CDX] Error: {e}")
        return None


def list_latest_html_subpages(preq, original_home, start_date, end_date, logfile):
    """
    Return latest HTML subpages for the host within the date range.
    """
    host = domain_from_url(original_home)

    params = {
        "url": f"{host}/*",
        "from": start_date.strftime("%Y%m%d"),
        "to": end_date.strftime("%Y%m%d"),
        "output": "json",
        "fl": "timestamp,original,mimetype",
        "collapse": "original",
        "sort": "reverse",
        "filter": "statuscode:200",
        "limit": str(MAX_SUBPAGES + 50),
        "gzip": "false",
        "matchType": "host",
    }

    try:
        r = preq.get(CDX_BASE, params=params)
        data = r.json()
    except Exception as e:
        log(logfile, f"[CDX] subpage error: {e}")
        return []

    if len(data) < 2:
        return []

    results = []
    home_norm = normalize_original_for_compare(original_home)

    for row in data[1:]:
        ts, original, mimetype = row[0], row[1], (row[2] or "").lower()

        # Keep only HTML pages
        if not (mimetype.startswith("text/html") or mimetype == "application/xhtml+xml"):
            continue

        # Avoid homepage duplication
        if normalize_original_for_compare(original) == home_norm:
            continue

        results.append((ts, original, mimetype))

        if len(results) >= MAX_SUBPAGES:
            break

    return results


# -------------------------------------------------------
# DOWNLOADING SNAPSHOTS
# -------------------------------------------------------
def safe_ext(ct: Optional[str]) -> str:
    """Choose robust extension from Content-Type."""
    if not ct:
        return ".bin"
    ct = ct.split(";")[0].strip().lower()

    if ct in ("text/html", "application/xhtml+xml"):
        return ".html"
    if ct.startswith("image/"):
        return "." + ct.split("/", 1)[1]
    if ct == "text/plain":
        return ".txt"
    return ".bin"


def download_file(preq, ts, original, fpath, logfile):
    """
    Download snapshot and write to disk.

    Returns:
        (final_file_path, content_type)
    """
    url = RAW_FMT.format(timestamp=ts, original=original)

    try:
        resp = preq.get(url, stream=True)

        if resp.status_code != 200:
            log(logfile, f"[GET] Failed: {url} HTTP {resp.status_code}")
            return None, None

        ct = resp.headers.get("Content-Type")
        ext = safe_ext(ct)

        final_path = fpath.with_suffix(ext)
        final_path.parent.mkdir(parents=True, exist_ok=True)

        with open(final_path, "wb") as f:
            for chunk in resp.iter_content(65536):
                if chunk:
                    f.write(chunk)

        log(logfile, f"[SAVE] {final_path}")
        return final_path, (ct.split(";")[0] if ct else None)

    except Exception as e:
        log(logfile, f"[ERROR] Download error: {e}")
        return None, None


# -------------------------------------------------------
# PROCESS A SINGLE YEAR
# -------------------------------------------------------
def process_year(preq, url, year, start_date, end_date, base_out_dir, logfile):
    """
    Download homepage + HTML subpages for a single calendar year.

    Folder structure:
        <out_dir>/<domain>/<YYYY>/
            index.html
            products/index.html
            team/index.html
            _captures.json
    """
    dom = domain_from_url(url)
    year_dir = base_out_dir / dom / str(year)
    year_dir.mkdir(parents=True, exist_ok=True)

    log(logfile, f"[YEAR] Processing {url} for {year}")

    # --- Homepage snapshot ---
    snap = latest_snapshot_for_range(preq, url, start_date, end_date, logfile)
    if not snap:
        log(logfile, f"[YEAR] No snapshot for {year}")
        return 0

    ts_home, original_home = snap
    home_path, home_ct = download_file(preq, ts_home, original_home, year_dir / "index.html", logfile)
    count = 1 if home_path else 0

    # --- Subpages ---
    subpages = list_latest_html_subpages(preq, original_home, start_date, end_date, logfile)
    subpage_records = []

    for ts, original, mimetype in subpages:
        rel = path_from_original(original)

        sp_path, sp_ct = download_file(preq, ts, original, year_dir / rel, logfile)

        if sp_path:
            count += 1
            subpage_records.append({
                "original": original,
                "timestamp": ts,
                "timestamp_human": humanize_ts(ts),
                "saved_path": str(sp_path.relative_to(year_dir)),
                "content_type": sp_ct or mimetype
            })

    # Manifest for reproducibility
    manifest = {
        "year": year,
        "range": {
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
        },
        "homepage": {
            "original": original_home,
            "timestamp": ts_home,
            "timestamp_human": humanize_ts(ts_home),
            "saved_path": str(home_path.relative_to(year_dir)) if home_path else None,
            "content_type": home_ct or "",
        },
        "subpages": subpage_records,
    }

    (year_dir / "_captures.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    log(logfile, f"[YEAR] Completed {year}: {count} files")

    return count


# -------------------------------------------------------
# BATCH PROCESSING (CSV)
# -------------------------------------------------------
def batch_process_csv(input_csv, output_csv, out_dir, preq, logfile):
    """
    Process an entire CSV of vendors:
        • iterate over rows
        • download snapshots year-by-year
        • write snapshot counts back into CSV
    """
    df = pd.read_csv(input_csv, dtype=str)

    if "snapshots_downloaded" not in df.columns:
        df["snapshots_downloaded"] = ""

    for idx, row in df.iterrows():

        url = (row.get("clean_link") or "").strip()
        if not url:
            continue

        # Skip rows with no historical snapshots
        try:
            sc_int = int(row.get("snapshot_count"))
        except:
            sc_int = 0

        if sc_int <= 0:
            log(logfile, f"[SKIP] {url} (snapshot_count=0)")
            df.at[idx, "snapshots_downloaded"] = 0
            continue

        # Parse snapshot date range
        try:
            first = parse_mmddyy(row["first_snapshot"])
            last = parse_mmddyy(row["last_snapshot"])
        except:
            log(logfile, f"[ERROR] Invalid dates for {url}")
            df.at[idx, "snapshots_downloaded"] = 0
            continue

        log(logfile, f"[PROCESS] {url} ({first} → {last})")

        total_downloaded = 0

        # Loop year-by-year
        for year in range(first.year, last.year + 1):
            y_start = dt.date(year, 1, 1)
            y_end = dt.date(year, 12, 31)

            # Clip range to actual first/last snapshot dates
            if year == first.year:
                y_start = first
            if year == last.year:
                y_end = last

            total_downloaded += process_year(
                preq, url, year, y_start, y_end, out_dir, logfile
            )

        df.at[idx, "snapshots_downloaded"] = total_downloaded

        log(logfile, f"[DONE] {url} → {total_downloaded} snapshots downloaded\n")

    # Write updated dataset
    df.to_csv(output_csv, index=False)
    log(logfile, f"[WRITE] Updated CSV written to {output_csv}")


# -------------------------------------------------------
# MAIN ENTRYPOINT
# -------------------------------------------------------
def main():
    """
    Command-line entry for year-by-year snapshot downloading.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True, help="Input CSV with snapshot info")
    ap.add_argument("--output", type=Path, required=True, help="Output CSV with counts")
    ap.add_argument("--out-dir", type=Path, required=True, help="Directory to save snapshots")
    ap.add_argument("--verbose", action="store_true")

    args = ap.parse_args()

    # Ensure snapshot directory exists
    args.out_dir.mkdir(parents=True, exist_ok=True)

    logfile = args.out_dir / "download_log.txt"

    # Prepare session + polite requester
    session = requests.Session()
    session.headers.update({"User-Agent": UA})

    preq = PoliteRequester(session=session, verbose=args.verbose, logfile=logfile)

    # Process vendors
    batch_process_csv(args.input, args.output, args.out_dir, preq, logfile)


if __name__ == "__main__":
    main()