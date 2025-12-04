#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Wayback Snapshot Downloader (Year-by-Year) + Logging + CSV Backup

Changes from previous version:
- Uses first_snapshot & last_snapshot from CSV to generate year-by-year ranges.
- Creates a backup copy of the entire CSV inside out_dir before modification.
- Adds snapshots_downloaded column to track actual downloaded files.
- Only processes rows where snapshot_count > 0.
- Saves each year's snapshots into: <out_dir>/<domain>/<YYYY>/
- Generates a log file: <out_dir>/download_log.txt
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


# ===== Tunables =====
UA = "SOC4994-Wayback-Downloader/2.0 (email=lakshyajain@vt.edu)"
CDX_BASE = "https://web.archive.org/cdx/search/cdx"
RAW_FMT = "https://web.archive.org/web/{timestamp}id_/{original}"
MAX_SUBPAGES = 300
# ====================


# ---------- UTILITIES ----------

def log(logfile: Path, msg: str):
    """Write logs to file and print to console."""
    timestamp = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {msg}"
    print(line)
    with open(logfile, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def parse_mmddyy(d: str) -> dt.date:
    """Parse snapshot dates in MM/DD/YY format."""
    return dt.datetime.strptime(d.strip(), "%m/%d/%y").date()


def humanize_ts(ts: str) -> str:
    try:
        return dt.datetime.strptime(ts, "%Y%m%d%H%M%S").strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return ts


def domain_from_url(url: str) -> str:
    m = re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://([^/]+)/?", url)
    return m.group(1) if m else "unknown-domain"


def normalize_original_for_compare(url: str) -> str:
    return url.rstrip("/")


def _safe_part(s: str) -> str:
    s = unquote(s)
    s = re.sub(r"[^\w\-.]+", "_", s)
    s = s.strip("_")
    return s or "_"


def path_from_original(original: str) -> str:
    """Map original URL path to filesystem path."""
    u = urlparse(original)
    p = u.path or "/"
    q = u.query

    if p.endswith("/"):
        rel = p[1:] + "index"
    else:
        rel = p[1:] or "index"

    if not os.path.splitext(rel)[1]:
        rel += ".html"

    if q:
        h = hashlib.sha1(q.encode("utf-8")).hexdigest()[:8]
        base, ext = os.path.splitext(rel)
        rel = f"{base}__q{h}{ext}"

    parts = [_safe_part(seg) for seg in rel.split("/")]
    return "/".join(parts)


# ---------- POLITE REQUESTER ----------

class PoliteRequester:
    def __init__(self, session: requests.Session,
                 base_sleep=2.0,
                 jitter=0.5,
                 max_retries=5,
                 backoff=2.0,
                 timeout=40.0,
                 verbose=False,
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
        delta = random.uniform(-self.jitter, self.jitter)
        time.sleep(max(0.0, self.base_sleep + delta))

    def _sleep_backoff(self, attempt: int):
        dur = (self.backoff ** attempt) + random.uniform(0.0, self.jitter)
        time.sleep(dur)

    def get(self, url: str, **kwargs) -> requests.Response:
        if "timeout" not in kwargs:
            kwargs["timeout"] = self.timeout
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

            if resp.status_code in (429, 503) or (500 <= resp.status_code < 600):
                if attempt >= self.max_retries:
                    return resp
                attempt += 1
                self._sleep_backoff(attempt)
                continue

            return resp


# ---------- WAYBACK QUERIES ----------

def latest_snapshot_for_range(preq, url, start_date, end_date, logfile) -> Optional[Tuple[str, str]]:
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


def list_latest_html_subpages(preq, original_home, start_date, end_date, logfile) -> List[Tuple[str, str, str]]:
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
        if not (mimetype.startswith("text/html") or mimetype == "application/xhtml+xml"):
            continue
        if normalize_original_for_compare(original) == home_norm:
            continue
        results.append((ts, original, mimetype))
        if len(results) >= MAX_SUBPAGES:
            break

    return results


# ---------- DOWNLOADING ----------

def safe_ext(ct: Optional[str]) -> str:
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


def download_file(preq, ts, original, fpath: Path, logfile) -> Tuple[Optional[Path], Optional[str]]:
    url = RAW_FMT.format(timestamp=ts, original=original)
    try:
        resp = preq.get(url, stream=True)
        if resp.status_code != 200:
            log(logfile, f"[GET] Failed: {url} HTTP {resp.status_code}")
            return None, None

        ct = resp.headers.get("Content-Type")
        ext = safe_ext(ct)
        base, _ = os.path.splitext(fpath.name)
        final_path = fpath.with_suffix(ext)

        final_path.parent.mkdir(parents=True, exist_ok=True)

        with open(final_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=65536):
                if chunk:
                    f.write(chunk)

        log(logfile, f"[SAVE] {final_path}")
        return final_path, (ct.split(";")[0] if ct else None)

    except Exception as e:
        log(logfile, f"[ERROR] Download error: {e}")
        return None, None


# ---------- PROCESS ONE YEAR ----------

def process_year(preq, url, year, start_date, end_date, base_out_dir, logfile):
    dom = domain_from_url(url)
    year_dir = base_out_dir / dom / str(year)
    year_dir.mkdir(parents=True, exist_ok=True)

    log(logfile, f"[YEAR] Processing {url} for {year}")

    snap = latest_snapshot_for_range(preq, url, start_date, end_date, logfile)
    if not snap:
        log(logfile, f"[YEAR] No snapshot for {year}")
        return 0

    ts_home, original_home = snap
    home_path, home_ct = download_file(preq, ts_home, original_home, year_dir / "index.html", logfile)

    count = 1 if home_path else 0

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

    manifest = {
        "year": year,
        "range": {
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat()
        },
        "homepage": {
            "original": original_home,
            "timestamp": ts_home,
            "timestamp_human": humanize_ts(ts_home),
            "saved_path": (str(home_path.relative_to(year_dir)) if home_path else None),
            "content_type": home_ct or ""
        },
        "subpages": subpage_records
    }

    (year_dir / "_captures.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    log(logfile, f"[YEAR] Completed {year}: {count} files")
    return count


# ---------- BATCH CSV MODE ----------

def batch_process_csv(input_csv, output_csv, out_dir, preq, logfile):
    df = pd.read_csv(input_csv, dtype=str)

    # ensure column exists
    if "snapshots_downloaded" not in df.columns:
        df["snapshots_downloaded"] = ""

    for idx, row in df.iterrows():
        url = (row.get("clean_link") or "").strip()
        if not url:
            continue

        snapshot_count = row.get("snapshot_count")
        try:
            sc_int = int(snapshot_count)
        except:
            sc_int = 0

        if sc_int <= 0:
            log(logfile, f"[SKIP] {url} (snapshot_count=0)")
            df.at[idx, "snapshots_downloaded"] = 0
            continue

        try:
            first = parse_mmddyy(row["first_snapshot"])
            last = parse_mmddyy(row["last_snapshot"])
        except:
            log(logfile, f"[ERROR] Invalid dates for {url}")
            df.at[idx, "snapshots_downloaded"] = 0
            continue

        log(logfile, f"[PROCESS] {url} ({first} → {last})")

        total_downloaded = 0

        for year in range(first.year, last.year + 1):
            y_start = dt.date(year, 1, 1)
            y_end = dt.date(year, 12, 31)

            if year == first.year:
                y_start = first
            if year == last.year:
                y_end = last

            total_downloaded += process_year(
                preq, url, year, y_start, y_end, out_dir, logfile
            )

        df.at[idx, "snapshots_downloaded"] = total_downloaded
        log(logfile, f"[DONE] {url} → {total_downloaded} snapshots downloaded\n")

    df.to_csv(output_csv, index=False)
    log(logfile, f"[WRITE] Updated CSV written to {output_csv}")


# ---------- MAIN ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, help="Input CSV dataset", required=True)
    ap.add_argument("--output", type=Path, help="Output CSV with results", required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--verbose", action="store_true")

    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    logfile = args.out_dir / "download_log.txt"

    session = requests.Session()
    session.headers.update({"User-Agent": UA})

    preq = PoliteRequester(
        session=session,
        verbose=args.verbose,
        logfile=logfile
    )

    batch_process_csv(args.input, args.output, args.out_dir, preq, logfile)


if __name__ == "__main__":
    main()