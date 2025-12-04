#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Wayback Snapshot Downloader (Decade-Based) + Logging

Features:
- Uses first_snapshot & last_snapshot to compute decades.
- For each decade, downloads MOST RECENT snapshot within:
      <decade_start>-01-01  →  <decade_end>-12-31
- Saves snapshots into: <out_dir>/<domain>/<1990s>/, <2000s>/, <2010s>/ ...
- Adds snapshots_downloaded column.
- Processes only rows with snapshot_count > 0.
- Logs everything to <out_dir>/download_log.txt.
"""

import argparse
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
UA = "SOC4994-Wayback-Downloader/3.1 (email=lakshyajain@vt.edu)"
CDX_BASE = "https://web.archive.org/cdx/search/cdx"
RAW_FMT = "https://web.archive.org/web/{timestamp}id_/{original}"
MAX_SUBPAGES = 300
# ====================


# ---------- UTILITIES ----------

def log(logfile: Path, msg: str):
    timestamp = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {msg}"
    print(line)
    with open(logfile, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def parse_mmddyy(d: str) -> dt.date:
    return dt.datetime.strptime(d.strip(), "%m/%d/%y").date()


def humanize_ts(ts: str) -> str:
    try:
        return dt.datetime.strptime(ts, "%Y%m%d%H%M%S").strftime("%Y-%m-%d %H:%M:%S")
    except:
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
    """Convert URL path to safe filesystem path."""
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

    return "/".join(_safe_part(seg) for seg in rel.split("/"))


# ---------- POLITE REQUESTER ----------

class PoliteRequester:
    def __init__(self, session,
                 base_sleep=2.0,
                 jitter=0.5,
                 max_retries=5,
                 backoff=2.0,
                 timeout=40.0,
                 verbose=False,
                 logfile=None):
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

    def _sleep_backoff(self, attempt):
        dur = (self.backoff ** attempt) + random.uniform(0.0, self.jitter)
        time.sleep(dur)

    def get(self, url, **kwargs):
        if "timeout" not in kwargs:
            kwargs["timeout"] = self.timeout
        attempt = 0

        while True:
            self._sleep_polite()

            try:
                resp = self.s.get(url, **kwargs)
            except Exception as e:
                if attempt >= self.max_retries:
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

def latest_snapshot_for_range(preq, url, start_date, end_date, logfile):
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
        r = preq.get(CDX_BASE, params=params)
        data = r.json()
        if len(data) < 2:
            log(logfile, f"[CDX] No snapshot in {start_date} → {end_date}")
            return None
        return data[1][0], data[1][1]
    except Exception as e:
        log(logfile, f"[CDX] Error: {e}")
        return None


def list_latest_html_subpages(preq, original_home, start_date, end_date, logfile):
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
        "matchType": "host",
    }

    try:
        r = preq.get(CDX_BASE, params=params)
        data = r.json()
    except:
        log(logfile, f"[CDX] Failed subpage list")
        return []

    if len(data) < 2:
        return []

    results = []
    home_norm = normalize_original_for_compare(original_home)

    for row in data[1:]:
        ts, original, mimetype = row
        mimetype = (mimetype or "").lower()

        if not (mimetype.startswith("text/html") or mimetype == "application/xhtml+xml"):
            continue

        if normalize_original_for_compare(original) == home_norm:
            continue

        results.append((ts, original, mimetype))
        if len(results) >= MAX_SUBPAGES:
            break

    return results


# ---------- DOWNLOADING ----------

def safe_ext(ct):
    if not ct:
        return ".bin"
    ct = ct.split(";")[0].lower()
    if ct in ("text/html", "application/xhtml+xml"):
        return ".html"
    if ct.startswith("image/"):
        return "." + ct.split("/", 1)[1]
    if ct == "text/plain":
        return ".txt"
    return ".bin"


def download_file(preq, ts, original, fpath, logfile):
    url = RAW_FMT.format(timestamp=ts, original=original)

    try:
        resp = preq.get(url, stream=True)
        if resp.status_code != 200:
            log(logfile, f"[DOWNLOAD] Failed {url} HTTP {resp.status_code}")
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
        log(logfile, f"[ERROR] Download exception: {e}")
        return None, None


# ---------- PROCESS DECADE ----------

def process_decade(preq, url, decade_start, decade_end, out_dir, logfile):
    decade_name = f"{decade_start}s"
    dom = domain_from_url(url)

    decade_dir = out_dir / dom / decade_name
    decade_dir.mkdir(parents=True, exist_ok=True)

    log(logfile, f"[DECADE] {url} — {decade_name}")

    snap = latest_snapshot_for_range(
        preq,
        url,
        dt.date(decade_start, 1, 1),
        dt.date(decade_end, 12, 31),
        logfile
    )

    if not snap:
        log(logfile, f"[DECADE] No snapshots found in {decade_name}")
        return 0

    ts_home, original_home = snap

    home_path, home_ct = download_file(
        preq, ts_home, original_home, decade_dir / "index.html", logfile
    )

    count = 1 if home_path else 0

    subpages = list_latest_html_subpages(
        preq,
        original_home,
        dt.date(decade_start, 1, 1),
        dt.date(decade_end, 12, 31),
        logfile
    )

    subpage_records = []

    for ts, original, mimetype in subpages:
        rel = path_from_original(original)
        sp_path, sp_ct = download_file(
            preq, ts, original, decade_dir / rel, logfile
        )
        if sp_path:
            count += 1
            subpage_records.append({
                "original": original,
                "timestamp": ts,
                "timestamp_human": humanize_ts(ts),
                "saved_path": str(sp_path.relative_to(decade_dir)),
                "content_type": sp_ct or mimetype
            })

    manifest = {
        "decade": decade_name,
        "range": {
            "start_date": f"{decade_start}-01-01",
            "end_date": f"{decade_end}-12-31",
        },
        "homepage": {
            "original": original_home,
            "timestamp": ts_home,
            "timestamp_human": humanize_ts(ts_home),
            "saved_path": (str(home_path.relative_to(decade_dir)) if home_path else None),
            "content_type": home_ct or ""
        },
        "subpages": subpage_records
    }

    (decade_dir / "_captures.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    log(logfile, f"[DECADE] Completed {decade_name}: {count} files")
    return count


# ---------- BATCH CSV MODE ----------

def batch_process_csv(input_csv, output_csv, out_dir, preq, logfile):
    df = pd.read_csv(input_csv, dtype=str)

    if "snapshots_downloaded" not in df.columns:
        df["snapshots_downloaded"] = ""

    for idx, row in df.iterrows():

        # ---------- SAFE CLEAN_LINK EXTRACTION ----------
        raw_url = row.get("clean_link")

        if pd.isna(raw_url):
            url = ""
        else:
            url = str(raw_url).strip()

        if not url:
            log(logfile, f"[SKIP] Row {idx}: empty or NaN clean_link")
            df.at[idx, "snapshots_downloaded"] = 0
            continue
        # --------------------------------------------------

        # snapshot_count filter
        snapshot_count = row.get("snapshot_count")
        try:
            sc_int = int(snapshot_count)
        except:
            sc_int = 0

        if sc_int <= 0:
            log(logfile, f"[SKIP] {url} (snapshot_count=0)")
            df.at[idx, "snapshots_downloaded"] = 0
            continue

        # date parsing
        try:
            first = parse_mmddyy(row["first_snapshot"])
            last = parse_mmddyy(row["last_snapshot"])
        except:
            log(logfile, f"[ERROR] Invalid dates for {url}")
            df.at[idx, "snapshots_downloaded"] = 0
            continue

        log(logfile, f"[PROCESS] {url} ({first.year} → {last.year})")

        total_downloaded = 0

        # Compute decades
        start_decade = (first.year // 10) * 10
        end_decade = (last.year // 10) * 10

        for decade_start in range(start_decade, end_decade + 1, 10):
            decade_end = decade_start + 9

            total_downloaded += process_decade(
                preq, url, decade_start, decade_end, out_dir, logfile
            )

        df.at[idx, "snapshots_downloaded"] = total_downloaded
        log(logfile, f"[DONE] {url}: {total_downloaded} snapshots downloaded\n")

    df.to_csv(output_csv, index=False)
    log(logfile, f"[WRITE] CSV saved to {output_csv}")


# ---------- MAIN ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
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

    batch_process_csv(
        args.input,
        args.output,
        args.out_dir,
        preq,
        logfile
    )


if __name__ == "__main__":
    main()