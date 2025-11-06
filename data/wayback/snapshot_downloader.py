#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Wayback Snapshot Downloader + Same-Day Subpage Crawl.

Behavior:
- Finds ONLY the single MOST-RECENT snapshot in [start_date, end_date] (status 200).
- Downloads that homepage snapshot.
- Then crawls same-day subpages (same host, same YYYYMMDD) via CDX (HTML only),
  picking the latest capture per original on that day, de-duplicated.
- Saves to:
    <out_dir>/<domain>/<YYYY-MM-DD>/<HHMMSS>.<ext>                 (homepage)
    <out_dir>/<domain>/<YYYY-MM-DD>/<path>/.../file.<ext>          (subpages)

Usage (no extra flags needed):
  python3 snapshot_downloader.py \
      --url "https://www.advanced-biometrics.com/ab/" \
      --start-date 2025-03-10 --end-date 2025-04-20 \
      --out-dir ./snapshots --verbose
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
from typing import Optional, Tuple, List
from urllib.parse import urlparse, unquote

try:
    import pandas as pd
except Exception as e:
    print("This script requires pandas. Try: pip install pandas", file=sys.stderr)
    raise

try:
    import requests
except Exception as e:
    print("This script requires requests. Try: pip install requests", file=sys.stderr)
    raise


# ===== Tunables (no CLI flags needed) =====
UA = "SOC4994-Wayback-Downloader/1.3 (+https://example.edu; email=student@vt.edu)"
CDX_BASE = "https://web.archive.org/cdx/search/cdx"
RAW_FMT = "https://web.archive.org/web/{timestamp}id_/{original}"   # raw passthrough
HUMAN_FMT = "https://web.archive.org/web/{timestamp}/{original}"    # rewritten HTML (not used for downloads)
SUBPAGE_MIME_FILTERS = ("mimetype:text/html", "mimetype:application/xhtml+xml")
MAX_SUBPAGES = 300  # change here if you want more/less pages
# =========================================


def parse_date(d: str) -> dt.date:
    return dt.datetime.strptime(d, "%Y-%m-%d").date()


def is_empty_notes(val) -> bool:
    if val is None:
        return True
    if isinstance(val, float):
        return False if pd.notna(val) else True
    s = str(val).strip()
    return len(s) == 0


def domain_from_url(url: str) -> str:
    m = re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://([^/]+)/?", url)
    return m.group(1) if m else "unknown-domain"


def normalize_original_for_compare(url: str) -> str:
    # remove trailing slash for equality checks
    return url.rstrip("/")


def _safe_part(s: str) -> str:
    # filesystem-safe piece
    s = unquote(s)
    s = re.sub(r"[^\w\-.]+", "_", s)
    s = s.strip("_")
    return s or "_"


def path_from_original(original: str) -> str:
    """
    Build a hierarchical relative path for saving a subpage HTML:
    - /a/b/c.html -> a/b/c.html
    - /a/b/       -> a/b/index.html
    - ?q=...      -> index.<hash>.html
    - includes short hash suffix for query strings to avoid collisions
    """
    u = urlparse(original)
    p = u.path or "/"
    q = u.query

    # ensure we have a filename
    if p.endswith("/"):
        rel = p[1:] + "index"  # drop leading slash
    else:
        rel = p[1:]  # drop leading slash; may be 'a/b/file.ext' or 'a'
        if rel == "":
            rel = "index"

    # ensure extension
    if not os.path.splitext(rel)[1]:
        # no extension -> treat as html
        rel = rel + ".html"

    # add query hash suffix if needed
    if q:
        h = hashlib.sha1(q.encode("utf-8")).hexdigest()[:8]
        base, ext = os.path.splitext(rel)
        rel = f"{base}__q{h}{ext}"

    # sanitize each segment
    parts = [_safe_part(seg) for seg in rel.split("/")]
    rel_safe = "/".join(parts)
    return rel_safe


class PoliteRequester:
    def __init__(self, session: requests.Session,
                 base_sleep: float = 2.0,
                 jitter: float = 0.5,
                 max_retries: int = 5,
                 backoff: float = 2.0,
                 timeout: float = 40.0,
                 verbose: bool = False):
        self.s = session
        self.base_sleep = max(0.0, base_sleep)
        self.jitter = max(0.0, jitter)
        self.max_retries = max_retries
        self.backoff = max(1.0, backoff)
        self.timeout = timeout
        self.verbose = verbose

    def _sleep_polite(self):
        delta = random.uniform(-self.jitter, self.jitter)
        dur = max(0.0, self.base_sleep + delta)
        time.sleep(dur)

    def _sleep_backoff(self, attempt: int, retry_after: Optional[float] = None):
        if retry_after is not None:
            if self.verbose:
                print(f"[polite] Honor Retry-After: sleeping {retry_after:.2f}s")
            time.sleep(max(0.0, retry_after))
            return
        dur = (self.backoff ** attempt) + random.uniform(0.0, self.jitter)
        if self.verbose:
            print(f"[polite] Backoff attempt {attempt}: sleeping {dur:.2f}s")
        time.sleep(dur)

    def get(self, url: str, **kwargs) -> requests.Response:
        if "timeout" not in kwargs:
            kwargs["timeout"] = self.timeout
        attempt = 0
        while True:
            self._sleep_polite()
            try:
                resp = self.s.get(url, **kwargs)
            except requests.RequestException as e:
                if attempt >= self.max_retries:
                    if self.verbose:
                        print(f"[polite] Network error, giving up: {e}")
                    raise
                if self.verbose:
                    print(f"[polite] Network error: {e} (attempt {attempt+1}/{self.max_retries})")
                attempt += 1
                self._sleep_backoff(attempt)
                continue

            if resp.status_code in (429, 503) or (500 <= resp.status_code < 600):
                if attempt >= self.max_retries:
                    if self.verbose:
                        print(f"[polite] HTTP {resp.status_code}, giving up.")
                    return resp
                retry_after_hdr = resp.headers.get("Retry-After")
                retry_after = None
                if retry_after_hdr:
                    try:
                        retry_after = float(retry_after_hdr)
                    except ValueError:
                        retry_after = None
                if self.verbose:
                    print(f"[polite] HTTP {resp.status_code} (attempt {attempt+1}/{self.max_retries})")
                attempt += 1
                self._sleep_backoff(attempt, retry_after=retry_after)
                continue

            return resp


def latest_snapshot_in_range(preq: PoliteRequester, url: str, start_date: dt.date, end_date: dt.date,
                             verbose: bool = False) -> Optional[Tuple[str, str]]:
    params = {
        "url": url,
        "from": start_date.strftime("%Y%m%d"),
        "to": end_date.strftime("%Y%m%d"),
        "output": "json",
        "filter": "statuscode:200",
        "fl": "timestamp,original,mimetype",
        "limit": "1",
        "gzip": "false",
        "sort": "reverse",
    }
    try:
        r = preq.get(CDX_BASE, params=params, allow_redirects=True)
        r.raise_for_status()
        data = r.json()
        if not data or len(data) < 2:
            if verbose:
                print(f"[CDX] No captures for {url} in range {params['from']}..{params['to']}")
            return None
        row = data[1]
        ts, original = row[0], row[1]
        return ts, original
    except requests.RequestException as e:
        if verbose:
            print(f"[CDX] Error for {url} {start_date}..{end_date}: {e}")
        return None
    except json.JSONDecodeError:
        if verbose:
            body = r.text[:200] if 'r' in locals() else ''
            print(f"[CDX] Non-JSON response for {url}: {body}")
        return None


def safe_ext_from_content_type(ct: Optional[str]) -> str:
    if not ct:
        return ".bin"
    ct = ct.split(";")[0].strip().lower()
    if ct in ("text/html", "application/xhtml+xml"):
        return ".html"
    if ct in ("text/plain",):
        return ".txt"
    if ct in ("application/json",):
        return ".json"
    if ct.startswith("image/"):
        sub = ct.split("/", 1)[1]
        return f".{sub}"
    if ct.startswith("text/"):
        return ".txt"
    return ".bin"


def download_snapshot_homepage(preq: PoliteRequester, ts: str, original: str, day_dir: Path,
                               verbose=False) -> Optional[Path]:
    """
    Download the homepage snapshot to:
        <day_dir>/<HHMMSS>.<ext>
    """
    url = RAW_FMT.format(timestamp=ts, original=original)
    hhmmss = f"{ts[8:10]}{ts[10:12]}{ts[12:14]}" if len(ts) >= 14 else "000000"
    try:
        resp = preq.get(url, allow_redirects=True, stream=True)
        if resp.status_code != 200:
            if verbose:
                print(f"[GET] {url} -> HTTP {resp.status_code}")
            return None

        ct = resp.headers.get("Content-Type")
        ext = safe_ext_from_content_type(ct)

        day_dir.mkdir(parents=True, exist_ok=True)
        fpath = day_dir / f"{hhmmss}{ext}"
        with open(fpath, "wb") as f:
            for chunk in resp.iter_content(chunk_size=65536):
                if chunk:
                    f.write(chunk)
        if verbose:
            print(f"[SAVE] homepage: {fpath}")
        return fpath

    except requests.RequestException as e:
        if verbose:
            print(f"[GET] Error {url}: {e}")
        return None


def download_snapshot_subpage(preq: PoliteRequester, ts: str, original: str, day_dir: Path,
                              verbose=False) -> Optional[Path]:
    """
    Download one subpage's raw HTML to hierarchical path under <day_dir>.
    """
    url = RAW_FMT.format(timestamp=ts, original=original)
    rel = path_from_original(original)
    fpath = day_dir / rel
    fpath.parent.mkdir(parents=True, exist_ok=True)

    try:
        resp = preq.get(url, allow_redirects=True, stream=True)
        if resp.status_code != 200:
            if verbose:
                print(f"[GET] {url} -> HTTP {resp.status_code}")
            return None

        # force .html if content type says html/xhtml, else derive
        ct = resp.headers.get("Content-Type")
        ext = safe_ext_from_content_type(ct)
        base, _ext = os.path.splitext(fpath.name)
        if ext and ext != _ext:
            # adjust extension to actual content-type
            fpath = fpath.with_name(base + ext)

        with open(fpath, "wb") as f:
            for chunk in resp.iter_content(chunk_size=65536):
                if chunk:
                    f.write(chunk)
        if verbose:
            print(f"[SAVE] subpage: {fpath}")
        return fpath

    except requests.RequestException as e:
        if verbose:
            print(f"[GET] Error {url}: {e}")
        return None


def crawl_same_day_subpages(preq: PoliteRequester, original_home: str, ts_home: str,
                            day_dir: Path, verbose=False) -> int:
    """
    Query CDX for same-day (YYYYMMDD) captures for the same host, HTML only.
    For each DISTINCT original URL, pick the latest capture that day (sort=reverse, collapse=original).
    Returns count of successfully downloaded subpages.
    """
    ymd = ts_home[:8]
    host = domain_from_url(original_home)
    # We use url=host/* and matchType=host to get same host paths (no foreign domains).
    params = {
        "url": f"{host}/*",
        "from": ymd,
        "to": ymd,
        "output": "json",
        "filter": "statuscode:200",
        "fl": "timestamp,original,mimetype",
        "sort": "reverse",
        "collapse": "original",
        "limit": str(MAX_SUBPAGES + 5),  # a little headroom; we'll trim ourselves
        "matchType": "host",
        "gzip": "false",
    }

    # add MIME filters (HTML only)
    # CDX supports multiple 'filter' params
    filters = ["statuscode:200"] + [f for f in SUBPAGE_MIME_FILTERS]
    # requests will handle list -> repeated keys
    params_list = []
    for k, v in params.items():
        if k != "filter":
            params_list.append((k, v))
    for f in filters:
        params_list.append(("filter", f))

    try:
        r = preq.get(CDX_BASE, params=params_list, allow_redirects=True)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        if verbose:
            print(f"[CDX] Subpage listing error for {host} on {ymd}: {e}")
        return 0

    if not data or len(data) < 2:
        if verbose:
            print(f"[CDX] No same-day subpages for {host} on {ymd}")
        return 0

    home_norm = normalize_original_for_compare(original_home)
    downloaded = 0
    seen = set()

    for row in data[1:]:
        if len(row) < 2:
            continue
        ts, original = row[0], row[1]
        if normalize_original_for_compare(original) == home_norm:
            # skip the homepage itself (we already saved it)
            continue
        if original in seen:
            continue
        seen.add(original)

        # Save only up to MAX_SUBPAGES
        if downloaded >= MAX_SUBPAGES:
            break

        ok_path = download_snapshot_subpage(preq, ts, original, day_dir, verbose=verbose)
        if ok_path is not None:
            downloaded += 1

    if verbose:
        print(f"[CDX] Subpages saved: {downloaded} (host={host}, date={ymd})")
    return downloaded


def process_url_latest_in_range(preq: PoliteRequester, url: str, start_date: dt.date, end_date: dt.date,
                                base_out_dir: Path, verbose=False) -> bool:
    """
    Find and download ONLY the single most-recent snapshot within [start_date, end_date],
    then crawl same-day subpages.
    """
    snap = latest_snapshot_in_range(preq, url, start_date, end_date, verbose=verbose)
    if not snap:
        return False
    ts, original = snap
    dom = domain_from_url(url)
    ymd_dir = f"{ts[:4]}-{ts[4:6]}-{ts[6:8]}" if len(ts) >= 8 else "unknown-date"
    day_dir = base_out_dir / dom / ymd_dir

    # 1) homepage
    home_ok = download_snapshot_homepage(preq, ts, original, day_dir, verbose=verbose)

    # 2) same-day subpages (HTML only)
    _ = crawl_same_day_subpages(preq, original, ts, day_dir, verbose=verbose)

    return home_ok is not None


def batch_process_csv(input_csv: Path, output_csv: Path, start_date: dt.date, end_date: dt.date,
                      out_dir: Path, preq: PoliteRequester,
                      between_rows_sleep: float = 1.0,
                      verbose=False) -> None:
    df = pd.read_csv(input_csv, dtype=str, keep_default_na=True, na_values=["", "NaN", "nan"])
    if "clean_link" not in df.columns:
        raise ValueError("Input CSV must contain a 'clean_link' column.")
    if "link_clean_notes" not in df.columns:
        raise ValueError("Input CSV must contain a 'link_clean_notes' column.")
    if "downloaded" not in df.columns:
        df["downloaded"] = ""

    mask = df["link_clean_notes"].apply(is_empty_notes)
    eligible = df[mask].copy()

    if verbose:
        print(f"Eligible rows (empty link_clean_notes): {len(eligible)} out of {len(df)}")

    url_to_result = {}

    for idx, row in eligible.iterrows():
        url = (row.get("clean_link") or "").strip()
        if not url:
            df.at[idx, "downloaded"] = "no"
            if between_rows_sleep > 0:
                time.sleep(between_rows_sleep)
            continue

        if url not in url_to_result:
            ok = process_url_latest_in_range(
                preq, url, start_date, end_date, Path(out_dir),
                verbose=verbose
            )
            url_to_result[url] = ok
        else:
            ok = url_to_result[url]

        df.at[idx, "downloaded"] = "yes" if ok else "no"

        if between_rows_sleep > 0:
            time.sleep(between_rows_sleep)

    df.to_csv(output_csv, index=False, quoting=csv.QUOTE_MINIMAL)
    if verbose:
        print(f"Wrote updated CSV to: {output_csv}")


def main():
    ap = argparse.ArgumentParser(description="Download the most-recent Wayback snapshot in a date range and same-day subpages (HTML) for cleaned links with empty link_clean_notes.")
    ap.add_argument("--input", type=Path, help="Input CSV (must have columns: clean_link, link_clean_notes).")
    ap.add_argument("--output", type=Path, help="Output CSV path (will be created/overwritten).")
    ap.add_argument("--url", type=str, help="Test a single URL (bypass CSV).")
    ap.add_argument("--start-date", required=True, type=str, help="Start date YYYY-MM-DD.")
    ap.add_argument("--end-date", required=True, type=str, help="End date YYYY-MM-DD.")
    ap.add_argument("--out-dir", required=True, type=Path, help="Directory to save snapshots.")
    ap.add_argument("--verbose", action="store_true", help="Verbose logging.")

    # Politeness / retry controls (tunable)
    ap.add_argument("--base-sleep", type=float, default=2.0,
                    help="Base seconds to sleep before EVERY HTTP request (default 2.0).")
    ap.add_argument("--jitter", type=float, default=0.5,
                    help="Uniform jitter (+/-) added to base sleep and backoff (default 0.5).")
    ap.add_argument("--max-retries", type=int, default=5,
                    help="Max retries on 429/5xx/network errors (default 5).")
    ap.add_argument("--backoff", type=float, default=2.0,
                    help="Exponential backoff base (default 2.0).")
    ap.add_argument("--timeout", type=float, default=40.0,
                    help="HTTP timeout seconds (default 40).")
    ap.add_argument("--between-rows-sleep", type=float, default=1.0,
                    help="Extra sleep after finishing each CSV row (default 1.0).")

    args = ap.parse_args()

    start_date = parse_date(args.start_date)
    end_date = parse_date(args.end_date)
    if end_date < start_date:
        print("end-date must be >= start-date", file=sys.stderr)
        sys.exit(2)

    session = requests.Session()
    session.headers.update({"User-Agent": UA})

    preq = PoliteRequester(
        session=session,
        base_sleep=args.base_sleep,
        jitter=args.jitter,
        max_retries=args.max_retries,
        backoff=args.backoff,
        timeout=args.timeout,
        verbose=args.verbose
    )

    if args.url:
        ok = process_url_latest_in_range(
            preq, args.url.strip(), start_date, end_date, Path(args.out_dir),
            verbose=args.verbose
        )
        print("downloaded=yes" if ok else "downloaded=no")
        return

    if not args.input or not args.output:
        print("For CSV mode, provide --input and --output.", file=sys.stderr)
        sys.exit(2)

    batch_process_csv(
        args.input, args.output, start_date, end_date, args.out_dir,
        preq=preq,
        between_rows_sleep=args.between_rows_sleep,
        verbose=args.verbose
    )


if __name__ == "__main__":
    main()