#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Wayback snapshot downloader for rows with empty 'link_clean_notes' using 'clean_link'.

- Input CSV must contain: clean_link, link_clean_notes
- Adds/overwrites a column: downloaded (yes/no)
- For each eligible row, for each date in [start_date, end_date], grabs the LATEST snapshot
  for that day (status 200) and downloads raw content.
- Saves files under: <out_dir>/<domain>/<YYYY-MM-DD>/<HHMMSS>.html (or .bin if unknown)

Usage examples:
  # Batch mode on a CSV
  python3 snapshot_downloader.py \
      --input data.csv --output data_with_downloads.csv \
      --start-date 2019-01-01 --end-date 2019-03-31 \
      --out-dir ./wayback

  # Test a single URL (no CSV writing)
  python3 snapshot_downloader.py \
      --url https://example.com \
      --start-date 2020-01-01 --end-date 2020-01-05 \
      --out-dir ./wayback --verbose
"""

import argparse
import csv
import datetime as dt
import json
import os
import re
import sys
import time
import random
from pathlib import Path
from typing import Optional, Tuple, List

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


UA = "SOC4994-Wayback-Downloader/1.1 (+https://example.edu; email=student@vt.edu)"
CDX_BASE = "https://web.archive.org/cdx/search/cdx"
RAW_FMT = "https://web.archive.org/web/{timestamp}id_/{original}"  # id_ = raw content passthrough


def parse_date(d: str) -> dt.date:
    return dt.datetime.strptime(d, "%Y-%m-%d").date()


def daterange(start: dt.date, end: dt.date):
    cur = start
    one = dt.timedelta(days=1)
    while cur <= end:
        yield cur
        cur += one


def is_empty_notes(val) -> bool:
    if val is None:
        return True
    if isinstance(val, float):  # NaN
        return False if pd.notna(val) else True
    s = str(val).strip()
    return len(s) == 0


def domain_from_url(url: str) -> str:
    # crude but robust
    m = re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://([^/]+)/?", url)
    return m.group(1) if m else "unknown-domain"


class PoliteRequester:
    """
    Adds jittered sleeps between requests and robust retries with exponential backoff.
    Designed to be 'polite' to the Wayback services.
    """
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
        # base +/- uniform jitter
        delta = random.uniform(-self.jitter, self.jitter)
        dur = max(0.0, self.base_sleep + delta)
        time.sleep(dur)

    def _sleep_backoff(self, attempt: int, retry_after: Optional[float] = None):
        if retry_after is not None:
            if self.verbose:
                print(f"[polite] Honor Retry-After: sleeping {retry_after:.2f}s")
            time.sleep(max(0.0, retry_after))
            return
        # exponential backoff with jitter
        dur = (self.backoff ** attempt) + random.uniform(0.0, self.jitter)
        if self.verbose:
            print(f"[polite] Backoff attempt {attempt}: sleeping {dur:.2f}s")
        time.sleep(dur)

    def get(self, url: str, **kwargs) -> requests.Response:
        """
        GET with polite delay, timeout, and retries on 429/5xx and common network errors.
        """
        # Ensure timeout is set
        if "timeout" not in kwargs:
            kwargs["timeout"] = self.timeout

        attempt = 0
        while True:
            # polite pre-request sleep every time (throttle rate)
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

            # Retry on 429, 503, and generic 5xx
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


def latest_snapshot_for_day(preq: PoliteRequester, url: str, day: dt.date, verbose=False) -> Optional[Tuple[str, str]]:
    """
    Query CDX for the LATEST (reverse sort) snapshot on a given day with status 200.
    Returns (timestamp14, original_url) or None.
    """
    ymd = day.strftime("%Y%m%d")
    params = {
        "url": url,
        "from": ymd,
        "to": ymd,
        "output": "json",
        "filter": "statuscode:200",
        "fl": "timestamp,original,mimetype",
        "limit": "1",
        "gzip": "false",
        "sort": "reverse",  # latest first
    }
    try:
        r = preq.get(CDX_BASE, params=params, allow_redirects=True)
        r.raise_for_status()
        data = r.json()
        if not data or len(data) < 2:
            if verbose:
                print(f"[CDX] No captures for {url} on {ymd}")
            return None
        row = data[1]
        ts, original = row[0], row[1]
        return ts, original
    except requests.RequestException as e:
        if verbose:
            print(f"[CDX] Error for {url} {ymd}: {e}")
        return None
    except json.JSONDecodeError:
        if verbose:
            # Truncate to keep logs readable
            body = r.text[:200] if 'r' in locals() else ''
            print(f"[CDX] Non-JSON response for {url} {ymd}: {body}")
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


def download_snapshot(preq: PoliteRequester, ts: str, original: str, out_dir: Path, verbose=False) -> Optional[Path]:
    """
    Download raw archived content for (ts, original) to out_dir.
    Returns the written file path or None if failed.
    """
    url = RAW_FMT.format(timestamp=ts, original=original)
    try:
        resp = preq.get(url, allow_redirects=True, stream=True)
        if resp.status_code != 200:
            if verbose:
                print(f"[GET] {url} -> HTTP {resp.status_code}")
            return None
        ct = resp.headers.get("Content-Type")
        ext = safe_ext_from_content_type(ct)
        hhmmss = ts[8:14] if len(ts) >= 14 else "000000"
        out_dir.mkdir(parents=True, exist_ok=True)
        fname = f"{hhmmss}{ext}"
        fpath = out_dir / fname
        with open(fpath, "wb") as f:
            for chunk in resp.iter_content(chunk_size=65536):
                if chunk:
                    f.write(chunk)
        if verbose:
            print(f"[SAVE] {fpath}")
        return fpath
    except requests.RequestException as e:
        if verbose:
            print(f"[GET] Error {url}: {e}")
        return None


def process_url_over_range(preq: PoliteRequester, url: str, start_date: dt.date, end_date: dt.date,
                           base_out_dir: Path, between_days_sleep: float = 0.0, verbose=False) -> bool:
    """
    Returns True if at least one snapshot downloaded.
    """
    dom = domain_from_url(url)
    any_saved = False
    for day in daterange(start_date, end_date):
        snap = latest_snapshot_for_day(preq, url, day, verbose=verbose)
        if not snap:
            if between_days_sleep > 0:
                time.sleep(between_days_sleep)
            continue
        ts, original = snap
        day_dir = base_out_dir / dom / day.strftime("%Y-%m-%d")
        saved = download_snapshot(preq, ts, original, day_dir, verbose=verbose)
        if saved:
            any_saved = True
        if between_days_sleep > 0:
            time.sleep(between_days_sleep)
    return any_saved


def batch_process_csv(input_csv: Path, output_csv: Path, start_date: dt.date, end_date: dt.date,
                      out_dir: Path, preq: PoliteRequester,
                      between_rows_sleep: float = 1.0, between_days_sleep: float = 0.0,
                      verbose=False) -> None:
    df = pd.read_csv(input_csv, dtype=str, keep_default_na=True, na_values=["", "NaN", "nan"])
    if "clean_link" not in df.columns:
        raise ValueError("Input CSV must contain a 'clean_link' column.")
    if "link_clean_notes" not in df.columns:
        raise ValueError("Input CSV must contain a 'link_clean_notes' column.")

    if "downloaded" not in df.columns:
        df["downloaded"] = ""

    # rows where link_clean_notes is empty
    mask = df["link_clean_notes"].apply(is_empty_notes)
    eligible = df[mask].copy()

    if verbose:
        print(f"Eligible rows (empty link_clean_notes): {len(eligible)} out of {len(df)}")

    # de-dup urls to avoid repeating work
    url_to_result = {}

    for idx, row in eligible.iterrows():
        url = (row.get("clean_link") or "").strip()
        if not url:
            df.at[idx, "downloaded"] = "no"
            if between_rows_sleep > 0:
                time.sleep(between_rows_sleep)
            continue

        if url not in url_to_result:
            ok = process_url_over_range(
                preq, url, start_date, end_date, Path(out_dir),
                between_days_sleep=between_days_sleep, verbose=verbose
            )
            url_to_result[url] = ok
        else:
            ok = url_to_result[url]

        df.at[idx, "downloaded"] = "yes" if ok else "no"

        # polite pause between CSV rows (helps when many companies)
        if between_rows_sleep > 0:
            time.sleep(between_rows_sleep)

    df.to_csv(output_csv, index=False, quoting=csv.QUOTE_MINIMAL)
    if verbose:
        print(f"Wrote updated CSV to: {output_csv}")


def main():
    ap = argparse.ArgumentParser(description="Download latest Wayback snapshots (per day) for cleaned links with empty link_clean_notes.")
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
    ap.add_argument("--between-days-sleep", type=float, default=0.0,
                    help="Extra sleep after finishing each day for a URL (default 0.0).")

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
        ok = process_url_over_range(
            preq, args.url.strip(), start_date, end_date, Path(args.out_dir),
            between_days_sleep=args.between_days_sleep, verbose=args.verbose
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
        between_days_sleep=args.between_days_sleep,
        verbose=args.verbose
    )


if __name__ == "__main__":
    main()