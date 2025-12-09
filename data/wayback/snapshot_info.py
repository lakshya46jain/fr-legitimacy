#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
snapshot_info.py
----------------
Query the Internet Archive (Wayback Machine) to compute snapshot
history for each vendor in the facial recognition dataset.

This script is designed to run AFTER:
    • website_cleaner.py  → produces clean_link
and BEFORE:
    • snapshot_downloader_yearly.py
    • snapshot_downloader_decade.py

Goal
----
For every vendor row:
    • Use clean_link (if valid and not flagged) to fetch:
          - snapshot_count
          - first_snapshot (YYYY-MM-DD)
          - last_snapshot  (YYYY-MM-DD)
    • Leave unprocessed rows untouched
    • Save all fields into `fr_vendors_snapshot_info.csv`

Features
--------
• Rate-limit handling with exponential backoff  
• Processes *all rows* but only queries rows eligible for processing  
• Saves progress every 20 rows  
• Safe URL normalization  
• Robust timestamp parsing  

Usage
-----
    python3 snapshot_info.py

Output
------
    fr_vendors_snapshot_info.csv  (merged with original vendor metadata)

"""

import pandas as pd
import random
import requests
import time
from datetime import datetime


# -------------------------------------------------------
# CONFIGURATION
# -------------------------------------------------------
DATASET_PATH = "../vendors/facial_recognition_vendors.csv"
URL_COLUMN = "clean_link"
NOTES_COLUMN = "link_clean_notes"
OUTPUT_CSV = "./fr_vendors_snapshot_info.csv"

print("Loading dataset...")
df = pd.read_csv(DATASET_PATH)

# Always keep ALL rows — we enrich, not replace
all_rows = df.copy()

print(f"Total rows loaded: {len(all_rows)}\n")


# -------------------------------------------------------
# URL & TIMESTAMP HELPERS
# -------------------------------------------------------
def clean_url(url):
    """
    Ensure URL begins with http/https and strip whitespace.

    Convert:
        "example.com" → "https://example.com"
        " http://abc.com " → "http://abc.com"
    """
    if not isinstance(url, str) or not url.strip():
        return None
    if not url.startswith("http"):
        url = "https://" + url
    return url.strip()


def format_date(ts):
    """
    Convert Wayback timestamp → YYYY-MM-DD.

    Input:
        "20191226084331"
    Output:
        "2019-12-26"
    """
    if ts is None:
        return None
    try:
        return datetime.strptime(ts, "%Y%m%d%H%M%S").strftime("%Y-%m-%d")
    except:
        return None


# -------------------------------------------------------
# WAYBACK SNAPSHOT QUERY
# -------------------------------------------------------
def get_snapshot_info(url, retries=5):
    """
    Query Wayback Machine CDX API to fetch snapshot history.

    Returns:
        (snapshot_count, first_snapshot, last_snapshot)

    Error handling:
        - Retries on network errors
        - Handles rate limiting (HTTP 429)
        - Returns zeros on repeated failure
    """
    base = "https://web.archive.org/cdx/search/cdx"
    params = {
        "url": url,
        "output": "json",
        "filter": "statuscode:200"
    }

    for attempt in range(retries):
        try:
            r = requests.get(base, params=params, timeout=20)

            # Rate limited → exponential backoff
            if r.status_code == 429:
                wait = (2 ** attempt) + random.uniform(0, 2)
                print(f"Rate limit hit. Waiting {wait:.1f}s before retry...")
                time.sleep(wait)
                continue

            r.raise_for_status()
            data = r.json()

            # No snapshot rows found
            if len(data) <= 1:
                return 0, None, None

            # Row format: [urlkey, timestamp, original, mimetype, status]
            timestamps = [row[1] for row in data[1:]]

            first_ts = format_date(min(timestamps))
            last_ts = format_date(max(timestamps))

            return len(timestamps), first_ts, last_ts

        except Exception as e:
            print(f"{url}: {e} (attempt {attempt+1}/{retries})")
            time.sleep(2)

    print(f"Skipping {url} after {retries} failed attempts.")
    return 0, None, None


# -------------------------------------------------------
# MAIN PROCESSING LOOP
# -------------------------------------------------------
results = []
start_time = time.time()

for idx, row in all_rows.iterrows():

    # Preserve original row
    record = row.to_dict()

    raw_clean_link = record.get(URL_COLUMN, None)
    link_notes = record.get(NOTES_COLUMN, "")

    # Initialize new fields
    record["processed_link"] = None
    record["snapshot_count"] = None
    record["first_snapshot"] = None
    record["last_snapshot"] = None

    # Determine whether record is eligible for processing
    should_process = (
        (pd.isna(link_notes) or str(link_notes).strip() == "") and
        (isinstance(raw_clean_link, str) and raw_clean_link.strip() != "")
    )

    if should_process:
        cleaned = clean_url(raw_clean_link)

        company_name = record.get("company", "")
        print(f"[{idx+1}] {company_name}: checking {cleaned}")

        if cleaned:
            count, first, last = get_snapshot_info(cleaned)
            record["processed_link"] = cleaned
            record["snapshot_count"] = count
            record["first_snapshot"] = first
            record["last_snapshot"] = last
        else:
            record["processed_link"] = None

    # Append processed row to results
    results.append(record)

    # Logging status for visibility
    if record["snapshot_count"] is None:
        print(f"✓ [{idx+1}] {record.get('company', '')}: no info (skipped or no clean link)")
    else:
        print(f"✓ [{idx+1}] {record.get('company', '')}: snapshots = {record['snapshot_count']}")

    # Save progress every 20 rows to avoid data loss
    if (len(results) % 20) == 0:
        pd.DataFrame(results).to_csv(OUTPUT_CSV, index=False)
        print(f"Progress saved ({len(results)} processed so far)")

    # Randomized delay to avoid overwhelming Wayback servers
    time.sleep(random.uniform(1.0, 2.2))


# -------------------------------------------------------
# FINAL SAVE
# -------------------------------------------------------
out_df = pd.DataFrame(results)
out_df.to_csv(OUTPUT_CSV, index=False)

elapsed = (time.time() - start_time) / 60
print(f"\nFinished! {len(results)} rows processed in {elapsed:.1f} minutes.")
print(f"Results saved to {OUTPUT_CSV}")