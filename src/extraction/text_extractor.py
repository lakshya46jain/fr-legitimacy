"""
text_extractor.py
-----------------
Parallel HTML text extraction for Wayback snapshots.
Uses multiprocessing for speed and optional progress bar.

OUTPUT → analysis/text/snapshot_texts.csv
"""

import os
import pandas as pd
from multiprocessing import Pool, cpu_count
from functools import partial
from tqdm import tqdm
from html_cleaning_utils import clean_html_text


SNAPSHOT_ROOT = "data/wayback/snapshots"
OUTPUT_DIR = "analysis/text"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "snapshot_texts.csv")

os.makedirs(OUTPUT_DIR, exist_ok=True)


def process_html_file(snapshot_root, relpath):
    """
    Worker: Given a relative path inside snapshot_root,
    return cleaned text + metadata dictionary.
    """
    parts = relpath.split(os.sep)
    if len(parts) < 3:
        return None  # unexpected format

    company = parts[0]
    decade = parts[1]
    fpath = os.path.join(snapshot_root, relpath)

    try:
        with open(fpath, "r", errors="ignore") as f:
            html = f.read()
        text = clean_html_text(html)
        return {
            "company": company,
            "decade": decade,
            "path": relpath,
            "text": text
        }
    except Exception:
        return None


def collect_html_paths(root):
    paths = []
    for base, dirs, files in os.walk(root):
        for fname in files:
            if fname.endswith(".html"):
                fpath = os.path.relpath(os.path.join(base, fname), root)
                paths.append(fpath)
    return paths


if __name__ == "__main__":

    print("Scanning for HTML files in:", SNAPSHOT_ROOT)
    html_files = collect_html_paths(SNAPSHOT_ROOT)
    total_files = len(html_files)
    print("Total HTML files found:", total_files)

    worker = partial(process_html_file, SNAPSHOT_ROOT)
    num_workers = max(1, cpu_count() - 1)

    print("Starting multiprocessing extraction using", num_workers, "workers.")

    results = []
    with Pool(num_workers) as pool:
        for record in tqdm(pool.imap_unordered(worker, html_files), total=total_files):
            if record is not None:
                results.append(record)

    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_FILE, index=False)

    print("---------------------------------------------")
    print("Extraction complete.")
    print("Total successfully processed files:", len(df))
    print("Saved dataset to:", OUTPUT_FILE)
    print("---------------------------------------------")