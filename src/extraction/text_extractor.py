"""
text_extractor.py
-----------------
Enhanced version with:
- deep HTML cleaning (from html_cleaning_utils.py)
- domain-specific filtering for facial recognition relevance
- language filtering (English only for BERTopic)
- size filtering (remove tiny/noisy pages)
- duplicate removal
- detailed cleaning statistics for final reporting

Outputs:
  analysis/text/snapshot_texts.csv       -> cleaned dataset
  analysis/text/cleaning_stats.csv       -> machine-readable stats
  analysis/text/cleaning_stats.txt       -> human-readable summary
"""

import os
import pandas as pd
from multiprocessing import Pool, cpu_count
from functools import partial
from tqdm import tqdm
from langdetect import detect, LangDetectException

from src.extraction.html_cleaning_utils import clean_html_text


# -------------------------------------------------------
# PATHS
# -------------------------------------------------------
SNAPSHOT_ROOT = "data/wayback/snapshots"
OUTPUT_DIR = "analysis/text"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "snapshot_texts.csv")
STATS_FILE = os.path.join(OUTPUT_DIR, "cleaning_stats.csv")
TXT_FILE = os.path.join(OUTPUT_DIR, "cleaning_stats.txt")

os.makedirs(OUTPUT_DIR, exist_ok=True)


# -------------------------------------------------------
# FACIAL RECOGNITION KEYWORDS (RELEVANCE FILTER)
# -------------------------------------------------------
FR_KEYWORDS = [

    # --- Direct FR keywords ---
    "face recognition", "facial recognition", "face biometric",
    "biometric identification", "biometric authentication",
    "face authentication", "facial authentication",
    "liveness detection", "spoof detection", "anti-spoofing",
    "face capture", "face matching", "face detection",

    # --- Identity / Verification ---
    "identity verification", "id verification", "idv",
    "kyc", "aml", "identity assurance",
    "identity management", "identity platform",
    "fraud prevention", "secure authentication",
    "digital identity", "onboarding", "verification workflow",

    # --- Computer Vision / ML ---
    "computer vision", "deep learning", "neural network",
    "vision ai", "image recognition", "pattern recognition",
    "object detection", "video analytics",
    "ai-powered", "machine learning model", "analytics platform",

    # --- Security / Access Control ---
    "access control", "entry management", "smart access",
    "contactless access", "touchless", "visitor management",
    "credentials", "credentialing", "door controller",
    "physical security", "biometric access",
    "time and attendance", "workforce management",

    # --- Surveillance ---
    "video surveillance", "cctv", "public safety",
    "real-time monitoring", "tracking system",
    "watchlist", "forensic", "crime prevention",

    # --- AI ethics and compliance ---
    "gdpr", "privacy", "responsible ai", "ai ethics",
    "fairness", "transparency", "bias mitigation",
    "compliance", "regulation", "ethical ai",
]


def contains_fr_keywords(text: str) -> bool:
    """Return True if text contains any FR keyword."""
    t = text.lower()
    return any(k in t for k in FR_KEYWORDS)


def detect_language(text: str) -> str:
    """Detect language using langdetect; return 'unknown' on failure."""
    try:
        snippet = text[:1000]  # langdetect doesn't need full doc
        return detect(snippet)
    except LangDetectException:
        return "unknown"
    except Exception:
        return "unknown"


# -------------------------------------------------------
# WORKER FUNCTION
# -------------------------------------------------------
def process_html_file(snapshot_root, relpath):
    """
    Process a single HTML file and return:
      {
        "record": {...} OR None,
        "stats": {
            "total_files": 1,
            "parsed_ok": 0/1,
            "removed_short": 0/1,
            "removed_non_english": 0/1,
            "removed_no_keywords": 0/1,
            "kept_final": 0/1,
        }
      }
    """
    local_stats = {
        "total_files": 1,
        "parsed_ok": 0,
        "removed_short": 0,
        "removed_non_english": 0,
        "removed_no_keywords": 0,
        "kept_final": 0,
    }

    parts = relpath.split(os.sep)
    if len(parts) < 3:
        # Still count the file, but nothing parsed
        return {"record": None, "stats": local_stats}

    company = parts[0]
    decade = parts[1]
    fpath = os.path.join(snapshot_root, relpath)

    try:
        # Read HTML
        with open(fpath, "r", errors="ignore") as f:
            html = f.read()

        # Clean HTML text using your enhanced utilities
        cleaned = clean_html_text(html)
        local_stats["parsed_ok"] = 1

        # Rule 1: Remove very short pages
        if len(cleaned.split()) < 30:
            local_stats["removed_short"] = 1
            return {"record": None, "stats": local_stats}

        # Rule 2: Language filter
        lang = detect_language(cleaned)
        if lang != "en":
            local_stats["removed_non_english"] = 1
            return {"record": None, "stats": local_stats}

        # Rule 3: Facial-recognition relevance filter
        if not contains_fr_keywords(cleaned):
            local_stats["removed_no_keywords"] = 1
            return {"record": None, "stats": local_stats}

        # Passed all filters!
        local_stats["kept_final"] = 1

        record = {
            "company": company,
            "decade": decade,
            "path": relpath,
            "lang": lang,
            "text": cleaned,
        }

        return {"record": record, "stats": local_stats}

    except Exception:
        # Could not parse/clean; parsed_ok stays 0
        return {"record": None, "stats": local_stats}


# -------------------------------------------------------
# COLLECT ALL HTML PATHS
# -------------------------------------------------------
def collect_html_paths(root):
    paths = []
    for base, dirs, files in os.walk(root):
        for fname in files:
            if fname.endswith(".html"):
                paths.append(os.path.relpath(os.path.join(base, fname), root))
    return paths


# -------------------------------------------------------
# MAIN EXECUTION
# -------------------------------------------------------
if __name__ == "__main__":

    print("\nScanning HTML files...")
    html_files = collect_html_paths(SNAPSHOT_ROOT)
    total = len(html_files)
    print(f"Total HTML files found: {total}\n")

    worker = partial(process_html_file, SNAPSHOT_ROOT)
    num_workers = max(1, cpu_count() - 1)

    print(f"Starting multiprocessing extraction using {num_workers} workers...\n")

    # Aggregate results + stats
    all_records = []
    agg_stats = {
        "total_files": 0,
        "parsed_ok": 0,
        "removed_short": 0,
        "removed_non_english": 0,
        "removed_no_keywords": 0,
        "kept_final": 0,
    }

    with Pool(num_workers) as pool:
        for result in tqdm(pool.imap_unordered(worker, html_files), total=total):
            if result is None:
                continue

            rec = result["record"]
            s = result["stats"]

            # Aggregate stats
            for k in agg_stats.keys():
                agg_stats[k] += s.get(k, 0)

            # Collect kept records
            if rec is not None:
                all_records.append(rec)

    # Create DataFrame
    df = pd.DataFrame(all_records)

    # Duplicate removal (important for snapshot archives)
    before_dedup = len(df)
    df = df.drop_duplicates(subset=["company", "decade", "text"])
    after_dedup = len(df)
    duplicates_removed = before_dedup - after_dedup

    # Save cleaned dataset
    df.to_csv(OUTPUT_FILE, index=False)

    # Add duplicates_removed to stats
    agg_stats["duplicates_removed"] = duplicates_removed

    # Save machine-readable stats CSV
    pd.DataFrame([agg_stats]).to_csv(STATS_FILE, index=False)

    # -------------------------------------------------------
    # BUILD HUMAN-READABLE TXT REPORT
    # -------------------------------------------------------
    total_final = len(df)
    total_files = agg_stats["total_files"]
    kept_pct = (total_final / total_files) * 100 if total_files else 0.0

    report = [
        "------------------------------------------------------------------",
        "             FACIAL RECOGNITION DATA CLEANING REPORT",
        "------------------------------------------------------------------",
        "",
        f"Total HTML files scanned       : {total_files}",
        f"Successfully parsed            : {agg_stats['parsed_ok']}",
        "",
        "REMOVED DURING CLEANING:",
        f"  - Too short (<30 words)      : {agg_stats['removed_short']}",
        f"  - Non-English                : {agg_stats['removed_non_english']}",
        f"  - No FR-related keywords     : {agg_stats['removed_no_keywords']}",
        "",
        f"Duplicates removed             : {duplicates_removed}",
        "",
        "FINAL DATASET:",
        f"  - Kept documents             : {total_final}",
        f"  - Percent kept               : {kept_pct:.2f}%",
        "",
    ]

    # Optional: display top companies
    if not df.empty:
        report.append("TOP COMPANIES BY DOCUMENT COUNT:")
        c_counts = df["company"].value_counts().head(10)
        for comp, count in c_counts.items():
            report.append(f"  - {comp:20s}: {count}")
        report.append("")

    # Optional: show decade distribution
    if "decade" in df.columns:
        report.append("DOCUMENTS BY DECADE:")
        d_counts = df["decade"].value_counts()
        for dec, count in d_counts.items():
            report.append(f"  - {dec:10s}: {count}")
        report.append("")

    report.append("------------------------------------------------------------------")
    report.append("   This report is automatically generated by text_extractor.py")
    report.append("Save this file for reproducibility and your SOC 4994 final report.")
    report.append("------------------------------------------------------------------")

    # Save TXT report
    with open(TXT_FILE, "w") as f:
        f.write("\n".join(report))

    print(f"\nSaved cleaned dataset to: {OUTPUT_FILE}")
    print(f"Saved machine-readable stats to: {STATS_FILE}")
    print(f"Saved human-readable report to: {TXT_FILE}\n")
    print("---------------------------------------------\n")