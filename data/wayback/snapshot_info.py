import pandas as pd
import random
import requests
import time

DATASET_PATH = "../vendors/facial_recognition_vendors.csv"
DATASET_PATH = "./test.csv"
URL_COLUMN = "clean_link"
NOTES_COLUMN = "link_clean_notes"
OUTPUT_CSV = "./fr_vendors_snapshot_info.csv"

print("Loading dataset...")
df = pd.read_csv(DATASET_PATH)

# Keep ALL rows
all_rows = df.copy()

print(f"Total rows loaded: {len(all_rows)}\n")

def clean_url(url):
    """Ensure URL starts with https://"""
    if not isinstance(url, str) or not url.strip():
        return None
    if not url.startswith("http"):
        url = "https://" + url
    return url.strip()

def get_snapshot_info(url, retries=5):
    """Fetch snapshot info from the Wayback Machine with retry/backoff."""
    base = "https://web.archive.org/cdx/search/cdx"
    params = {"url": url, "output": "json", "filter": "statuscode:200"}

    for attempt in range(retries):
        try:
            r = requests.get(base, params=params, timeout=20)
            if r.status_code == 429:
                wait = (2 ** attempt) + random.uniform(0, 2)
                print(f"Rate limit hit. Waiting {wait:.1f}s before retry...")
                time.sleep(wait)
                continue

            r.raise_for_status()
            data = r.json()

            if len(data) <= 1:
                return 0, None, None

            timestamps = [row[1] for row in data[1:]]
            return len(timestamps), min(timestamps), max(timestamps)

        except Exception as e:
            print(f"{url}: {e} (attempt {attempt+1}/{retries})")
            time.sleep(2)

    print(f"Skipping {url} after {retries} failed attempts.")
    return 0, None, None


# === PROCESSING ON ALL ROWS ===
results = []
start_time = time.time()

for idx, row in all_rows.iterrows():
    base_record = row.to_dict()

    raw_clean_link = base_record.get("clean_link", None)
    link_notes = base_record.get("link_clean_notes", "")

    # Default new fields
    base_record["processed_link"] = None
    base_record["snapshot_count"] = None
    base_record["first_snapshot"] = None
    base_record["last_snapshot"] = None

    # Determine if row should be processed
    should_process = (
        (pd.isna(link_notes) or str(link_notes).strip() == "") and
        (isinstance(raw_clean_link, str) and raw_clean_link.strip() != "")
    )

    if should_process:
        cleaned = clean_url(raw_clean_link)

        print(f"[{idx+1}] {base_record.get('company', '')}: checking {cleaned}")

        if cleaned:
            count, first, last = get_snapshot_info(cleaned)
            base_record["processed_link"] = cleaned
            base_record["snapshot_count"] = count
            base_record["first_snapshot"] = first
            base_record["last_snapshot"] = last
        else:
            base_record["processed_link"] = None

    else:
        # Not processed — leave processed_link as None
        pass

    results.append(base_record)

    # Print a status message for each row
    if base_record["snapshot_count"] is None:
        print(f"✓ [{idx+1}] {base_record.get('company', '')}: no info (skipped or no clean link)")
    else:
        print(f"✓ [{idx+1}] {base_record.get('company', '')}: snapshots = {base_record['snapshot_count']}")

    # Save progress every 20 rows
    if (len(results) % 20) == 0:
        pd.DataFrame(results).to_csv(OUTPUT_CSV, index=False)
        print(f"Progress saved ({len(results)} processed so far)")

    # Safe randomized delay between requests
    time.sleep(random.uniform(1.0, 2.2))


# === Final save ===
out_df = pd.DataFrame(results)
out_df.to_csv(OUTPUT_CSV, index=False)

elapsed = (time.time() - start_time) / 60
print(f"\nFinished! {len(results)} rows processed in {elapsed:.1f} minutes.")
print(f"Results saved to {OUTPUT_CSV}")