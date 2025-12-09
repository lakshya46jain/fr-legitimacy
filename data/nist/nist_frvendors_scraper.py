"""
nist_frvendors_scraper.py
------------------------
Scraper for the NIST FRVT 1:1 Performance Summary Table.

This module collects:
    • Company names
    • Country codes
    • Reportcard URLs (one per row)
    • Placeholder metadata columns
    • Stable unique IDs for ordering

Core steps:
    1. Load the FRVT webpage using Selenium
    2. Locate the correct results table
    3. Expand pagination ("All" rows if possible)
    4. Iterate through visible rows, de-duplicating links
    5. Parse vendor name + country from the <a title="..."> attribute
    6. Save structured results to CSV

Output:
    nist_frvendors_scraped.csv  → canonical vendor list for downstream steps

This script is part of the SOC 4994 research pipeline and should be run
BEFORE:
    - website_cleaner.py
    - Wayback snapshot collection
    - text extraction
"""

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait, Select
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.chrome.options import Options

import re
import pandas as pd
import time
import pycountry


# -------------------------------------------------------
# CONFIGURATION
# -------------------------------------------------------
URL = "https://pages.nist.gov/frvt/html/frvt11.html#_FRTE_1:1_Performance_Summary_Table"
BASE_URL = "https://face.nist.gov/frte/reportcards/11/"


# -------------------------------------------------------
# TITLE PARSING HELPERS
# -------------------------------------------------------
def parse_title(text: str):
    """
    Extract company name + ISO country code from the <a title="..."> field.

    Example
    -------
    Input:  "QazSmartVision.AI (KZ)"
    Output: ("QazSmartVision.AI", "KZ")
    """
    match = re.match(r"^(.*?)\s*\(([^)]+)\)\s*$", text or "")
    if match:
        return match.group(1).strip(), match.group(2).strip()
    # fallback — poorly formatted or missing country
    return text or "", ""


def iso_to_country(iso_code: str) -> str:
    """
    Convert a two-letter ISO code (e.g., 'KZ') into a full country name.
    Falls back to original code if unrecognized.
    """
    try:
        country = pycountry.countries.get(alpha_2=iso_code.upper())
        return country.name if country else iso_code
    except Exception:
        return iso_code


# -------------------------------------------------------
# SELENIUM DRIVER SETUP
# -------------------------------------------------------
chrome_options = Options()
# chrome_options.add_argument("--headless=new")  # Uncomment for headless mode

driver = webdriver.Chrome(options=chrome_options)
wait = WebDriverWait(driver, 15)

driver.get(URL)


# -------------------------------------------------------
# TABLE DISCOVERY
# -------------------------------------------------------
def find_frvt_table():
    """
    Locate the FRVT 1:1 results table.

    Strategy:
        1. Prefer the table immediately following the FRTE anchor ID
        2. Fallback: search for any table containing reportcard links

    Returns
    -------
    (table_element, wrapper_element)
    """
    # Wait for page load
    WebDriverWait(driver, 10).until(
        lambda d: d.execute_script("return document.readyState") == "complete"
    )

    try:
        # Attempt anchor-based lookup
        anchor = WebDriverWait(driver, 5).until(
            EC.presence_of_element_located((
                By.XPATH,
                "//*[@id='_FRTE_1:1_Performance_Summary_Table' "
                "or @name='_FRTE_1:1_Performance_Summary_Table']"
            ))
        )
        table = anchor.find_element(By.XPATH, "following::table[1]")

    except TimeoutException:
        # Fallback: find ANY table containing reportcard links
        table = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((
                By.XPATH,
                f"//table[.//a[starts-with(@href, '{BASE_URL}')]]"
            ))
        )

    # DataTables wrapper (needed for pagination detection)
    wrapper = table.find_element(
        By.XPATH,
        "ancestor::div[contains(@class,'dataTables_wrapper')]"
    )

    return table, wrapper


table, wrapper = find_frvt_table()


# -------------------------------------------------------
# TRY TO SHOW ALL ROWS
# -------------------------------------------------------
try:
    length_select = wrapper.find_element(
        By.XPATH, ".//select[contains(@name,'_length')]"
    )
    sel = Select(length_select)
    options = [o.text.strip() for o in sel.options]

    # Prefer an "All" option if available
    if any("All" in o for o in options):
        sel.select_by_visible_text(next(o for o in options if "All" in o))
    else:
        # Otherwise select the largest numeric option
        nums = [int(o) for o in options if o.isdigit()]
        if nums:
            sel.select_by_visible_text(str(max(nums)))

    time.sleep(0.3)  # allow redraw

except Exception:
    # Some tables do not offer a length selector → pagination fallback
    pass


# -------------------------------------------------------
# SCRAPING STATE
# -------------------------------------------------------
all_rows = []
seen_hrefs = set()
sequential_id = 1  # deterministic row numbering


# -------------------------------------------------------
# ROW SCRAPING
# -------------------------------------------------------
def scrape_visible_rows():
    """
    Extract all *visible* table rows (DataTables hides off-page rows).
    Ensures:
        - Only 1 reportcard link per row
        - No duplicate hrefs across pages
    """
    global sequential_id

    rows = table.find_elements(By.XPATH, ".//tbody/tr")

    for r in rows:
        if not r.is_displayed():
            continue

        links = r.find_elements(
            By.XPATH,
            f".//a[starts-with(@href, '{BASE_URL}') and @title]"
        )
        if not links:
            continue

        # Use the first relevant link (canonical for this row)
        a = links[0]
        title = (a.get_attribute("title") or "").strip()
        href = (a.get_attribute("href") or "").strip()

        if not href or href in seen_hrefs:
            continue

        seen_hrefs.add(href)
        name, country = parse_title(title)

        all_rows.append({
            "id": sequential_id,
            "href": href,
            "title": title,
            "company": name,
            "country": iso_to_country(country),

            # Placeholders populated later in the pipeline
            "year": "",
            "status": "",
            "bio": "",
            "hist": "",
            "org": "",
            "media": "",
            "social": "",
            "gov": "",
            "link": "",
            "text": "",
        })

        sequential_id += 1


# -------------------------------------------------------
# PAGINATION HANDLER
# -------------------------------------------------------
def click_next_if_possible():
    """
    Click the DataTables "Next" button if it is active.

    Returns
    -------
    True  → page advanced
    False → pagination complete or unable to advance
    """
    try:
        next_btn = wrapper.find_element(
            By.CSS_SELECTOR,
            ".dataTables_paginate .next, a.paginate_button.next, li.paginate_button.next"
        )
    except Exception:
        return False

    cls = (next_btn.get_attribute("class") or "").lower()
    aria_disabled = (next_btn.get_attribute("aria-disabled") or "").lower()

    if "disabled" in cls or aria_disabled == "true":
        return False

    # Detect page change via staleness of first visible row
    visible_rows = [
        tr for tr in table.find_elements(By.XPATH, ".//tbody/tr")
        if tr.is_displayed()
    ]
    anchor_row = visible_rows[0] if visible_rows else None

    next_btn.click()

    if anchor_row:
        try:
            wait.until(EC.staleness_of(anchor_row))
        except TimeoutException:
            return False

    time.sleep(0.2)
    return True


# -------------------------------------------------------
# MAIN SCRAPING LOOP
# -------------------------------------------------------
try:
    wait.until(EC.presence_of_element_located((By.XPATH, ".//tbody/tr")))
    scrape_visible_rows()

    for _ in range(500):  # large safeguard limit
        if click_next_if_possible():
            scrape_visible_rows()
        else:
            break

finally:
    driver.quit()


# -------------------------------------------------------
# SAVE RESULTS
# -------------------------------------------------------
df = pd.DataFrame(all_rows)

if not df.empty:
    df = (
        df.drop_duplicates(subset=["href"], keep="first")
          .sort_values("id")
          .reset_index(drop=True)
          .drop(columns=["href", "title"])  # cleaned structure
    )

print(df.head())
df.to_csv("nist_frvendors_scraped.csv", index=False)
print(f"Saved {len(df)} entries to nist_frvendors_scraped.csv")