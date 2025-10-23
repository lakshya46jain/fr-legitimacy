"""
Scraper for NIST FRVT performance summary table.
- Visits the NIST FRVT webpage
- Extracts company titles like "QazSmartVision.AI (KZ)"
- Splits them into 'name' and 'country'
- Navigates through all paginated pages (or selects 'All' if available)
- Saves results to a CSV file
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

# ---------------- Config ---------------- #
URL = "https://pages.nist.gov/frvt/html/frvt11.html#_FRTE_1:1_Performance_Summary_Table"
BASE_URL = "https://face.nist.gov/frte/reportcards/11/"

def parse_title(text: str):
    """
    Parse the title attribute text into name and country.
    Example: "QazSmartVision.AI (KZ)" -> ("QazSmartVision.AI", "KZ")
    """
    match = re.match(r"^(.*?)\s*\(([^)]+)\)\s*$", text or "")
    if match:
        return match.group(1).strip(), match.group(2).strip()
    return text or "", ""  # fallback

def iso_to_country(iso_code: str) -> str:
    try:
        country = pycountry.countries.get(alpha_2=iso_code.upper())
        return country.name if country else iso_code
    except Exception:
        return iso_code

# ---------------- Driver Setup ---------------- #
chrome_options = Options()
# chrome_options.add_argument("--headless=new")  # uncomment for headless
driver = webdriver.Chrome(options=chrome_options)
wait = WebDriverWait(driver, 15)

driver.get(URL)

# ---------------- Utilities ---------------- #
def find_frvt_table():
    """
    Prefer the table right after the FRTE anchor if present; otherwise
    fall back to the first table that contains reportcard links.
    """
    # Wait for the page to finish initial load (defensive)
    WebDriverWait(driver, 10).until(
        lambda d: d.execute_script("return document.readyState") == "complete"
    )

    # Try the anchor-based approach first
    try:
        anchor = WebDriverWait(driver, 5).until(EC.presence_of_element_located((
            By.XPATH, "//*[@id='_FRTE_1:1_Performance_Summary_Table' or @name='_FRTE_1:1_Performance_Summary_Table']"
        )))
        table = anchor.find_element(By.XPATH, "following::table[1]")
    except TimeoutException:
        # Fallback: find the table that actually has the links we want
        table = WebDriverWait(driver, 10).until(EC.presence_of_element_located((
            By.XPATH, f"//table[.//a[starts-with(@href, '{BASE_URL}')]]"
        )))

    wrapper = table.find_element(By.XPATH, "ancestor::div[contains(@class,'dataTables_wrapper')]")
    return table, wrapper

table, wrapper = find_frvt_table()

# Try to show all rows to avoid pagination if possible
try:
    length_select = wrapper.find_element(By.XPATH, ".//select[contains(@name,'_length')]")
    sel = Select(length_select)
    options = [o.text.strip() for o in sel.options]
    if any("All" in o for o in options):
        sel.select_by_visible_text(next(o for o in options if "All" in o))
    else:
        # pick the largest numeric option
        nums = [int(o) for o in options if o.isdigit()]
        if nums:
            sel.select_by_visible_text(str(max(nums)))
    # wait briefly for table to redraw
    time.sleep(0.3)
except Exception:
    pass  # no length control; we'll paginate

all_rows = []
seen_hrefs = set()
sequential_id = 1  # running order index

def scrape_visible_rows():
    """
    Scrape visible tbody rows of the FRVT table.
    Keep exactly one reportcard link per row that starts with BASE_URL.
    """
    global sequential_id
    rows = table.find_elements(By.XPATH, ".//tbody/tr")
    for r in rows:
        if not r.is_displayed():
            continue  # DataTables keeps off-page rows hidden
        links = r.find_elements(By.XPATH, f".//a[starts-with(@href, '{BASE_URL}') and @title]")
        if not links:
            continue
        a = links[0]  # first matching link in this row
        title = (a.get_attribute("title") or "").strip()
        href  = (a.get_attribute("href") or "").strip()
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
            "year": "", # need to populate manually
            "status": "", # need to populate manually
            "bio": "",  # placeholder for future use
            "hist": "",  # placeholder for future use
            "org": "",  # placeholder for future use
            "media": "",  # placeholder for future use
            "social": "",  # placeholder for future use
            "gov": "",  # placeholder for future use
            "link": "",  # need to populate manually
            "text": "",  # need to populate manually
        })
        sequential_id += 1

def click_next_if_possible():
    """
    Click the 'Next' button within THIS table's paginator.
    Returns True if we advanced, False if no further pages.
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

    # Use staleness of first visible row to detect page change
    visible_rows = [tr for tr in table.find_elements(By.XPATH, ".//tbody/tr") if tr.is_displayed()]
    anchor_row = visible_rows[0] if visible_rows else None
    next_btn.click()
    if anchor_row:
        try:
            wait.until(EC.staleness_of(anchor_row))
        except TimeoutException:
            return False
    time.sleep(0.2)
    return True

# ---------------- Main Scraping ---------------- #
try:
    # Ensure table is present
    wait.until(EC.presence_of_element_located((By.XPATH, ".//tbody/tr")))
    scrape_visible_rows()

    # If not all rows are visible, paginate through this specific table
    paged = False
    for _ in range(500):  # generous cap
        if click_next_if_possible():
            paged = True
            scrape_visible_rows()
        else:
            break

finally:
    driver.quit()

# ---------------- Save Results ---------------- #
df = pd.DataFrame(all_rows)
if not df.empty:
    df = df.drop_duplicates(subset=["href"], keep="first").sort_values("id").reset_index(drop=True)
    df = df.drop(columns=["href", "title"])

print(df.head())
df.to_csv("nist_frvendors_scraped.csv", index=False)
print(f"Saved {len(df)} entries to nist_frvendors_scraped.csv")