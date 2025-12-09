# Facial Recognition Research

**SOC 4994 — Undergraduate Research, Virginia Tech**
**Instructor: Prof. Zhuofan Li**

This repository contains a full end-to-end research pipeline that **identifies**, **tracks**, and **analyzes** how global facial-recognition (FR) companies present, frame, and legitimize their technologies over time.

In line with SOC 4994 learning outcomes, the project integrates:

* Large-scale **web scraping** (NIST FRVT vendors + company homepages)
* Automated **Wayback Machine archival extraction** (full decades & yearly snapshots)
* Advanced **HTML cleaning & data processing**
* Topic modeling using **BERTopic**
* Corporate-level **ethical / surveillance / market framing analysis**
* **Per-company profiles**, topic interpretations, and interactive visual outputs
* Reproducible, open-source research design

---

# Repository Structure

```
facial_recognition_research/
│
├── data/
│   ├── nist/                         # NIST FRVT scraping
│   ├── vendors/                      # Vendor list + homepage cleaner
│   └── wayback/                      # Snapshot metadata + downloading
│
├── src/
│   ├── extraction/                   # HTML cleaning + text extraction
│   ├── nlp/                          # BERTopic, framing, profiles, visuals
│   └── utils/                        # Shared logger
│
├── analysis/
│   ├── text/                         # Cleaned text + stats
│   ├── topics/                       # BERTopic outputs + model
│   ├── framing/                      # Corporate framing results
│   ├── company_profiles/             # JSON profiles per company
│   ├── interpretations/              # Topic summaries
│   └── visuals/                      # HTML visualizations
│
└── notebooks/                        # Optional exploration notebooks
```

---

# Project Overview

This project answers a central research question:

### **How do companies that develop facial-recognition technology frame their products—ethically, economically, or through surveillance—and how has this framing evolved over time?**

To support this, the pipeline:

1. Scrapes **NIST FRVT vendor tables** to extract the universe of FR actors
2. Normalizes & cleans homepage URLs
3. Downloads **decade-wise + year-wise snapshots** from the Wayback Machine
4. Applies aggressive HTML cleaning & relevance filtering
5. Performs **BERTopic modeling** to uncover thematic structures
6. Computes **legitimacy framing metrics**
7. Generates **per-company JSON profiles**
8. Produces **interactive visualizations**

All scripts are reproducible and modular.

---

# 1. Web Scraping & Vendor Identification

## **NIST FRVT Scraper**

`data/nist/nist_frvendors_scraper.py`

* Navigates the FRVT Performance Summary Table using Selenium
* Extracts company names + country codes
* Handles pagination & table redraw events
* Outputs a clean CSV: `nist_frvendors_scraped.csv`

## **Homepage Cleaner**

`data/vendors/website_cleaner.py`

* Normalizes vendor URLs
* Resolves canonical/og/homepage links
* Detects popups, cookie banners, deep pages
* Outputs cleaned URLs + review flags

This dataset becomes the basis for Wayback snapshot retrieval.

---

# 2. Wayback Machine Snapshot Collection

### Scripts:

* `snapshot_info.py`
* `snapshot_downloader_decade.py`
* `snapshot_downloader_yearly.py`

The system:

* Queries CDX API for snapshot counts
* Records **first/last snapshot, snapshot_count**
* Downloads **homepage + up to 300 subpages** per decade/year
* Saves files under:

```
data/wayback/snapshots/<domain>/<decade or year>/...
```

* Produces download logs + updated metadata CSVs

This forms the **historical dataset** for all NLP modules.

---

# 3. HTML Cleaning & Text Extraction

### Key Module:

`src/extraction/html_cleaning_utils.py`

Removes:

* Scripts, styles, nav, footer, forms
* GDPR/cookie banners
* Modals, ads, newsletter popups
* Non-content HTML elements

Extracts only text from:

* `article`, `main`, `section`, `p`, `li`, `h1–h4`

Cleans boilerplate, Unicode, whitespace, and noise.

### Extraction Pipeline:

`src/extraction/text_extractor.py`

Applies:

* HTML → cleaned text
* **Language filtering** (English only)
* **Relevance filtering** using 50+ FR-specific keywords
* **Length filtering**
* **Duplicate removal**
* Generates:

  * `analysis/text/snapshot_texts.csv`
  * `cleaning_stats.csv` (machine readable)
  * `cleaning_stats.txt` (human readable report)

---

# 4. Topic Modeling (BERTopic)

### Pipeline:

`src/nlp/bertopic_pipeline.py`

Steps:

1. Loads cleaned dataset
2. Builds embedding + UMAP + HDBSCAN clusterer
3. Fits BERTopic with multiple representation models (MMR, KeyBERT)
4. Reduces topics to ~25 macro-themes
5. Computes:

   * `topics.csv` (document → topic assignment)
   * `topic_over_time.csv`
   * `topics_per_decade.csv`
6. Saves the model
7. Generates visualizations:

   * barchart
   * overview
   * topics-over-time

Outputs are stored in:

```
analysis/topics/
analysis/visuals/
```

---

# 5. Topic Interpretation

### Module:

`src/nlp/topic_interpretation.py`

For each topic:

* Extracts top words
* Lists top associated companies
* Computes **ethical/surveillance/market framing scores**
* Selects representative documents
* Saves:

  * `topic_summaries.json`
  * `topic_summaries.csv`

Used for your final report and dashboard narrative.

---

# 6. Corporate Framing Analysis

### Script:

`src/nlp/corporate_framing_analysis.py`

Computes framing intensities based on curated keyword dictionaries:

* **ETHICAL** – responsible AI, fairness, privacy, governance
* **SURVEILLANCE** – public safety, law enforcement, threat detection
* **MARKET** – enterprise-grade, compliance-ready, onboarding, trust

Outputs:

```
analysis/framing/company_framing_scores.csv
analysis/framing/company_framing_by_year.csv
```

These metrics plug directly into your SOC 4994 final deliverables.

---

# 7. Company Profile Generation

### Script:

`src/nlp/company_profiles.py`

Combines:

* Dominant topics
* Topic labels
* Framing scores
* Topic timeline (year-by-year)
* Sample documents

Outputs **one JSON per company**, e.g.:

```
analysis/company_profiles/www_accelrobotics_com.json
analysis/company_profiles/company_profiles_index.csv
```

These profiles power:

* Interactive dashboards
* Case studies
* Cross-company comparison sections in the final report

---

# 8. Visualizations

### Script:

`src/nlp/topic_visualizations.py`

Generates HTML plots:

* Topic hierarchy
* Topic heatmap
* Term-rank plot
* Topics-over-time

Saved under:

```
analysis/visuals/
```

These fulfill the SOC 4994 requirement for **interpretable NLP visualizations**.

---

# 9. Utilities

## Logger

`src/utils/logger.py`

Standardized logging across all modules with timestamps & multi-handler support.

---

# Running the Pipeline

### 1. Scrape NIST Vendors

```bash
python data/nist/nist_frvendors_scraper.py
```

### 2. Clean Homepages

```bash
python data/vendors/website_cleaner.py
```

### 3. Collect Snapshot Metadata

```bash
python data/wayback/snapshot_info.py
```

### 4. Download Snapshots (Decade or Year Mode)

```bash
python data/wayback/snapshot_downloader_decade.py --input ... --output ... --out-dir ...
# or
python data/wayback/snapshot_downloader_yearly.py --input ... --output ... --out-dir ...
```

### 5. Extract Cleaned Text

```bash
python src/extraction/text_extractor.py
```

### 6. Topic Modeling

```bash
python -m src.nlp.bertopic_pipeline
```

### 7. Corporate Framing Analysis

```bash
python -m src.nlp.corporate_framing_analysis
```

### 8. Topic Interpretation

```bash
python -m src.nlp.topic_interpretation
```

### 9. Company Profile Generation

```bash
python -m src.nlp.company_profiles
```

### 10. Visualization Generation

```bash
python -m src.nlp.topic_visualizations
```

---

# Ethical Considerations

This research adheres to SOC 4994's emphasis on evaluating the **ethical implications of automated data collection and analysis**:

* Only publicly available company web content is used
* Web requests are rate-limited (PoliteRequester)
* Archival analysis preserves historical transparency
* Framing analysis is interpretive, not evaluative
* Outputs should **not** be used to classify companies as “ethical” or “unethical”; rather, they reveal **patterns of corporate discourse**