"""
corporate_framing_analysis.py
-----------------------------
Compute company-level framing scores for the CS 4994 project:

For each company, we estimate:
    - ethical framing intensity
    - surveillance framing intensity
    - market (enterprise/compliance) framing intensity

Optionally, we also compute framing over time (company-year).

Outputs are written to:
    analysis/framing/<version>/company_framing_scores.csv
    analysis/framing/<version>/company_framing_by_year.csv

Run from the project root:

    python -m src.nlp.corporate_framing_analysis
"""

import os
import pandas as pd

from src.utils.logger import get_logger

# -------------------------------------------------------
# CONFIGURATION
# -------------------------------------------------------
TOPIC_DIR = os.path.join("analysis", "topics")
OUTPUT_DIR = os.path.join("analysis", "framing")

TOPIC_CSV = os.path.join(TOPIC_DIR, "topics.csv")

os.makedirs(OUTPUT_DIR, exist_ok=True)

logger = get_logger("CorporateFramingAnalysis")


# -------------------------------------------------------
# LEGITIMACY FRAMING KEYWORDS
# -------------------------------------------------------
ETHICAL_KEYWORDS = [
    "responsible ai", "trustworthy ai", "ethical ai",
    "fairness", "transparency", "accountability", "explainability",
    "privacy-first", "privacy by design", "privacy by default",
    "bias mitigation", "risk management", "human rights",
    "data protection", "gdpr", "compliance", "regulation"
]

SURVEILLANCE_KEYWORDS = [
    "public safety", "threat detection", "situational awareness",
    "law enforcement", "crime prevention", "forensic",
    "suspect identification", "real-time monitoring",
    "watchlist", "border control", "security camera",
]

MARKET_KEYWORDS = [
    "enterprise-grade", "compliance-ready", "regulatory compliant",
    "industry standard", "scalable", "high availability",
    "frictionless onboarding", "seamless onboarding",
    "customer experience", "digital trust", "identity assurance",
    "kyc", "aml"
]


def _count_keyword_hits(text: str, keywords: list[str]) -> int:
    """Count occurrences of any keyword from the list in a given text."""
    t = text.lower()
    return sum(t.count(kw) for kw in keywords)


def _compute_scores_for_group(texts: list[str]) -> dict:
    """
    Compute normalized framing scores for a set of documents
    (e.g., all docs for a company, or company-year combination).
    """
    total_ethical = 0
    total_surv = 0
    total_market = 0

    for txt in texts:
        total_ethical += _count_keyword_hits(txt, ETHICAL_KEYWORDS)
        total_surv += _count_keyword_hits(txt, SURVEILLANCE_KEYWORDS)
        total_market += _count_keyword_hits(txt, MARKET_KEYWORDS)

    total = total_ethical + total_surv + total_market
    if total == 0:
        return {"ethical": 0.0, "surveillance": 0.0, "market": 0.0}

    return {
        "ethical": total_ethical / total,
        "surveillance": total_surv / total,
        "market": total_market / total,
    }


# -------------------------------------------------------
# MAIN ANALYSIS FUNCTIONS
# -------------------------------------------------------
def load_topics_df() -> pd.DataFrame:
    """
    Load topics.csv created by the BERTopic pipeline.

    Expected columns:
        - company
        - text
        - year (optional but recommended)
        - topic
    """
    logger.info(f"Loading topic assignments from {TOPIC_CSV}")
    df = pd.read_csv(TOPIC_CSV)

    # Drop rows without company or text
    df = df.dropna(subset=["company", "text"])
    return df


def compute_company_framing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute framing scores for each company across all years.

    Returns a DataFrame with:
        - company
        - framing_ethical
        - framing_surveillance
        - framing_market
        - num_docs
    """
    logger.info("Computing overall framing per company...")

    rows = []
    for company, group in df.groupby("company"):
        texts = group["text"].tolist()
        scores = _compute_scores_for_group(texts)

        rows.append(
            {
                "company": company,
                "framing_ethical": scores["ethical"],
                "framing_surveillance": scores["surveillance"],
                "framing_market": scores["market"],
                "num_docs": len(texts),
            }
        )

    company_df = pd.DataFrame(rows)
    return company_df


def compute_company_framing_by_year(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute framing scores for each company-year combination,
    if a 'year' column exists.

    Returns a DataFrame with:
        - company
        - year
        - framing_ethical
        - framing_surveillance
        - framing_market
        - num_docs
    """
    if "year" not in df.columns:
        logger.warning("No 'year' column found; skipping company-by-year framing.")
        return pd.DataFrame()

    logger.info("Computing framing per company-year...")

    # Drop rows without year
    df_year = df.dropna(subset=["year"])
    rows = []

    for (company, year), group in df_year.groupby(["company", "year"]):
        texts = group["text"].tolist()
        scores = _compute_scores_for_group(texts)

        rows.append(
            {
                "company": company,
                "year": year,
                "framing_ethical": scores["ethical"],
                "framing_surveillance": scores["surveillance"],
                "framing_market": scores["market"],
                "num_docs": len(texts),
            }
        )

    return pd.DataFrame(rows)


def save_results(company_df: pd.DataFrame, company_year_df: pd.DataFrame):
    """Save the framing DataFrames to CSV in the framing output directory."""
    out_company = os.path.join(OUTPUT_DIR, "company_framing_scores.csv")
    logger.info(f"Saving company-level framing scores to {out_company}")
    company_df.to_csv(out_company, index=False)

    if not company_year_df.empty:
        out_company_year = os.path.join(OUTPUT_DIR, "company_framing_by_year.csv")
        logger.info(f"Saving company-by-year framing scores to {out_company_year}")
        company_year_df.to_csv(out_company_year, index=False)
    else:
        logger.info("No company-year framing scores to save.")


# -------------------------------------------------------
# MAIN
# -------------------------------------------------------
if __name__ == "__main__":
    logger.info("Starting corporate framing analysis...")
    df_topics = load_topics_df()
    company_df = compute_company_framing(df_topics)
    company_year_df = compute_company_framing_by_year(df_topics)
    save_results(company_df, company_year_df)
    logger.info("Corporate framing analysis completed.")