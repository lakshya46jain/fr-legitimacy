"""
company_profiles.py
-------------------
Generate per-company profiles that combine:

    - dominant topics
    - short topic descriptions (based on top words)
    - framing scores (ethical, surveillance, market)
    - simple temporal information (topic counts by year)

These profiles can be used directly in:
    - an interactive dashboard
    - the final SOC 4994 report (case studies)
    - qualitative comparison of companies

Run from the project root:

    python -m src.nlp.company_profiles
"""

import os
import re
import json
import pandas as pd
from bertopic import BERTopic

from src.utils.logger import get_logger

# -------------------------------------------------------
# CONFIGURATION
# -------------------------------------------------------
TOPIC_DIR = os.path.join("analysis", "topics")
FRAMING_DIR = os.path.join("analysis", "framing")
OUTPUT_DIR = os.path.join("analysis", "company_profiles")

MODEL_DIR = os.path.join(TOPIC_DIR, "bertopic_model")
TOPIC_CSV = os.path.join(TOPIC_DIR, "topics.csv")
FRAMING_CSV = os.path.join(FRAMING_DIR, "company_framing_scores.csv")

os.makedirs(OUTPUT_DIR, exist_ok=True)

logger = get_logger("CompanyProfiles")


# -------------------------------------------------------
# HELPERS
# -------------------------------------------------------
def slugify(name: str) -> str:
    """
    Convert a company name into a safe filename:
    - lowercase
    - replace non-alphanumeric chars with underscore
    - strip leading/trailing underscores
    """
    name = name.lower()
    name = re.sub(r"[^a-z0-9]+", "_", name)
    return name.strip("_") or "unknown_company"


def load_model_and_data():
    """
    Load BERTopic model, topics.csv, and company framing scores
    (if available).
    """
    logger.info(f"Loading BERTopic model from {MODEL_DIR}")
    model = BERTopic.load(MODEL_DIR)

    logger.info(f"Loading topic assignments from {TOPIC_CSV}")
    df_topics = pd.read_csv(TOPIC_CSV)

    # Framing scores are optional: script will still run without them
    framing_df = None
    if os.path.exists(FRAMING_CSV):
        logger.info(f"Loading company framing scores from {FRAMING_CSV}")
        framing_df = pd.read_csv(FRAMING_CSV)
    else:
        logger.warning("No framing scores found; company profiles will omit framing.")

    return model, df_topics, framing_df


def get_topic_label(model: BERTopic, topic_id: int, top_n: int = 5) -> str:
    """
    Create a short human-readable label for a topic by joining
    its top N words.
    """
    words_scores = model.get_topic(topic_id) or []
    top_words = [w for w, _ in words_scores[:top_n]]
    return ", ".join(top_words)


def build_company_profile(
    company: str,
    company_df: pd.DataFrame,
    model: BERTopic,
    framing_df: pd.DataFrame | None,
) -> dict:
    """
    Build a structured profile for a single company.

    Includes:
        - dominant_topics: top topics by document count
        - topic_descriptions: mapping topic_id → comma-separated top words
        - framing_scores: from framing_df if available
        - topic_timeline: simple topic counts by year
        - sample_docs: a few example texts
    """
    # Basic info
    profile = {"company": company}

    # 1) Dominant topics for this company
    topic_counts = (
        company_df["topic"]
        .value_counts()
        .head(5)
        .to_dict()
    )
    profile["dominant_topics"] = list(topic_counts.keys())
    profile["topic_counts"] = topic_counts

    # 2) Topic descriptions (short labels)
    topic_descriptions = {}
    for topic_id in topic_counts.keys():
        if topic_id == -1:
            continue  # skip outlier topic
        topic_descriptions[str(topic_id)] = get_topic_label(model, int(topic_id))

    profile["topic_descriptions"] = topic_descriptions

    # 3) Framing scores (overall)
    framing_scores = None
    if framing_df is not None:
        row = framing_df[framing_df["company"] == company]
        if not row.empty:
            row = row.iloc[0]
            framing_scores = {
                "framing_ethical": float(row["framing_ethical"]),
                "framing_surveillance": float(row["framing_surveillance"]),
                "framing_market": float(row["framing_market"]),
                "num_docs": int(row["num_docs"]),
            }
    profile["framing_scores"] = framing_scores

    # 4) Simple topic timeline by year (if year exists)
    timeline = {}
    if "year" in company_df.columns:
        df_year = company_df.dropna(subset=["year"])
        if not df_year.empty:
            grouped = (
                df_year.groupby(["year", "topic"])
                .size()
                .reset_index(name="count")
            )
            for _, row in grouped.iterrows():
                year = str(int(row["year"]))
                topic_id = int(row["topic"])
                count = int(row["count"])
                year_dict = timeline.setdefault(year, {})
                year_dict[str(topic_id)] = count

    profile["topic_timeline"] = timeline

    # 5) Sample documents (lightly truncated for readability)
    sample_texts = (
        company_df["text"]
        .dropna()
        .head(5)
        .tolist()
    )
    profile["sample_docs"] = sample_texts

    return profile


def save_company_profile(profile: dict):
    """Save a single company profile to a JSON file."""
    company = profile.get("company", "unknown")
    slug = slugify(company)
    out_path = os.path.join(OUTPUT_DIR, f"{slug}.json")

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(profile, f, indent=2, ensure_ascii=False)

    return out_path


# -------------------------------------------------------
# MAIN
# -------------------------------------------------------
if __name__ == "__main__":
    logger.info("Starting company profile generation...")

    model, df_topics, framing_df = load_model_and_data()

    # Drop rows without company or topic
    df_topics = df_topics.dropna(subset=["company", "topic", "text"])

    # Keep an index of all companies and profile paths
    index_rows = []

    for company, group in df_topics.groupby("company"):
        logger.info(f"Building profile for company: {company}")
        profile = build_company_profile(company, group, model, framing_df)
        path = save_company_profile(profile)

        index_rows.append(
            {
                "company": company,
                "profile_path": path,
                "num_docs": len(group),
            }
        )

    # Save a master index file for convenience
    index_df = pd.DataFrame(index_rows)
    index_path = os.path.join(OUTPUT_DIR, "company_profiles_index.csv")
    index_df.to_csv(index_path, index=False)

    logger.info(f"Saved company profile index to {index_path}")
    logger.info("Company profile generation completed.")