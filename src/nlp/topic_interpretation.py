"""
topic_interpretation.py
-----------------------
Produce human-readable summaries of each BERTopic topic for the
CS 4994 facial recognition legitimacy project.

This script:
    1. Loads the trained BERTopic model and topics.csv
    2. For each topic:
        - extracts top words
        - finds representative documents
        - identifies top companies associated with the topic
        - computes ethical / surveillance / market framing scores
    3. Saves results to CSV and JSON for use in your final report
       and dashboard.

Run from the project root as:

    python -m src.nlp.topic_interpretation
"""

import os
import json
import pandas as pd
from bertopic import BERTopic

from src.utils.logger import get_logger

# -------------------------------------------------------
# CONFIGURATION
# -------------------------------------------------------
TOPIC_DIR = os.path.join("analysis", "topics")
OUTPUT_DIR = os.path.join("analysis", "interpretations")

MODEL_DIR = os.path.join(TOPIC_DIR, "bertopic_model")
TOPIC_CSV = os.path.join(TOPIC_DIR, "topics.csv")

os.makedirs(OUTPUT_DIR, exist_ok=True)

logger = get_logger("TopicInterpretation")


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
    """Count how many times any of the keywords appear in the text."""
    t = text.lower()
    return sum(t.count(kw) for kw in keywords)


def _compute_framing_scores(texts: list[str]) -> dict:
    """
    Compute normalized framing scores for a list of documents
    belonging to the same topic.

    Returns a dict with keys:
        - ethical
        - surveillance
        - market
    and values in [0, 1] representing relative proportions.
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
        # No framing keywords detected; return zeros
        return {"ethical": 0.0, "surveillance": 0.0, "market": 0.0}

    return {
        "ethical": total_ethical / total,
        "surveillance": total_surv / total,
        "market": total_market / total,
    }


# -------------------------------------------------------
# LOAD MODEL AND DATA
# -------------------------------------------------------
def load_model_and_data():
    """Load BERTopic model and topics.csv from the selected version."""
    logger.info(f"Loading BERTopic model from {MODEL_DIR}")
    model = BERTopic.load(MODEL_DIR)

    logger.info(f"Loading topic assignments from {TOPIC_CSV}")
    df = pd.read_csv(TOPIC_CSV)

    return model, df


# -------------------------------------------------------
# MAIN TOPIC INTERPRETATION LOGIC
# -------------------------------------------------------
def interpret_topics(model: BERTopic, df: pd.DataFrame):
    """
    Build a list of topic summaries.

    For each topic (excluding -1 outliers), we compute:
        - top_words: list of (word, score)
        - top_companies: most frequent companies associated with topic
        - framing_scores: normalized ethical/surveillance/market scores
        - representative_docs: sample of representative documents
    """
    docs_by_topic = (
        df.groupby("topic")["text"]
        .apply(list)
        .to_dict()
    )

    companies_by_topic = (
        df.groupby("topic")["company"]
        .apply(list)
        .to_dict()
    )

    # All topics known by the model (excluding -1 outlier)
    topic_ids = [t for t in model.get_topics().keys() if t != -1]

    summaries = []

    for topic_id in topic_ids:
        logger.info(f"Interpreting topic {topic_id}...")

        # 1) Top words for this topic
        words_scores = model.get_topic(topic_id) or []
        top_words = [w for w, _ in words_scores]
        top_word_scores = words_scores

        # 2) Documents belonging to this topic in topics.csv
        topic_texts = docs_by_topic.get(topic_id, [])

        # 3) Compute framing scores based on all texts in this topic
        framing_scores = _compute_framing_scores(topic_texts)

        # 4) Top companies associated with this topic
        topic_companies = companies_by_topic.get(topic_id, [])
        if topic_companies:
            company_counts = (
                pd.Series(topic_companies)
                .value_counts()
                .head(10)
                .to_dict()
            )
        else:
            company_counts = {}

        # 5) Representative docs: try to use model's built-in helper if available
        representative_docs = []
        try:
            representative_docs = model.get_representative_docs(topic_id)
        except Exception:
            # Fallback: take up to 5 texts from docs_by_topic
            representative_docs = topic_texts[:5]

        summaries.append(
            {
                "topic_id": topic_id,
                "top_words": top_words,
                "top_word_scores": top_word_scores,
                "top_companies": company_counts,
                "framing_ethical": framing_scores["ethical"],
                "framing_surveillance": framing_scores["surveillance"],
                "framing_market": framing_scores["market"],
                "representative_docs": representative_docs,
            }
        )

    return summaries


def save_summaries(summaries: list[dict]):
    """
    Save topic summaries to JSON and CSV for both
    machine consumption (dashboard) and human reading (report).
    """
    json_path = os.path.join(OUTPUT_DIR, "topic_summaries.json")
    csv_path = os.path.join(OUTPUT_DIR, "topic_summaries.csv")

    logger.info(f"Saving topic summaries to {json_path} and {csv_path}")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summaries, f, indent=2, ensure_ascii=False)

    # Flatten for CSV: drop large text fields
    flat_rows = []
    for row in summaries:
        flat_rows.append(
            {
                "topic_id": row["topic_id"],
                "top_words": ", ".join(row["top_words"][:10]),
                "top_companies": "; ".join(
                    f"{k} ({v})" for k, v in row["top_companies"].items()
                ),
                "framing_ethical": row["framing_ethical"],
                "framing_surveillance": row["framing_surveillance"],
                "framing_market": row["framing_market"],
            }
        )

    pd.DataFrame(flat_rows).to_csv(csv_path, index=False)


# -------------------------------------------------------
# MAIN
# -------------------------------------------------------
if __name__ == "__main__":
    logger.info("Starting topic interpretation script...")
    model, df = load_model_and_data()
    summaries = interpret_topics(model, df)
    save_summaries(summaries)
    logger.info("Topic interpretation completed.")