"""
bertopic_pipeline.py
--------------------------------------
End-to-end BERTopic pipeline with improved topic granularity.

This updated pipeline:
    - Produces more topics (instead of collapsing into Topic 0 & 1)
    - Distributes documents more evenly across clusters
    - Improves UMAP and HDBSCAN sensitivity
    - Uses a fixed nr_topics target (40) for clearer interpretability

Run:
    python -m src.nlp.bertopic_pipeline
"""

import os
import pandas as pd

from bertopic import BERTopic
from bertopic.representation import (
    MaximalMarginalRelevance,
    KeyBERTInspired,
)
from umap import UMAP
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer

from src.utils.logger import get_logger


# -------------------------------------------------------
# PATHS
# -------------------------------------------------------
INPUT_FILE = "./analysis/text/snapshot_texts.csv"

TOPIC_DIR = "./analysis/topics"
VIS_DIR = "./analysis/visuals"

TOPIC_CSV = os.path.join(TOPIC_DIR, "topics.csv")
TOPIC_TIME_CSV = os.path.join(TOPIC_DIR, "topic_over_time.csv")
TOPIC_PER_DECADE_CSV = os.path.join(TOPIC_DIR, "topics_per_decade.csv")
MODEL_DIR = os.path.join(TOPIC_DIR, "bertopic_model")

os.makedirs(TOPIC_DIR, exist_ok=True)
os.makedirs(VIS_DIR, exist_ok=True)

logger = get_logger("BERTopicPipeline")


# -------------------------------------------------------
# FR KEYWORDS
# -------------------------------------------------------
FR_KEYWORDS = [
    "face recognition", "facial recognition", "face biometric",
    "biometric identification", "biometric authentication",
    "face authentication", "facial authentication",
    "liveness detection", "spoof detection", "anti-spoofing",
    "face capture", "face matching", "face detection",

    "identity verification", "id verification", "idv",
    "kyc", "aml", "identity assurance",
    "identity management", "identity platform",
    "fraud prevention", "secure authentication",
    "digital identity", "onboarding", "verification workflow",

    "computer vision", "deep learning", "neural network",
    "vision ai", "image recognition", "pattern recognition",
    "object detection", "video analytics",

    "access control", "visitor management", "workforce management",

    "cctv", "public safety", "video surveillance",

    "gdpr", "privacy", "responsible ai", "ai ethics"
]


def contains_fr_keywords(text: str) -> bool:
    """Return True if text contains any FR keyword."""
    t = str(text).lower()
    return any(k in t for k in FR_KEYWORDS)


# -------------------------------------------------------
# LOAD & CLEAN DATA
# -------------------------------------------------------
def load_data() -> pd.DataFrame:
    """
    Load, filter, and dedupe cleaned FR-related documents.

    NOTE: Additional optional cleaning is provided below
    (commented out) for cases where too many irrelevant
    pages create large noisy topics.
    """
    logger.info(f"Loading cleaned text from {INPUT_FILE}")
    df = pd.read_csv(INPUT_FILE)

    # drop missing text
    df = df.dropna(subset=["text"])

    # remove extremely short documents
    df = df[df["text"].str.split().str.len() >= 40]

    # keep only FR-relevant docs
    df = df[df["text"].apply(contains_fr_keywords)]

    # limit to English (if available)
    if "lang" in df.columns:
        df = df[df["lang"] == "en"]

    # remove duplicates across company+decade+text
    df = df.drop_duplicates(subset=["company", "decade", "text"]).reset_index(drop=True)

    logger.info(f"Loaded {len(df)} documents after filtering.")
    return df


def add_year_column(df: pd.DataFrame) -> pd.DataFrame:
    """Infer year from path field."""
    logger.info("Extracting year from path...")

    if "path" not in df.columns:
        df["year"] = None
        return df

    df["year"] = df["path"].astype(str).str.extract(r"(19|20)\d{2}").astype(float)
    logger.info(f"Extracted year for {df['year'].notna().sum()} documents.")
    return df


# -------------------------------------------------------
# IMPROVED MODEL COMPONENTS (for more granular topics)
# -------------------------------------------------------
def build_low_memory_models():
    """
    Build new UMAP + HDBSCAN + Vectorizer models
    that produce *more topics* and reduce Topic 0/1 dominance.
    """

    # ----------------------------
    # UMAP: smaller neighborhoods = more fine-grained clusters
    # ----------------------------
    umap_model = UMAP(
        n_neighbors=8,         # ↓ from 15
        n_components=5,
        min_dist=0.05,         # slightly spread topics
        metric="cosine",
        low_memory=True,
        verbose=True,
    )

    # ----------------------------
    # HDBSCAN: more sensitive clustering
    # ----------------------------
    hdbscan_model = HDBSCAN(
        min_cluster_size=10,   # ↓ from 20
        min_samples=3,         # ↓ from 10
        metric="euclidean",
        cluster_selection_method="eom",
        prediction_data=False,
    )

    # ----------------------------
    # Vectorizer: reduce noise, improve separation
    # ----------------------------
    vectorizer_model = CountVectorizer(
        stop_words="english",
        ngram_range=(1, 2),    # narrowed to avoid massive trigrams
        min_df=3,
        max_features=30000,
    )

    return umap_model, hdbscan_model, vectorizer_model


def build_representation_model():
    """Representation for improving topic labels."""
    mmr = MaximalMarginalRelevance(diversity=0.3)
    keybert = KeyBERTInspired()
    return {"MMR": mmr, "KeyBERT": keybert}


# -------------------------------------------------------
# RUN BERTOPIC WITH IMPROVED GRANULARITY
# -------------------------------------------------------
def run_bertopic(docs, timestamps):
    """
    Fit BERTopic with new, more fine-grained clustering settings.
    Distribute docs across ~40 topics rather than collapsing.
    """

    logger.info("Initializing BERTopic...")

    umap_model, hdbscan_model, vectorizer_model = build_low_memory_models()
    representation_model = build_representation_model()

    topic_model = BERTopic(
        embedding_model="all-MiniLM-L6-v2",
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model,
        representation_model=representation_model,
        calculate_probabilities=False,

        # IMPORTANT: force more topics
        nr_topics=40,

        verbose=True,
    )

    logger.info("Fitting model with improved settings...")
    topics, _ = topic_model.fit_transform(docs)

    # ----------------------
    # Topics-over-time (if possible)
    # ----------------------
    topics_over_time = None
    if timestamps and any(t is not None for t in timestamps):
        try:
            topics_over_time = topic_model.topics_over_time(docs, timestamps=timestamps)
        except Exception as e:
            logger.warning(f"topics_over_time failed: {e}")

    # ----------------------
    # Diversity (optional)
    # ----------------------
    eval_scores = {"diversity": None}

    try:
        unique_words = {
            w for tid, words in topic_model.get_topics().items()
            if tid != -1
            for (w, _) in words
        }
        eval_scores["diversity"] = len(unique_words)
    except Exception:
        pass

    return topic_model, topics, topics_over_time, eval_scores


# -------------------------------------------------------
# SAVE OUTPUTS
# -------------------------------------------------------
def save_topic_outputs(df, topics, topics_over_time):
    df = df.copy()
    df["topic"] = topics

    df.to_csv(TOPIC_CSV, index=False)
    logger.info(f"Saved topic assignments → {TOPIC_CSV}")

    if topics_over_time is not None:
        topics_over_time.to_csv(TOPIC_TIME_CSV, index=False)
        logger.info(f"Saved topic-over-time → {TOPIC_TIME_CSV}")

    return df


def save_topics_per_decade(topic_model, df):
    if "decade" not in df.columns:
        return

    try:
        topics_per_decade = topic_model.topics_per_class(
            df["text"].tolist(),
            classes=df["decade"].tolist()
        )
        topics_per_decade.to_csv(TOPIC_PER_DECADE_CSV, index=False)
    except Exception as e:
        logger.warning(f"topics_per_class failed: {e}")


def save_model_and_visuals(topic_model, topics_over_time):
    topic_model.save(MODEL_DIR)
    logger.info(f"Saved model → {MODEL_DIR}")

    try:
        topic_model.visualize_barchart().write_html(
            os.path.join(VIS_DIR, "topic_barchart.html")
        )
    except Exception:
        pass

    try:
        topic_model.visualize_topics().write_html(
            os.path.join(VIS_DIR, "topics_overview.html")
        )
    except Exception:
        pass

    if topics_over_time is not None:
        try:
            topic_model.visualize_topics_over_time(topics_over_time).write_html(
                os.path.join(VIS_DIR, "topics_over_time.html")
            )
        except Exception:
            pass


# -------------------------------------------------------
# MAIN
# -------------------------------------------------------
if __name__ == "__main__":
    logger.info("Starting BERTopic Pipeline...")

    df = load_data()
    df = add_year_column(df)

    docs = df["text"].tolist()
    timestamps = df["year"].tolist()

    topic_model, topics, topics_over_time, eval_scores = run_bertopic(docs, timestamps)

    df = save_topic_outputs(df, topics, topics_over_time)
    save_topics_per_decade(topic_model, df)
    save_model_and_visuals(topic_model, topics_over_time)

    logger.info(
        f"Pipeline complete. Diversity={eval_scores.get('diversity')}"
    )