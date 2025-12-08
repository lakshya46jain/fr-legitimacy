"""
bertopic_pipeline.py
--------------------
End-to-end BERTopic pipeline for analyzing how companies working
on facial recognition and adjacent technologies frame and legitimize
their products over time.

Compatible with Python 3.13 and all BERTopic versions.
Evaluation module removed (not available in many versions).
A safe diversity proxy is used instead.

Run:
    python -m src.nlp.bertopic_pipeline
"""

import os
import pandas as pd

from bertopic import BERTopic
from bertopic.representation import (
    MaximalMarginalRelevance,
    KeyBERTInspired,
    PartOfSpeech,
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
# FACIAL RECOGNITION KEYWORDS
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
    "ai-powered", "machine learning model", "analytics platform",

    "access control", "entry management", "smart access",
    "contactless access", "touchless", "visitor management",
    "credentials", "credentialing", "door controller",
    "physical security", "biometric access",
    "time and attendance", "workforce management",

    "video surveillance", "cctv", "public safety",
    "real-time monitoring", "tracking system",
    "watchlist", "forensic", "crime prevention",

    "gdpr", "privacy", "responsible ai", "ai ethics",
    "fairness", "transparency", "bias mitigation",
    "compliance", "regulation", "ethical ai",
]


def contains_fr_keywords(text: str) -> bool:
    """Return True if text contains any FR keyword."""
    t = str(text).lower()
    return any(k in t for k in FR_KEYWORDS)


# -------------------------------------------------------
# LOAD & CLEAN DATA
# -------------------------------------------------------
def load_data() -> pd.DataFrame:
    """Load, filter, and dedupe FR-related documents."""
    logger.info(f"Loading cleaned text from {INPUT_FILE}")
    df = pd.read_csv(INPUT_FILE)

    df = df.dropna(subset=["text"])
    df = df[df["text"].str.split().str.len() >= 40]
    df = df[df["text"].apply(contains_fr_keywords)]

    if "lang" in df.columns:
        df = df[df["lang"] == "en"]

    df = df.drop_duplicates(subset=["company", "decade", "text"]).reset_index(drop=True)

    logger.info(f"Loaded {len(df)} documents after filtering.")
    return df


def add_year_column(df: pd.DataFrame) -> pd.DataFrame:
    """Infer year from `path` column."""
    logger.info("Extracting year from path...")

    if "path" not in df.columns:
        logger.warning("No 'path' column found; setting year=None")
        df["year"] = None
        return df

    df["year"] = (
        df["path"].astype(str).str.extract(r"(19|20)\d{2}")
    ).astype(float)

    logger.info(f"Extracted year for {df['year'].notna().sum()} documents.")
    return df


# -------------------------------------------------------
# MODEL COMPONENTS
# -------------------------------------------------------
def build_low_memory_models():
    """Construct UMAP, HDBSCAN, CountVectorizer."""
    umap_model = UMAP(
        n_neighbors=15,
        n_components=5,
        min_dist=0.0,
        metric="cosine",
        low_memory=True,
        verbose=True,
    )

    hdbscan_model = HDBSCAN(
        min_cluster_size=20,
        min_samples=10,
        metric="euclidean",
        cluster_selection_method="eom",
        prediction_data=False,
    )

    vectorizer_model = CountVectorizer(
        stop_words="english",
        ngram_range=(1, 3),
        min_df=3,
        max_features=30000,
    )

    return umap_model, hdbscan_model, vectorizer_model


def build_representation_model():
    """Combine diverse representation models for interpretability."""
    mmr = MaximalMarginalRelevance(diversity=0.3)
    keybert = KeyBERTInspired()

    return {"MMR": mmr, "KeyBERT": keybert}


# -------------------------------------------------------
# RUN BERTOPIC
# -------------------------------------------------------
def run_bertopic(docs, timestamps):
    """
    Fit BERTopic, reduce topics, compute topics-over-time,
    and compute a diversity metric.
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
        nr_topics="auto",
        verbose=True,
    )

    logger.info("Fitting model...")
    topics, _ = topic_model.fit_transform(docs)

    # ----------------------
    # Reduce topics
    # ----------------------
    try:
        logger.info("Reducing to ~25 macro-topics...")
        topic_model, topics = topic_model.reduce_topics(
            docs, topics, nr_topics=25
        )
    except Exception as e:
        logger.warning(f"Topic reduction failed: {e}")

    # ----------------------
    # Topics-over-time
    # ----------------------
    topics_over_time = None
    if timestamps and any(t is not None for t in timestamps):
        try:
            logger.info("Computing topics-over-time...")
            topics_over_time = topic_model.topics_over_time(
                docs, timestamps=timestamps
            )
        except Exception as e:
            logger.warning(f"topics_over_time failed: {e}")

    # ----------------------
    # Diversity Proxy
    # ----------------------
    eval_scores = {"coherence": None, "diversity": None}

    try:
        logger.info("Computing diversity proxy score...")
        unique_words = set()
        for topic_id, word_scores in topic_model.get_topics().items():
            if topic_id == -1:
                continue
            for w, _ in word_scores:
                unique_words.add(w)

        diversity_score = len(unique_words)
        eval_scores["diversity"] = diversity_score

        logger.info(f"Diversity score: {diversity_score}")

    except Exception as e:
        logger.warning(f"Diversity proxy failed: {e}")

    return topic_model, topics, topics_over_time, eval_scores


# -------------------------------------------------------
# SAVE OUTPUTS
# -------------------------------------------------------
def save_topic_outputs(df, topics, topics_over_time):
    """Save topics.csv and topic_over_time.csv."""
    df = df.copy()
    df["topic"] = topics

    df.to_csv(TOPIC_CSV, index=False)
    logger.info(f"Saved topic assignments → {TOPIC_CSV}")

    if topics_over_time is not None:
        topics_over_time.to_csv(TOPIC_TIME_CSV, index=False)
        logger.info(f"Saved topic-over-time → {TOPIC_TIME_CSV}")

    return df


def save_topics_per_decade(topic_model, df):
    """Optional: topic distribution by decade."""
    if "decade" not in df.columns:
        return

    try:
        logger.info("Computing topics-per-decade...")
        topics_per_decade = topic_model.topics_per_class(
            df["text"].tolist(),
            classes=df["decade"].tolist(),
        )
        topics_per_decade.to_csv(TOPIC_PER_DECADE_CSV, index=False)
        logger.info(f"Saved → {TOPIC_PER_DECADE_CSV}")
    except Exception as e:
        logger.warning(f"topics_per_class failed: {e}")


def save_model_and_visuals(topic_model, topics_over_time):
    """Save BERTopic model + HTML visualizations."""
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
    logger.info("Starting BERTopic pipeline...")

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