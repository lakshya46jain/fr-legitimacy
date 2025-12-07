import os
import pandas as pd

from bertopic import BERTopic
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
MODEL_DIR = os.path.join(TOPIC_DIR, "bertopic_model")

os.makedirs(TOPIC_DIR, exist_ok=True)
os.makedirs(VIS_DIR, exist_ok=True)

logger = get_logger("BERTopicPipeline")


# -------------------------------------------------------
# LOAD DATA
# -------------------------------------------------------
def load_data():
    logger.info(f"Loading cleaned text from {INPUT_FILE}")
    df = pd.read_csv(INPUT_FILE)
    df = df.dropna(subset=["text"])

    # Optional: Filter out extremely short documents (< 50 chars)
    df = df[df["text"].str.len() > 50]

    logger.info(f"Loaded {len(df)} documents after filtering.")
    return df


# -------------------------------------------------------
# BUILD LOW-MEMORY MODELS
# -------------------------------------------------------
def build_low_memory_models():
    umap_model = UMAP(
        n_neighbors=10,
        n_components=5,
        min_dist=0.0,
        metric="cosine",
        low_memory=True,
        verbose=True
    )

    hdbscan_model = HDBSCAN(
        min_cluster_size=40,
        metric='euclidean',
        cluster_selection_method='eom',
        prediction_data=False      # Saves huge memory
    )

    vectorizer_model = CountVectorizer(
        stop_words="english",
        min_df=15,                  # Shrinks vocabulary, saves memory
    )

    return umap_model, hdbscan_model, vectorizer_model


# -------------------------------------------------------
# RUN BERTOPIC
# -------------------------------------------------------
def run_bertopic(docs):
    logger.info("Initializing low-memory BERTopic model...")

    umap_model, hdbscan_model, vectorizer_model = build_low_memory_models()

    topic_model = BERTopic(
        embedding_model="all-MiniLM-L6-v2",
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model,
        calculate_probabilities=False,   # Avoid massive NxK matrix (saves 4–10GB)
        nr_topics="auto",
        verbose=True
    )

    logger.info("Fitting topic model...")
    topics, _ = topic_model.fit_transform(docs)

    return topic_model, topics


# -------------------------------------------------------
# SAVE OUTPUTS
# -------------------------------------------------------
def save_topic_outputs(df, topics):
    df["topic"] = topics

    df.to_csv(TOPIC_CSV, index=False)
    logger.info(f"Saved topic assignments to {TOPIC_CSV}")

    # Extract year if possible
    df["year"] = df["path"].str.extract(r"(19|20)\d{2}").astype("float")

    topic_year = df.groupby(["year", "topic"]).size().reset_index(name="count")
    topic_year.to_csv(TOPIC_TIME_CSV, index=False)

    logger.info(f"Saved topic timeline to {TOPIC_TIME_CSV}")

    return df


# -------------------------------------------------------
# SAVE MODEL & BASIC VISUALS
# -------------------------------------------------------
def save_model_and_visuals(topic_model):
    topic_model.save(MODEL_DIR)
    logger.info(f"Saved BERTopic model to {MODEL_DIR}")

    fig = topic_model.visualize_barchart()
    fig.write_html(os.path.join(VIS_DIR, "topic_barchart.html"))

    fig = topic_model.visualize_topics()
    fig.write_html(os.path.join(VIS_DIR, "topics_overview.html"))

    logger.info(f"Saved basic visualizations to {VIS_DIR}")


# -------------------------------------------------------
# MAIN
# -------------------------------------------------------
if __name__ == "__main__":
    df = load_data()

    docs = df["text"].tolist()
    topic_model, topics = run_bertopic(docs)

    df = save_topic_outputs(df, topics)
    save_model_and_visuals(topic_model)

    logger.info("Low-memory BERTopic pipeline completed successfully.")