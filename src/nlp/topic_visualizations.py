import os
import pandas as pd
from bertopic import BERTopic
from src.utils.logger import get_logger

# -------------------------------------------------------
# PATHS
# -------------------------------------------------------
MODEL_DIR = "analysis/topics/bertopic_model"
TOPIC_CSV = "analysis/topics/topics.csv"
VIS_DIR = "analysis/visuals"

os.makedirs(VIS_DIR, exist_ok=True)

logger = get_logger("TopicVisualizations")


# -------------------------------------------------------
# LOAD MODEL AND DATA
# -------------------------------------------------------
def load_model_and_data():
    logger.info(f"Loading BERTopic model from {MODEL_DIR}")
    model = BERTopic.load(MODEL_DIR)

    logger.info(f"Loading topic assignments from {TOPIC_CSV}")
    df = pd.read_csv(TOPIC_CSV)

    return model, df


# -------------------------------------------------------
# GENERATE VISUALIZATIONS
# -------------------------------------------------------
def generate_visuals(model, df):
    logger.info("Generating BERTopic visualizations...")

    fig = model.visualize_hierarchy()
    fig.write_html(os.path.join(VIS_DIR, "topic_hierarchy.html"))

    fig = model.visualize_heatmap()
    fig.write_html(os.path.join(VIS_DIR, "topic_heatmap.html"))

    fig = model.visualize_term_rank()
    fig.write_html(os.path.join(VIS_DIR, "topic_term_rank.html"))

    if "year" in df.columns and df["year"].notna().sum() > 0:
        fig = model.visualize_topics_over_time(df)
        fig.write_html(os.path.join(VIS_DIR, "topic_over_time.html"))
        logger.info("Generated topic-over-time visualization.")

    logger.info("All visualizations saved.")


# -------------------------------------------------------
# MAIN
# -------------------------------------------------------
if __name__ == "__main__":
    model, df = load_model_and_data()
    generate_visuals(model, df)
    logger.info("Visualization generation completed.")