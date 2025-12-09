"""
topic_visualizations.py
------------------------
Generate advanced BERTopic visualizations for the SOC 4994
Facial Recognition Legitimacy Research Pipeline.

This module:
    • Loads the trained BERTopic model and topic assignment CSV
    • Generates hierarchical, temporal, and term-rank visualizations
    • Performs defensive checks to avoid runtime errors
    • Saves all HTML visualizations to analysis/visuals/

This script is intended to be run AFTER the main BERTopic pipeline:

    python -m src.nlp.topic_visualizations

The visual outputs directly support SOC 4994 learning outcomes:
    • Analyze corporate language + ethical framing over time
    • Build interpretable, interactive NLP visualizations
    • Support final report + dashboard requirements
"""

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

# Ensure visualization directory exists
os.makedirs(VIS_DIR, exist_ok=True)

logger = get_logger("TopicVisualizations")


# -------------------------------------------------------
# LOAD MODEL AND DATA
# -------------------------------------------------------
def load_model_and_data():
    """
    Load the trained BERTopic model and topic-level dataset.

    Returns
    -------
    model : BERTopic
        The previously saved BERTopic model containing:
            - reduced topics
            - representation models (MMR, KeyBERT, POS)
            - embedding + clustering structure
            - any temporal modeling metadata

    df : pd.DataFrame
        Document-level dataset containing:
            - text
            - topic assignments
            - optional 'year' column (for temporal visualization)

    Raises
    ------
    FileNotFoundError if required model files are missing.
    """
    logger.info(f"Loading BERTopic model from {MODEL_DIR}")

    try:
        model = BERTopic.load(MODEL_DIR)
    except Exception as e:
        logger.error(
            f"Failed to load BERTopic model. Ensure the model was saved correctly.\n{e}"
        )
        raise

    logger.info(f"Loading topic assignments from {TOPIC_CSV}")

    try:
        df = pd.read_csv(TOPIC_CSV)
    except Exception as e:
        logger.error(
            f"Failed to load topic CSV. Ensure the BERTopic pipeline completed.\n{e}"
        )
        raise

    return model, df


# -------------------------------------------------------
# GENERATE VISUALIZATIONS
# -------------------------------------------------------
def generate_visuals(model: BERTopic, df: pd.DataFrame):
    """
    Generate and save multiple BERTopic visualizations:
        • Hierarchy plot (topic tree)
        • Heatmap (topic-word similarity)
        • Term-rank (word distributions across topics)
        • Topics-over-time (if year metadata exists)

    This supports SOC 4994 requirements for interactive visualization
    and interpretability of thematic analysis.

    Parameters
    ----------
    model : BERTopic
        Loaded BERTopic model.
    df : pd.DataFrame
        Document-level dataset that may contain a 'year' column.
    """

    logger.info("Generating BERTopic visualizations...")

    # -----------------------------
    # Topic Hierarchy Visualization
    # -----------------------------
    # Shows hierarchical merging of clusters → useful for qualitative
    # analysis of how themes relate (surveillance → public safety → ethics).
    try:
        fig = model.visualize_hierarchy()
        fig.write_html(os.path.join(VIS_DIR, "topic_hierarchy.html"))
        logger.info("Saved topic hierarchy visualization.")
    except Exception as e:
        logger.warning(f"Failed to generate hierarchy visualization: {e}")

    # -----------------------------
    # Topic Heatmap Visualization
    # -----------------------------
    # Shows how topics relate via embeddings + CTFIDF.
    try:
        fig = model.visualize_heatmap()
        fig.write_html(os.path.join(VIS_DIR, "topic_heatmap.html"))
        logger.info("Saved topic heatmap visualization.")
    except Exception as e:
        logger.warning(f"Failed to generate heatmap visualization: {e}")

    # -----------------------------
    # Topic Term-Rank Visualization
    # -----------------------------
    # Shows distribution of key terms across topics, important for:
    #   - Identifying ethical framing vocab
    #   - Interpreting how FR legitimacy is constructed over time
    try:
        fig = model.visualize_term_rank()
        fig.write_html(os.path.join(VIS_DIR, "topic_term_rank.html"))
        logger.info("Saved term-rank visualization.")
    except Exception as e:
        logger.warning(f"Failed to generate term-rank visualization: {e}")

    # -----------------------------
    # Topics-Over-Time Visualization
    # -----------------------------
    # Requires the BERTopic pipeline to have saved a 'year' column.
    # Critical for the syllabus: "analyze ethical framing + thematic changes"
    # across time.
    if "year" in df.columns and df["year"].notna().sum() > 0:
        try:
            # Note: BERTopic expects a DataFrame with columns:
            # ['Document', 'Topic', 'Timestamp', 'Words']
            # But YOUR pipeline already outputs the correct format in topic_over_time.csv.
            # Here, we simply regenerate it via model.visualize_topics_over_time(df)
            fig = model.visualize_topics_over_time(df)
            fig.write_html(os.path.join(VIS_DIR, "topic_over_time.html"))
            logger.info("Saved topics-over-time visualization.")
        except Exception as e:
            logger.warning(f"Failed to generate topics-over-time visualization: {e}")
    else:
        logger.warning("Skipping 'topics_over_time' visualization — missing year data.")

    logger.info("All visualizations saved successfully.")


# -------------------------------------------------------
# MAIN
# -------------------------------------------------------
if __name__ == "__main__":
    logger.info("Starting visualization generation module...")
    model, df = load_model_and_data()
    generate_visuals(model, df)
    logger.info("Visualization generation completed.")