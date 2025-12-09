"""
logger.py
---------
Utility for creating consistent, formatted loggers across the SOC 4994
Facial Recognition Research Pipeline.

This module is used by:
    - BERTopic pipeline
    - company profile generation
    - framing analysis
    - topic interpretation
    - visualization modules
    - any script that requires standardized logging

Features:
    - Timestamped console logs for visibility during long-running processes
    - Optional file logging for reproducibility and debugging
    - Uniform log formatting across all project components

Usage example:
    from src.utils.logger import get_logger

    logger = get_logger("MyScript")
    logger.info("Processing started.")

Run:
    (import only — not executable as standalone)
"""

import logging
import os


def get_logger(name, log_file=None, level=logging.INFO):
    """
    Create and configure a formatted logger shared across all project scripts.

    Parameters
    ----------
    name : str
        The name of the logger (typically the module or script name).
    
    log_file : str | None
        Optional path to a log file where logs should be written.
        If provided:
            - Ensures the directory exists
            - Adds a FileHandler to the logger
    
    level : int
        Logging level (default: logging.INFO)

    Returns
    -------
    logger : logging.Logger
        A configured logger instance with:
            - console handler
            - optional file handler
            - timestamped + consistent formatting
    """

    # Create a logger instance for the given name
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Standard formatter used across the pipeline
    # Example output:
    #   2025-12-08 23:12:45 [INFO] BERTopicPipeline: Starting model...
    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # -----------------------------
    # Console Handler (always enabled)
    # -----------------------------
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # -----------------------------
    # File Handler (optional)
    # -----------------------------
    if log_file:
        # Ensure the folder for log files exists
        os.makedirs(os.path.dirname(log_file), exist_ok=True)

        # Create a file handler and apply the same formatting
        fh = logging.FileHandler(log_file)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger