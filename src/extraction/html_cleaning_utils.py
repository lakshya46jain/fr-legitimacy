"""
html_cleaning_utils.py
----------------------
Utility functions for cleaning HTML text extracted from archived
Wayback Machine snapshots.

This module is imported by text_extractor.py and should not be run standalone.
"""

from bs4 import BeautifulSoup
import re


# Tags that should always be removed from extracted HTML
REMOVE_TAGS = [
    "script", "style", "nav", "footer", "header",
    "form", "noscript", "iframe", "svg", "canvas"
]

# Common CSS selectors used by websites for banners, popups, cookie notices, etc.
REMOVE_CSS_PATTERNS = [
    "cookie", "gdpr", "privacy", "banner", "popup", "consent",
    "subscribe", "newsletter", "advert", "ad-container", "modal"
]


def remove_unwanted_tags(soup):
    """Remove script, style, nav, footer, and other irrelevant tags."""
    for tag in soup(REMOVE_TAGS):
        try:
            tag.decompose()
        except:
            pass
    return soup


def remove_unwanted_css(soup):
    """
    Remove unwanted divs and sections based on class/id name patterns
    (common GDPR banners, cookie popups, subscription popups).
    """
    for element in soup.find_all(True):
        try:
            classes = " ".join(element.get("class", [])).lower()
            element_id = (element.get("id") or "").lower()

            if any(pattern in classes for pattern in REMOVE_CSS_PATTERNS):
                element.decompose()
            elif any(pattern in element_id for pattern in REMOVE_CSS_PATTERNS):
                element.decompose()

        except:
            continue

    return soup


def clean_html_text(html):
    """
    Complete HTML → cleaned text pipeline:
    1. Parse with BeautifulSoup
    2. Remove scripts/styles/nav/footer/gdpr/cookie elements
    3. Extract visible text
    4. Normalize whitespace
    """
    soup = BeautifulSoup(html, "html.parser")

    # Remove scripts, styles, nav bars, cookie banners
    soup = remove_unwanted_tags(soup)
    soup = remove_unwanted_css(soup)

    # Extract text using spaces as separators
    text = soup.get_text(separator=" ")

    # Squash excessive whitespace
    text = re.sub(r"\s+", " ", text)

    return text.strip()