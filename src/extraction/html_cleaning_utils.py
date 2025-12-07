"""
html_cleaning_utils.py
----------------------
Utility functions for cleaning HTML text extracted from archived
Wayback Machine snapshots.

This module is imported by text_extractor.py and should not be run standalone.
"""

from bs4 import BeautifulSoup
import re
import ftfy

# Tags that should always be removed from extracted HTML
REMOVE_TAGS = [
    "script", "style", "nav", "footer", "header",
    "form", "noscript", "iframe", "svg", "canvas",
    "button", "input"
]

# Common CSS selectors used by websites for banners, popups, cookie notices, etc.
REMOVE_CSS_PATTERNS = [
    "cookie", "gdpr", "privacy", "banner", "popup", "consent",
    "subscribe", "newsletter", "advert", "ad-container", "modal",
    "promo", "hero", "sidebar", "breadcrumbs"
]

# Tags we *do* trust for real content
CONTENT_TAGS = ["article", "main", "section", "p", "li", "h1", "h2", "h3", "h4"]

# Boilerplate phrases / junk patterns to strip out of otherwise valid text
JUNK_PATTERNS = [
    r"©\s*\d{4}.*?all rights reserved",
    r"©\s*\d{4}",
    r"all rights reserved",
    r"privacy policy",
    r"terms(?: and conditions| of service)?",
    r"cookies? settings?",
    r"click here",
    r"learn more",
    r"back to top",
    r"home\s+about\s+contact",
    r"sign in|sign up|log in|log out",
    r"404 not found",
    r"page not found",
    r"javascript.*?enable",
]

# Compile once
JUNK_REGEXES = [re.compile(pat, flags=re.I) for pat in JUNK_PATTERNS]


def remove_unwanted_tags(soup):
    """Remove script, style, nav, footer, and other irrelevant tags."""
    for tag in soup(REMOVE_TAGS):
        try:
            tag.decompose()
        except Exception:
            pass
    return soup


def remove_unwanted_css(soup):
    """
    Remove unwanted divs and sections based on class/id name patterns
    (common GDPR banners, cookie popups, subscription popups, etc.).
    """
    for element in soup.find_all(True):
        try:
            classes = " ".join(element.get("class", [])).lower()
            element_id = (element.get("id") or "").lower()

            if any(pattern in classes for pattern in REMOVE_CSS_PATTERNS):
                element.decompose()
            elif any(pattern in element_id for pattern in REMOVE_CSS_PATTERNS):
                element.decompose()
        except Exception:
            continue
    return soup


def extract_visible_text(soup):
    """
    Extract text only from content-bearing tags (whitelist).
    This drops most navigation, menus, and layout junk.
    """
    chunks = []
    for tag in soup.find_all(CONTENT_TAGS):
        txt = tag.get_text(" ", strip=True)
        # Ignore ultra-short fragments (e.g., menu items)
        if len(txt.split()) >= 4:
            chunks.append(txt)
    return " ".join(chunks)


def _strip_boilerplate(text: str) -> str:
    """Remove common boilerplate phrases / junk patterns."""
    out = text
    for rx in JUNK_REGEXES:
        out = rx.sub(" ", out)
    return out


def clean_html_text(html: str) -> str:
    """
    Complete HTML → cleaned text pipeline:
    1. Parse with BeautifulSoup
    2. Remove scripts/styles/nav/footer/gdpr/cookie elements
    3. Extract text only from content-bearing tags
    4. Fix broken Unicode
    5. Remove boilerplate phrases
    6. Normalize whitespace
    """
    soup = BeautifulSoup(html, "html.parser")

    # Remove scripts, styles, nav bars, cookie banners
    soup = remove_unwanted_tags(soup)
    soup = remove_unwanted_css(soup)

    # Extract only visible content text
    text = extract_visible_text(soup)

    # Fix encoding issues (Â©, ð, etc.)
    text = ftfy.fix_text(text)

    # Lowercase for easier downstream matching, but keep numbers & tech terms
    text = text.strip()

    # Strip boilerplate / UI phrases
    text = _strip_boilerplate(text)

    # Squash excessive whitespace
    text = re.sub(r"\s+", " ", text)

    return text.strip()