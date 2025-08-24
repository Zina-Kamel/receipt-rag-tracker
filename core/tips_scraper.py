import requests
from bs4 import BeautifulSoup
from typing import List, Tuple

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def fetch_page(url: str, headers: dict = None, timeout: int = 10) -> str:
    """Fetch HTML content from a URL with error handling."""
    try:
        headers = headers or {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers, timeout=timeout)
        response.raise_for_status()
        return response.text
    except Exception as e:
        logger.error(f"Failed to fetch {url}: {e}")
        return ""


def scrape_eringobler() -> Tuple[List[str], str]:
    """Scrape personal finance tips from Erin Gobler."""
    url = "https://eringobler.com/personal-finance-tips/"
    html = fetch_page(url)
    if not html:
        return [], "ErinGobler"

    soup = BeautifulSoup(html, "html.parser")
    container = soup.find("article", class_="post-content entry-content")
    if not container:
        logger.warning("No content container found on ErinGobler")
        return [], "ErinGobler"

    tips = []
    headers = container.find_all("h2")
    for header in headers:
        desc_parts = []
        for sibling in header.find_next_siblings():
            if sibling.name == "h2":
                break
            if sibling.name == "p":
                desc_parts.append(sibling.get_text(strip=True))
        text = f"{header.get_text(strip=True)}: {' '.join(desc_parts)}"
        if len(text.split()) > 4:
            tips.append(text)

    return tips, "ErinGobler"


def scrape_manulife() -> Tuple[List[str], str]:
    """Scrape personal finance tips from Manulife."""
    url = (
        "https://www.manulife.ca/personal/plan-and-learn/healthy-finances/"
        "financial-planning/ten-simple-money-management-tips.html"
    )
    html = fetch_page(url)
    if not html:
        return [], "Manulife"

    soup = BeautifulSoup(html, "html.parser")
    container = soup.find("div", class_="enable-margin aem-GridColumn aem-GridColumn--default--8 aem-GridColumn--phone--12")
    if not container:
        logger.warning("No content container found on Manulife")
        return [], "Manulife"

    tips = []
    titles = container.find_all("div", class_="heading")
    descs = container.find_all("div", class_="rich-text parbase")
    for title, desc in zip(titles, descs):
        text = f"{title.get_text(strip=True)}: {desc.get_text(strip=True)}"
        if len(text.split()) > 4:
            tips.append(text)

    return tips, "Manulife"


def fetch_all_sources() -> List[Tuple[str, str]]:
    """
    Fetch tips from all sources. If no tips are found, return fallback tips.
    Returns:
        List of tuples (tip_text, source_name)
    """
    all_tips: List[Tuple[str, str]] = []
    scrapers = [scrape_eringobler, scrape_manulife]

    for scraper in scrapers:
        tips, source = scraper()
        all_tips.extend([(tip, source) for tip in tips])

    if not all_tips:
        logger.info("No tips scraped; using fallback tips.")
        fallback = [
            ("Track your spending weekly to identify patterns.", "Fallback"),
            ("Set aside 20% of your income for savings and investments.", "Fallback"),
            ("Avoid impulse purchases by waiting 24 hours before buying.", "Fallback"),
            ("Automate bill payments to avoid late fees.", "Fallback"),
            ("Review subscriptions every 3 months to cut unused ones.", "Fallback"),
        ]
        all_tips = fallback

    return all_tips
