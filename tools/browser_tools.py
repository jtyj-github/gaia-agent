"""
Browser automation tools using Playwright.
Provides web page fetching and content extraction.
"""

from playwright.sync_api import sync_playwright
from bs4 import BeautifulSoup
from typing import Optional, Dict
from utils import setup_logger

logger = setup_logger("browser_tools")


def fetch_page_content(url: str, timeout: int = 30) -> Optional[str]:
    """
    Fetch page content using Playwright.

    Args:
        url: URL to fetch
        timeout: Timeout in seconds

    Returns:
        HTML content of the page, or None if failed
    """
    logger.info(f"Fetching: {url}")

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            page.goto(url, timeout=timeout * 1000)

            # Wait for content to load
            page.wait_for_load_state("networkidle", timeout=timeout * 1000)

            content = page.content()
            browser.close()

            logger.info(f"Successfully fetched: {url}")
            return content

    except Exception as e:
        logger.error(f"Error fetching {url}: {e}")
        return None


def extract_text_from_html(html: str) -> str:
    """
    Extract clean text from HTML.

    Args:
        html: HTML content

    Returns:
        Cleaned text extracted from HTML
    """
    soup = BeautifulSoup(html, 'lxml')

    # Remove script and style elements
    for script in soup(["script", "style"]):
        script.decompose()

    # Get text
    text = soup.get_text()

    # Clean up whitespace
    lines = (line.strip() for line in text.splitlines())
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    text = '\n'.join(chunk for chunk in chunks if chunk)

    return text


def scrape_page(url: str, timeout: int = 30) -> Optional[Dict]:
    """
    Scrape page and return structured data.

    Args:
        url: URL to scrape
        timeout: Timeout in seconds

    Returns:
        Dictionary containing url, title, text, and links
    """
    html = fetch_page_content(url, timeout)
    if not html:
        return None

    soup = BeautifulSoup(html, 'lxml')

    return {
        'url': url,
        'title': soup.title.string if soup.title else '',
        'text': extract_text_from_html(html),
        'links': [a.get('href') for a in soup.find_all('a', href=True)][:20]
    }
