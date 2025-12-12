"""
Web search tools using DuckDuckGo.
Provides search capabilities without requiring API keys.
"""

from duckduckgo_search import DDGS
from typing import List, Dict
from utils import setup_logger, SimpleCache

logger = setup_logger("search_tools")
cache = SimpleCache()


def web_search(query: str, max_results: int = 5) -> List[Dict]:
    """
    Search the web using DuckDuckGo.

    Args:
        query: Search query string
        max_results: Maximum number of results to return

    Returns:
        List of search result dictionaries with title, href, and body
    """
    # Check cache
    cache_key = f"search:{query}:{max_results}"
    cached = cache.get(cache_key)
    if cached:
        logger.info(f"Cache hit for query: {query}")
        return cached

    logger.info(f"Searching: {query}")

    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))

        # Cache results
        cache.set(cache_key, results)
        logger.info(f"Found {len(results)} results for: {query}")
        return results

    except Exception as e:
        logger.error(f"Search error: {e}")
        return []


def search_news(query: str, max_results: int = 5) -> List[Dict]:
    """
    Search news articles using DuckDuckGo News.

    Args:
        query: Search query string
        max_results: Maximum number of news results

    Returns:
        List of news article dictionaries
    """
    logger.info(f"Searching news: {query}")

    try:
        with DDGS() as ddgs:
            results = list(ddgs.news(query, max_results=max_results))

        logger.info(f"Found {len(results)} news results for: {query}")
        return results

    except Exception as e:
        logger.error(f"News search error: {e}")
        return []
