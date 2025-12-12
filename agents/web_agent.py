"""
Web Research Agent.
Searches the web and extracts information from pages.
"""

from typing import List, Dict, Any
from langchain_core.messages import SystemMessage, HumanMessage
from tools.search_tools import web_search
from tools.browser_tools import scrape_page
from utils import setup_logger

logger = setup_logger("web_agent")


class WebAgent:
    """Agent for web search and browsing."""

    def __init__(self, llm, config: Dict):
        """
        Initialize web agent.

        Args:
            llm: Language model instance
            config: Configuration dictionary with agent settings
        """
        self.llm = llm
        self.config = config
        self.max_search_results = config.get('max_search_results', 5)
        self.max_pages = config.get('max_pages_to_visit', 3)
        self.system_prompt = config.get('system_prompt', '')

    def search_and_extract(self, query: str) -> Dict[str, Any]:
        """
        Search web and extract relevant information.

        Args:
            query: Search query or question

        Returns:
            Dictionary with success status, answer, sources, and raw data
        """
        logger.info(f"Web agent processing query: {query}")

        # Step 1: Search
        search_results = web_search(query, max_results=self.max_search_results)

        if not search_results:
            logger.warning("No search results found")
            return {
                'success': False,
                'message': 'No search results found',
                'answer': '',
                'sources': []
            }

        # Step 2: Visit top pages
        scraped_data = []
        for i, result in enumerate(search_results[:self.max_pages]):
            url = result.get('href') or result.get('link')
            if url:
                logger.info(f"Scraping page {i+1}/{self.max_pages}: {url}")
                page_data = scrape_page(url)
                if page_data:
                    scraped_data.append(page_data)

        # Step 3: LLM summarization
        context = self._format_context(search_results, scraped_data)

        prompt = f"""Based on the following web search results and page contents, answer the question: {query}

Search Results:
{context}

Provide a clear, concise answer based on the information found. If the information is not sufficient, state what additional information is needed.
"""

        try:
            messages = [
                SystemMessage(content=self.system_prompt),
                HumanMessage(content=prompt)
            ]

            response = self.llm.invoke(messages)
            answer = response.content if hasattr(response, 'content') else str(response)

            logger.info(f"Web agent generated answer: {answer[:100]}...")

            return {
                'success': True,
                'answer': answer,
                'sources': [r.get('href') or r.get('link') for r in search_results[:3]],
                'raw_data': {
                    'search_results': search_results,
                    'scraped_pages': scraped_data
                }
            }

        except Exception as e:
            logger.error(f"Error during LLM invocation: {e}")
            return {
                'success': False,
                'message': f'LLM error: {str(e)}',
                'answer': '',
                'sources': []
            }

    def _format_context(self, search_results: List[Dict], scraped_data: List[Dict]) -> str:
        """
        Format search and scrape data for LLM.

        Args:
            search_results: List of search result dictionaries
            scraped_data: List of scraped page data

        Returns:
            Formatted context string
        """
        context = []

        # Add search result snippets
        for i, result in enumerate(search_results):
            context.append(f"[{i+1}] {result.get('title', 'No title')}")
            context.append(f"    URL: {result.get('href') or result.get('link')}")
            context.append(f"    Snippet: {result.get('body', 'No description')}")
            context.append("")

        # Add full page contents
        if scraped_data:
            context.append("\n--- Page Contents ---\n")
            for i, page in enumerate(scraped_data):
                context.append(f"Page {i+1}: {page['title']}")
                # Limit text to avoid token overflow
                page_text = page['text'][:2000]
                context.append(f"{page_text}...")
                context.append("")

        return '\n'.join(context)
