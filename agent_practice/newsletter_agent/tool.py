"""Tool for searching news articles."""

import streamlit as st
from tavily import AsyncTavilyClient, TavilyClient


class NewsletterTool:
    """Tool for searching news articles."""

    def __init__(self) -> None:
        self.client = TavilyClient()
        self.async_client = AsyncTavilyClient()

    def search_recent_news(self, keyword: str) -> list:
        """Search for recent news articles based on the keyword.

        Args:
            keyword (str): The keyword to search for.

        Returns:
            list: A list of titles of the search results.
        """
        search_result = self.client.search(
            query=keyword,
            max_results=5,
            topic="news",
            days=5,
        )
        titles = [result["title"] for result in search_result["results"]]
        return titles

    async def search_news_for_subtheme(self, subtheme: str) -> dict:
        """Search for recent news articles based on the sub-theme.

        Args:
            subtheme (str): The sub-theme to search for.

        Returns:
            dict: A dictionary containing the search results.
        """
        search_params = {
            "query": subtheme,
            "max_results": 3,
            "topic": "news",
            "days": 7,
            "include_images": True,
            "include_raw_content": True,
        }

        try:
            with st.status(
                label=f"Searching '{subtheme}' related news...",
                expanded=False,
            ) as status:
                response = await self.async_client.search(**search_params)
                images = response.get("images", [])
                results = response.get("results", [])

                article_info = []
                for i, result in enumerate(results):
                    article_info.append(
                        {
                            "title": result.get("title", ""),
                            "image_url": images[i] if i < len(images) else "",
                            "raw_content": result.get("raw_content", ""),
                        }
                    )

                if article_info:
                    status.update(
                        label=f"Found {len(article_info)} articles related to '{subtheme}'.",
                        state="complete",
                        expanded=False,
                    )
                    for article in article_info:
                        status.markdown(f"- {article['title']}")
                else:
                    status.update(
                        label=f"No articles found related to '{subtheme}'.",
                        state="error",
                        expanded=False,
                    )
            return {subtheme: article_info}

        except Exception as e:
            st.write(f"Error in search_news_for_subtheme: {e}")
            return {subtheme: []}
