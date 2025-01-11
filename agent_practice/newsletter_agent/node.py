"""Node for the newsletter agent."""

import asyncio
from typing import TYPE_CHECKING

from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from prompt import NewsletterPrompt
from pydantic import BaseModel, Field
from state import State
from tool import NewsletterTool


class NewsletterThemeOutput(BaseModel):
    """Output model for structured theme and sub-theme generation."""

    theme: str = Field(
        description="The main newsletter theme based on the provided article titles."
    )
    sub_themes: list[str] = Field(
        description="List of sub-themes or key news items to investigate under the main theme, ensuring they are specific and researchable."
    )


class NewsletterNode:
    """Node for the newsletter agent."""

    def __init__(self, llm: ChatOpenAI) -> None:
        self.llm = llm
        self.tool = NewsletterTool()

    def search_keyword_news(self, state: State) -> State:
        """Search for recent news articles based on the keyword.

        Args:
            state (State): The current state of the agent.

        Returns:
            State: The updated state of the agent.
        """
        keyword = state["keyword"]
        article_titles = self.tool.search_recent_news(keyword)
        return {"article_titles": article_titles}

    def generate_themes(self, state: State) -> State:
        """Generate newsletter themes.

        Args:
            state (State): The current state of the agent.

        Returns:
            State: The updated state of the agent.
        """
        article_titles = state["article_titles"]
        language = state["language"]
        newsletter_theme = self.llm.with_structured_output(NewsletterThemeOutput)
        theme_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", NewsletterPrompt.generate_themes),
                ("human", "Article titles: \n\n {article_titles}"),
            ]
        )

        # Chain together the system prompt and the structured output model
        subtheme_chain = theme_prompt | newsletter_theme
        newsletter_theme = subtheme_chain.invoke(
            {"article_titles": "\n".join(article_titles), "language": language}
        )
        newsletter_theme.sub_themes = newsletter_theme.sub_themes[:5]
        return {"newsletter_theme": newsletter_theme}

    async def search_sub_theme_articles(self, state: State) -> State:
        """Search for recent news articles based on the sub-theme.

        Args:
            state (State): The current state of the agent.

        Returns:
            State: The updated state of the agent.
        """
        subthemes = state["newsletter_theme"].sub_themes
        results = await asyncio.gather(
            *[self.tool.search_news_for_subtheme(subtheme) for subtheme in subthemes]
        )

        sub_theme_articles = {}
        for result in results:
            sub_theme_articles.update(result)

        if not any(sub_theme_articles.values()):
            raise ValueError(
                "No articles found for any sub-theme. Please try a different keyword."
            )
        return {"sub_theme_articles": sub_theme_articles}

    def write_section(self, state: State, sub_theme: str) -> State:
        """Write a newsletter section for the sub-theme.

        Args:
            state (State): The current state of the agent.
            sub_theme (str): The sub-theme to write a section for.

        Returns:
            State: The updated state of the agent.
        """
        return asyncio.run(self.write_section_async(state, sub_theme))

    async def write_section_async(self, state: State, sub_theme: str) -> State:
        """Write a newsletter section for the sub-theme asynchronously.

        Args:
            state (State): The current state of the agent.
            sub_theme (str): The sub-theme to write a section for.

        Returns:
            State: The updated state of the agent.
        """
        articles = state["sub_theme_articles"][sub_theme]
        language = state["language"]

        # Prepare article references with proper image markdown
        article_references = "\n".join(
            [
                f"Title: {article['title']}\n"
                + (
                    f"![Article Image]({article['image_url']})\n"
                    if article["image_url"]
                    else ""
                )
                + f"Content: {article['raw_content']}..."
                for article in articles
            ]
        )

        prompt = NewsletterPrompt.write_section.format(
            sub_theme=sub_theme,
            article_references=article_references,
            language=language,
        )
        messages = [HumanMessage(content=prompt)]
        response = await self.llm.ainvoke(messages)
        return {"results": {sub_theme: response.content}}

    def aggregate_results(self, state: State) -> State:
        """Aggregate the results of the newsletter sections.

        Args:
            state (State): The current state of the agent.

        Returns:
            State: The updated state of the agent.
        """
        theme = state["newsletter_theme"].theme
        combined_newsletter = f"# {theme}\n\n"
        for sub_theme, content in state["results"].items():
            combined_newsletter += f"## {sub_theme}\n{content}\n\n"
        return {"messages": [HumanMessage(content=combined_newsletter)]}

    def edit_newsletter(self, state: State) -> State:
        """Edit the newsletter.

        Args:
            state (State): The current state of the agent.

        Returns:
            State: The updated state of the agent.
        """
        theme = state["newsletter_theme"].theme
        language = state["language"]
        combined_newsletter = state["messages"][-1].content

        prompt = NewsletterPrompt.edit_newsletter.format(
            theme=theme, combined_newsletter=combined_newsletter, language=language
        )
        messages = [HumanMessage(content=prompt)]
        response = self.llm.invoke(messages)
        return {"messages": [HumanMessage(content=response.content)]}
