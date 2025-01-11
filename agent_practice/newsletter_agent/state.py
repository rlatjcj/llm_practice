"""State for the newsletter agent."""

from typing import Annotated, TypedDict

from langgraph.graph.message import add_messages
from pydantic import BaseModel


def merge_dicts(left: dict, right: dict) -> dict:
    return {**left, **right}


class NewsletterThemeOutput(BaseModel):
    """Output model for structured theme and sub-theme generation."""

    theme: str
    sub_themes: list[str]


class State(TypedDict):
    """State for the newsletter agent."""

    keyword: str
    article_titles: list[str]
    newsletter_theme: NewsletterThemeOutput
    sub_theme_articles: dict[str, list[dict]]
    results: Annotated[dict[str, str], merge_dicts]
    messages: Annotated[list, add_messages]
    language: str
