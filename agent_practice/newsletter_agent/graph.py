"""Graph for the newsletter agent."""

import logging

from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from node import NewsletterNode
from state import State

logger = logging.getLogger(__name__)


def create_newsletter_graph() -> StateGraph:
    """Create a newsletter graph."""

    logger.info("Create newsletter graph...")

    llm = ChatOpenAI(model="gpt-4o-mini")
    workflow = StateGraph(State)
    node = NewsletterNode(llm)

    # Add nodes
    workflow.add_node("search_news", node.search_keyword_news)
    workflow.add_node("generate_themes", node.generate_themes)
    workflow.add_node("search_sub_theme_articles", node.search_sub_theme_articles)
    for i in range(5):
        node_name = f"write_section_{i}"
        workflow.add_node(
            node_name,
            lambda s, i=i: node.write_section(s, s["newsletter_theme"].sub_themes[i]),
        )
    workflow.add_node("aggregate", node.aggregate_results)
    workflow.add_node("edit_newsletter", node.edit_newsletter)

    # Add edges
    workflow.add_edge(START, "search_news")
    workflow.add_edge("search_news", "generate_themes")
    workflow.add_edge("generate_themes", "search_sub_theme_articles")
    for i in range(5):
        workflow.add_edge("search_sub_theme_articles", f"write_section_{i}")
        workflow.add_edge(f"write_section_{i}", "aggregate")
    workflow.add_edge("aggregate", "edit_newsletter")
    workflow.add_edge("edit_newsletter", END)

    logger.info("Newsletter graph is created successfully!")
    return workflow.compile()
