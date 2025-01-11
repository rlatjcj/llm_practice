import functools

from langchain_core.messages import BaseMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import create_react_agent

from agent import StockTickerAnalysisAgent
from state import State, MEMBERS


def create_stock_ticker_analysis_graph() -> StateGraph:
    """Create the stock ticker analysis graph."""

    llm = ChatOpenAI(model="gpt-4o-mini")

    workflow = StateGraph(State)
    agent = StockTickerAnalysisAgent(llm)

    # Add nodes
    workflow.add_node("Researcher", agent.researcher_agent)
    workflow.add_node("Stock_Analyzer", agent.stock_analyzer_agent)
    workflow.add_node("Chart_Generator", agent.chart_generator_agent)
    workflow.add_node("supervisor", agent.supervisor_agent)

    # Add edges
    workflow.add_edge(START, "supervisor")
    for member in MEMBERS:
        workflow.add_edge(member, "supervisor")  # member -> supervisor

    conditional_map = {k: k for k in MEMBERS}
    conditional_map["FINISH"] = END
    workflow.add_conditional_edges("supervisor", lambda x: x["next"], conditional_map)

    return workflow.compile()
