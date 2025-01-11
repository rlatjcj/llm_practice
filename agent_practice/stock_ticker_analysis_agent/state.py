import operator
from typing import Annotated, Sequence, TypedDict, Literal

from langchain_core.messages import BaseMessage
from pydantic import BaseModel


MEMBERS = ["Researcher", "Stock_Analyzer", "Chart_Generator"]


class State(TypedDict):
    """The state of the agent."""

    messages: Annotated[Sequence[BaseMessage], operator.add]
    next: str
    language: Literal["한글", "English"]


class RouteResponse(BaseModel):
    next: Literal["FINISH", *MEMBERS]  # type: ignore
