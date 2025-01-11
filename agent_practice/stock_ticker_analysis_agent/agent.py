import functools
from datetime import datetime
from typing import Callable, TypeVar

import streamlit as st
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from prompt import StockTickerAnalysisPrompt
from state import MEMBERS, RouteResponse, State
from tool import StockTickerAnalysisTool

T = TypeVar("T")


def with_status(
    agent_name: str, max_trials: int | None = None
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator to show status while agent is running."""

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> T:
            # Get instance (self) from args
            instance = args[0] if args else None
            # Use instance max_trials if not specified in decorator
            actual_max_trials = (
                max_trials if max_trials is not None else instance.max_trials
            )

            # Get state from args (first argument after self)
            state = args[1] if len(args) > 1 else kwargs.get("state")
            language = state.get("language", "English")

            # Get status text based on language
            status_text = {
                "한글": {
                    "working": "작업 중... (시도: {}/{})",
                    "completed": "완료! (시도: {}/{})",
                },
                "English": {
                    "working": "is working... (trial: {}/{})",
                    "completed": "completed! (trial: {}/{})",
                },
            }[language]

            # Initialize containers and state in session state if not exists
            if "status_containers" not in st.session_state:
                st.session_state.status_containers = {}
            if "markdown_containers" not in st.session_state:
                st.session_state.markdown_containers = {}
            if "current_agent" not in st.session_state:
                st.session_state.current_agent = None
            if "last_result" not in st.session_state:
                st.session_state.last_result = None
            if "agent_trials" not in st.session_state:
                st.session_state.agent_trials = {}

            # Clean agent name for display (remove underscores)
            display_name = agent_name.replace("_", " ")

            # Reset trial count when switching to a new agent
            if st.session_state.current_agent != display_name:
                st.session_state.agent_trials[display_name] = 1
            else:
                # Increment trial count for the same agent
                st.session_state.agent_trials[display_name] += 1

            current_trial = st.session_state.agent_trials[display_name]

            # If we're switching to a different agent, complete the previous one
            if (
                st.session_state.current_agent
                and st.session_state.current_agent != display_name
                and st.session_state.last_result
            ):
                prev_agent = st.session_state.current_agent
                prev_agent_status = st.session_state.status_containers.get(prev_agent)
                prev_agent_markdown = st.session_state.markdown_containers.get(
                    prev_agent
                )

                if prev_agent_status and prev_agent_markdown:
                    prev_trials = st.session_state.agent_trials[prev_agent]
                    prev_agent_status.update(
                        label=f"{prev_agent} {status_text['completed'].format(prev_trials, actual_max_trials)}",
                        state="complete",
                    )
                    if "messages" in st.session_state.last_result:
                        content = st.session_state.last_result["messages"][0].content
                        try:
                            if "![Chart]" in content:
                                # Extract the chart data from the content
                                import json
                                import plotly.graph_objects as go

                                # Find the JSON data between ```json and ```
                                start_idx = content.find("```json\n") + 8
                                end_idx = content.find("```", start_idx)
                                chart_data = json.loads(content[start_idx:end_idx])

                                # Create and display the chart
                                fig = go.Figure(data=chart_data)
                                st.plotly_chart(fig, use_container_width=True)
                            else:
                                prev_agent_markdown.markdown(content)
                        except Exception as e:
                            st.error(f"Error displaying chart: {str(e)}")
                            prev_agent_markdown.markdown(content)

            # Reuse existing status container or create new one
            if display_name not in st.session_state.status_containers:
                status = st.status(
                    f"{display_name} {status_text['working'].format(current_trial, actual_max_trials)}",
                    expanded=False,
                )
                st.session_state.status_containers[display_name] = status
                markdown_container = status.empty()
                st.session_state.markdown_containers[display_name] = markdown_container
            else:
                status = st.session_state.status_containers[display_name]
                markdown_container = st.session_state.markdown_containers[display_name]
                status.update(
                    label=f"{display_name} {status_text['working'].format(current_trial, actual_max_trials)}",
                    state="running",
                )
                markdown_container.empty()

            # Update current agent and execute function
            st.session_state.current_agent = display_name
            result = func(*args, **kwargs)
            st.session_state.last_result = result

            # Only complete if next agent is different or FINISH
            next_agent = None
            if isinstance(result, dict):
                if "next" in result:
                    next_agent = result["next"]
                elif "messages" in result and len(result["messages"]) > 0:
                    next_message = result["messages"][-1]
                    if hasattr(next_message, "additional_kwargs"):
                        next_agent = next_message.additional_kwargs.get("next")

            # Update status if moving to different agent or finishing
            if next_agent and (next_agent != display_name or next_agent == "FINISH"):
                status.update(
                    label=f"{display_name} {status_text['completed'].format(current_trial, actual_max_trials)}",
                    state="complete",
                )
                if "messages" in result:
                    content = result["messages"][0].content
                    try:
                        if "![Chart]" in content:
                            # Extract the chart data from the content
                            import json
                            import plotly.graph_objects as go

                            # Find the JSON data between ```json and ```
                            start_idx = content.find("```json\n") + 8
                            end_idx = content.find("```", start_idx)
                            chart_data = json.loads(content[start_idx:end_idx])

                            # Create and display the chart
                            fig = go.Figure(data=chart_data)
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            markdown_container.markdown(content)
                    except Exception as e:
                        st.error(f"Error displaying chart: {str(e)}")
                        markdown_container.markdown(content)

            return result

        return wrapper

    return decorator


class StockTickerAnalysisAgent:
    def __init__(self, llm: ChatOpenAI) -> None:
        self.llm = llm
        self.prompt = StockTickerAnalysisPrompt()
        self.tool = StockTickerAnalysisTool()
        self.max_trials = 1

    def supervisor_agent(self, state: State) -> RouteResponse:
        # Get current date
        current_date = datetime.now().strftime("%Y-%m-%d")

        # Create a proper prompt template
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    self.prompt.system_prompt.format(
                        members=", ".join(MEMBERS),
                        language=state.get("language", "English"),
                        current_date=current_date,
                    ),
                ),
                ("human", "{input}"),
            ]
        )

        # Create the chain
        supervisor_chain = prompt | self.llm.with_structured_output(RouteResponse)

        # Get the last message content for the input
        last_message = state["messages"][-1]
        if isinstance(last_message, dict):
            message_content = last_message.get("content", "")
        else:
            message_content = last_message.content

        # Get supervisor's decision
        response = supervisor_chain.invoke({"input": message_content})
        next_agent = response.next

        # If this is the first call or coming from supervisor, start with first member
        if next_agent == "supervisor" or (
            "current_agent" not in st.session_state
            or st.session_state.current_agent == "supervisor"
        ):
            next_agent = MEMBERS[0]
            return RouteResponse(next=next_agent)

        # Check if the next agent has exceeded max trials
        if "agent_trials" in st.session_state and next_agent != "FINISH":
            # Get the display name for checking trials
            display_name = next_agent.replace("_", " ")  # Convert to display name
            current_trials = st.session_state.agent_trials.get(display_name, 0)

            if current_trials >= self.max_trials:
                try:
                    current_idx = MEMBERS.index(next_agent)
                    if current_idx < len(MEMBERS) - 1:
                        # Move to next agent in sequence
                        next_agent = MEMBERS[current_idx + 1]
                    else:
                        # If we're at the last agent, finish
                        next_agent = "FINISH"

                    # Log the transition
                    print(
                        f"Agent {display_name} exceeded max trials ({current_trials}/{self.max_trials}). Moving to {next_agent}"
                    )
                except ValueError:
                    # If agent not found in MEMBERS, finish
                    next_agent = "FINISH"

        # If next_agent is FINISH, check if we've gone through all members
        if next_agent == "FINISH":
            current_agent_member = st.session_state.current_agent.replace(" ", "_")
            try:
                current_idx = MEMBERS.index(current_agent_member)
                if current_idx < len(MEMBERS) - 1:
                    # If we haven't gone through all members, force next member
                    next_agent = MEMBERS[current_idx + 1]
                else:
                    # If we're at the last member, confirm FINISH
                    return RouteResponse(next="FINISH")
            except ValueError:
                # If current agent not found in MEMBERS, start with first member
                next_agent = MEMBERS[0]

        return RouteResponse(next=next_agent)

    @with_status("Researcher")
    def researcher_agent(self, state: State) -> State:
        current_date = datetime.now().strftime("%Y-%m-%d")
        research_agent = create_react_agent(
            self.llm,
            tools=[self.tool.tavily_tool],
            state_modifier=self.prompt.researcher_prompt.format(
                language=state.get("language", "English"),
                current_date=current_date,
            ),
        )
        return self.agent_node(state, research_agent, "Researcher")

    @with_status("Stock_Analyzer")
    def stock_analyzer_agent(self, state: State) -> State:
        current_date = datetime.now().strftime("%Y-%m-%d")
        stock_agent = create_react_agent(
            self.llm,
            tools=[self.tool.analyze_stock_ticker],
            state_modifier=self.prompt.stock_analyzer_prompt.format(
                language=state.get("language", "English"),
                current_date=current_date,
            ),
        )
        return self.agent_node(state, stock_agent, "Stock_Analyzer")

    @with_status("Chart_Generator")
    def chart_generator_agent(self, state: State) -> State:
        current_date = datetime.now().strftime("%Y-%m-%d")
        chart_agent = create_react_agent(
            self.llm,
            tools=[self.tool.python_repl_tool],
            state_modifier=self.prompt.chart_generator_prompt.format(
                language=state.get("language", "English"),
                current_date=current_date,
            ),
        )
        return self.agent_node(state, chart_agent, "Chart_Generator")

    def agent_node(self, state: State, agent: ChatOpenAI, name: str) -> State:
        result = agent.invoke(state)

        last_message = result["messages"][-1]
        if isinstance(last_message, dict):
            content = last_message.get("content", "")
        else:
            content = last_message.content

        return {"messages": [HumanMessage(content=content, name=name)]}
