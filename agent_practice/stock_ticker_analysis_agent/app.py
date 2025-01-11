"""Streamlit app for stock analysis agent."""

import asyncio
import streamlit as st
from dotenv import load_dotenv
from langchain.schema import HumanMessage

from graph import create_stock_ticker_analysis_graph

# UI text dictionary
UI_TEXT = {
    "한글": {
        "title": "주식 분석 어시스턴트 📈",
        "input_label": "주식에 대해 무엇이 궁금하신가요?",
        "input_placeholder": "예시: 애플 주식을 사야할까요? AAPL 분석 부탁드립니다.",
        "language_label": "언어 선택:",
        "warning": "질문을 입력해주세요.",
        "analyze_button": "분석 시작",
        "current_step": "현재 단계",
        "analysis_completed": "분석 완료!",
        "analysis_failed": "분석 실패",
        "error_occurred": "오류가 발생했습니다",
    },
    "English": {
        "title": "Stock Analysis Assistant 📈",
        "input_label": "What would you like to know about a stock?",
        "input_placeholder": "Example: Should I buy Apple stock? Please analyze AAPL.",
        "language_label": "Select Language:",
        "warning": "Please enter your question.",
        "analyze_button": "Analyze Stock",
        "current_step": "Current Step",
        "analysis_completed": "Analysis completed!",
        "analysis_failed": "Analysis failed",
        "error_occurred": "An error occurred",
    },
}


async def run_graph(inputs: dict) -> None:
    """Run the stock analysis graph."""
    graph = create_stock_ticker_analysis_graph()
    language = inputs.get("language", "English")
    text = UI_TEXT[language]

    # Create status containers
    status_container = st.container()

    with status_container:
        col1, col2 = st.columns([2, 1])
        with col1:
            status_text = st.empty()
        with col2:
            progress_bar = st.progress(0)

    # Create log container
    with st.expander("Detailed logs", expanded=False):
        log_container = st.empty()
        logs = []

    step_dict = {
        "supervisor": 0,
        "Researcher": 1,
        "Stock_Analyzer": 2,
        "Chart_Generator": 3,
    }
    total_steps = 4

    try:
        for output in graph.stream(inputs):
            # Add log entry for each output
            for key, value in output.items():
                # Format the output for logging
                log_entry = f"**{key}**:\n```python\n{value}\n```"
                logs.append(log_entry)
                log_container.markdown("\n\n".join(logs))

                # Update progress
                if "next" in value:
                    next_agent = value["next"]
                    if next_agent == "FINISH":
                        progress_bar.progress(1.0)
                        status_text.success(text["analysis_completed"])
                        break
                    progress_bar.progress(step_dict[next_agent] / total_steps)
                    status_text.text(
                        f"{text['current_step']}: {next_agent.replace('_', ' ')}"
                    )

    except Exception as e:
        status_text.error(text["analysis_failed"])
        st.error(f"{text['error_occurred']}: {str(e)}")
        logs.append(f"**Error**:\n```python\n{str(e)}\n```")
        log_container.markdown("\n\n".join(logs))


if __name__ == "__main__":
    load_dotenv(override=True)

    # Initialize or clear session state when rerunning
    if st.button("New Analysis"):
        # Clear all session state
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

    # Get language selection
    language = st.selectbox(
        "Select Language / 언어 선택:",
        ["한글", "English"],
        index=0,
    )

    text = UI_TEXT[language]
    st.title(text["title"])

    # User inputs
    question = st.text_input(
        text["input_label"],
        placeholder=text["input_placeholder"],
    )

    if question.strip() == "":
        st.warning(text["warning"])
        st.stop()

    if st.button(text["analyze_button"]):
        # Clear previous analysis state
        for key in list(st.session_state.keys()):
            if key not in ["language"]:  # Keep language preference
                del st.session_state[key]

        asyncio.run(
            run_graph(
                {
                    "messages": [HumanMessage(content=question)],
                    "next": "supervisor",
                    "language": language,
                }
            )
        )
