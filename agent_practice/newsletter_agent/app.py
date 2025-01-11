"""Streamlit app for the newsletter agent."""

import asyncio

import streamlit as st
from dotenv import load_dotenv
from graph import create_newsletter_graph

if __name__ == "__main__":
    load_dotenv(override=True)

    st.title("Newsletter Agent")

    keyword = st.text_input("Enter a keyword for the newsletter:")
    language = st.selectbox(
        "Select newsletter language:",
        ["Korean", "English"],
        index=0,
    )

    if keyword.strip() == "":
        st.warning("Please enter a valid keyword.")
        st.stop()

    async def run_graph(inputs: dict) -> None:
        """Run the newsletter graph."""

        graph = create_newsletter_graph()

        # Create a status container for progress tracking
        status_container = st.container()

        with status_container:
            col1, col2 = st.columns([2, 1])
            with col1:
                status_text = st.empty()
            with col2:
                progress_bar = st.progress(0)

            # Create expandable status sections
            with st.expander("Detailed Progress", expanded=True):
                search_status = st.empty()
                theme_status = st.empty()
                subtheme_status = st.empty()
                write_status = st.empty()
                aggregate_status = st.empty()
                edit_status = st.empty()

        step = 0
        total_steps = 10  # Total number of steps in our graph

        try:
            async for output in graph.astream(inputs):
                for key, value in output.items():
                    step += 1
                    progress_bar.progress(step / total_steps)
                    status_text.text(f"Current Step: {key}")

                    # Update detailed status based on the current step
                    if key == "search_news":
                        search_status.success("✅ Article search is completed!")
                    elif key == "generate_themes":
                        theme_status.success("✅ Theme generation is completed!")
                    elif key == "search_sub_theme_articles":
                        subtheme_status.success("✅ Sub-theme research is completed!")
                    elif key.startswith("write_section"):
                        write_status.success(f"✅ Section {key[-1]} is written!")
                    elif key == "aggregate":
                        aggregate_status.success("✅ Draft compilation is completed!")
                        with st.expander("Draft Newsletter", expanded=False):
                            st.markdown(value["messages"][0].content)
                    elif key == "edit_newsletter":
                        edit_status.success("✅ Final editing is completed!")
                        st.markdown("## Final Newsletter")
                        st.markdown(value["messages"][0].content)

            status_text.success("Newsletter generation completed!")

        except Exception as e:
            status_text.error("Newsletter generation failed.")
            with st.expander("Error Details"):
                st.error(f"An error occurred: {e}")
                import traceback

                st.code(traceback.format_exc())

    if st.button("Generate Newsletter"):
        asyncio.run(run_graph({"keyword": keyword, "language": language}))
