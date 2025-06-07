import streamlit as st
from langgraph_client import LangGraphClient
import pandas as pd
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize client in session state if not present
if "client" not in st.session_state:
    logger.info("Initializing LangGraphClient...")
    st.session_state["client"] = LangGraphClient()
    st.session_state["client"].assistant_id = st.session_state["client"].create_assistant("research")
    st.session_state["response"] = None

st.title("AI Research Assistant v1.0")

if not st.session_state["response"]:
    # User inputs
    topic = st.text_area("Enter research topic:", height=100, value="Climate change")
    max_analysts = st.number_input("Number of analysts:", min_value=1, max_value=3, value=3)

    if st.button("Generate Analysts"):
        logger.info("Generating analysts...")
        with st.spinner("Generating analysts..."):
            # Prepare input data for backend
            input_data = {
                "topic": topic,
                "max_analysts": max_analysts,
            }
            st.session_state["response"] = st.session_state["client"].run_graph(input_data=input_data)
            st.rerun()
            
if st.session_state["response"]:
    if "final_analysts" not in st.session_state["response"]:
        feedback = st.text_area("Optional editorial feedback:", height=70, value="Add tech specialist")
        if st.button("Add Feedback"):
            logger.info("Adding feedback...")
            with st.spinner("Generating more analysts..."):
                input_data = {
                    "human_analyst_feedback": [feedback] if feedback else [],
                }
                st.session_state["response"] = st.session_state["client"].run_graph_resume(input_data=input_data)
                logger.info("Feedback added, updating analysts...")

        analysts = st.session_state["response"]["analysts"]
        data = pd.DataFrame(analysts)
        data.index += 1
        st.dataframe(data, use_container_width=True)
        
        if st.button("Select Final Analysts"):
            logger.info("Selecting final analysts...")
            with st.spinner("Selecting final analysts..."):
                input_data = {
                    "human_analyst_feedback": ["approved"],
                }
                st.session_state["response"] = st.session_state["client"].run_graph_resume(input_data=input_data)
                logger.info("Analysts selected, updating final analysts...")
                st.rerun()
    
    if "final_analysts" in st.session_state["response"]:
        st.header("Final Selected Analysts")
        final_analysts = st.session_state["response"]["final_analysts"]
        data_final = pd.DataFrame(final_analysts)
        data_final.index += 1
        st.dataframe(data_final, use_container_width=True)

        if st.button("Start Research"):
            logger.info("Starting research...")
            with st.spinner("Starting research..."):
                # Prepare input data for backend
                input_data = {}
                with st.container(height=300):
                    st.write_stream(st.session_state["client"].run_graph_stream(input_data=input_data))
            
            logger.info("Research completed, updating final report...")
            client_state = st.session_state["client"].get_state()
            final_report = client_state["final_report"]
            st.header("Final Report")
            st.markdown(final_report)
                