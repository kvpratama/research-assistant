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
    st.session_state["client"].assistant_id = st.session_state["client"].create_assistant("conduct_interviews")
    st.session_state["response"] = None

st.title("Conduct Interview with AI Analyst")

if not st.session_state["response"]:
    # User inputs

    if st.button("Conduct Interview"):
        logger.info("Conducting Interview...")
        with st.spinner("Conducting Interview..."):
            # Prepare input data for backend
            input_data = {
                "messages": ["Ask questions"],
            }
            st.write_stream(st.session_state["client"].run_graph_stream(input_data=input_data))
            
if st.session_state["response"]:
    question = st.session_state["response"]["messages"][-1]["content"]
    st.text(question)
    