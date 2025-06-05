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
    # topic = st.text_area("Enter research topic:", height=100, value="Climate change")
    # max_analysts = st.number_input("Number of analysts:", min_value=1, max_value=10, value=3)

    if st.button("Generate Question"):
        logger.info("Generating question...")
        with st.spinner("Generating question..."):
            # Prepare input data for backend
            input_data = {
                "messages": ["Ask questions"],
            }
            st.session_state["response"] = st.session_state["client"].run_graph(input_data=input_data)
            st.rerun() 
            
if st.session_state["response"]:
    question = st.session_state["response"]["messages"][-1]["content"]
    st.text(question)
    