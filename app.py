import streamlit as st
from langgraph_client import LangGraphLocalClient
import pandas as pd
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Get API Key from user
if "api_key_entered" not in st.session_state:
    st.session_state.api_key_entered = False

if not st.session_state.api_key_entered:
    with st.sidebar:
        st.header("API Configuration")
        google_api_key = st.text_input("Enter your Google API Key:", type="password", key="google_api_key_input", value="AIzaSyDKr22mO6Wt9xfKWK1_AOgbkgKNqHV1ydg")
        tavily_api_key = st.text_input("Enter your Tavily API Key:", type="password", key="tavily_api_key_input", value="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")
        if st.button("Set API Key"):
            if google_api_key and tavily_api_key:
                logger.info("Initializing LangGraphLocalClient...")
                st.session_state["client"] = LangGraphLocalClient(google_api_key, tavily_api_key)
                st.session_state["response"] = None
                st.session_state.api_key_entered = True
                st.success("API Key set successfully!") 
                st.rerun()
            else:
                st.error("Please enter a valid API Key.") 
    st.markdown("""
    # AI Research Assistant v1.0 (Mini)

    Welcome to our lightweight, AI-powered research assistant!  
    This app helps you explore complex topics quickly and thoroughly using a team of collaborative AI analysts.

    ---

    ## How It Works

    ### 1. Enter a Topic
    Start by providing a topic you'd like to research.

    ### 2. Generate a Team of Analysts
    With a single click, the system creates a **team of AI analysts**, each focusing on a specific sub-topic.

    ### 3. Refine with Feedback
    You can review the analysts' focus areas and provide feedback.  
    The system uses your input to generate **additional analysts** to explore new directions or dig deeper.

    ### 4. Automatic Selection
    From the full set of analysts, the system automatically selects the **most relevant** ones for the research phase.

    ---

    ## AI Interviews

    Each selected analyst:

    - Conducts an **interview with an expert AI**, equipped with:
      - A **web browser**
      - **Wikipedia**
      - _(More expert tools coming soon!)_

    - Engages in a **dynamic, two-turn conversation** to explore the sub-topic
      - Includes follow-up questions and clarifications
      - Aims to extract rich, focused insights

    - Runs **in parallel** with the other analysts to speed up the process

    ---

    ## Final Report

    Once all interviews are complete:

    - The system **synthesizes the insights** from every analyst
    - Then generates a **final, structured report**  
      - Clean, concise, and easy to read  
      - Ready for review, sharing, or further analysis

    ---

    ## Inspired By

    Based on research [Assisting in Writing Wikipedia-like Articles From Scratch with Large Language Models](https://arxiv.org/abs/2402.14207).
    
    """)
    st.stop() 

st.title("AI Research Assistant v1.0")

if not st.session_state["response"]:
    with st.sidebar:
        st.markdown("""
        ### 1. Enter your research topic
        - Start by providing a topic you'd like to research.

        ### 2. Generate a Team of Analysts
        - With a single click, the system creates a **team of AI analysts**, each focusing on a specific sub-topic.
        """
        )
    
    # User inputs
    topic = st.text_area("Enter your research topic:", height=100, value="Renewable Energy Solutions for a Sustainable Future")
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
        with st.sidebar:
            st.markdown("""
            ### 3. Refine with Feedback
            - You can review the analysts' focus areas and provide feedback.  
            - The system uses your input to generate **additional analysts** to explore new directions.
            - **Or, if you're happy with the team, move on to analyst selection.**

            ### 4. Automatic Selection
            - From the full set of analysts, the system automatically selects the **most relevant** ones for the research phase.
            """
            )
        feedback = st.text_area("Optional feedback:", height=70, value="Add more analysts focused on emerging technologies.")
        if st.button("Add Feedback"):
            logger.info("Adding feedback...")
            with st.spinner("Generating more analysts..."):
                input_data = {
                    "human_analyst_feedback": [feedback] if feedback else [],
                }
                st.session_state["response"] = st.session_state["client"].run_graph_resume(input_data=input_data)
                logger.info("Feedback added, updating analysts...")

        analysts = st.session_state["response"]["analysts"]
        data = pd.DataFrame([analyst.to_dict for analyst in analysts])
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
        with st.sidebar:
            st.markdown("""
            ## AI Research and Interviews

            Each selected analyst:

            - Interviews an expert AI with access to a **web browser** and **Wikipedia**
            - Has a **conversation** with follow-up questions for deeper insights

            ---

            ## Final Report

            After the research is complete:

            - Insights are **synthesized** into a **clear, structured report**

            ---

            ## Disclaimer

            1. While this AI system is designed to assist with research and provide useful insights, 
            it may occasionally produce inaccurate or outdated information.  
            
            2. Always double-check key facts and consult reliable sources before making decisions based on the results.
            
            ---

            *Note:* The research may take **2–3 minutes** to complete.  
            Sit back, relax, and let your team of AI analysts do the heavy lifting!
            """
            )
        st.header("Final Selected Analysts")
        final_analysts = st.session_state["response"]["final_analysts"]
        data_final = pd.DataFrame([analyst.to_dict for analyst in final_analysts])
        data_final.index += 1
        st.dataframe(data_final, use_container_width=True)

        if st.button("Start Research"):
            logger.info("Starting research...")
            with st.spinner("Conducting research..."):
                input_data = {}
                with st.container(height=300):
                    st.write_stream(st.session_state["client"].run_graph_stream(input_data=input_data))
            
            logger.info("Research completed, updating final report...")
            client_state = st.session_state["client"].get_state()
            logger.debug(f"Client state: {client_state}")
            final_report = client_state["final_report"]
            st.header("Final Report")
            st.markdown(final_report)
                