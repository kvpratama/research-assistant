import streamlit as st
from langgraph_client import LangGraphClient

st.title("AI Analyst Persona Generator")

# User inputs
topic = st.text_area("Enter research topic:", height=100)
max_analysts = st.number_input("Number of analysts:", min_value=1, max_value=10, value=3)

# Initialize client in session state if not present
if "client" not in st.session_state:
    print("Initializing LangGraphClient...")
    st.session_state["client"] = LangGraphClient()
    st.session_state["response"] = {}

if st.button("Generate Analysts"):
    print("Generating analysts...")
    with st.spinner("Generating analysts..."):
        # Prepare state for backend
        input_data = {
            "topic": topic,
            "max_analysts": max_analysts,
        }
        st.session_state["response"] = st.session_state["client"].run_graph(input_data=input_data)
        analysts = st.session_state["response"]["analysts"]
        st.subheader("Generated Analyst Personas")
        # for idx, analyst in enumerate(analysts, 1):
        #     st.markdown(f"**Analyst {idx}:**")
        #     st.markdown(f"- **Name:** {analyst["name"]}")
        #     st.markdown(f"- **Role:** {analyst["role"]}")
        #     st.markdown(f"- **Affiliation:** {analyst["affiliation"]}")
        #     st.markdown(f"- **Description:** {analyst["description"]}")
        #     st.markdown("---")
            
if st.session_state["response"]:
    print("Response already exists in session state.")
    analysts = st.session_state["response"]["analysts"]
    st.subheader("Generated Analyst Personas")
    for idx, analyst in enumerate(analysts, 1):
        st.markdown(f"**Analyst {idx}:**")
        st.markdown(f"- **Name:** {analyst["name"]}")
        st.markdown(f"- **Role:** {analyst["role"]}")
        st.markdown(f"- **Affiliation:** {analyst["affiliation"]}")
        st.markdown(f"- **Description:** {analyst["description"]}")
        st.markdown("---")
    
    feedback = st.text_area("Optional editorial feedback:", height=70)

    if st.button("Add Feedback"):
        st.write("Feedback added, updating analysts...")
        print("Adding feedback...")
        with st.spinner("Generating more analysts..."):
            input_data = {
                "human_analyst_feedback": [feedback] if feedback else [],
            }
            st.session_state["response"] = st.session_state["client"].run_graph_resume(input_data=input_data)
            print("Feedback added, updating analysts...")
            analysts = st.session_state["response"]["analysts"]
            st.subheader("Generated Analyst Personas")
            for idx, analyst in enumerate(analysts, 1):
                st.markdown(f"**Analyst {idx}:**")
                st.markdown(f"- **Name:** {analyst["name"]}")
                st.markdown(f"- **Role:** {analyst["role"]}")
                st.markdown(f"- **Affiliation:** {analyst["affiliation"]}")
                st.markdown(f"- **Description:** {analyst["description"]}")
                st.markdown("---")
        