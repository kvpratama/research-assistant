from state import ResearchState
import time
from prompts import load_prompt
from langchain_core.messages import HumanMessage, SystemMessage
from llm_model import get_default_llm, get_versatile_llm, get_creative_llm


def write_report(state: ResearchState, config: dict):
    time.sleep(15)  # to prevent rate limit

    sections = state["sections"]
    topic = state["topic"]
    google_api_key = config["configurable"]["google_api_key"]

    # Concat all sections together
    formatted_str_sections = "\n\n".join([f"{section}" for section in sections])
    
    # Summarize the sections into a final report
    report_writer_instructions = load_prompt("report_writer_instructions")
    system_message = report_writer_instructions.format(topic=topic, context=formatted_str_sections)    
    report = get_default_llm(google_api_key).invoke([SystemMessage(content=system_message)]+[HumanMessage(content=f"Write a report based upon these memos.")]) 
    return {"content": report.content}


def write_introduction(state: ResearchState, config: dict):
    time.sleep(15)  # to prevent rate limit
    
    sections = state["sections"]
    topic = state["topic"]
    google_api_key = config["configurable"]["google_api_key"]

    # Concat all sections together
    formatted_str_sections = "\n\n".join([f"{section}" for section in sections])
    
    # Summarize the sections into a final report
    intro_conclusion_instructions = load_prompt("intro_conclusion_instructions")
    instructions = intro_conclusion_instructions.format(topic=topic, formatted_str_sections=formatted_str_sections)    
    intro = get_creative_llm(google_api_key).invoke([instructions]+[HumanMessage(content=f"Write the report introduction")]) 
    return {"introduction": intro.content}


def write_conclusion(state: ResearchState, config: dict):
    time.sleep(15)  # to prevent rate limit
    
    sections = state["sections"]
    topic = state["topic"]
    google_api_key = config["configurable"]["google_api_key"]

    # Concat all sections together
    formatted_str_sections = "\n\n".join([f"{section}" for section in sections])
    
    # Summarize the sections into a final report
    intro_conclusion_instructions = load_prompt("intro_conclusion_instructions")
    instructions = intro_conclusion_instructions.format(topic=topic, formatted_str_sections=formatted_str_sections)    
    conclusion = get_versatile_llm(google_api_key).invoke([instructions]+[HumanMessage(content=f"Write the report conclusion")]) 
    return {"conclusion": conclusion.content}


def finalize_report(state: ResearchState):
    """ The is the "reduce" step where we gather all the sections, combine them, and reflect on them to write the intro/conclusion """
    # Save full final report
    content = state["content"]
    if content.startswith("## Insights"):
        content = content.strip("## Insights")
    if "## Sources" in content:
        try:
            content, sources = content.split("\n## Sources\n")
        except:
            sources = None
    else:
        sources = None

    final_report = state["introduction"] + "\n\n---\n\n" + content + "\n\n---\n\n" + state["conclusion"]
    if sources is not None:
        final_report += "\n\n## Sources\n" + sources
    return {"final_report": final_report}