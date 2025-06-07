from langgraph.graph import START, END, StateGraph
from state import ResearchState
from create_analysts import create_analysts, human_feedback, select_analysts, should_continue
from conduct_interviews import interview_builder, initiate_all_interviews
from generate_report import write_report, write_introduction, write_conclusion, finalize_report

builder = StateGraph(ResearchState)
builder.add_node("create_analysts", create_analysts)
builder.add_node("human_feedback", human_feedback)
builder.add_node("select_analysts", select_analysts)
builder.add_node("human_conduct_interview", human_feedback)
builder.add_node("conduct_interview", interview_builder.compile())
builder.add_node("write_report",write_report)
builder.add_node("write_introduction",write_introduction)
builder.add_node("write_conclusion",write_conclusion)
builder.add_node("finalize_report",finalize_report)

# Logic
builder.add_edge(START, "create_analysts")
builder.add_edge("create_analysts", "human_feedback")
builder.add_conditional_edges("human_feedback", should_continue, ["create_analysts", "select_analysts"])
builder.add_edge("select_analysts", "human_conduct_interview")
builder.add_conditional_edges("human_conduct_interview", initiate_all_interviews, ["create_analysts", "conduct_interview"])
builder.add_edge("conduct_interview", "write_report")
builder.add_edge("conduct_interview", "write_introduction")
builder.add_edge("conduct_interview", "write_conclusion")
builder.add_edge(["write_conclusion", "write_report", "write_introduction"], "finalize_report")
builder.add_edge("finalize_report", END)

# Compile
# memory = MemorySaver()
graph = builder.compile(interrupt_before=['human_feedback', 'human_conduct_interview'])