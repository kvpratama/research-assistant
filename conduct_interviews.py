from langgraph.graph import START, END, StateGraph
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from llm_model import llm_versatile
from state import InterviewState, ResearchState
from generate_answer import search_web, search_wikipedia, generate_answer, save_interview, write_section, route_messages
from prompts import load_prompt
from langgraph.constants import Send

def generate_question(state: InterviewState):
    """ Node to generate a question """

    # Get state
    print("Generating question")
    print("State:", state)
    analyst = state["analyst"]
    messages = state["messages"]
    topic = state["topic"]

    # If no messages, start with a system message
    if not messages:
        messages = [HumanMessage(f"Considering your expertise, ask your first question about {topic}?")]
    elif isinstance(messages[-1], AIMessage):
        # If the last message is from the AI, we need to add a human message
        messages.append(HumanMessage(content="Considering your expertise and prior responses, formulate an insightful follow-up question that delves deeper into the topic."))

    question_instructions = load_prompt("question_instructions")
    # system_message = question_instructions.format(name=analyst["name"], role=analyst["role"], affiliation=analyst["affiliation"], description=analyst["description"])
    system_message = question_instructions.format(name=analyst.name, role=analyst.role, affiliation=analyst.affiliation, description=analyst.description)
    question = llm_versatile.invoke([SystemMessage(content=system_message)]+messages)
        
    # Write messages to state
    return {"messages": [question]}


def initiate_all_interviews(state: ResearchState):
    """ This is the "map" step where we run each interview sub-graph using Send API """

    print("Initiating interviews for topic:", state["topic"])
    print("Final analysts:", state["final_analysts"])

    # return {"analyst": state["final_analysts"][0]}
    return [Send("conduct_interview", {"analyst": analyst, "topic": state["topic"]}) for analyst in state["final_analysts"]]
    # return {"conduct_interview": [Send("conduct_interview", {"analyst": analyst,
    #                                     "messages": []}) for analyst in state["final_analysts"]]}

interview_builder = StateGraph(InterviewState)
interview_builder.add_node("generate_question", generate_question)
interview_builder.add_node("search_web", search_web)
interview_builder.add_node("search_wikipedia", search_wikipedia)
interview_builder.add_node("generate_answer", generate_answer)
interview_builder.add_node("save_interview", save_interview)
interview_builder.add_node("write_section", write_section)

# Flow
interview_builder.add_edge(START, "generate_question")
interview_builder.add_edge("generate_question", "search_web")
interview_builder.add_edge("generate_question", "search_wikipedia")
interview_builder.add_edge("search_web", "generate_answer")
interview_builder.add_edge("search_wikipedia", "generate_answer")
interview_builder.add_conditional_edges("generate_answer", route_messages,['generate_question','save_interview'])
interview_builder.add_edge("save_interview", "write_section")
interview_builder.add_edge("write_section", END)

# Interview 
# memory = MemorySaver()
interview_graph = interview_builder.compile().with_config(run_name="Conduct Interviews")