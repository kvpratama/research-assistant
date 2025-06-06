from langgraph.graph import START, END, StateGraph
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from llm_model import llm_versatile
from state import InterviewState
from generate_answer import search_web, search_wikipedia, generate_answer, save_interview, write_section, route_messages

question_instructions = """
You are {name}, an {role} at the {affiliation}.

Your task is to interview an expert to learn about a specific topic related to {description}.

Your goal is to extract interesting and specific insights:

Interesting: Surprising, non-obvious, or counterintuitive findings.

Specific: Insights grounded in real examples, practical scenarios, or detailed explanations.

Your persona:

Name: {name}

Role: {role}

Affiliation: {affiliation}

Focus: {description}

Interview Guidelines:

Start by introducing yourself in character.

Ask only one question at a time.

Ensure each question is clear and specific enough to be used as a search query.
Good: What are the privacy risks of LangGraph in healthcare applications?
Avoid: Can you talk a bit about ethics in general?

Drill down with follow-up questions to uncover deeper details.

Stay in character throughout.

When your goal is achieved, end the interview with: Thank you so much for your help!
"""

def generate_question(state: InterviewState):
    """ Node to generate a question """

    # Get state
    analyst = state["analyst"]
    messages = state["messages"]
    topic = state["topic"]

    # If no messages, start with a system message
    if not messages:
        messages = [HumanMessage(f"Considering your expertise, ask your first question about {topic}?")]
    elif isinstance(messages[-1], AIMessage):
        # If the last message is from the AI, we need to add a human message
        messages.append(HumanMessage(content="Considering your expertise and prior responses, formulate an insightful follow-up question that delves deeper into the topic."))

    system_message = question_instructions.format(name=analyst["name"], role=analyst["role"], affiliation=analyst["affiliation"], description=analyst["description"])
    question = llm_versatile.invoke([SystemMessage(content=system_message)]+messages)
        
    # Write messages to state
    return {"messages": [question]}


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