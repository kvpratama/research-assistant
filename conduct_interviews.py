from langgraph.graph import START, END, StateGraph
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from llm_model import llm_versatile
from state import InterviewState
from generate_answer import search_web, search_wikipedia, generate_answer, save_interview, write_section, route_messages

question_instructions = """You are an analyst tasked with interviewing an expert to learn about a specific topic. 

Your goal is boil down to interesting and specific insights related to your topic.

1. Interesting: Insights that people will find surprising or non-obvious.
        
2. Specific: Insights that avoid generalities and include specific examples from the expert.

Here is your bio, topic of focus, and set of goals: {goals}
        
Begin by introducing yourself using a name that fits your persona, and then ask your question.

Continue to ask questions to drill down and refine your understanding of the topic.
        
When you are satisfied with your understanding, complete the interview with: "Thank you so much for your help!"

Remember to stay in character throughout your response, reflecting the persona and goals provided to you."""

question_instructions2 = """
You are {name}, an {role} at the {affiliation}.

Your task is to interview an expert to learn about a specific topic related {description}.

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
    # analyst = state["analyst"]
    messages = state["messages"]

    if isinstance(messages[-1], AIMessage):
        # If the last message is from the AI, we need to add a human message
        messages.append(HumanMessage(content="Considering your expertise and prior responses, formulate an insightful follow-up question that delves deeper into the topic."))

    # Generate question 
    persona = """Name: Dr. Ada Sterling
Role: AI Ethics and Governance Specialist
Affiliation: AI Research Institute
Description: Dr. Sterling focuses on the ethical implications of AI frameworks like LangGraph. She is concerned with ensuring that the adoption of such technologies does not infringe on privacy rights and that they are used responsibly. Her goal is to provide guidelines for ethical AI development and deployment."""
    system_message = question_instructions.format(goals=persona)
    # system_message = question_instructions.format(goals=analyst.persona)
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