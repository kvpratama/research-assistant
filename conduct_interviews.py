from typing import List, Annotated
from typing_extensions import TypedDict
from pydantic import BaseModel, Field
from operator import add
from langgraph.graph import START, END, StateGraph
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from llm_model import llm, llm_creative, llm_versatile
from create_analysts import Analyst
from langgraph.graph import MessagesState

class InterviewState(MessagesState):
    max_num_turns: int # Number turns of conversation
    context: Annotated[list, add] # Source docs
    analyst: Analyst # Analyst asking questions
    interview: str # Interview transcript
    sections: list # Final key we duplicate in outer state for Send() API

class SearchQuery(BaseModel):
    search_query: str = Field(None, description="Search query for retrieval.")

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


# Add nodes and edges 
builder = StateGraph(InterviewState)
builder.add_node("generate_question", generate_question)

builder.add_edge(START, "generate_question")
builder.add_edge("generate_question", END)

# Compile
graph = builder.compile()