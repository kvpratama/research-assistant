from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from state import InterviewState, SearchQuery
from llm_model import get_default_llm, get_versatile_llm, get_creative_llm
from langchain_core.messages import get_buffer_string
from langgraph.graph import START, END, StateGraph
from prompts import load_prompt
# Web search tool
from langchain_community.tools.tavily_search import TavilySearchResults
tavily_search = TavilySearchResults(max_results=3)

# Wikipedia search tool
from langchain_community.document_loaders import WikipediaLoader
import time


def search_web(state: InterviewState):
    
    """ Retrieve docs from web search """
    time.sleep(15) # to prevent rate limit
    # Search query
    messages = state["messages"]
    search_instructions = load_prompt("search_instructions")
    system_message = search_instructions.format(search_engine="tavily")
    structured_llm = get_creative_llm().with_structured_output(SearchQuery)
    # search_query = structured_llm.invoke([search_instructions]+state['messages'])
    human_message = HumanMessage(content=messages[-1].content)
    search_query = structured_llm.invoke([SystemMessage(content=system_message)] + [human_message])
    
    # Search
    search_docs = tavily_search.invoke(search_query.search_query)

     # Format
    formatted_search_docs = "\n\n---\n\n".join(
        [
            f'<Document href="{doc["url"]}"/>\n{doc["content"]}\n</Document>'
            for doc in search_docs
        ]
    )

    return {"context": [formatted_search_docs]} 


def search_wikipedia(state: InterviewState):
    
    """ Retrieve docs from wikipedia """

    time.sleep(20)  # to prevent rate limit
    # Search query
    messages = state["messages"]
    search_instructions = load_prompt("search_instructions")
    system_message = search_instructions.format(search_engine="wikipedia")
    structured_llm = get_versatile_llm().with_structured_output(SearchQuery)
    # search_query = structured_llm.invoke([search_instructions]+state['messages'])
    human_message = HumanMessage(content=messages[-1].content)
    search_query = structured_llm.invoke([SystemMessage(content=system_message)] + [human_message])
    
    # Search
    search_docs = WikipediaLoader(query=search_query.search_query, 
                                  load_max_docs=2).load()

     # Format
    formatted_search_docs = "\n\n---\n\n".join(
        [
            f'<Document source="{doc.metadata["source"]}" page="{doc.metadata.get("page", "")}"/>\n{doc.page_content}\n</Document>'
            for doc in search_docs
        ]
    )

    return {"context": [formatted_search_docs]} 


def generate_answer(state: InterviewState):
    
    """ Node to answer a question """

    # Get state
    analyst = state["analyst"]
    messages = state["messages"]
    context = state["context"]

    # Answer question
    answer_instructions = load_prompt("answer_instructions")
    # system_message = answer_instructions.format(goals=analyst["description"], context=context)
    system_message = answer_instructions.format(goals=analyst.description, context=context)
    human_message = HumanMessage(content=messages[-1].content)
    answer = get_default_llm().invoke([SystemMessage(content=system_message)] + [human_message])
            
    # Name the message as coming from the expert
    answer.name = "expert"
    
    time.sleep(30)  # to prevent rate limit
    # Append it to state
    return {"messages": [answer]}


def route_messages(state: InterviewState, 
                   name: str = "expert"):

    """ Route between question and answer """
    
    # Get messages
    messages = state["messages"]
    max_num_turns = state.get('max_num_turns',2)

    # Check the number of expert answers 
    num_responses = len(
        [m for m in messages if isinstance(m, AIMessage) and m.name == name]
    )

    # End if expert has answered more than the max turns
    if num_responses >= max_num_turns:
        return 'save_interview'

    # This router is run after each question - answer pair 
    # Get the last question asked to check if it signals the end of discussion
    last_question = messages[-2]
    
    if "Thank you so much for your help" in last_question.content:
        return 'save_interview'
    return "generate_question"


def save_interview(state: InterviewState):
    
    """ Save interviews """

    # Get messages
    messages = state["messages"]
    
    # Convert interview to a string
    interview = get_buffer_string(messages)
    
    # Save to interviews key
    return {"interview": interview}


def write_section(state: InterviewState):

    """ Node to answer a question """

    time.sleep(20)  # to prevent rate limit
    # Get state
    interview = state["interview"]
    context = state["context"]
    analyst = state["analyst"]
   
    # Write section using either the gathered source docs from interview (context) or the interview itself (interview)
    section_writer_instructions = load_prompt("section_writer_instructions")
    # system_message = section_writer_instructions.format(focus=analyst["description"])
    system_message = section_writer_instructions.format(focus=analyst.description)
    section = get_default_llm().invoke([SystemMessage(content=system_message)]+[HumanMessage(content=f"Use this source to write your section: {interview}")]) 
                
    # Append it to state
    return {"sections": [section.content]}
