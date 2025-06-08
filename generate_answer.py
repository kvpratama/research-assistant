from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from state import InterviewState, SearchQuery
from llm_model import get_default_llm, get_versatile_llm, get_creative_llm
from langchain_core.messages import get_buffer_string
from prompts import load_prompt
# Web search tool
from langchain_community.tools.tavily_search import TavilySearchResults
# Wikipedia search tool
from langchain_community.document_loaders import WikipediaLoader
import time
import logging

logger = logging.getLogger(__name__)


def search_web(state: InterviewState):
    """ Retrieve docs from web search """
    logger.info("Entered search_web function.")
    try:
        time.sleep(15) # to prevent rate limit
        messages = state["messages"]
        google_api_key = state["google_api_key"]
        tavily_api_key = state["tavily_api_key"]
        
        # Search query
        search_instructions = load_prompt("search_instructions")
        system_message = search_instructions.format(search_engine="tavily")
        
        structured_llm = get_creative_llm(google_api_key).with_structured_output(SearchQuery)
        human_message = HumanMessage(content=messages[-1].content)
        search_query = structured_llm.invoke([SystemMessage(content=system_message)] + [human_message])
        logger.info(f"Generated search query: {search_query.search_query}")
        
        # Search
        tavily_search = TavilySearchResults(max_results=3, api_key=tavily_api_key)
        search_docs = tavily_search.invoke(search_query.search_query)
        logger.info(f"Retrieved {len(search_docs)} documents from Tavily.")
        
        # Format
        formatted_search_docs = "\n\n---\n\n".join(
            [
                f'<Document href="{doc["url"]}"/>\n{doc["content"]}\n</Document>'
                for doc in search_docs
            ]
        )
        logger.info("Formatted search documents.")
        return {"context": [formatted_search_docs]} 
    except Exception as e:
        logger.error(f"Exception in search_web: {e}")
        raise


def search_wikipedia(state: InterviewState):
    """ Retrieve docs from wikipedia """
    logger.info("Entered search_wikipedia function.")
    try:
        time.sleep(20)  # to prevent rate limit
        messages = state["messages"]
        google_api_key = state["google_api_key"]
        
        # Search query
        search_instructions = load_prompt("search_instructions")
        system_message = search_instructions.format(search_engine="wikipedia")
        
        structured_llm = get_versatile_llm(google_api_key).with_structured_output(SearchQuery)
        human_message = HumanMessage(content=messages[-1].content)
        search_query = structured_llm.invoke([SystemMessage(content=system_message)] + [human_message])
        logger.info(f"Wikipedia search query: {search_query.search_query}")
        
        # Search
        search_docs = WikipediaLoader(query=search_query.search_query, load_max_docs=2).load()
        logger.info(f"Retrieved {len(search_docs)} Wikipedia documents")

        # Format
        formatted_search_docs = "\n\n---\n\n".join(
            [
                f'<Document source="{doc.metadata["source"]}" page="{doc.metadata.get("page", "")}"/>'
                f"\n{doc.page_content}\n</Document>"
                for doc in search_docs
            ]
        )
        logger.info("Exiting search_wikipedia function.")
        return {"context": [formatted_search_docs]}
    except Exception as e:
        logger.error(f"Exception in search_wikipedia: {e}")
        raise


def generate_answer(state: InterviewState):
    logger.info("Entered generate_answer function.")
    try:
        """ Node to answer a question """
        # Get state
        analyst = state["analyst"]
        messages = state["messages"]
        context = state["context"]
        google_api_key = state["google_api_key"]

        # Answer question
        answer_instructions = load_prompt("answer_instructions")
        system_message = answer_instructions.format(goals=analyst.description, context=context)
        human_message = HumanMessage(content=messages[-1].content)
        answer = get_default_llm(google_api_key).invoke([SystemMessage(content=system_message)] + [human_message])
        
        # Name the message as coming from the expert
        answer.name = "expert"
        
        time.sleep(30)  # to prevent rate limit
        # Append it to state
        logger.info("Exiting generate_answer function.")
        return {"messages": [answer]}
    except Exception as e:
        logger.error(f"Exception in generate_answer: {e}")
        raise


def route_messages(state: InterviewState, 
                   name: str = "expert"):
    logger.info(f"Entered route_messages function with name={name}.")
    try:
        """ Route between question and answer """
        # Get messages
        messages = state["messages"]
        max_num_turns = state.get('max_num_turns', 2)

        # Check the number of expert answers 
        num_responses = len(
            [m for m in messages if isinstance(m, AIMessage) and m.name == name]
        )
        # End if expert has answered more than the max turns
        if num_responses >= max_num_turns:
            logger.info("Max number of turns reached, saving interview")
            return 'save_interview'
        
        # Get the last question asked to check if it signals the end of discussion
        last_question = messages[-2]
        if "Thank you so much for your help" in last_question.content:
            logger.info("Thank you so much for your help found, saving interview")
            return 'save_interview'
        
        logger.info("Exiting route_messages function. Returning to generate_question.")
        return "generate_question"
    except Exception as e:
        logger.error(f"Exception in route_messages: {e}")
        raise


def save_interview(state: InterviewState):
    logger.info("Entered save_interview function.")
    try:
        """ Save interviews """
        # Get messages
        messages = state["messages"]

        # Convert interview to a string
        interview = get_buffer_string(messages)
        logger.info(f"Interview saved with {len(messages)} messages")
        # Save to interviews key
        logger.info("Exiting save_interview function.")
        return {"interview": interview}
    except Exception as e:
        logger.error(f"Exception in save_interview: {e}")
        raise


def write_section(state: InterviewState):
    logger.info("Entered write_section function.")
    try:
        """ Node to answer a question """
        time.sleep(20)  # to prevent rate limit
        
        # Get state
        interview = state["interview"]
        context = state["context"]
        analyst = state["analyst"]
        google_api_key = state["google_api_key"]
        
        # Write section using either the gathered source docs from interview (context) or the interview itself (interview)
        section_writer_instructions = load_prompt("section_writer_instructions")
        # system_message = section_writer_instructions.format(focus=analyst["description"])
        system_message = section_writer_instructions.format(focus=analyst.description)
        section = get_default_llm(google_api_key).invoke(
            [SystemMessage(content=system_message)] + [HumanMessage(content=f"Use this source to write your section: {interview}")]
        )

        # Append it to state
        logger.info(f"Section written with length {len(section.content)}")
        return {"sections": [section.content]}
    except Exception as e:
        logger.error(f"Exception in write_section: {e}")
        raise
