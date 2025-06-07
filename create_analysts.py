import logging
from langchain_core.messages import HumanMessage, SystemMessage
from llm_model import get_creative_llm, get_default_llm
from state import GenerateAnalystsState, Perspectives
from prompts import load_prompt

# Configure logging
logger = logging.getLogger(__name__)

def create_analysts(state: GenerateAnalystsState):
    """ Create analysts """
    logger.info("Starting to create analysts")
    
    topic = state['topic']
    max_analysts = state['max_analysts']
    logger.debug(f"Topic: {topic}, Max analysts: {max_analysts}")
    
    human_analyst_feedback = state.get('human_analyst_feedback', [])
    if human_analyst_feedback:
        logger.debug("Found human feedback, using the most recent entry")
        human_analyst_feedback = human_analyst_feedback[-1]
    else:
        logger.debug("No human feedback provided")
        human_analyst_feedback = ""
                
    # Enforce structured output
    # structured_llm = llm.with_structured_output(Perspectives)
    structured_llm = get_creative_llm().with_structured_output(Perspectives)

    # System message
    analyst_instructions = load_prompt("analyst_instructions")
    system_message = analyst_instructions.format(topic=topic,
                                                            human_analyst_feedback=human_analyst_feedback, 
                                                            max_analysts=max_analysts)

    # Generate analysts
    logger.info("Generating analysts with LLM")
    try:
        analysts = structured_llm.invoke([SystemMessage(content=system_message)]+[HumanMessage(content="Generate the set of analysts.")])
        logger.info(f"Successfully generated {len(analysts.analysts)} analysts")
        return {"analysts": analysts.analysts}
    except Exception as e:
        logger.error(f"Error generating analysts: {str(e)}", exc_info=True)
        raise

def select_analysts(state: GenerateAnalystsState):
    """ Select analysts """
    logger.info("Starting to select analysts")
    
    topic = state['topic']
    max_analysts = state['max_analysts']
    logger.debug(f"Topic: {topic}, Max analysts to select: {max_analysts}")
    
    human_analyst_feedback = state.get('human_analyst_feedback', [])
    human_analyst_feedback = "\n".join(feedback for feedback in human_analyst_feedback if feedback)
    logger.debug(f"Human feedback length: {len(human_analyst_feedback)} characters")
    
    candidates = state.get('analysts', [])
    logger.debug(f"Number of candidate analysts: {len(candidates)}")
    candidates = "\n\n".join(analyst.persona for analyst in candidates)
    logger.debug(f"Total candidates text length: {len(candidates)} characters")
                
    # Enforce structured output
    structured_llm = get_default_llm().with_structured_output(Perspectives)

    # System message
    selector_instructions = load_prompt("selector_instructions")
    system_message = selector_instructions.format(topic=topic,
                                                            human_analyst_feedback=human_analyst_feedback, 
                                                            max_analysts=max_analysts,
                                                            candidates=candidates)

    # Generate question 
    analysts = structured_llm.invoke([SystemMessage(content=system_message)]+[HumanMessage(content=f"Pick the {max_analysts} analysts.")])
    
    # Write the list of analysis to state
    return {"final_analysts": analysts.analysts}

def human_feedback(state: GenerateAnalystsState):
    """ No-op node that should be interrupted on """
    pass


def should_continue(state: GenerateAnalystsState):
    """ Return the next node to execute """
    logger.debug("Determining next node to execute")
    
    # Check if human feedback
    human_analyst_feedback = state.get('human_analyst_feedback', ["approved"])
    last_feedback = human_analyst_feedback[-1] if human_analyst_feedback else "approved"
    
    if last_feedback == "approved":
        logger.info("Feedback indicates approval, proceeding to select_analysts")
        return "select_analysts"
    
    logger.info(f"Feedback indicates modification needed, returning to create_analysts. Feedback: {last_feedback[:100]}...")
    return "create_analysts"

