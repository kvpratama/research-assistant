from langgraph.graph import START, END, StateGraph
from langchain_core.messages import HumanMessage, SystemMessage
from llm_model import llm, llm_creative
from state import GenerateAnalystsState, Perspectives
from prompts import load_prompt

def create_analysts(state: GenerateAnalystsState):
    
    """ Create analysts """
    
    topic=state['topic']
    max_analysts=state['max_analysts']
    # human_analyst_feedback=state.get('human_analyst_feedback', '')
    human_analyst_feedback=state.get('human_analyst_feedback', [])
    if human_analyst_feedback:
        human_analyst_feedback = human_analyst_feedback[-1]
    else:
        human_analyst_feedback = ""
                
    # Enforce structured output
    # structured_llm = llm.with_structured_output(Perspectives)
    structured_llm = llm_creative.with_structured_output(Perspectives)

    # System message
    analyst_instructions = load_prompt("analyst_instructions")
    system_message = analyst_instructions.format(topic=topic,
                                                            human_analyst_feedback=human_analyst_feedback, 
                                                            max_analysts=max_analysts)

    # Generate question 
    analysts = structured_llm.invoke([SystemMessage(content=system_message)]+[HumanMessage(content="Generate the set of analysts.")])
    
    # Write the list of analysis to state
    return {"analysts": analysts.analysts}

def select_analysts(state: GenerateAnalystsState):
    
    """ Select analysts """
    
    topic=state['topic']
    max_analysts=state['max_analysts']
    human_analyst_feedback=state.get('human_analyst_feedback', [])
    human_analyst_feedback = "\n".join(feedback for feedback in human_analyst_feedback if feedback)
    candidates = state.get('analysts', [])
    candidates = "\n\n".join(analyst.persona for analyst in candidates)
                
    # Enforce structured output
    structured_llm = llm.with_structured_output(Perspectives)

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

    # Check if human feedback
    human_analyst_feedback=state.get('human_analyst_feedback', ["approved"])
    if human_analyst_feedback[-1] == "approved":
        return "select_analysts"
    return "create_analysts"


# Add nodes and edges 
builder = StateGraph(GenerateAnalystsState)
builder.add_node("create_analysts", create_analysts)
builder.add_node("human_feedback", human_feedback)
builder.add_node("select_analysts", select_analysts)
builder.add_edge(START, "create_analysts")
builder.add_edge("create_analysts", "human_feedback")
builder.add_conditional_edges("human_feedback", should_continue, ["create_analysts", "select_analysts"])
builder.add_edge("select_analysts", END)

# Compile
# memory = MemorySaver()
graph = builder.compile(interrupt_before=['human_feedback'])
