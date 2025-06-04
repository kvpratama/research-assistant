from typing import List, Annotated
from typing_extensions import TypedDict
from pydantic import BaseModel, Field
from operator import add
from langgraph.graph import START, END, StateGraph
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from llm_model import llm, llm_creative

class Analyst(BaseModel):
    affiliation: str = Field(
        description="Primary affiliation of the analyst.",
    )
    name: str = Field(
        description="Name of the analyst."
    )
    role: str = Field(
        description="Role of the analyst in the context of the topic.",
    )
    description: str = Field(
        description="Description of the analyst focus, concerns, and motives.",
    )
    @property
    def persona(self) -> str:
        return f"Name: {self.name}\nRole: {self.role}\nAffiliation: {self.affiliation}\nDescription: {self.description}\n"

class Perspectives(BaseModel):
    analysts: List[Analyst] = Field(
        description="Comprehensive list of analysts with their roles and affiliations.",
    )

class GenerateAnalystsState(TypedDict):
    topic: str # Research topic
    max_analysts: int # Number of analysts
    # human_analyst_feedback: str # Human feedback
    human_analyst_feedback: Annotated[List[str], add]  # Human feedback
    # analysts: List[Analyst] # Analyst asking questions
    analysts: Annotated[List[Analyst], add] 
    final_analysts: List[Analyst] #= Field(..., min_items=3, max_items=3, description="Exactly 3 related sub-topics")# Analyst asking questions

analyst_instructions="""You are tasked with creating a set of AI analyst personas. Follow these instructions carefully:

1. First, review the research topic:
{topic}
        
2. Examine any editorial feedback that has been optionally provided to guide creation of the analysts: 
        
{human_analyst_feedback}
    
3. Determine the most interesting themes based upon documents and / or feedback above.
                    
4. Pick the top {max_analysts} themes.

5. Assign one analyst to each theme."""

selector_instructions="""You are tasked with selecting {max_analysts} AI analyst personas. Follow these instructions carefully:

1. First, review the research topic:
{topic}
        
2. Examine any editorial feedback that has been optionally provided to guide creation of the analysts: 
        
{human_analyst_feedback}
    
3. Here are the candidates:
{candidates}
                    
4. Pick {max_analysts} analysts suitable for researching about {topic}. Make sure to select no more and no less than {max_analysts} analysts.
"""

def create_analysts(state: GenerateAnalystsState):
    
    """ Create analysts """
    
    topic=state['topic']
    max_analysts=state['max_analysts']
    # human_analyst_feedback=state.get('human_analyst_feedback', '')
    # import pdb; pdb.set_trace()
    human_analyst_feedback=state.get('human_analyst_feedback', [])
    if human_analyst_feedback:
        human_analyst_feedback = human_analyst_feedback[-1]
    else:
        human_analyst_feedback = ""
                
    # Enforce structured output
    # structured_llm = llm.with_structured_output(Perspectives)
    structured_llm = llm_creative.with_structured_output(Perspectives)

    # System message
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
