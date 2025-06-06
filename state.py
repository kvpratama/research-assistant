from pydantic import BaseModel, Field
from langgraph.graph import MessagesState
from typing import List, Annotated
from typing_extensions import TypedDict
from operator import add

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

class InterviewState(MessagesState):
    topic: str # Research topic
    max_num_turns: int # Number turns of conversation
    context: Annotated[list, add] # Source docs
    analyst: Analyst # Analyst asking questions
    interview: str # Interview transcript
    sections: list # Final key we duplicate in outer state for Send() API

class InterviewStateOutput(MessagesState):
    sections: list # Final key we duplicate in outer state for Send() API

class SearchQuery(BaseModel):
    search_query: str = Field(None, description="Search query for retrieval.")

class ResearchState(MessagesState):
    topic: str # Research topic
    max_analysts: int # Number of analysts
    human_analyst_feedback: Annotated[List[str], add]  # Human feedback
    analysts: Annotated[List[Analyst], add] 
    final_analysts: List[Analyst]
    sections: Annotated[list, add]
    introduction: str # Introduction for the final report
    content: str # Content for the final report
    conclusion: str # Conclusion for the final report
    final_report: str # Final report