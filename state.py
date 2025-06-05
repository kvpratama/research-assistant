from pydantic import BaseModel, Field
from langgraph.graph import MessagesState
from typing import List, Annotated
from typing_extensions import TypedDict
from operator import add
from create_analysts import Analyst

class InterviewState(MessagesState):
    max_num_turns: int # Number turns of conversation
    context: Annotated[list, add] # Source docs
    analyst: Analyst # Analyst asking questions
    interview: str # Interview transcript
    sections: list # Final key we duplicate in outer state for Send() API

class SearchQuery(BaseModel):
    search_query: str = Field(None, description="Search query for retrieval.")