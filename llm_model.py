import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools.tavily_search import TavilySearchResults

os.environ["LANGSMITH_PROJECT"] = "research-assistant"

class LLMConfig:
    DEFAULT = "gemini-2.0-flash"
    VERSATILE = "gemini-2.0-flash-lite"
    CREATIVE = "gemini-2.0-flash-lite"

def create_llm(model_name="gemini-2.0-flash", temperature=0, google_api_key=None):
    return ChatGoogleGenerativeAI(model=model_name, temperature=temperature, google_api_key=google_api_key)

def get_default_llm(google_api_key):
    return create_llm(LLMConfig.DEFAULT, temperature=0, google_api_key=google_api_key)

def get_versatile_llm(google_api_key):
    return create_llm(LLMConfig.VERSATILE, temperature=0.5, google_api_key=google_api_key)

def get_creative_llm(google_api_key):
    return create_llm(LLMConfig.CREATIVE, temperature=1.0, google_api_key=google_api_key)

def get_tavily_search(tavily_api_key):
    return TavilySearchResults(max_results=3, api_key=tavily_api_key)
    