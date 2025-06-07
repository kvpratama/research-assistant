import os
from langchain_google_genai import ChatGoogleGenerativeAI


if "GOOGLE_API_KEY" not in os.environ:
        os.environ["GOOGLE_API_KEY"] = os.environ["GEMINI_API_KEY"]

os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_PROJECT"] = "langchain-academy"

class LLMConfig:
    DEFAULT = "gemini-2.0-flash"
    VERSATILE = "gemini-2.0-flash-lite"
    CREATIVE = "gemini-2.0-flash-lite"

def create_llm(model_name="gemini-2.0-flash", temperature=0):
    return ChatGoogleGenerativeAI(model=model_name, temperature=temperature)

def get_default_llm():
    return create_llm(LLMConfig.DEFAULT, temperature=0)

def get_versatile_llm():
    return create_llm(LLMConfig.VERSATILE, temperature=0.5)

def get_creative_llm():
    return create_llm(LLMConfig.CREATIVE, temperature=1.0)