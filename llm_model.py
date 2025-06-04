import os
if "GOOGLE_API_KEY" not in os.environ:
        os.environ["GOOGLE_API_KEY"] = os.environ["GEMINI_API_KEY"]

# if "MISTRAL_API_KEY" not in os.environ:
#     os.environ["MISTRAL_API_KEY"] = os.environ["MISTRAL_API_KEY"]

from langchain_google_genai import ChatGoogleGenerativeAI
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash" ,temperature=0)
# llm_versatile = ChatGoogleGenerativeAI(model="gemma-3n-e4b-it" ,temperature=0.5)
# llm_versatile = ChatGoogleGenerativeAI(model="gemma-3-27b-it" ,temperature=0.5)
llm_versatile = ChatGoogleGenerativeAI(model="gemini-2.0-flash-lite" ,temperature=0.5)

# from langchain_mistralai import ChatMistralAI
# llm_creative = ChatMistralAI(model="mistral-small-2503", temperature=1)
llm_creative = ChatGoogleGenerativeAI(model="gemini-2.0-flash-lite" ,temperature=1)

# _set_env("LANGSMITH_API_KEY")
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_PROJECT"] = "langchain-academy"
