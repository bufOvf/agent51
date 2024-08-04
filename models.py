import os
from langchain_groq import ChatGroq
from dotenv import load_dotenv

def get_api_key(service):
    load_dotenv()
    return os.getenv(f"{service}_API_KEY")

def get_groq_llama_31_405b_reasoning(temperature=DEFAULT_TEMPERATURE):
    api_key = get_api_key("GROQ")
    return ChatGroq(model_name="llama-3.1-405b-reasoning", temperature=temperature, api_key=api_key)

def get_groq_llama_31_70b_versatile(temperature=DEFAULT_TEMPERATURE):
    api_key = get_api_key("GROQ")
    return ChatGroq(model_name="llama-3.1-70b-versatile", temperature=temperature, api_key=api_key)

def get_groq_llama_31_8b_instant(temperature=DEFAULT_TEMPERATURE):
    api_key = get_api_key("GROQ")
    return ChatGroq(model_name="llama-3.1-8b-instant", temperature=temperature, api_key=api_key)

def get_groq_llama3_groq_70b_tool_use_preview(temperature=DEFAULT_TEMPERATURE):
    api_key = get_api_key("GROQ")
    return ChatGroq(model_name="llama3-groq-70b-8192-tool-use-preview", temperature=temperature, api_key=api_key)

def get_groq_llama3_groq_8b_tool_use_preview(temperature=DEFAULT_TEMPERATURE):
    api_key = get_api_key("GROQ")
    return ChatGroq(model_name="llama3-groq-8b-8192-tool-use-preview", temperature=temperature, api_key=api_key)

def get_mistral_mixtral_8x7b(temperature=DEFAULT_TEMPERATURE):
    api_key = get_api_key("GROQ")
    return ChatMistral(model_name="mixtral-8x7b-32768", temperature=temperature, api_key=api_key)

def get_whisper_whisper_large_v3(temperature=DEFAULT_TEMPERATURE):
    api_key = get_api_key("GROQ")
    return Whisper(model_name="whisper-large-v3", temperature=temperature, api_key=api_key)


    
