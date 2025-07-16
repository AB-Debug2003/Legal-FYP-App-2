import requests
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq

# --- CONFIG ---
API_URL = "http://107.170.35.62:8000/summarize"

# --- Load LLaMA3 (Groq) ---
@st.cache_resource
def load_llm():
    try:
        return ChatGroq(
            model_name="llama3-70b-8192",
            api_key=st.secrets["GROQ_API_KEY"],
            request_timeout=60
        )
    except Exception as e:
        st.error(f"Failed to load LLM: {str(e)}")
        return None

llm = load_llm()


# --- Pegasus Raw Summary ---
def generate_raw_summary_with_pegasus(text: str) -> str:
    """Splits the doc and sends chunks to Pegasus FastAPI"""
    if not text:
        return "No content to summarize."

    try:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=3000,
            chunk_overlap=200,
            separators=["\n\n", "\n", ". ", "! ", "? ", " "]
        )
        chunks = splitter.split_text(text)

        all_summaries = []
        for chunk in chunks:
            response = requests.post(API_URL, json={
                "text": chunk,
                "max_length": 512,
                "min_length": 100
            }, timeout=120)

            if response.status_code == 200:
                summary = response.json().get("summary", "").strip()
                all_summaries.append(summary)
            else:
                all_summaries.append("[Error: Summarization failed for a chunk]")

        return "\n\n".join(all_summaries)

    except Exception as e:
        return f"Summary generation failed: {str(e)}"


# --- Format with LLaMA3 using Prompt ---
def format_summary_with_llama3(raw_summary: str, full_doc: str) -> str:
    prompt_template = """
You are a legal analyst. Format the following raw legal summary and original legal document using this exact structure:

⚠️ STRICT FORMAT INSTRUCTIONS:
- Each heading MUST appear on its own line.
- The explanation or list must come on the line right after the heading.
- Do NOT combine heading and content on the same line.
- Use the headings provided. If a heading is not applicable, you may skip it.
- Use both the raw summary and original document to complete missing information.

HEADINGS TO USE:
Case Background  
Appellants & Defendants  
Important Dates  
Type of Case  
Legal Issues Raised  
Jurisdictional Challenge  
Evidentiary Support  
Arguments by Petitioners  
Arguments by Respondents  
Case Laws Applied  
Lower Court Findings  
Court’s Decision  
Relief Sought  

Raw Summary:
{raw_summary}

Original Document:
{full_doc}

Formatted Structured Summary:
"""

    formatted_prompt = prompt_template.format(raw_summary=raw_summary, full_doc=full_doc)

    if not llm:
        return "LLM not available."

    try:
        response = llm.invoke(formatted_prompt)
        return response.strip()
    except Exception as e:
        return f"Formatting failed: {str(e)}"


# --- Final Orchestration ---
def generate_structured_legal_summary(doc_text: str) -> str:
    raw = generate_raw_summary_with_pegasus(doc_text)
    structured = format_summary_with_llama3(raw_summary=raw, full_doc=doc_text)
    return structured
