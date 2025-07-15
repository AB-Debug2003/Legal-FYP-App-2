import requests
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter

API_URL = "http://107.170.35.62:8000/summarize"

def generate_summary(text: str) -> str:
    """Call Pegasus FastAPI to summarize a legal document."""
    if not text:
        return "No content to summarize"

    try:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=3000,
            chunk_overlap=200,
            separators=["\n\n", "\n", ". ", "! ", "? ", " "]
        )
        chunks = text_splitter.split_text(text)

        all_summaries = []
        for chunk in chunks:
            payload = {
                "text": chunk,
                "max_length": 256,
                "min_length": 30
            }
            response = requests.post(API_URL, json=payload, timeout=60)
            if response.status_code == 200:
                summary = response.json().get("summary", "")
                all_summaries.append(summary)
            else:
                all_summaries.append("[Error: Summarization failed for a chunk]")

        return "\n\n".join(all_summaries)

    except Exception as e:
        return f"Summary generation failed: {str(e)}"
