from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import LLMChain, MapReduceDocumentsChain, ReduceDocumentsChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.prompts import PromptTemplate
import streamlit as st
from langchain_groq import ChatGroq
import os

@st.cache_resource
def load_llm():
    try:
        return ChatGroq(
            model_name="llama3-70b-8192", 
            api_key=os.getenv("GROQ_API_KEY"),
            request_timeout=30
        )
    except Exception as e:
        st.error(f"Failed to load LLM: {str(e)}")
        return None

llm = load_llm()

def generate_summary(text: str) -> str:
    """Structured legal summary with clean headings and detailed content formatting."""
    if not text:
        return "No content to summarize"

    try:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=4000,
            chunk_overlap=200,
            separators=["\n\n", "\n", ". ", "! ", "? ", " "]
        )
        docs = text_splitter.create_documents([text])

        map_template = """
                You are a legal analyst. Summarize the legal document chunk using the exact format below.

                ⚠️ STRICT FORMATTING INSTRUCTIONS:
                - Each heading should be on its own line.
                - The content (description or list) should be on the next line only.
                - Do NOT place the content on the same line as the heading.
                - Use the provided headings only if relevant to the document.

                HEADINGS TO USE (if applicable):
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

                Text:
                {docs}

                Structured Summary:
                """
        map_prompt = PromptTemplate.from_template(map_template)

        reduce_template = """
You are a legal analyst. Combine and clean up the summaries below into one final summary with this strict format:

⚠️ STRICT FORMAT RULES:
- Each heading should appear on one line.
- The detail or list should appear on the line directly after it.
- Do not merge heading and content into a single line.
- Repeat this format for each relevant heading.
- Do not invent or hallucinate sections. Just merge and clean what is already present.

Summaries:
{doc_summaries}

Final Structured Summary:
"""
        reduce_prompt = PromptTemplate.from_template(reduce_template)

        map_chain = LLMChain(llm=llm, prompt=map_prompt)
        reduce_chain = LLMChain(llm=llm, prompt=reduce_prompt)

        combine_documents_chain = StuffDocumentsChain(
            llm_chain=reduce_chain,
            document_variable_name="doc_summaries"
        )

        reduce_documents_chain = ReduceDocumentsChain(
            combine_documents_chain=combine_documents_chain,
            collapse_documents_chain=combine_documents_chain,
            token_max=4000
        )

        return MapReduceDocumentsChain(
            llm_chain=map_chain,
            reduce_documents_chain=reduce_documents_chain,
            document_variable_name="docs"
        ).run(docs)
    except Exception as e:
        return f"Summary generation failed: {str(e)}"