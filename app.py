import plotly.express as px
import nltk
nltk.download(['punkt', 'punkt_tab', 'averaged_perceptron_tagger', 'vader_lexicon'])
import streamlit as st
import os
import base64
import faiss
import numpy as np
import fitz  # PyMuPDF
import pandas as pd
from fpdf import FPDF
from io import BytesIO
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, MapReduceDocumentsChain, ReduceDocumentsChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from typing import Tuple, List, Dict
from langchain.text_splitter import RecursiveCharacterTextSplitter
import difflib
import requests
from bs4 import BeautifulSoup
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail, Attachment, FileContent, FileName, FileType, Disposition
import base64

# Initialize NLP resources
import nltk
import os
nltk.data.path.append(os.path.join(os.getcwd(), "nltk_data")) # Or your preferred path

try:
    nltk.download('vader_lexicon', download_dir=os.path.join(os.getcwd(), "nltk_data")) #optional
except LookupError as e:
    st.error(f"Error downloading NLTK resources: {e}.  Please check your internet connection and try again.")
    st.stop()

from nltk.sentiment import SentimentIntensityAnalyzer


from document_processing import extract_text_from_pdf
from document_processing import chunk_text, create_faiss_index
from rag import generate_rag_response
from summarization import generate_summary
from utils import initialize_session_state

# Initialize session state
initialize_session_state()


def main():
    # Custom CSS styling
    st.markdown("""
    <style>
        .main {background-color: #f5f7fb;}
        .stButton>button {border-radius: 8px; padding: 0.5rem 1rem;}
        .stDownloadButton>button {width: 100%;}
        .stExpander .st-emotion-cache-1hynsf2 {border-radius: 10px;}
        .metric-box {padding: 20px; border-radius: 10px; background: white; box-shadow: 0 2px 8px rgba(0,0,0,0.1);}
        .update-card {
            background: white;
            border-radius: 8px;
            padding: 15px;
            margin-bottom: 10px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.05);
        }
        .update-title {
            font-weight: bold;
            color: #2c3e50;
            margin-bottom: 5px;
        }
        .update-source {
            color: #7f8c8d;
            font-size: 0.8rem;
        }
        .update-snippet {
            color: #34495e;
            font-size: 0.9rem;
            margin-top: 5px;
        }
        .regulation-item {
            background: #e8f4f8;
            padding: 6px 10px;
            margin-right: 5px;
            margin-bottom: 5px;
            border-radius: 4px;
            display: inline-block;
            font-size: 0.9em;
        }
    </style>
    """, unsafe_allow_html=True)
    # Main Layout
    tab1, tab2 = st.tabs([
        "Document Summary",
        "Q&A Chat"
    ])

    # Document Processing Section
    with tab1:
        st.header("Document Processing")
        with st.container(border=True):
            uploaded_file = st.file_uploader("Upload Legal Document (PDF)", type=["pdf"])
            if uploaded_file and not st.session_state.document_processed:
                if st.button("Analyze Document", type="primary"):
                    with st.status("Processing document...", expanded=True) as status:
                        try:
                            st.write("Extracting text...")
                            st.session_state.full_text = extract_text_from_pdf(uploaded_file)

                            st.write("Chunking text...")
                            st.session_state.text_chunks = chunk_text(st.session_state.full_text)

                            st.write("Creating search index...")
                            st.session_state.faiss_index = create_faiss_index(st.session_state.text_chunks)

                            st.write("Generating summary...")
                            st.session_state.summaries['document'] = generate_summary(st.session_state.full_text)

                            status.update(label="Analysis Complete!", state="complete", expanded=False)
                            st.session_state.document_processed = True
                            st.rerun()
                        except Exception as e:
                            st.error(f"Processing failed: {str(e)}")
                            st.session_state.document_processed = False

        if st.session_state.document_processed:
            with st.container(border=True):
                st.subheader("Document Summary")
                st.write(st.session_state.summaries.get('document', "No summary available"))

                # Document classification
                if st.session_state.document_categories:
                    st.subheader("Document Classification")
                    for category, confidence in st.session_state.document_categories[:3]:
                        st.write(f"- {category}: {confidence:.2f} confidence")


    # Chat Interface (Now as tab3)
    with tab2:
        st.header("Document Q&A")
        if st.session_state.document_processed:
            # Display chat history
            chat_container = st.container()
            with chat_container:
                for role, msg in st.session_state.chat_history:
                    with st.chat_message(role):
                        st.write(msg)
            
            # Input for new questions
            st.write("---")
            query = st.chat_input("Ask about the document...")
            if query:
                with st.spinner("Analyzing..."):
                    response = generate_rag_response(query, st.session_state.faiss_index, st.session_state.text_chunks)
                    st.session_state.chat_history.extend([
                        ("user", query),
                        ("assistant", response)
                    ])
                    st.rerun()
        else:
            st.info("Please upload and analyze a document first to use the Q&A feature.")
            
            # Sample Q&A to show functionality
            with st.expander("Sample Q&A"):
                st.markdown("""
                <div style="background-color: #f8f9fa; padding: 15px; border-radius: 5px; margin-top: 10px;">
                    <h4 style="margin-top: 0;">Example Questions You Can Ask:</h4>
                    <ul>
                        <li>What are the key obligations in this contract?</li>
                        <li>Explain the termination clause in simple terms.</li>
                        <li>What are the payment terms?</li>
                        <li>Are there any concerning liability clauses?</li>
                        <li>Summarize the confidentiality requirements.</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
