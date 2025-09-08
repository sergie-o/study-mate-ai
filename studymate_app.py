# app.py
import io
import os
import tempfile
import logging
import warnings
from typing import List, Dict, Any

import pandas as pd
import streamlit as st
import pypdf as PyPDF2

from langdetect import detect, DetectorFactory
from langdetect.lang_detect_exception import LangDetectException

from langchain.schema import Document
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_chroma import Chroma
from langchain_ollama import ChatOllama, OllamaEmbeddings
# If you need other community loaders later:
# from langchain_community.document_loaders import UnstructuredFileLoader

# -------------------------------
# Pretty CSS
# -------------------------------


def load_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');
    .main { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); font-family: 'Poppins', sans-serif; }
    .main-header { background: linear-gradient(90deg, #ff6b6b, #feca57, #48dbfb, #ff9ff3);
                   background-size: 400% 400%; animation: gradient 3s ease infinite;
                   padding: 2rem; border-radius: 20px; margin-bottom: 2rem; text-align: center;
                   box-shadow: 0 10px 30px rgba(0,0,0,0.3); }
    @keyframes gradient { 0%{background-position:0% 50%}50%{background-position:100% 50%}100%{background-position:0% 50%} }
    .main-title { color: #fff; font-size: 3rem; font-weight: 700; margin: 0; text-shadow: 2px 2px 4px rgba(0,0,0,0.3);}
    .main-subtitle { color: #fff; font-size: 1.2rem; margin: 0.5rem 0 0 0; opacity: 0.9; }
    .stButton > button { background: linear-gradient(45deg, #ff6b6b, #feca57); color: white; border: none; border-radius: 25px;
                         padding: 0.75rem 2rem; font-weight: 600; transition: all 0.3s ease;
                         box-shadow: 0 5px 15px rgba(0,0,0,0.2); }
    .stButton > button:hover { transform: translateY(-2px); box-shadow: 0 8px 25px rgba(0,0,0,0.3); }
    .floating { animation: floating 3s ease-in-out infinite; }
    @keyframes floating { 0% { transform: translate(0,0px) } 50% { transform: translate(0,-10px) } 100% { transform: translate(0,0px) } }
    .sparkle { color: #feca57; animation: sparkle 1.5s linear infinite; }
    @keyframes sparkle { 0%,100% { opacity: 1 } 50% { opacity: 0.3 } }
    </style>
    """, unsafe_allow_html=True)


# -------------------------------
# Logging / warnings
# -------------------------------
logging.getLogger(
    "streamlit.runtime.scriptrunner.script_run_context").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", message=".*ScriptRunContext.*")

# Language detection seed
DetectorFactory.seed = 0

# -------------------------------
# Languages supported
# -------------------------------
LANGUAGES = {'en': 'English', 'de': 'German', 'fr': 'French'}
LANGUAGE_CODES = {v: k for k, v in LANGUAGES.items()}

# -------------------------------
# Core RAG class
# -------------------------------


class MultiLanguageRAG:
    def __init__(self, ollama_base_url: str | None = None):
        self.embeddings = None
        self.vectorstore = None
        self.qa_chain = None
        self.documents = []
        self.ollama_base_url = ollama_base_url  # e.g., "http://localhost:11434"

    def initialize_components(self):
        """Initialize Ollama embeddings (no API keys needed)."""
        self.embeddings = OllamaEmbeddings(
            model="mxbai-embed-large",
            base_url=self.ollama_base_url
        )

    @staticmethod
    def detect_language(text: str) -> str:
        try:
            sample_text = text[:1000] if len(text) > 1000 else text
            detected = detect(sample_text)
            return detected if detected in LANGUAGES else 'en'
        except LangDetectException:
            return 'en'

    def process_pdf(self, uploaded_file) -> List[Document]:
        docs: List[Document] = []
        try:
            pdf_reader = PyPDF2.PdfReader(uploaded_file)
            text_content = ""
            for page in pdf_reader.pages:
                page_text = page.extract_text() or ""
                text_content += page_text + "\n"

            if text_content.strip():
                doc_language = self.detect_language(text_content)
                docs.append(Document(
                    page_content=text_content,
                    metadata={
                        'source': getattr(uploaded_file, "name", "uploaded.pdf"),
                        'language': doc_language,
                        'type': 'pdf'
                    }
                ))
        except Exception as e:
            st.error(
                f"Error processing PDF {getattr(uploaded_file, 'name', 'file')}: {e}")
        return docs

    def process_excel(self, uploaded_file) -> List[Document]:
        docs: List[Document] = []
        try:
            all_sheets = pd.read_excel(uploaded_file, sheet_name=None)
            for sheet_name, df in all_sheets.items():
                text_content = f"Sheet: {sheet_name}\n\n{df.to_string(index=False)}"
                if text_content.strip():
                    doc_language = self.detect_language(text_content)
                    docs.append(Document(
                        page_content=text_content,
                        metadata={
                            'source': f"{getattr(uploaded_file, 'name', 'workbook.xlsx')} - {sheet_name}",
                            'language': doc_language,
                            'type': 'excel',
                            'sheet': sheet_name
                        }
                    ))
        except Exception as e:
            st.error(
                f"Error processing Excel {getattr(uploaded_file, 'name', 'file')}: {e}")
        return docs

    @staticmethod
    def chunk_documents(documents: List[Document]) -> List[Document]:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        chunks: List[Document] = []
        for d in documents:
            chunks.extend(splitter.split_documents([d]))
        return chunks

    def create_vectorstore(self, documents: List[Document]):
        if not documents:
            st.warning("No documents to process.")
            return
        try:
            persist_directory = tempfile.mkdtemp(prefix="chroma_")
            self.vectorstore = Chroma.from_documents(
                documents=documents,
                embedding=self.embeddings,
                persist_directory=persist_directory
            )
            st.success(f"Successfully processed {len(documents)} chunks.")
        except Exception as e:
            st.error(f"Error creating vectorstore: {e}")

    def setup_qa_chain(self, output_language_code: str):
        if not self.vectorstore:
            st.error("Vector store not initialized.")
            return

        language_name = LANGUAGES.get(output_language_code, 'English')
        prompt_template = f"""You are a knowledgeable and friendly study tutor. Your goal is to provide clear, detailed explanations that help students truly understand the material.

When answering questions:
- Explain concepts step-by-step in a conversational tone
- Use examples and analogies when helpful
- Break down complex ideas into digestible parts
- Connect different concepts when relevant
- Be thorough but engaging

IMPORTANT: Respond in {language_name} regardless of the language of the source documents.

If you cannot answer the question based on the provided context, say "I cannot find information about this in the provided documents" in {language_name}.

Context: {{context}}

Question: {{question}}

Answer in {language_name}:"""

        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )

        self.qa_chain = RetrievalQA.from_chain_type(
            llm=ChatOllama(
                model="llama3.1:8b",
                temperature=0.3,
                # num_ctx=8192,           # optional: increase context if your model supports it
                base_url=self.ollama_base_url
            ),
            chain_type="stuff",
            retriever=self.vectorstore.as_retriever(search_kwargs={"k": 4}),
            chain_type_kwargs={"prompt": prompt},
            return_source_documents=True
        )

    def query(self, question: str, output_language_code: str) -> Dict[str, Any]:
        if not self.qa_chain:
            return {"error": "System not initialized"}
        try:
            query_lang = self.detect_language(question)
            result = self.qa_chain({"query": question})
            return {
                "answer": result["result"],
                "source_documents": result.get("source_documents", []),
                "query_language": query_lang,
                "output_language": output_language_code
            }
        except Exception as e:
            return {"error": f"Error processing query: {e}"}

# -------------------------------
# Streamlit App
# -------------------------------


def main():
    st.set_page_config(
        page_title="🌸 StudyMate AI",
        page_icon="🌸",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    load_css()
    st.markdown("""
    <div class="main-header floating">
        <h1 class="main-title">🌸 StudyMate AI <span class="sparkle">✨</span></h1>
        <p class="main-subtitle">Your Beautiful AI Study Companion 💖</p>
    </div>
    """, unsafe_allow_html=True)

    st.title("📚 StudyMate AI")
    st.markdown(
        "Upload your study materials and ask questions in multiple languages!")

    # --- Session state
    if 'rag_system' not in st.session_state:
        # If your Ollama is at a non-default URL, pass base_url here:
        st.session_state.rag_system = MultiLanguageRAG(ollama_base_url=None)
    if 'documents_processed' not in st.session_state:
        st.session_state.documents_processed = False
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    # Initialize Ollama embeddings once
    if st.session_state.rag_system.embeddings is None:
        try:
            st.session_state.rag_system.initialize_components()
        except Exception as e:
            st.error(f"Failed to initialize Ollama embeddings: {e}")
            st.stop()

    # --- Sidebar
    with st.sidebar:
        st.header("Configuration")
        st.success("✅ Using local Ollama (no API key required)")

        output_language = st.selectbox(
            "Choose output language:",
            options=list(LANGUAGES.values()),
            index=0
        )
        output_lang_code = LANGUAGE_CODES[output_language]

        st.divider()
        st.header("Upload Documents")
        uploaded_files = st.file_uploader(
            "Choose PDF or Excel files",
            type=['pdf', 'xlsx', 'xls'],
            accept_multiple_files=True
        )

        if st.button("Process Documents"):
            if not uploaded_files:
                st.warning("Please upload at least one file.")
            else:
                with st.spinner("Processing documents..."):
                    all_docs: List[Document] = []
                    for f in uploaded_files:
                        mime = getattr(f, "type", "")
                        if mime == "application/pdf":
                            all_docs.extend(
                                st.session_state.rag_system.process_pdf(f))
                        elif mime in (
                            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            "application/vnd.ms-excel"
                        ):
                            all_docs.extend(
                                st.session_state.rag_system.process_excel(f))

                    if all_docs:
                        chunks = st.session_state.rag_system.chunk_documents(
                            all_docs)
                        st.session_state.rag_system.create_vectorstore(chunks)
                        st.session_state.rag_system.setup_qa_chain(
                            output_lang_code)
                        st.session_state.documents_processed = True

                        st.success("Documents processed successfully!")
                        # Language summary
                        counts = {}
                        for d in all_docs:
                            lang = d.metadata.get("language", "unknown")
                            name = LANGUAGES.get(lang, lang)
                            counts[name] = counts.get(name, 0) + 1
                        st.write("**Detected Languages:**")
                        for k, v in counts.items():
                            st.write(f"- {k}: {v} document(s)")
                    else:
                        st.warning(
                            "No usable text extracted from the uploaded files.")

    # --- Main area
    if not st.session_state.documents_processed:
        st.info(
            "👈 Please upload and process documents in the sidebar to start asking questions.")
        return

    st.header("Ask Questions")

    # Chat history display
    for chat in st.session_state.chat_history:
        with st.chat_message("user"):
            st.write(chat["question"])
            if chat.get("query_language"):
                st.caption(
                    f"Detected language: {LANGUAGES.get(chat['query_language'], 'Unknown')}")
        with st.chat_message("assistant"):
            st.write(chat["answer"])
            if chat.get("sources"):
                with st.expander("Sources"):
                    for i, src in enumerate(chat["sources"], 1):
                        st.write(f"**Source {i}:** {src['source']}")
                        st.write(
                            f"**Language:** {LANGUAGES.get(src['language'], 'Unknown')}")
                        st.write(f"**Content:** {src['content'][:200]}...")
                        st.divider()

    # Input
    question = st.chat_input("Ask a question about your documents...")
    if question:
        with st.chat_message("user"):
            st.write(question)
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                result = st.session_state.rag_system.query(
                    question, output_lang_code)
                if "error" in result:
                    st.error(result["error"])
                else:
                    st.write(result["answer"])
                    if result.get("query_language"):
                        st.caption(
                            f"Detected query language: {LANGUAGES.get(result['query_language'], 'Unknown')}")

                    sources = []
                    if result.get("source_documents"):
                        with st.expander("Sources"):
                            for i, d in enumerate(result["source_documents"], 1):
                                src = d.metadata.get('source', 'Unknown')
                                lang = d.metadata.get('language', 'unknown')
                                st.write(f"**Source {i}:** {src}")
                                st.write(
                                    f"**Language:** {LANGUAGES.get(lang, 'Unknown')}")
                                st.write(
                                    f"**Content:** {d.page_content[:200]}...")
                                st.divider()
                                sources.append({
                                    "source": src,
                                    "language": lang,
                                    "content": d.page_content
                                })

                    st.session_state.chat_history.append({
                        "question": question,
                        "answer": result.get("answer", ""),
                        "query_language": result.get("query_language"),
                        "sources": sources
                    })


if __name__ == "__main__":
    main()
