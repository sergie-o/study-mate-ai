import io
import PyPDF
from langchain.schema import Document
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
import streamlit as st
import os
from typing import List, Dict, Any
import tempfile
import pandas as pd
import warnings
import logging
from langdetect import detect, DetectorFactory
from langdetect.lang_detect_exception import LangDetectException
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Custom CSS for beautiful design


def load_css():
    st.markdown("""
    <style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');
    
    /* Main app styling */
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        font-family: 'Poppins', sans-serif;
    }
    
    /* Header styling */
    .main-header {
        background: linear-gradient(90deg, #ff6b6b, #feca57, #48dbfb, #ff9ff3);
        background-size: 400% 400%;
        animation: gradient 3s ease infinite;
        padding: 2rem;
        border-radius: 20px;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 10px 30px rgba(0,0,0,0.3);
    }
    
    @keyframes gradient {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    .main-title {
        color: white;
        font-size: 3rem;
        font-weight: 700;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .main-subtitle {
        color: white;
        font-size: 1.2rem;
        margin: 0.5rem 0 0 0;
        opacity: 0.9;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #ff9a9e 0%, #fecfef 50%, #fecfef 100%);
    }
    
    /* Chat message styling */
    .stChatMessage {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 15px;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.2);
        margin: 1rem 0;
    }
    
    /* File uploader styling */
    .uploadedFile {
        border: 2px dashed #ff6b6b;
        border-radius: 15px;
        padding: 1rem;
        background: rgba(255, 255, 255, 0.1);
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(45deg, #ff6b6b, #feca57);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.3);
    }
    
    /* Success message styling */
    .stSuccess {
        background: linear-gradient(45deg, #56ab2f, #a8e6cf);
        color: white;
        border-radius: 15px;
        border: none;
    }
    
    /* Info message styling */
    .stInfo {
        background: linear-gradient(45deg, #667eea, #764ba2);
        color: white;
        border-radius: 15px;
        border: none;
    }
    
    /* Selectbox styling */
    .stSelectbox > div > div {
        background: rgba(255, 255, 255, 0.9);
        border-radius: 10px;
        border: 2px solid #ff6b6b;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background: linear-gradient(45deg, #ffecd2, #fcb69f);
        border-radius: 10px;
        color: #333;
        font-weight: 600;
    }
    
    /* Chat input styling */
    .stChatInput > div {
        border-radius: 25px;
        border: 2px solid #ff6b6b;
        background: rgba(255, 255, 255, 0.9);
    }
    
    /* Floating elements animation */
    .floating {
        animation: floating 3s ease-in-out infinite;
    }
    
    @keyframes floating {
        0% { transform: translate(0, 0px); }
        50% { transform: translate(0, -10px); }
        100% { transform: translate(0, 0px); }
    }
    
    /* Sparkle animation for extra fun */
    .sparkle {
        color: #feca57;
        animation: sparkle 1.5s linear infinite;
    }
    
    @keyframes sparkle {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.3; }
    }
    
    /* Custom divider */
    .custom-divider {
        height: 3px;
        background: linear-gradient(90deg, #ff6b6b, #feca57, #48dbfb, #ff9ff3);
        border-radius: 3px;
        margin: 2rem 0;
    }
    </style>
    """, unsafe_allow_html=True)


# Suppress ScriptRunContext warnings
logging.getLogger(
    "streamlit.runtime.scriptrunner.script_run_context").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", message=".*ScriptRunContext.*")

# Set seed for consistent language detection
DetectorFactory.seed = 0

# Language mapping
LANGUAGES = {
    'en': 'English',
    'de': 'German',
    'fr': 'French'
}

LANGUAGE_CODES = {v: k for k, v in LANGUAGES.items()}


class MultiLanguageRAG:
    def __init__(self):
        self.embeddings = None
        self.vectorstore = None
        self.qa_chain = None
        self.documents = []

    def initialize_components(self):
        """Initialize OpenAI components using environment variable"""
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY not found in environment variables")

        os.environ["OPENAI_API_KEY"] = api_key
        self.embeddings = OpenAIEmbeddings()

    def detect_language(self, text: str) -> str:
        """Detect language of text"""
        try:
            # Use first 1000 chars for detection (more reliable)
            sample_text = text[:1000] if len(text) > 1000 else text
            detected = detect(sample_text)
            return detected if detected in LANGUAGES else 'en'
        except LangDetectException:
            return 'en'  # Default to English

    def process_pdf(self, uploaded_file) -> List[Document]:
        """Process PDF file and extract text"""
        documents = []
        try:
            # Read PDF
            pdf_reader = PyPDF.PdfReader(uploaded_file)
            text_content = ""

            for page_num, page in enumerate(pdf_reader.pages):
                page_text = page.extract_text()
                text_content += page_text + "\n"

            if text_content.strip():
                # Detect language
                doc_language = self.detect_language(text_content)

                # Create document with metadata
                doc = Document(
                    page_content=text_content,
                    metadata={
                        'source': uploaded_file.name,
                        'language': doc_language,
                        'type': 'pdf'
                    }
                )
                documents.append(doc)

        except Exception as e:
            st.error(f"Error processing PDF {uploaded_file.name}: {str(e)}")

        return documents

    def process_excel(self, uploaded_file) -> List[Document]:
        """Process Excel file and extract text"""
        documents = []
        try:
            # Read Excel file - Read all sheets
            df = pd.read_excel(uploaded_file, sheet_name=None)

            for sheet_name, sheet_df in df.items():
                # Convert DataFrame to text
                text_content = f"Sheet: {sheet_name}\n\n"
                text_content += sheet_df.to_string(index=False)

                if text_content.strip():
                    # Detect language
                    doc_language = self.detect_language(text_content)

                    # Create document with metadata
                    doc = Document(
                        page_content=text_content,
                        metadata={
                            'source': f"{uploaded_file.name} - {sheet_name}",
                            'language': doc_language,
                            'type': 'excel',
                            'sheet': sheet_name
                        }
                    )
                    documents.append(doc)

        except Exception as e:
            st.error(f"Error processing Excel {uploaded_file.name}: {str(e)}")

        return documents

    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into chunks while preserving metadata"""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

        chunked_docs = []
        for doc in documents:
            chunks = text_splitter.split_documents([doc])
            chunked_docs.extend(chunks)

        return chunked_docs

    def create_vectorstore(self, documents: List[Document]):
        """Create Chroma vectorstore from documents"""
        if not documents:
            st.warning("No documents to process")
            return

        try:
            # Create temporary directory for Chroma
            persist_directory = tempfile.mkdtemp()

            # Create vectorstore
            self.vectorstore = Chroma.from_documents(
                documents=documents,
                embedding=self.embeddings,
                persist_directory=persist_directory
            )

            st.success(
                f"Successfully processed {len(documents)} document chunks")

        except Exception as e:
            st.error(f"Error creating vectorstore: {str(e)}")

    def setup_qa_chain(self, output_language: str):
        """Setup QA chain with language-specific prompt"""
        if not self.vectorstore:
            return

        # Create language-specific prompt
        language_name = LANGUAGES.get(output_language, 'English')

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

        # Create QA chain
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=ChatOpenAI(
                model='gpt-4o',
                temperature=0.7,
                max_tokens=4000,
                top_p=0.9,
                frequency_penalty=0.4,
                presence_penalty=0.3
            ),
            chain_type="stuff",
            retriever=self.vectorstore.as_retriever(search_kwargs={"k": 4}),
            chain_type_kwargs={"prompt": prompt},
            return_source_documents=True
        )

    def query(self, question: str, output_language: str) -> Dict[str, Any]:
        """Query the RAG system"""
        if not self.qa_chain:
            return {"error": "System not initialized"}

        try:
            # Detect query language
            query_language = self.detect_language(question)

            # Get response
            result = self.qa_chain({"query": question})

            return {
                "answer": result["result"],
                "source_documents": result["source_documents"],
                "query_language": query_language,
                "output_language": output_language
            }

        except Exception as e:
            return {"error": f"Error processing query: {str(e)}"}


def main():
    st.set_page_config(
        page_title="KatusiLearn AI ✨",
        page_icon="🌸",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Load the beautiful CSS styling
    load_css()  # <-- ADD THIS LINE!

    # Beautiful header
    st.markdown("""
    <div class="main-header floating">
        <h1 class="main-title">🌸 KatusiLearn AI <span class="sparkle">✨</span></h1>
        <p class="main-subtitle">Your Beautiful AI Study Companion 💖</p>
    </div>
    """, unsafe_allow_html=True)


def main():
    st.set_page_config(
        page_title="KatusiLearn AI ✨",
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

    # Initialize session state variables first
    if 'rag_system' not in st.session_state:
        st.session_state.rag_system = MultiLanguageRAG()
    if 'documents_processed' not in st.session_state:
        st.session_state.documents_processed = False
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'system_initialized' not in st.session_state:
        st.session_state.system_initialized = False

    # Try to initialize the system automatically (only if not already initialized)
    if not st.session_state.system_initialized:
        try:
            st.session_state.rag_system.initialize_components()
            st.session_state.system_initialized = True
        except ValueError as e:
            # Don't return here, just mark as not initialized
            st.session_state.system_initialized = False

    # Sidebar configuration
    with st.sidebar:
        st.header("Configuration")

        # Show API key status
        if st.session_state.system_initialized:
            st.success("✅ OpenAI API Key loaded from .env")
        else:
            st.error("❌ OpenAI API Key not found")
            st.info("Make sure your .env file contains: OPENAI_API_KEY=your_key")

        # Output language selection
        output_language = st.selectbox(
            "Choose output language:",
            options=list(LANGUAGES.values()),
            index=0
        )
        output_lang_code = LANGUAGE_CODES[output_language]

        st.divider()

        # File upload
        st.header("Upload Documents")
        uploaded_files = st.file_uploader(
            "Choose PDF or Excel files",
            type=['pdf', 'xlsx', 'xls'],
            accept_multiple_files=True
        )

        if st.button("Process Documents") and uploaded_files and st.session_state.system_initialized:
            with st.spinner("Processing documents..."):
                all_documents = []

                for uploaded_file in uploaded_files:
                    if uploaded_file.type == "application/pdf":
                        docs = st.session_state.rag_system.process_pdf(
                            uploaded_file)
                    elif uploaded_file.type in ["application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                                "application/vnd.ms-excel"]:
                        docs = st.session_state.rag_system.process_excel(
                            uploaded_file)
                    else:
                        continue

                    all_documents.extend(docs)

                if all_documents:
                    # Chunk documents
                    chunked_docs = st.session_state.rag_system.chunk_documents(
                        all_documents)

                    # Create vectorstore
                    st.session_state.rag_system.create_vectorstore(
                        chunked_docs)

                    # Setup QA chain
                    st.session_state.rag_system.setup_qa_chain(
                        output_lang_code)

                    st.session_state.documents_processed = True

                    # Display document info
                    st.success("Documents processed successfully!")

                    # Show document languages
                    doc_languages = {}
                    for doc in all_documents:
                        lang = doc.metadata.get('language', 'unknown')
                        lang_name = LANGUAGES.get(lang, lang)
                        if lang_name not in doc_languages:
                            doc_languages[lang_name] = 0
                        doc_languages[lang_name] += 1

                    st.write("**Detected Languages:**")
                    for lang, count in doc_languages.items():
                        st.write(f"- {lang}: {count} document(s)")

    # Main content area
    if not st.session_state.system_initialized:
        st.info("🔑 Please check your .env file contains a valid OPENAI_API_KEY")
        st.code("""
Create a .env file in the same folder as your app.py with:
OPENAI_API_KEY=your_actual_api_key_here
        """)
        return

    if not st.session_state.documents_processed:
        st.info(
            "👈 Please upload and process documents in the sidebar to start asking questions.")
        return

    # Chat interface
    st.header("Ask Questions")

    # Display chat history
    for chat in st.session_state.chat_history:
        with st.chat_message("user"):
            st.write(chat["question"])
            if chat.get("query_language"):
                st.caption(
                    f"Detected language: {LANGUAGES.get(chat['query_language'], 'Unknown')}")

        with st.chat_message("assistant"):
            st.write(chat["answer"])

            # Show sources
            if chat.get("sources"):
                with st.expander("Sources"):
                    for i, source in enumerate(chat["sources"], 1):
                        st.write(f"**Source {i}:** {source['source']}")
                        st.write(
                            f"**Language:** {LANGUAGES.get(source['language'], 'Unknown')}")
                        st.write(f"**Content:** {source['content'][:200]}...")
                        st.divider()

    # Question input
    question = st.chat_input("Ask a question about your documents...")

    if question:
        # Add user message to chat
        with st.chat_message("user"):
            st.write(question)

        # Get response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                result = st.session_state.rag_system.query(
                    question, output_lang_code)

                if "error" in result:
                    st.error(result["error"])
                else:
                    st.write(result["answer"])

                    # Show detected query language
                    if result.get("query_language"):
                        st.caption(
                            f"Detected query language: {LANGUAGES.get(result['query_language'], 'Unknown')}")

                    # Show sources
                    if result.get("source_documents"):
                        with st.expander("Sources"):
                            for i, doc in enumerate(result["source_documents"], 1):
                                st.write(
                                    f"**Source {i}:** {doc.metadata.get('source', 'Unknown')}")
                                st.write(
                                    f"**Language:** {LANGUAGES.get(doc.metadata.get('language'), 'Unknown')}")
                                st.write(
                                    f"**Content:** {doc.page_content[:200]}...")
                                st.divider()

                    # Add to chat history
                    sources = []
                    if result.get("source_documents"):
                        sources = [
                            {
                                "source": doc.metadata.get('source', 'Unknown'),
                                "language": doc.metadata.get('language', 'unknown'),
                                "content": doc.page_content
                            }
                            for doc in result["source_documents"]
                        ]

                    st.session_state.chat_history.append({
                        "question": question,
                        "answer": result["answer"],
                        "query_language": result.get("query_language"),
                        "sources": sources
                    })


if __name__ == "__main__":
    main()
