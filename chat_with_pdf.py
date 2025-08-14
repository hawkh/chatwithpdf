import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import logging
from io import BytesIO
import base64
from langchain.schema import Document
from rag_utils import build_vectorstore, load_vectorstore, get_retriever, extract_documents_from_pdfs, split_documents

# Set up logging for debugging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    st.error("Google API key not found. Please set it in the .env file.")
    st.stop()
genai.configure(api_key=api_key)

# Function to extract text from PDFs
def get_pdf_text(pdf_docs):
    """Extract text from uploaded PDF files."""
    text = ""
    try:
        for pdf in pdf_docs:
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                text += page.extract_text() or ""
        logger.info(f"Extracted text from {len(pdf_docs)} PDFs")
    except Exception as e:
        logger.error(f"Error extracting text from PDFs: {e}")
        st.error(f"Error extracting text from PDFs: {e}")
    return text

# Function to split text into chunks
def get_text_chunks(text):
    """Split text into chunks for processing."""
    try:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
        chunks = text_splitter.split_text(text)
        logger.info(f"Text split into {len(chunks)} chunks")
        return chunks
    except Exception as e:
        logger.error(f"Error splitting text: {e}")
        st.error(f"Error splitting text: {e}")
        return []

# Function to create and save vector store
def get_vector_store_from_documents(pdf_docs, prefer: str = "chroma"):
    """Create and save a vector store by extracting documents from PDFs and splitting into chunks."""
    try:
        documents = extract_documents_from_pdfs(pdf_docs)
        chunks = split_documents(documents, chunk_size=1500, chunk_overlap=200)
        build_vectorstore(chunks, prefer=prefer)
        logger.info("Vector store created and saved successfully")
    except Exception as e:
        logger.error(f"Error creating vector store: {e}")
        st.error(f"Error creating vector store: {e}")

# Function to define conversational chain
def get_conversational_chain():
    """Define the conversational chain for question answering."""
    prompt_template = """
    Answer the question as detailed as possible from the provided context. If the answer is not in the provided context, 
    state "answer is not available in the context". Do not provide incorrect answers.\n\n
    Context:\n {context}\n
    Question: \n{question}\n
    Answer:
    """
    try:
        model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
        prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
        chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
        logger.info("Conversational chain initialized successfully")
        return chain
    except Exception as e:
        logger.error(f"Error initializing conversational chain: {e}")
        st.error(f"Error initializing conversational chain: {e}")
        return None

# Function to handle user input and generate responses
def user_input(user_question, k: int = 6, prefer: str = "chroma"):
    """Process user question and generate a response using the configured vector DB."""
    try:
        vector_store = load_vectorstore(prefer=prefer)
        if vector_store is None:
            st.error("Vector store not found. Please upload and process PDF files first.")
            return None
        retriever = get_retriever(vector_store, k=k) or vector_store.as_retriever(search_kwargs={"k": k})
        docs = retriever.get_relevant_documents(user_question)

        chain = get_conversational_chain()
        if chain is None:
            return None

        response = chain(
            {"input_documents": docs, "question": user_question},
            return_only_outputs=True
        )
        answer_text = response["output_text"]
        logger.info(f"Generated response for question: {user_question}")
        st.write("### Reply:")
        st.success(answer_text)
        return answer_text
    except Exception as e:
        logger.error(f"Error processing user input: {e}")
        st.error(f"Error processing user input: {e}")
        return None

# Function to generate a downloadable report
def generate_report(user_question, response_text):
    """Generate a downloadable report of the question and answer."""
    report = f"Chat with PDF Report\n\n"
    report += f"Question: {user_question}\n"
    report += f"Answer: {response_text}\n"
    
    buffer = BytesIO()
    buffer.write(report.encode())
    buffer.seek(0)
    return buffer

# Streamlit app
def main():
    # Set page configuration for a professional look
    st.set_page_config(page_title="Chat with PDF using Gemini", page_icon="📄", layout="wide")

    # Custom CSS for better styling
    st.markdown("""
        <style>
        .main-title {
            font-size: 36px;
            font-weight: bold;
            color: #1E88E5;
            text-align: center;
            margin-bottom: 20px;
        }
        .sidebar .sidebar-content {
            background-color: #f0f2f6;
        }
        .stButton>button {
            background-color: #1E88E5;
            color: white;
            border-radius: 5px;
        }
        </style>
    """, unsafe_allow_html=True)

    # Title and introduction
    st.markdown('<div class="main-title">Chat with PDF using Gemini</div>', unsafe_allow_html=True)
    st.write("""
        Welcome to this interactive app that allows you to chat with your PDF documents using Google's Gemini model. 
        Upload your PDFs, ask questions, and get detailed answers extracted from the documents. This app leverages 
        advanced NLP techniques, including embeddings, vector stores, and conversational AI.
    """)

    # Sidebar for navigation and additional information
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Chat with PDF", "About"])

    # Page: Chat with PDF
    if page == "Chat with PDF":
        st.subheader("Chat with Your PDFs")
        st.write("""
            Upload your PDF files using the sidebar menu, then ask questions about the content. 
            The app will process the PDFs, extract relevant information, and provide detailed answers.
        """)

        user_question = st.text_input("Ask a Question from the PDF Files", key="user_question")

        if user_question:
            if not os.path.exists("chroma_db") and not os.path.exists("faiss_index"):
                st.warning("Please upload and process PDF files before asking questions.")
            else:
                response = user_input(
                    user_question,
                    k=st.session_state.get("retriever_k", 6),
                    prefer=st.session_state.get("vector_preference", "chroma"),
                )
                if response:
                    # Offer downloadable report
                    report_buffer = generate_report(user_question, response)
                    st.download_button(
                        label="Download Answer Report",
                        data=report_buffer,
                        file_name="answer_report.txt",
                        mime="text/plain"
                    )

        with st.sidebar:
            st.title("Menu:")
            vector_db_option = st.selectbox(
                "Vector DB",
                ["Auto (Chroma preferred)", "Chroma (persistent)", "FAISS (local)"]
            )
            prefer_map = {
                "Auto (Chroma preferred)": "chroma",
                "Chroma (persistent)": "chroma",
                "FAISS (local)": "faiss",
            }
            prefer = prefer_map.get(vector_db_option, "chroma")
            st.session_state["vector_preference"] = prefer

            k = st.slider("Retriever k", min_value=3, max_value=12, value=6, step=1, key="retriever_k_slider")
            st.session_state["retriever_k"] = k

            pdf_docs = st.file_uploader(
                "Upload your PDF Files and Click on the Submit & Process Button",
                accept_multiple_files=True,
                key="pdf_uploader"
            )
            if st.button("Submit & Process", key="process_button"):
                if pdf_docs:
                    with st.spinner("Processing..."):
                        logger.info("Starting PDF processing...")
                        # Extract documents and split into chunks, then build the store
                        get_vector_store_from_documents(pdf_docs, prefer=prefer)
                        st.success("Done! You can now ask questions.")
                else:
                    st.warning("Please upload at least one PDF file.")

    # Page: About
    elif page == "About":
        st.subheader("About This Project")
        st.write("""
            This project demonstrates the power of modern NLP techniques for document-based question answering. 
            The app processes PDF documents, extracts text, splits it into chunks, creates embeddings using Google's 
            Gemini model, and stores them in a persistent vector store (Chroma when available, with FAISS fallback) for efficient retrieval. Users can then ask questions, 
            and the app retrieves relevant document chunks to generate detailed answers.

            **Key Features:**
            - PDF text extraction and chunking.
            - Embedding generation using Google Gemini.
            - Persistent similarity search using Chroma (with FAISS fallback).
            - MMR-based retriever for diverse, relevant context.
            - Conversational question answering with detailed responses.
            - Downloadable answer reports.

            **Technologies Used:**
            - Python, Streamlit, PyPDF2, LangChain, Google Generative AI, Chroma, FAISS, dotenv.

            **Developer:** Sai Ruthvik  
            This project is part of my portfolio to showcase my expertise in machine learning, natural language processing, 
            and software engineering. For more details, feel free to connect with me on [LinkedIn](https://www.linkedin.com/in/sai-ruthvik) 
            or check out my [GitHub](https://github.com/hawkh).
        """)

if __name__ == "__main__":
    main()
