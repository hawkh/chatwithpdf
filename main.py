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
from langchain.schema import Document
from rag_utils import build_vectorstore, load_vectorstore, get_retriever, extract_documents_from_pdfs, split_documents

# Load environment variables
load_dotenv()

def get_pdf_text(pdf_docs):
    """Extract text from uploaded PDF files."""
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    """Split text into manageable chunks."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store_from_documents(pdf_docs, prefer: str = "chroma"):
    """Extract documents from PDFs, split, and build the vector store."""
    documents = extract_documents_from_pdfs(pdf_docs)
    chunks = split_documents(documents, chunk_size=1500, chunk_overlap=200)
    build_vectorstore(chunks, prefer=prefer)

def get_conversational_chain():
    """Create conversational chain for question answering."""
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details. 
    If the answer is not in the provided context, just say "answer is not available in the context", 
    don't provide a wrong answer.

    Context:\n {context}\n
    Question: \n{question}\n
    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def user_input(user_question, k: int = 6, prefer: str = "chroma"):
    """Process user question against uploaded PDF context using the configured vector DB."""
    try:
        vector_store = load_vectorstore(prefer=prefer)
        if vector_store is None:
            st.error("Vector store not found. Please upload PDFs and process them first.")
            return
        retriever = get_retriever(vector_store, k=k) or vector_store.as_retriever(search_kwargs={"k": k})
        docs = retriever.get_relevant_documents(user_question)
        chain = get_conversational_chain()
        response = chain(
            {"input_documents": docs, "question": user_question}, 
            return_only_outputs=True
        )
        st.write("Reply: ", response["output_text"])
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.error("Please upload PDFs and process them first.")

def main():
    """Main Streamlit application."""
    st.set_page_config("Chat PDF", page_icon="📄")
    st.header("Chat with PDF using Gemini 💁")

    # Sidebar for PDF upload and settings
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

        k = st.slider("Retriever k", min_value=3, max_value=12, value=6, step=1)
        st.session_state["retriever_k"] = k

        pdf_docs = st.file_uploader("Upload your PDF Files", 
                                    type=['pdf'], 
                                    accept_multiple_files=True)
        
        if st.button("Process PDFs"):
            if pdf_docs:
                with st.spinner("Processing PDFs..."):
                    # Build vector store directly from PDFs
                    get_vector_store_from_documents(pdf_docs, prefer=prefer)
                    st.success("PDFs processed successfully!")
            else:
                st.warning("Please upload PDF files first.")

    # Main chat interface
    user_question = st.text_input("Ask a question about your PDF:")
    
    if user_question:
        user_input(
            user_question,
            k=st.session_state.get("retriever_k", 6),
            prefer=st.session_state.get("vector_preference", "chroma"),
        )

if __name__ == "__main__":
    main()
