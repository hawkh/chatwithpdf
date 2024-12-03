import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

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
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    """Create and save vector store from text chunks."""
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

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

def user_input(user_question):
    """Process user question against uploaded PDF context."""
    try:
        # Verify embeddings and vector store
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        
        # Perform similarity search
        docs = new_db.similarity_search(user_question)
        
        # Get conversational chain
        chain = get_conversational_chain()
        
        # Get response
        response = chain(
            {"input_documents": docs, "question": user_question}, 
            return_only_outputs=True
        )
        
        # Display response
        st.write("Reply: ", response["output_text"])
    
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.error("Please upload PDFs and process them first.")

def main():
    """Main Streamlit application."""
    st.set_page_config("Chat PDF", page_icon="📄")
    st.header("Chat with PDF using Gemini 💁")

    # Sidebar for PDF upload
    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files", 
                                    type=['pdf'], 
                                    accept_multiple_files=True)
        
        if st.button("Process PDFs"):
            if pdf_docs:
                with st.spinner("Processing PDFs..."):
                    # Extract text from PDFs
                    raw_text = get_pdf_text(pdf_docs)
                    
                    # Create text chunks
                    text_chunks = get_text_chunks(raw_text)
                    
                    # Create vector store
                    get_vector_store(text_chunks)
                    
                    st.success("PDFs processed successfully!")
            else:
                st.warning("Please upload PDF files first.")

    # Main chat interface
    user_question = st.text_input("Ask a question about your PDF:")
    
    if user_question:
        user_input(user_question)

if __name__ == "__main__":
    main()
