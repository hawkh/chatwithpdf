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

# Load environment variables
load_dotenv()

# Configuration from environment
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
VECTOR_DB = os.getenv("VECTOR_DB", "CHROMA").upper()
PERSIST_DIR = os.getenv("PERSIST_DIR", "chroma_db")
TOP_K = int(os.getenv("TOP_K", "5"))


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
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = text_splitter.split_text(text)
    return chunks


def get_embeddings():
    return GoogleGenerativeAIEmbeddings(model="models/embedding-001")


def build_or_update_vector_store(text_chunks):
    """Create or update vector store from text chunks with persistence."""
    embeddings = get_embeddings()
    if VECTOR_DB == "CHROMA":
        from langchain_community.vectorstores import Chroma
        vs = Chroma.from_texts(
            texts=text_chunks,
            embedding=embeddings,
            collection_name="pdf_chunks",
            persist_directory=PERSIST_DIR,
        )
        vs.persist()
    else:
        from langchain_community.vectorstores import FAISS
        index_dir = "faiss_index"
        if os.path.exists(index_dir):
            vs = FAISS.load_local(index_dir, embeddings, allow_dangerous_deserialization=True)
            vs.add_texts(text_chunks)
            vs.save_local(index_dir)
        else:
            vs = FAISS.from_texts(text_chunks, embedding=embeddings)
            vs.save_local(index_dir)


def load_vector_store():
    embeddings = get_embeddings()
    if VECTOR_DB == "CHROMA":
        from langchain_community.vectorstores import Chroma
        return Chroma(collection_name="pdf_chunks", persist_directory=PERSIST_DIR, embedding_function=embeddings)
    from langchain_community.vectorstores import FAISS
    return FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)


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
        # Load vector store according to backend
        new_db = load_vector_store()

        # Perform similarity search via retriever
        retriever = new_db.as_retriever(search_kwargs={"k": TOP_K})
        docs = retriever.get_relevant_documents(user_question)

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
    st.caption(f"Vector DB backend: {VECTOR_DB}")

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
                    
                    # Create or update vector store
                    build_or_update_vector_store(text_chunks)
                    
                    st.success("PDFs processed successfully!")
            else:
                st.warning("Please upload PDF files first.")

    # Main chat interface
    user_question = st.text_input("Ask a question about your PDF:")
    
    if user_question:
        user_input(user_question)

if __name__ == "__main__":
    main()
