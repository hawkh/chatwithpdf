"""
Optimized PDF Chat Application - Simplified Version
A streamlined version with key optimizations and modern architecture
"""
import streamlit as st
import time
import json
import hashlib
import os
from typing import List, Dict, Any, Optional
from io import BytesIO
import pandas as pd

# Core imports
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
class Config:
    """Application configuration"""
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200
    MODEL_NAME = "gemini-pro"
    TEMPERATURE = 0.3
    MAX_TOKENS = 2048
    CACHE_TTL = 3600  # 1 hour
    
    if not GOOGLE_API_KEY:
        st.error("❌ GOOGLE_API_KEY not found in environment variables")
        st.stop()

# Simple caching system
class SimpleCache:
    """Simple in-memory cache with TTL"""
    
    def __init__(self):
        self.cache = {}
        self.timestamps = {}
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache if not expired"""
        if key in self.cache:
            if time.time() - self.timestamps[key] < Config.CACHE_TTL:
                return self.cache[key]
            else:
                # Remove expired entry
                del self.cache[key]
                del self.timestamps[key]
        return None
    
    def set(self, key: str, value: Any) -> None:
        """Set value in cache with timestamp"""
        self.cache[key] = value
        self.timestamps[key] = time.time()
    
    def clear(self) -> None:
        """Clear all cache entries"""
        self.cache.clear()
        self.timestamps.clear()

# Global cache instance
cache = SimpleCache()

class OptimizedDocumentProcessor:
    """Optimized document processor with caching and better text extraction"""
    
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
    
    def extract_text_from_pdf(self, pdf_file) -> Dict[str, Any]:
        """Extract text from PDF with improved error handling"""
        try:
            start_time = time.time()
            
            # Generate cache key based on file content
            file_content = pdf_file.read()
            pdf_file.seek(0)  # Reset file pointer
            cache_key = f"pdf_text_{hashlib.md5(file_content).hexdigest()}"
            
            # Check cache first
            cached_result = cache.get(cache_key)
            if cached_result:
                st.info("📋 Using cached text extraction")
                return cached_result
            
            # Extract text
            pdf_reader = PdfReader(pdf_file)
            text = ""
            
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
            
            # Generate metadata
            metadata = {
                "filename": pdf_file.name,
                "file_size": len(file_content),
                "pages": len(pdf_reader.pages),
                "extraction_time": time.time() - start_time,
                "word_count": len(text.split()),
                "character_count": len(text)
            }
            
            result = {
                "text": text.strip(),
                "metadata": metadata,
                "success": True
            }
            
            # Cache the result
            cache.set(cache_key, result)
            
            return result
            
        except Exception as e:
            st.error(f"❌ Error extracting text from {pdf_file.name}: {str(e)}")
            return {
                "text": "",
                "metadata": {"filename": pdf_file.name, "error": str(e)},
                "success": False
            }
    
    def create_chunks(self, text: str, metadata: Dict[str, Any]) -> List[Document]:
        """Create optimized text chunks with metadata"""
        try:
            if not text.strip():
                return []
            
            # Create chunks
            chunks = self.text_splitter.split_text(text)
            
            # Create Document objects with metadata
            documents = []
            for i, chunk in enumerate(chunks):
                chunk_metadata = metadata.copy()
                chunk_metadata.update({
                    "chunk_id": i,
                    "total_chunks": len(chunks),
                    "chunk_size": len(chunk)
                })
                
                document = Document(
                    page_content=chunk,
                    metadata=chunk_metadata
                )
                documents.append(document)
            
            return documents
            
        except Exception as e:
            st.error(f"❌ Error creating chunks: {str(e)}")
            return []

class OptimizedVectorDB:
    """Optimized vector database manager"""
    
    def __init__(self):
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=Config.GOOGLE_API_KEY
        )
        self.vector_store = None
        self.index_path = "./vector_db_index"
    
    def add_documents(self, documents: List[Document]) -> bool:
        """Add documents to vector database with optimization"""
        try:
            if not documents:
                return False
            
            # Create or update vector store
            if self.vector_store:
                self.vector_store.add_documents(documents)
            else:
                self.vector_store = FAISS.from_documents(
                    documents, 
                    self.embeddings
                )
            
            # Save index
            os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
            self.vector_store.save_local(self.index_path)
            
            return True
            
        except Exception as e:
            st.error(f"❌ Error adding documents to vector database: {str(e)}")
            return False
    
    def similarity_search(self, query: str, k: int = 4) -> List[Document]:
        """Perform similarity search with caching"""
        try:
            if not self.vector_store:
                return []
            
            # Generate cache key for search
            cache_key = f"search_{hashlib.md5(query.encode()).hexdigest()}"
            
            # Check cache
            cached_result = cache.get(cache_key)
            if cached_result:
                return cached_result
            
            # Perform search
            results = self.vector_store.similarity_search(query, k=k)
            
            # Cache results
            cache.set(cache_key, results)
            
            return results
            
        except Exception as e:
            st.error(f"❌ Error performing similarity search: {str(e)}")
            return []
    
    def load_existing_index(self) -> bool:
        """Load existing vector database index"""
        try:
            if os.path.exists(self.index_path):
                self.vector_store = FAISS.load_local(
                    self.index_path, 
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )
                return True
            return False
        except Exception as e:
            st.error(f"❌ Error loading existing index: {str(e)}")
            return False

class OptimizedLLMService:
    """Optimized LLM service with better prompt management"""
    
    def __init__(self):
        self.model = ChatGoogleGenerativeAI(
            model=Config.MODEL_NAME,
            temperature=Config.TEMPERATURE,
            max_output_tokens=Config.MAX_TOKENS,
            google_api_key=Config.GOOGLE_API_KEY
        )
        self.chain = self._create_chain()
    
    def _create_chain(self):
        """Create optimized question-answering chain"""
        prompt_template = """
        You are a helpful AI assistant that answers questions based on the provided context.
        
        Instructions:
        1. Answer the question as detailed as possible from the provided context
        2. If the answer is not in the provided context, say "I cannot find the answer in the provided context"
        3. Do not make up information or provide incorrect answers
        4. Use clear, concise language
        5. If relevant, cite specific parts of the context
        
        Context:
        {context}
        
        Question: {question}
        
        Answer:"""
        
        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        
        return load_qa_chain(
            self.model,
            chain_type="stuff",
            prompt=prompt,
            verbose=False
        )
    
    def generate_response(self, question: str, context_docs: List[Document]) -> Dict[str, Any]:
        """Generate response with caching and error handling"""
        try:
            if not question.strip():
                return {
                    "success": False,
                    "error": "Empty question provided",
                    "response": ""
                }
            
            if not context_docs:
                return {
                    "success": False,
                    "error": "No context documents provided",
                    "response": ""
                }
            
            # Generate cache key
            cache_key = f"llm_response_{hashlib.md5(question.encode()).hexdigest()}"
            
            # Check cache
            cached_result = cache.get(cache_key)
            if cached_result:
                st.info("🤖 Using cached response")
                return cached_result
            
            # Generate response
            start_time = time.time()
            
            chain_input = {
                "input_documents": context_docs,
                "question": question
            }
            
            result = self.chain(chain_input, return_only_outputs=True)
            response_text = result.get("output_text", "")
            
            processing_time = time.time() - start_time
            
            response_data = {
                "success": True,
                "response": response_text,
                "metadata": {
                    "processing_time": processing_time,
                    "context_docs_count": len(context_docs),
                    "model": Config.MODEL_NAME
                }
            }
            
            # Cache the response
            cache.set(cache_key, response_data)
            
            return response_data
            
        except Exception as e:
            st.error(f"❌ Error generating response: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "response": "I apologize, but I encountered an error while processing your request."
            }

class OptimizedPDFChatApp:
    """Main optimized PDF Chat Application"""
    
    def __init__(self):
        self.document_processor = OptimizedDocumentProcessor()
        self.vector_db = OptimizedVectorDB()
        self.llm_service = OptimizedLLMService()
        self.setup_page_config()
        self.setup_session_state()
        self.setup_custom_css()
    
    def setup_page_config(self):
        """Setup Streamlit page configuration"""
        st.set_page_config(
            page_title="AI PDF Chat Assistant - Optimized",
            page_icon="🤖",
            layout="wide",
            initial_sidebar_state="expanded"
        )
    
    def setup_session_state(self):
        """Initialize session state variables"""
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []
        
        if "uploaded_files" not in st.session_state:
            st.session_state.uploaded_files = []
        
        if "processing_stats" not in st.session_state:
            st.session_state.processing_stats = {}
    
    def setup_custom_css(self):
        """Setup custom CSS for better UI"""
        st.markdown("""
        <style>
        .main-header {
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            padding: 2rem;
            border-radius: 10px;
            color: white;
            text-align: center;
            margin-bottom: 2rem;
        }
        
        .metric-card {
            background: white;
            padding: 1rem;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin: 0.5rem 0;
        }
        
        .chat-message {
            padding: 1rem;
            border-radius: 10px;
            margin: 0.5rem 0;
        }
        
        .user-message {
            background: #e3f2fd;
            border-left: 4px solid #2196f3;
        }
        
        .ai-message {
            background: #f3e5f5;
            border-left: 4px solid #9c27b0;
        }
        
        .stButton > button {
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            border-radius: 5px;
            padding: 0.5rem 1rem;
        }
        
        .stButton > button:hover {
            background: linear-gradient(90deg, #5a6fd8 0%, #6a4190 100%);
        }
        </style>
        """, unsafe_allow_html=True)
    
    def render_header(self):
        """Render the main header"""
        st.markdown("""
        <div class="main-header">
            <h1>🤖 AI PDF Chat Assistant - Optimized</h1>
            <p>Advanced document analysis with performance optimizations and intelligent caching</p>
        </div>
        """, unsafe_allow_html=True)
    
    def render_sidebar(self):
        """Render the sidebar with file upload and controls"""
        with st.sidebar:
            st.title("📁 Document Management")
            
            # File upload section
            st.subheader("Upload Documents")
            uploaded_files = st.file_uploader(
                "Choose PDF files",
                type=['pdf'],
                accept_multiple_files=True,
                help="Upload one or more PDF files to analyze"
            )
            
            if uploaded_files:
                st.session_state.uploaded_files = uploaded_files
                
                # Show uploaded files
                st.write("**Uploaded Files:**")
                for file in uploaded_files:
                    st.write(f"📄 {file.name} ({file.size / 1024:.1f} KB)")
                
                # Process button
                if st.button("🚀 Process Documents", type="primary"):
                    self.process_documents(uploaded_files)
            
            # System info
            st.subheader("🔧 System Info")
            st.write(f"**Model:** {Config.MODEL_NAME}")
            st.write(f"**Chunk Size:** {Config.CHUNK_SIZE}")
            st.write(f"**Cache TTL:** {Config.CACHE_TTL}s")
            
            # Clear data button
            if st.button("🗑️ Clear All Data"):
                self.clear_all_data()
    
    def process_documents(self, files: List):
        """Process uploaded documents with optimizations"""
        try:
            with st.spinner("🔄 Processing documents..."):
                start_time = time.time()
                
                all_documents = []
                total_words = 0
                total_chunks = 0
                
                for file in files:
                    # Extract text
                    result = self.document_processor.extract_text_from_pdf(file)
                    
                    if result["success"] and result["text"]:
                        # Create chunks
                        chunks = self.document_processor.create_chunks(
                            result["text"], 
                            result["metadata"]
                        )
                        all_documents.extend(chunks)
                        
                        total_words += result["metadata"]["word_count"]
                        total_chunks += len(chunks)
                        
                        st.success(f"✅ Processed {file.name}: {len(chunks)} chunks")
                    else:
                        st.warning(f"⚠️ Failed to process {file.name}")
                
                if all_documents:
                    # Add to vector database
                    success = self.vector_db.add_documents(all_documents)
                    
                    if success:
                        duration = time.time() - start_time
                        
                        # Update stats
                        st.session_state.processing_stats = {
                            "total_files": len(files),
                            "total_chunks": total_chunks,
                            "total_words": total_words,
                            "processing_time": duration
                        }
                        
                        st.success(f"✅ Documents processed successfully in {duration:.2f} seconds!")
                        st.info(f"📊 Processed {total_chunks} chunks from {len(files)} files")
                        
                        # Show processing stats
                        self.render_processing_stats()
                    else:
                        st.error("❌ Failed to add documents to vector database")
                else:
                    st.warning("⚠️ No text could be extracted from the uploaded files")
                    
        except Exception as e:
            st.error(f"❌ Error processing documents: {str(e)}")
    
    def render_processing_stats(self):
        """Render document processing statistics"""
        if st.session_state.processing_stats:
            stats = st.session_state.processing_stats
            
            st.subheader("📊 Processing Statistics")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Files Processed", stats.get("total_files", 0))
                st.metric("Total Chunks", stats.get("total_chunks", 0))
            
            with col2:
                st.metric("Total Words", stats.get("total_words", 0))
                st.metric("Processing Time", f"{stats.get('processing_time', 0):.2f}s")
            
            with col3:
                st.metric("Cache Status", "Active")
                st.metric("Vector DB", "Ready")
    
    def render_chat_interface(self):
        """Render the main chat interface"""
        st.subheader("💬 Chat with Your Documents")
        
        # Try to load existing index
        if not hasattr(self.vector_db, 'vector_store') or not self.vector_db.vector_store:
            self.vector_db.load_existing_index()
        
        # Chat input
        user_question = st.text_input(
            "Ask a question about your documents:",
            placeholder="e.g., What are the main topics discussed in the documents?",
            key="user_question"
        )
        
        if user_question:
            self.process_user_question(user_question)
        
        # Chat history
        if st.session_state.chat_history:
            st.subheader("📝 Chat History")
            self.render_chat_history()
    
    def process_user_question(self, question: str):
        """Process user question and generate response"""
        try:
            with st.spinner("🤔 Thinking..."):
                # Search for relevant documents
                context_docs = self.vector_db.similarity_search(question, k=4)
                
                if context_docs:
                    # Generate response
                    result = self.llm_service.generate_response(question, context_docs)
                    
                    if result["success"]:
                        # Add to chat history
                        st.session_state.chat_history.append({
                            "question": question,
                            "answer": result["response"],
                            "timestamp": time.time(),
                            "metadata": result.get("metadata", {})
                        })
                        
                        # Display response
                        st.success("✅ Answer generated!")
                        st.markdown(f"""
                        <div class="chat-message ai-message">
                            <strong>AI Assistant:</strong><br>
                            {result["response"]}
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Show metadata
                        with st.expander("📋 Response Details"):
                            metadata = result.get("metadata", {})
                            st.json(metadata)
                    else:
                        st.error(f"❌ Error generating response: {result.get('error', 'Unknown error')}")
                else:
                    st.warning("⚠️ No relevant documents found. Please upload and process documents first.")
                    
        except Exception as e:
            st.error(f"❌ Error processing your question: {str(e)}")
    
    def render_chat_history(self):
        """Render chat history"""
        for i, chat in enumerate(st.session_state.chat_history):
            with st.container():
                # User question
                st.markdown(f"""
                <div class="chat-message user-message">
                    <strong>You:</strong><br>
                    {chat["question"]}
                </div>
                """, unsafe_allow_html=True)
                
                # AI answer
                st.markdown(f"""
                <div class="chat-message ai-message">
                    <strong>AI Assistant:</strong><br>
                    {chat["answer"]}
                </div>
                """, unsafe_allow_html=True)
                
                # Timestamp
                timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(chat["timestamp"]))
                st.caption(f"📅 {timestamp}")
                
                st.divider()
    
    def clear_all_data(self):
        """Clear all stored data"""
        try:
            # Clear cache
            cache.clear()
            
            # Clear session state
            st.session_state.chat_history = []
            st.session_state.uploaded_files = []
            st.session_state.processing_stats = {}
            
            # Reset vector database
            self.vector_db.vector_store = None
            
            st.success("✅ All data cleared successfully!")
            
        except Exception as e:
            st.error(f"❌ Error clearing data: {str(e)}")
    
    def render_advanced_features(self):
        """Render advanced features section"""
        st.subheader("🔬 Advanced Features")
        
        tab1, tab2 = st.tabs(["📈 Analytics", "📋 Export"])
        
        with tab1:
            self.render_analytics_tab()
        
        with tab2:
            self.render_export_tab()
    
    def render_analytics_tab(self):
        """Render analytics tab"""
        st.write("**Performance Metrics**")
        
        # Processing stats
        if st.session_state.processing_stats:
            st.json(st.session_state.processing_stats)
        
        # Cache info
        st.write("**Cache Information**")
        st.write(f"Cache entries: {len(cache.cache)}")
        st.write(f"Cache TTL: {Config.CACHE_TTL} seconds")
    
    def render_export_tab(self):
        """Render export tab"""
        st.write("**Export Options**")
        
        if st.session_state.chat_history:
            # Export chat history
            chat_data = json.dumps(st.session_state.chat_history, indent=2)
            st.download_button(
                label="📥 Export Chat History (JSON)",
                data=chat_data,
                file_name="chat_history.json",
                mime="application/json"
            )
            
            # Export as CSV
            df = pd.DataFrame(st.session_state.chat_history)
            csv_data = df.to_csv(index=False)
            st.download_button(
                label="📥 Export Chat History (CSV)",
                data=csv_data,
                file_name="chat_history.csv",
                mime="text/csv"
            )
        else:
            st.info("No chat history to export")
    
    def run(self):
        """Main application runner"""
        try:
            # Render header
            self.render_header()
            
            # Render sidebar
            self.render_sidebar()
            
            # Main content area
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # Chat interface
                self.render_chat_interface()
            
            with col2:
                # Quick stats
                if st.session_state.processing_stats:
                    st.subheader("📊 Quick Stats")
                    stats = st.session_state.processing_stats
                    
                    st.metric("Files", stats.get("total_files", 0))
                    st.metric("Chunks", stats.get("total_chunks", 0))
                    st.metric("Words", stats.get("total_words", 0))
            
            # Advanced features
            self.render_advanced_features()
            
        except Exception as e:
            st.error(f"❌ Application error: {str(e)}")

def main():
    """Main entry point"""
    try:
        app = OptimizedPDFChatApp()
        app.run()
    except Exception as e:
        st.error(f"Failed to start application: {str(e)}")

if __name__ == "__main__":
    main()