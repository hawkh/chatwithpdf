"""
Optimized PDF Chat Application
Main application with improved architecture, performance, and user experience
"""
import streamlit as st
import time
import json
from typing import List, Dict, Any
from io import BytesIO
import pandas as pd

# Local imports
from config import app_config, vector_db_config
from utils.logger import logger, log_performance
from utils.cache import cache_manager
from vector_db.manager import vector_db_manager
from processors.document_processor import document_processor
from services.llm_service import llm_service

class PDFChatApp:
    """Main PDF Chat Application with optimized features"""
    
    def __init__(self):
        self.setup_page_config()
        self.setup_session_state()
        self.setup_custom_css()
    
    def setup_page_config(self):
        """Setup Streamlit page configuration"""
        st.set_page_config(
            page_title="AI PDF Chat Assistant",
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
        
        if "vector_db_stats" not in st.session_state:
            st.session_state.vector_db_stats = {}
    
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
        
        .sidebar .sidebar-content {
            background-color: #f8f9fa;
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
        
        .file-upload-area {
            border: 2px dashed #ccc;
            border-radius: 10px;
            padding: 2rem;
            text-align: center;
            background: #f9f9f9;
        }
        
        .status-indicator {
            display: inline-block;
            width: 10px;
            height: 10px;
            border-radius: 50%;
            margin-right: 8px;
        }
        
        .status-healthy { background-color: #4caf50; }
        .status-warning { background-color: #ff9800; }
        .status-error { background-color: #f44336; }
        </style>
        """, unsafe_allow_html=True)
    
    def render_header(self):
        """Render the main header"""
        st.markdown("""
        <div class="main-header">
            <h1>🤖 AI PDF Chat Assistant</h1>
            <p>Advanced document analysis powered by Google Gemini and optimized vector databases</p>
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
            
            # System status
            st.subheader("🔧 System Status")
            self.render_system_status()
            
            # Clear data button
            if st.button("🗑️ Clear All Data"):
                self.clear_all_data()
    
    def render_system_status(self):
        """Render system status indicators"""
        try:
            # Vector DB status
            vdb_stats = vector_db_manager.health_check()
            vdb_status = vdb_stats.get("status", "unknown")
            
            status_color = {
                "healthy": "status-healthy",
                "unhealthy": "status-error",
                "unknown": "status-warning"
            }.get(vdb_status, "status-warning")
            
            st.markdown(f"""
            <div class="metric-card">
                <span class="status-indicator {status_color}"></span>
                <strong>Vector Database:</strong> {vdb_status.title()}
                <br><small>Type: {vdb_stats.get('db_type', 'Unknown')}</small>
            </div>
            """, unsafe_allow_html=True)
            
            # LLM status
            llm_stats = llm_service.health_check()
            llm_status = llm_stats.get("status", "unknown")
            
            llm_color = "status-healthy" if llm_status == "healthy" else "status-error"
            
            st.markdown(f"""
            <div class="metric-card">
                <span class="status-indicator {llm_color}"></span>
                <strong>Language Model:</strong> {llm_status.title()}
                <br><small>Model: {llm_stats.get('model', 'Unknown')}</small>
            </div>
            """, unsafe_allow_html=True)
            
            # Cache status
            cache_status = "Healthy" if cache_manager.redis_client or cache_manager.memory_cache else "Limited"
            cache_color = "status-healthy" if cache_status == "Healthy" else "status-warning"
            
            st.markdown(f"""
            <div class="metric-card">
                <span class="status-indicator {cache_color}"></span>
                <strong>Cache System:</strong> {cache_status}
            </div>
            """, unsafe_allow_html=True)
            
        except Exception as e:
            logger.error(f"Error rendering system status: {e}")
            st.error("Unable to load system status")
    
    def process_documents(self, files: List):
        """Process uploaded documents"""
        try:
            with st.spinner("🔄 Processing documents..."):
                start_time = time.time()
                
                # Process documents
                documents = document_processor.process_documents(files)
                
                if documents:
                    # Add to vector database
                    success = vector_db_manager.add_documents(documents)
                    
                    if success:
                        # Update stats
                        st.session_state.processing_stats = document_processor.get_processing_stats(documents)
                        st.session_state.vector_db_stats = vector_db_manager.get_stats()
                        
                        duration = time.time() - start_time
                        
                        st.success(f"✅ Documents processed successfully in {duration:.2f} seconds!")
                        st.info(f"📊 Processed {len(documents)} chunks from {len(files)} files")
                        
                        # Show processing stats
                        self.render_processing_stats()
                    else:
                        st.error("❌ Failed to add documents to vector database")
                else:
                    st.warning("⚠️ No text could be extracted from the uploaded files")
                    
        except Exception as e:
            logger.error(f"Error processing documents: {e}")
            st.error(f"❌ Error processing documents: {str(e)}")
    
    def render_processing_stats(self):
        """Render document processing statistics"""
        if st.session_state.processing_stats:
            stats = st.session_state.processing_stats
            
            st.subheader("📊 Processing Statistics")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Chunks", stats.get("total_chunks", 0))
                st.metric("Total Words", stats.get("total_words", 0))
            
            with col2:
                st.metric("Total Characters", stats.get("total_characters", 0))
                st.metric("Files Processed", stats.get("unique_files", 0))
            
            with col3:
                st.metric("Avg Chunk Size", f"{stats.get('average_chunk_size', 0)} chars")
                st.metric("Processing Time", "Optimized")
    
    def render_chat_interface(self):
        """Render the main chat interface"""
        st.subheader("💬 Chat with Your Documents")
        
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
                context_docs = vector_db_manager.similarity_search(question, k=4)
                
                if context_docs:
                    # Generate response
                    result = llm_service.generate_response(question, context_docs)
                    
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
            logger.error(f"Error processing question: {e}")
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
            # Clear vector database
            vector_db_manager.delete_collection()
            
            # Clear cache
            cache_manager.clear()
            
            # Clear session state
            st.session_state.chat_history = []
            st.session_state.uploaded_files = []
            st.session_state.processing_stats = {}
            st.session_state.vector_db_stats = {}
            
            st.success("✅ All data cleared successfully!")
            
        except Exception as e:
            logger.error(f"Error clearing data: {e}")
            st.error(f"❌ Error clearing data: {str(e)}")
    
    def render_advanced_features(self):
        """Render advanced features section"""
        st.subheader("🔬 Advanced Features")
        
        tab1, tab2, tab3 = st.tabs(["📈 Analytics", "⚙️ Settings", "📋 Export"])
        
        with tab1:
            self.render_analytics_tab()
        
        with tab2:
            self.render_settings_tab()
        
        with tab3:
            self.render_export_tab()
    
    def render_analytics_tab(self):
        """Render analytics tab"""
        st.write("**System Performance Metrics**")
        
        # Vector DB stats
        if st.session_state.vector_db_stats:
            st.json(st.session_state.vector_db_stats)
        
        # Processing stats
        if st.session_state.processing_stats:
            st.json(st.session_state.processing_stats)
    
    def render_settings_tab(self):
        """Render settings tab"""
        st.write("**Application Settings**")
        
        # Vector DB type selection
        db_type = st.selectbox(
            "Vector Database Type",
            ["chroma", "faiss"],
            index=0 if vector_db_config.type == "chroma" else 1
        )
        
        if st.button("Apply Settings"):
            st.info("Settings updated! Restart the application to apply changes.")
    
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
                    
                    st.metric("Documents", stats.get("unique_files", 0))
                    st.metric("Chunks", stats.get("total_chunks", 0))
                    st.metric("Words", stats.get("total_words", 0))
            
            # Advanced features
            self.render_advanced_features()
            
        except Exception as e:
            logger.error(f"Application error: {e}")
            st.error(f"❌ Application error: {str(e)}")

def main():
    """Main entry point"""
    try:
        app = PDFChatApp()
        app.run()
    except Exception as e:
        logger.error(f"Failed to start application: {e}")
        st.error("Failed to start application. Please check the logs.")

if __name__ == "__main__":
    main()