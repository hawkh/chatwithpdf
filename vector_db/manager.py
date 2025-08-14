"""
Vector Database Manager for the PDF Chat Application
Supports multiple vector database backends with optimized performance
"""
import os
import time
from typing import List, Dict, Any, Optional, Union
from abc import ABC, abstractmethod
import numpy as np

# Vector database imports
from langchain_community.vectorstores import FAISS, Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.schema import Document
import chromadb
from chromadb.config import Settings

# Local imports
from config import vector_db_config, embedding_config
from utils.logger import logger, log_performance, log_error
from utils.cache import cached

class VectorDBInterface(ABC):
    """Abstract interface for vector database operations"""
    
    @abstractmethod
    def add_documents(self, documents: List[Document]) -> bool:
        """Add documents to the vector database"""
        pass
    
    @abstractmethod
    def similarity_search(self, query: str, k: int = 4) -> List[Document]:
        """Perform similarity search"""
        pass
    
    @abstractmethod
    def delete_collection(self) -> bool:
        """Delete the entire collection"""
        pass
    
    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """Get database statistics"""
        pass

class ChromaVectorDB(VectorDBInterface):
    """Chroma vector database implementation"""
    
    def __init__(self):
        self.persist_directory = vector_db_config.persist_directory
        self.collection_name = vector_db_config.collection_name
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model=embedding_config.model_name
        )
        self._setup_database()
    
    def _setup_database(self):
        """Setup Chroma database"""
        try:
            # Ensure directory exists
            os.makedirs(self.persist_directory, exist_ok=True)
            
            # Initialize Chroma client
            self.client = chromadb.PersistentClient(
                path=self.persist_directory,
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            
            # Get or create collection
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            
            # Create LangChain vector store
            self.vector_store = Chroma(
                client=self.client,
                collection_name=self.collection_name,
                embedding_function=self.embeddings,
                persist_directory=self.persist_directory
            )
            
            logger.info(f"Chroma database initialized: {self.persist_directory}")
            
        except Exception as e:
            log_error(e, {"operation": "chroma_setup"})
            raise
    
    @cached(ttl=3600, key_prefix="chroma_search")
    def similarity_search(self, query: str, k: int = 4) -> List[Document]:
        """Perform similarity search with caching"""
        try:
            start_time = time.time()
            results = self.vector_store.similarity_search(query, k=k)
            duration = time.time() - start_time
            
            log_performance(
                operation="chroma_similarity_search",
                duration=duration,
                query_length=len(query),
                results_count=len(results)
            )
            
            return results
            
        except Exception as e:
            log_error(e, {"operation": "chroma_search", "query": query})
            return []
    
    def add_documents(self, documents: List[Document]) -> bool:
        """Add documents to Chroma database"""
        try:
            start_time = time.time()
            
            # Add documents to vector store
            self.vector_store.add_documents(documents)
            
            # Persist changes
            self.vector_store.persist()
            
            duration = time.time() - start_time
            
            log_performance(
                operation="chroma_add_documents",
                duration=duration,
                documents_count=len(documents)
            )
            
            logger.info(f"Added {len(documents)} documents to Chroma")
            return True
            
        except Exception as e:
            log_error(e, {"operation": "chroma_add_documents"})
            return False
    
    def delete_collection(self) -> bool:
        """Delete the entire collection"""
        try:
            self.client.delete_collection(self.collection_name)
            logger.info("Chroma collection deleted")
            return True
        except Exception as e:
            log_error(e, {"operation": "chroma_delete_collection"})
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get Chroma database statistics"""
        try:
            count = self.collection.count()
            return {
                "type": "chroma",
                "collection_name": self.collection_name,
                "document_count": count,
                "persist_directory": self.persist_directory
            }
        except Exception as e:
            log_error(e, {"operation": "chroma_get_stats"})
            return {"type": "chroma", "error": str(e)}

class FAISSVectorDB(VectorDBInterface):
    """FAISS vector database implementation"""
    
    def __init__(self):
        self.index_path = os.path.join(vector_db_config.persist_directory, "faiss_index")
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model=embedding_config.model_name
        )
        self.vector_store = None
        self._load_or_create_index()
    
    def _load_or_create_index(self):
        """Load existing FAISS index or create new one"""
        try:
            if os.path.exists(self.index_path):
                self.vector_store = FAISS.load_local(
                    self.index_path, 
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )
                logger.info(f"Loaded existing FAISS index: {self.index_path}")
            else:
                logger.info("Creating new FAISS index")
                
        except Exception as e:
            log_error(e, {"operation": "faiss_load_index"})
            self.vector_store = None
    
    @cached(ttl=3600, key_prefix="faiss_search")
    def similarity_search(self, query: str, k: int = 4) -> List[Document]:
        """Perform similarity search with caching"""
        if not self.vector_store:
            logger.warning("FAISS index not available")
            return []
        
        try:
            start_time = time.time()
            results = self.vector_store.similarity_search(query, k=k)
            duration = time.time() - start_time
            
            log_performance(
                operation="faiss_similarity_search",
                duration=duration,
                query_length=len(query),
                results_count=len(results)
            )
            
            return results
            
        except Exception as e:
            log_error(e, {"operation": "faiss_search", "query": query})
            return []
    
    def add_documents(self, documents: List[Document]) -> bool:
        """Add documents to FAISS database"""
        try:
            start_time = time.time()
            
            if self.vector_store:
                # Add to existing index
                self.vector_store.add_documents(documents)
            else:
                # Create new index
                self.vector_store = FAISS.from_documents(
                    documents, 
                    self.embeddings
                )
            
            # Save index
            os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
            self.vector_store.save_local(self.index_path)
            
            duration = time.time() - start_time
            
            log_performance(
                operation="faiss_add_documents",
                duration=duration,
                documents_count=len(documents)
            )
            
            logger.info(f"Added {len(documents)} documents to FAISS")
            return True
            
        except Exception as e:
            log_error(e, {"operation": "faiss_add_documents"})
            return False
    
    def delete_collection(self) -> bool:
        """Delete the FAISS index"""
        try:
            if os.path.exists(self.index_path):
                import shutil
                shutil.rmtree(self.index_path)
                self.vector_store = None
                logger.info("FAISS index deleted")
                return True
            return False
        except Exception as e:
            log_error(e, {"operation": "faiss_delete_collection"})
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get FAISS database statistics"""
        try:
            if self.vector_store:
                return {
                    "type": "faiss",
                    "index_path": self.index_path,
                    "index_exists": True,
                    "embedding_dimension": self.vector_store.index.ntotal if hasattr(self.vector_store.index, 'ntotal') else 'unknown'
                }
            else:
                return {
                    "type": "faiss",
                    "index_path": self.index_path,
                    "index_exists": False
                }
        except Exception as e:
            log_error(e, {"operation": "faiss_get_stats"})
            return {"type": "faiss", "error": str(e)}

class VectorDBManager:
    """Main vector database manager"""
    
    def __init__(self, db_type: str = None):
        self.db_type = db_type or vector_db_config.type
        self.vector_db = self._initialize_database()
    
    def _initialize_database(self) -> VectorDBInterface:
        """Initialize the appropriate vector database"""
        try:
            if self.db_type == "chroma":
                return ChromaVectorDB()
            elif self.db_type == "faiss":
                return FAISSVectorDB()
            else:
                logger.warning(f"Unknown vector DB type: {self.db_type}, falling back to Chroma")
                return ChromaVectorDB()
                
        except Exception as e:
            log_error(e, {"operation": "vector_db_initialization", "db_type": self.db_type})
            # Fallback to FAISS
            logger.info("Falling back to FAISS")
            return FAISSVectorDB()
    
    def add_documents(self, documents: List[Document]) -> bool:
        """Add documents to vector database"""
        return self.vector_db.add_documents(documents)
    
    def similarity_search(self, query: str, k: int = 4) -> List[Document]:
        """Perform similarity search"""
        return self.vector_db.similarity_search(query, k)
    
    def delete_collection(self) -> bool:
        """Delete the entire collection"""
        return self.vector_db.delete_collection()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get database statistics"""
        return self.vector_db.get_stats()
    
    def health_check(self) -> Dict[str, Any]:
        """Perform health check on vector database"""
        try:
            stats = self.get_stats()
            stats["status"] = "healthy"
            stats["db_type"] = self.db_type
            return stats
        except Exception as e:
            log_error(e, {"operation": "vector_db_health_check"})
            return {
                "status": "unhealthy",
                "db_type": self.db_type,
                "error": str(e)
            }

# Global vector database manager instance
vector_db_manager = VectorDBManager()