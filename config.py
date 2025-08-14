"""
Configuration settings for the PDF Chat Application
"""
import os
from typing import Optional
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()

@dataclass
class VectorDBConfig:
    """Vector database configuration"""
    type: str = "chroma"  # Options: chroma, faiss, pinecone, qdrant
    persist_directory: str = "./vector_db"
    collection_name: str = "pdf_documents"
    
    # Pinecone specific
    pinecone_api_key: Optional[str] = os.getenv("PINECONE_API_KEY")
    pinecone_environment: Optional[str] = os.getenv("PINECONE_ENVIRONMENT")
    pinecone_index_name: str = "pdf-chat-index"
    
    # Qdrant specific
    qdrant_url: str = os.getenv("QDRANT_URL", "http://localhost:6333")
    qdrant_api_key: Optional[str] = os.getenv("QDRANT_API_KEY")

@dataclass
class LLMConfig:
    """Language model configuration"""
    model_name: str = "gemini-pro"
    temperature: float = 0.3
    max_tokens: int = 2048
    top_p: float = 0.9
    top_k: int = 40

@dataclass
class EmbeddingConfig:
    """Embedding model configuration"""
    model_name: str = "models/embedding-001"
    chunk_size: int = 1000
    chunk_overlap: int = 200
    max_chunk_size: int = 10000

@dataclass
class CacheConfig:
    """Caching configuration"""
    enabled: bool = True
    redis_url: str = os.getenv("REDIS_URL", "redis://localhost:6379")
    cache_ttl: int = 3600  # 1 hour
    max_cache_size: int = 1000

@dataclass
class AppConfig:
    """Main application configuration"""
    debug: bool = os.getenv("DEBUG", "False").lower() == "true"
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    max_file_size: int = 50 * 1024 * 1024  # 50MB
    supported_formats: list = None
    
    def __post_init__(self):
        if self.supported_formats is None:
            self.supported_formats = [".pdf", ".txt", ".docx"]

@dataclass
class SecurityConfig:
    """Security configuration"""
    enable_rate_limiting: bool = True
    max_requests_per_minute: int = 60
    enable_file_scanning: bool = True
    allowed_file_types: list = None
    
    def __post_init__(self):
        if self.allowed_file_types is None:
            self.allowed_file_types = ["application/pdf", "text/plain"]

# Global configuration instances
vector_db_config = VectorDBConfig()
llm_config = LLMConfig()
embedding_config = EmbeddingConfig()
cache_config = CacheConfig()
app_config = AppConfig()
security_config = SecurityConfig()

# Google API Key
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY environment variable is required")