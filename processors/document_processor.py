"""
Document Processing Module for the PDF Chat Application
Optimized text extraction, chunking, and metadata handling
"""
import os
import time
import hashlib
from typing import List, Dict, Any, Optional, Union
from io import BytesIO
import fitz  # PyMuPDF
import pdfplumber
from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
import pandas as pd

# Local imports
from config import embedding_config, app_config
from utils.logger import logger, log_performance, log_error
from utils.cache import cached

class DocumentProcessor:
    """Optimized document processor with multiple extraction methods"""
    
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=embedding_config.chunk_size,
            chunk_overlap=embedding_config.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
    
    def extract_text_from_pdf(self, pdf_file, filename: str) -> Dict[str, Any]:
        """Extract text from PDF using multiple methods for better results"""
        try:
            start_time = time.time()
            
            # Method 1: PyMuPDF (fitz) - Best for most PDFs
            text_fitz = self._extract_with_fitz(pdf_file)
            
            # Method 2: pdfplumber - Better for tables and complex layouts
            text_plumber = self._extract_with_pdfplumber(pdf_file)
            
            # Method 3: PyPDF2 - Fallback method
            text_pypdf = self._extract_with_pypdf(pdf_file)
            
            # Combine and clean text
            combined_text = self._combine_text_methods(
                text_fitz, text_plumber, text_pypdf
            )
            
            # Generate metadata
            metadata = self._generate_metadata(pdf_file, filename, combined_text)
            
            duration = time.time() - start_time
            
            log_performance(
                operation="pdf_text_extraction",
                duration=duration,
                filename=filename,
                text_length=len(combined_text)
            )
            
            return {
                "text": combined_text,
                "metadata": metadata,
                "success": True
            }
            
        except Exception as e:
            log_error(e, {"operation": "pdf_extraction", "filename": filename})
            return {
                "text": "",
                "metadata": {"filename": filename, "error": str(e)},
                "success": False
            }
    
    def _extract_with_fitz(self, pdf_file) -> str:
        """Extract text using PyMuPDF (fitz)"""
        try:
            # Reset file pointer
            pdf_file.seek(0)
            
            doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
            text = ""
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                text += page.get_text()
            
            doc.close()
            return text.strip()
            
        except Exception as e:
            logger.warning(f"Fitz extraction failed: {e}")
            return ""
    
    def _extract_with_pdfplumber(self, pdf_file) -> str:
        """Extract text using pdfplumber"""
        try:
            # Reset file pointer
            pdf_file.seek(0)
            
            with pdfplumber.open(pdf_file) as pdf:
                text = ""
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
            
            return text.strip()
            
        except Exception as e:
            logger.warning(f"Pdfplumber extraction failed: {e}")
            return ""
    
    def _extract_with_pypdf(self, pdf_file) -> str:
        """Extract text using PyPDF2"""
        try:
            # Reset file pointer
            pdf_file.seek(0)
            
            reader = PdfReader(pdf_file)
            text = ""
            
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
            
            return text.strip()
            
        except Exception as e:
            logger.warning(f"PyPDF2 extraction failed: {e}")
            return ""
    
    def _combine_text_methods(self, text_fitz: str, text_plumber: str, text_pypdf: str) -> str:
        """Combine text from multiple extraction methods"""
        texts = [text_fitz, text_plumber, text_pypdf]
        
        # Filter out empty texts
        valid_texts = [text for text in texts if text and len(text.strip()) > 0]
        
        if not valid_texts:
            return ""
        
        # If only one method worked, use it
        if len(valid_texts) == 1:
            return valid_texts[0]
        
        # If multiple methods worked, combine them intelligently
        # Use the longest text as base and supplement with unique content from others
        base_text = max(valid_texts, key=len)
        
        # Add unique content from other texts
        for text in valid_texts:
            if text != base_text:
                # Find unique sentences or paragraphs
                unique_content = self._extract_unique_content(base_text, text)
                if unique_content:
                    base_text += "\n\n" + unique_content
        
        return base_text.strip()
    
    def _extract_unique_content(self, base_text: str, additional_text: str) -> str:
        """Extract unique content from additional text"""
        # Simple approach: find sentences that don't appear in base text
        base_sentences = set(base_text.lower().split('.'))
        additional_sentences = additional_text.lower().split('.')
        
        unique_sentences = []
        for sentence in additional_sentences:
            sentence = sentence.strip()
            if sentence and sentence not in base_sentences:
                unique_sentences.append(sentence)
        
        return '. '.join(unique_sentences)
    
    def _generate_metadata(self, pdf_file, filename: str, text: str) -> Dict[str, Any]:
        """Generate comprehensive metadata for the document"""
        try:
            # Reset file pointer
            pdf_file.seek(0)
            
            # Basic file info
            file_size = len(pdf_file.read())
            pdf_file.seek(0)
            
            # Try to get PDF info
            try:
                reader = PdfReader(pdf_file)
                num_pages = len(reader.pages)
                pdf_file.seek(0)
            except:
                num_pages = "unknown"
            
            # Generate content hash
            content_hash = hashlib.md5(text.encode()).hexdigest()
            
            # Estimate reading time (average 200 words per minute)
            word_count = len(text.split())
            reading_time_minutes = max(1, word_count // 200)
            
            return {
                "filename": filename,
                "file_size_bytes": file_size,
                "file_size_mb": round(file_size / (1024 * 1024), 2),
                "num_pages": num_pages,
                "word_count": word_count,
                "character_count": len(text),
                "content_hash": content_hash,
                "reading_time_minutes": reading_time_minutes,
                "extraction_method": "combined",
                "timestamp": time.time()
            }
            
        except Exception as e:
            log_error(e, {"operation": "metadata_generation", "filename": filename})
            return {
                "filename": filename,
                "error": str(e),
                "timestamp": time.time()
            }
    
    @cached(ttl=7200, key_prefix="text_chunking")
    def create_chunks(self, text: str, metadata: Dict[str, Any]) -> List[Document]:
        """Create optimized text chunks with metadata"""
        try:
            start_time = time.time()
            
            if not text.strip():
                logger.warning("Empty text provided for chunking")
                return []
            
            # Create chunks
            chunks = self.text_splitter.split_text(text)
            
            # Create Document objects with metadata
            documents = []
            for i, chunk in enumerate(chunks):
                chunk_metadata = metadata.copy()
                chunk_metadata.update({
                    "chunk_id": i,
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    "chunk_size": len(chunk),
                    "chunk_hash": hashlib.md5(chunk.encode()).hexdigest()
                })
                
                document = Document(
                    page_content=chunk,
                    metadata=chunk_metadata
                )
                documents.append(document)
            
            duration = time.time() - start_time
            
            log_performance(
                operation="text_chunking",
                duration=duration,
                original_text_length=len(text),
                chunks_created=len(documents)
            )
            
            logger.info(f"Created {len(documents)} chunks from text")
            return documents
            
        except Exception as e:
            log_error(e, {"operation": "text_chunking"})
            return []
    
    def process_documents(self, files: List[Any]) -> List[Document]:
        """Process multiple documents and return chunks"""
        all_documents = []
        
        for file in files:
            try:
                # Extract text and metadata
                result = self.extract_text_from_pdf(file, file.name)
                
                if result["success"] and result["text"]:
                    # Create chunks
                    chunks = self.create_chunks(result["text"], result["metadata"])
                    all_documents.extend(chunks)
                    
                    logger.info(f"Processed {file.name}: {len(chunks)} chunks")
                else:
                    logger.warning(f"Failed to process {file.name}")
                    
            except Exception as e:
                log_error(e, {"operation": "document_processing", "filename": file.name})
                continue
        
        logger.info(f"Total documents processed: {len(all_documents)}")
        return all_documents
    
    def get_processing_stats(self, documents: List[Document]) -> Dict[str, Any]:
        """Get statistics about processed documents"""
        if not documents:
            return {"error": "No documents provided"}
        
        total_chunks = len(documents)
        total_words = sum(len(doc.page_content.split()) for doc in documents)
        total_chars = sum(len(doc.page_content) for doc in documents)
        
        # Get unique files
        unique_files = set(doc.metadata.get("filename", "unknown") for doc in documents)
        
        # Average chunk size
        avg_chunk_size = total_chars / total_chunks if total_chunks > 0 else 0
        
        return {
            "total_chunks": total_chunks,
            "total_words": total_words,
            "total_characters": total_chars,
            "unique_files": len(unique_files),
            "average_chunk_size": round(avg_chunk_size, 2),
            "files_processed": list(unique_files)
        }

# Global document processor instance
document_processor = DocumentProcessor()