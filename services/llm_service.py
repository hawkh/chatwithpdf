"""
LLM Service for the PDF Chat Application
Optimized language model interactions with caching and error handling
"""
import time
from typing import List, Dict, Any, Optional
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain.schema import Document

# Local imports
from config import llm_config, GOOGLE_API_KEY
from utils.logger import logger, log_performance, log_error
from utils.cache import cached

class LLMService:
    """Optimized LLM service with caching and error handling"""
    
    def __init__(self):
        self.model = self._initialize_model()
        self.chains = {}
        self._setup_prompts()
    
    def _initialize_model(self) -> ChatGoogleGenerativeAI:
        """Initialize the language model"""
        try:
            model = ChatGoogleGenerativeAI(
                model=llm_config.model_name,
                temperature=llm_config.temperature,
                max_output_tokens=llm_config.max_tokens,
                top_p=llm_config.top_p,
                top_k=llm_config.top_k,
                google_api_key=GOOGLE_API_KEY
            )
            logger.info(f"LLM model initialized: {llm_config.model_name}")
            return model
        except Exception as e:
            log_error(e, {"operation": "llm_initialization"})
            raise
    
    def _setup_prompts(self):
        """Setup different prompt templates for various use cases"""
        self.prompts = {
            "qa": PromptTemplate(
                template="""
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
                
                Answer:""",
                input_variables=["context", "question"]
            ),
            
            "summary": PromptTemplate(
                template="""
                You are a helpful AI assistant that creates concise summaries.
                
                Please provide a comprehensive summary of the following text:
                
                {text}
                
                Summary:""",
                input_variables=["text"]
            ),
            
            "analysis": PromptTemplate(
                template="""
                You are a helpful AI assistant that provides detailed analysis.
                
                Please analyze the following text and provide insights:
                
                {text}
                
                Analysis:""",
                input_variables=["text"]
            )
        }
    
    def _get_chain(self, chain_type: str = "qa"):
        """Get or create a chain for the specified type"""
        if chain_type not in self.chains:
            try:
                prompt = self.prompts.get(chain_type, self.prompts["qa"])
                chain = load_qa_chain(
                    self.model,
                    chain_type="stuff",
                    prompt=prompt,
                    verbose=False
                )
                self.chains[chain_type] = chain
                logger.info(f"Created chain for type: {chain_type}")
            except Exception as e:
                log_error(e, {"operation": "chain_creation", "chain_type": chain_type})
                raise
        
        return self.chains[chain_type]
    
    @cached(ttl=1800, key_prefix="llm_response")  # 30 minutes cache
    def generate_response(self, question: str, context_docs: List[Document], 
                         chain_type: str = "qa") -> Dict[str, Any]:
        """Generate response with caching and error handling"""
        try:
            start_time = time.time()
            
            # Validate inputs
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
            
            # Get the appropriate chain
            chain = self._get_chain(chain_type)
            
            # Prepare input
            chain_input = {
                "input_documents": context_docs,
                "question": question
            }
            
            # Generate response
            result = chain(chain_input, return_only_outputs=True)
            response_text = result.get("output_text", "")
            
            duration = time.time() - start_time
            
            # Log performance
            log_performance(
                operation="llm_response_generation",
                duration=duration,
                question_length=len(question),
                context_docs_count=len(context_docs),
                response_length=len(response_text),
                chain_type=chain_type
            )
            
            return {
                "success": True,
                "response": response_text,
                "metadata": {
                    "chain_type": chain_type,
                    "context_docs_count": len(context_docs),
                    "processing_time": duration,
                    "model": llm_config.model_name
                }
            }
            
        except Exception as e:
            log_error(e, {
                "operation": "llm_response_generation",
                "question": question,
                "chain_type": chain_type
            })
            return {
                "success": False,
                "error": str(e),
                "response": "I apologize, but I encountered an error while processing your request."
            }
    
    def generate_summary(self, text: str) -> Dict[str, Any]:
        """Generate a summary of the provided text"""
        try:
            start_time = time.time()
            
            if not text.strip():
                return {
                    "success": False,
                    "error": "Empty text provided",
                    "summary": ""
                }
            
            # Create a simple document for summarization
            doc = Document(page_content=text, metadata={})
            
            # Use the summary chain
            result = self.generate_response(
                question="Please provide a comprehensive summary",
                context_docs=[doc],
                chain_type="summary"
            )
            
            duration = time.time() - start_time
            
            log_performance(
                operation="text_summarization",
                duration=duration,
                text_length=len(text)
            )
            
            return result
            
        except Exception as e:
            log_error(e, {"operation": "text_summarization"})
            return {
                "success": False,
                "error": str(e),
                "summary": ""
            }
    
    def analyze_document(self, text: str) -> Dict[str, Any]:
        """Analyze the provided text and provide insights"""
        try:
            start_time = time.time()
            
            if not text.strip():
                return {
                    "success": False,
                    "error": "Empty text provided",
                    "analysis": ""
                }
            
            # Create a document for analysis
            doc = Document(page_content=text, metadata={})
            
            # Use the analysis chain
            result = self.generate_response(
                question="Please analyze this text and provide insights",
                context_docs=[doc],
                chain_type="analysis"
            )
            
            duration = time.time() - start_time
            
            log_performance(
                operation="document_analysis",
                duration=duration,
                text_length=len(text)
            )
            
            return result
            
        except Exception as e:
            log_error(e, {"operation": "document_analysis"})
            return {
                "success": False,
                "error": str(e),
                "analysis": ""
            }
    
    def health_check(self) -> Dict[str, Any]:
        """Perform health check on the LLM service"""
        try:
            # Test with a simple question
            test_doc = Document(
                page_content="This is a test document for health check.",
                metadata={"test": True}
            )
            
            result = self.generate_response(
                question="What is this document about?",
                context_docs=[test_doc]
            )
            
            return {
                "status": "healthy" if result["success"] else "unhealthy",
                "model": llm_config.model_name,
                "test_response": result.get("response", ""),
                "error": result.get("error", "")
            }
            
        except Exception as e:
            log_error(e, {"operation": "llm_health_check"})
            return {
                "status": "unhealthy",
                "model": llm_config.model_name,
                "error": str(e)
            }
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model"""
        return {
            "model_name": llm_config.model_name,
            "temperature": llm_config.temperature,
            "max_tokens": llm_config.max_tokens,
            "top_p": llm_config.top_p,
            "top_k": llm_config.top_k
        }

# Global LLM service instance
llm_service = LLMService()