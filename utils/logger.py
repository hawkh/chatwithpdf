"""
Structured logging configuration for the PDF Chat Application
"""
import structlog
import logging
import sys
from typing import Any, Dict
from config import app_config

def setup_logger() -> structlog.BoundLogger:
    """Setup structured logging with proper configuration"""
    
    # Configure structlog
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer()
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
    
    # Configure standard library logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, app_config.log_level.upper()),
    )
    
    return structlog.get_logger()

# Global logger instance
logger = setup_logger()

def log_function_call(func_name: str, **kwargs) -> structlog.BoundLogger:
    """Decorator helper for logging function calls"""
    return logger.bind(
        function=func_name,
        **kwargs
    )

def log_performance(operation: str, duration: float, **kwargs) -> None:
    """Log performance metrics"""
    logger.info(
        "Performance metric",
        operation=operation,
        duration_ms=duration * 1000,
        **kwargs
    )

def log_error(error: Exception, context: Dict[str, Any] = None) -> None:
    """Log errors with context"""
    logger.error(
        "Application error",
        error_type=type(error).__name__,
        error_message=str(error),
        context=context or {},
        exc_info=True
    )