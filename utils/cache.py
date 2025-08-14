"""
Caching utilities for the PDF Chat Application
"""
import hashlib
import json
import time
from typing import Any, Optional, Union
from functools import wraps
import redis
from cachetools import TTLCache, LRUCache
from config import cache_config
from utils.logger import logger, log_performance

class CacheManager:
    """Manages caching with multiple backends"""
    
    def __init__(self):
        self.redis_client = None
        self.memory_cache = None
        self._setup_caches()
    
    def _setup_caches(self):
        """Setup cache backends"""
        if cache_config.enabled:
            # Setup Redis if available
            try:
                self.redis_client = redis.from_url(cache_config.redis_url)
                self.redis_client.ping()  # Test connection
                logger.info("Redis cache initialized successfully")
            except Exception as e:
                logger.warning(f"Redis cache unavailable: {e}")
                self.redis_client = None
            
            # Setup in-memory cache as fallback
            self.memory_cache = TTLCache(
                maxsize=cache_config.max_cache_size,
                ttl=cache_config.cache_ttl
            )
            logger.info("Memory cache initialized")
    
    def _generate_key(self, *args, **kwargs) -> str:
        """Generate cache key from function arguments"""
        key_data = {
            'args': args,
            'kwargs': sorted(kwargs.items())
        }
        key_string = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        if not cache_config.enabled:
            return None
        
        # Try Redis first
        if self.redis_client:
            try:
                value = self.redis_client.get(key)
                if value:
                    logger.debug(f"Cache hit (Redis): {key}")
                    return json.loads(value)
            except Exception as e:
                logger.warning(f"Redis get error: {e}")
        
        # Fallback to memory cache
        if self.memory_cache and key in self.memory_cache:
            logger.debug(f"Cache hit (Memory): {key}")
            return self.memory_cache[key]
        
        logger.debug(f"Cache miss: {key}")
        return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in cache"""
        if not cache_config.enabled:
            return False
        
        ttl = ttl or cache_config.cache_ttl
        
        # Try Redis first
        if self.redis_client:
            try:
                self.redis_client.setex(
                    key, 
                    ttl, 
                    json.dumps(value, default=str)
                )
                logger.debug(f"Cache set (Redis): {key}")
                return True
            except Exception as e:
                logger.warning(f"Redis set error: {e}")
        
        # Fallback to memory cache
        if self.memory_cache:
            self.memory_cache[key] = value
            logger.debug(f"Cache set (Memory): {key}")
            return True
        
        return False
    
    def delete(self, key: str) -> bool:
        """Delete value from cache"""
        if not cache_config.enabled:
            return False
        
        success = False
        
        # Delete from Redis
        if self.redis_client:
            try:
                self.redis_client.delete(key)
                success = True
            except Exception as e:
                logger.warning(f"Redis delete error: {e}")
        
        # Delete from memory cache
        if self.memory_cache and key in self.memory_cache:
            del self.memory_cache[key]
            success = True
        
        return success
    
    def clear(self) -> None:
        """Clear all caches"""
        if self.redis_client:
            try:
                self.redis_client.flushdb()
                logger.info("Redis cache cleared")
            except Exception as e:
                logger.warning(f"Redis clear error: {e}")
        
        if self.memory_cache:
            self.memory_cache.clear()
            logger.info("Memory cache cleared")

# Global cache manager instance
cache_manager = CacheManager()

def cached(ttl: Optional[int] = None, key_prefix: str = ""):
    """Decorator for caching function results"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key
            cache_key = f"{key_prefix}:{cache_manager._generate_key(*args, **kwargs)}"
            
            # Try to get from cache
            cached_result = cache_manager.get(cache_key)
            if cached_result is not None:
                return cached_result
            
            # Execute function and cache result
            start_time = time.time()
            result = func(*args, **kwargs)
            duration = time.time() - start_time
            
            # Cache the result
            cache_manager.set(cache_key, result, ttl)
            
            # Log performance
            log_performance(
                operation=f"cached_{func.__name__}",
                duration=duration,
                cache_hit=False
            )
            
            return result
        return wrapper
    return decorator

def cache_key_generator(*args, **kwargs) -> str:
    """Generate a cache key from arguments"""
    return cache_manager._generate_key(*args, **kwargs)