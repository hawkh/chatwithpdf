#!/usr/bin/env python3
"""
Test script for the optimized PDF Chat Application
Verifies that all components are working correctly
"""
import os
import sys
import time
from pathlib import Path

def test_imports():
    """Test that all modules can be imported"""
    print("🔍 Testing imports...")
    
    try:
        # Test configuration
        from config import (
            vector_db_config, 
            llm_config, 
            embedding_config, 
            cache_config, 
            app_config, 
            security_config
        )
        print("✅ Configuration modules imported successfully")
        
        # Test utilities
        from utils.logger import logger, log_performance, log_error
        from utils.cache import cache_manager, cached
        print("✅ Utility modules imported successfully")
        
        # Test vector database
        from vector_db.manager import vector_db_manager
        print("✅ Vector database modules imported successfully")
        
        # Test processors
        from processors.document_processor import document_processor
        print("✅ Document processor modules imported successfully")
        
        # Test services
        from services.llm_service import llm_service
        print("✅ LLM service modules imported successfully")
        
        # Test monitoring
        from monitoring.metrics import metrics_collector, track_metrics
        print("✅ Monitoring modules imported successfully")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error during imports: {e}")
        return False

def test_configuration():
    """Test configuration loading"""
    print("\n🔧 Testing configuration...")
    
    try:
        from config import vector_db_config, llm_config, embedding_config
        
        # Test vector DB config
        print(f"✅ Vector DB type: {vector_db_config.type}")
        print(f"✅ Vector DB directory: {vector_db_config.persist_directory}")
        
        # Test LLM config
        print(f"✅ LLM model: {llm_config.model_name}")
        print(f"✅ LLM temperature: {llm_config.temperature}")
        
        # Test embedding config
        print(f"✅ Chunk size: {embedding_config.chunk_size}")
        print(f"✅ Chunk overlap: {embedding_config.chunk_overlap}")
        
        return True
        
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False

def test_logging():
    """Test logging functionality"""
    print("\n📝 Testing logging...")
    
    try:
        from utils.logger import logger, log_performance, log_error
        
        # Test basic logging
        logger.info("Test info message")
        logger.warning("Test warning message")
        
        # Test performance logging
        log_performance("test_operation", 0.1, test_param="value")
        
        # Test error logging
        try:
            raise ValueError("Test error")
        except Exception as e:
            log_error(e, {"test_context": "value"})
        
        print("✅ Logging functionality working")
        return True
        
    except Exception as e:
        print(f"❌ Logging test failed: {e}")
        return False

def test_cache():
    """Test caching functionality"""
    print("\n💾 Testing caching...")
    
    try:
        from utils.cache import cache_manager, cached
        
        # Test cache manager
        test_key = "test_key"
        test_value = {"test": "data"}
        
        # Set value
        success = cache_manager.set(test_key, test_value)
        print(f"✅ Cache set: {success}")
        
        # Get value
        retrieved = cache_manager.get(test_key)
        print(f"✅ Cache get: {retrieved == test_value}")
        
        # Test cached decorator
        @cached(ttl=60, key_prefix="test")
        def test_function(x):
            return x * 2
        
        result1 = test_function(5)
        result2 = test_function(5)  # Should be cached
        print(f"✅ Cached function: {result1 == result2 == 10}")
        
        return True
        
    except Exception as e:
        print(f"❌ Cache test failed: {e}")
        return False

def test_vector_db():
    """Test vector database functionality"""
    print("\n🗄️ Testing vector database...")
    
    try:
        from vector_db.manager import vector_db_manager
        
        # Test health check
        health = vector_db_manager.health_check()
        print(f"✅ Vector DB health: {health.get('status', 'unknown')}")
        
        # Test stats
        stats = vector_db_manager.get_stats()
        print(f"✅ Vector DB stats: {stats.get('type', 'unknown')}")
        
        return True
        
    except Exception as e:
        print(f"❌ Vector DB test failed: {e}")
        return False

def test_document_processor():
    """Test document processor functionality"""
    print("\n📄 Testing document processor...")
    
    try:
        from processors.document_processor import document_processor
        
        # Test text chunking
        test_text = "This is a test document. " * 100  # Create longer text
        test_metadata = {"filename": "test.pdf", "test": True}
        
        chunks = document_processor.create_chunks(test_text, test_metadata)
        print(f"✅ Created {len(chunks)} chunks")
        
        if chunks:
            print(f"✅ First chunk size: {len(chunks[0].page_content)} characters")
            print(f"✅ Metadata preserved: {chunks[0].metadata.get('filename') == 'test.pdf'}")
        
        return True
        
    except Exception as e:
        print(f"❌ Document processor test failed: {e}")
        return False

def test_llm_service():
    """Test LLM service functionality"""
    print("\n🤖 Testing LLM service...")
    
    try:
        from services.llm_service import llm_service
        
        # Test model info
        model_info = llm_service.get_model_info()
        print(f"✅ Model info: {model_info.get('model_name', 'unknown')}")
        
        # Test health check (without actual API call)
        try:
            health = llm_service.health_check()
            print(f"✅ LLM health: {health.get('status', 'unknown')}")
        except Exception as e:
            print(f"⚠️  LLM health check failed (expected without API key): {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ LLM service test failed: {e}")
        return False

def test_monitoring():
    """Test monitoring functionality"""
    print("\n📊 Testing monitoring...")
    
    try:
        from monitoring.metrics import metrics_collector
        
        # Test metrics collection
        metrics_collector.record_request(0.1)
        metrics_collector.record_cache_hit()
        metrics_collector.record_vector_db_operation()
        
        # Test system health
        health = metrics_collector.get_system_health()
        print(f"✅ System health: {health.get('status', 'unknown')}")
        
        # Test performance summary
        summary = metrics_collector.get_performance_summary(hours=1)
        print(f"✅ Performance summary available: {bool(summary)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Monitoring test failed: {e}")
        return False

def test_directory_structure():
    """Test that required directories exist"""
    print("\n📁 Testing directory structure...")
    
    required_dirs = [
        "data",
        "data/vector_db",
        "data/cache",
        "logs",
        "temp"
    ]
    
    for directory in required_dirs:
        if os.path.exists(directory):
            print(f"✅ {directory}/ exists")
        else:
            print(f"⚠️  {directory}/ missing - creating...")
            try:
                os.makedirs(directory, exist_ok=True)
                print(f"✅ Created {directory}/")
            except Exception as e:
                print(f"❌ Failed to create {directory}/: {e}")
                return False
    
    return True

def main():
    """Main test function"""
    print("=" * 60)
    print("🧪 PDF Chat Application - Component Test")
    print("=" * 60)
    
    tests = [
        ("Imports", test_imports),
        ("Configuration", test_configuration),
        ("Logging", test_logging),
        ("Caching", test_cache),
        ("Vector Database", test_vector_db),
        ("Document Processor", test_document_processor),
        ("LLM Service", test_llm_service),
        ("Monitoring", test_monitoring),
        ("Directory Structure", test_directory_structure)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                print(f"❌ {test_name} test failed")
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
    
    print("\n" + "=" * 60)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The optimized application is ready to use.")
        print("\n📋 Next steps:")
        print("1. Set up your .env file with API keys")
        print("2. Run: streamlit run app.py")
        print("3. Start using the optimized application!")
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
        print("📚 See README_OPTIMIZED.md for troubleshooting.")
    
    print("=" * 60)

if __name__ == "__main__":
    main()