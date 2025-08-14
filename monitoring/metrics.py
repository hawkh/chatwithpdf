"""
Monitoring and Metrics Service for the PDF Chat Application
Performance tracking, system health monitoring, and analytics
"""
import time
import psutil
import threading
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from collections import defaultdict, deque
import json

# Local imports
from utils.logger import logger, log_performance
from config import app_config

@dataclass
class SystemMetrics:
    """System performance metrics"""
    timestamp: float
    cpu_percent: float
    memory_percent: float
    memory_available: int
    disk_usage_percent: float
    network_io: Dict[str, int]

@dataclass
class ApplicationMetrics:
    """Application-specific metrics"""
    timestamp: float
    active_requests: int
    total_requests: int
    average_response_time: float
    cache_hit_rate: float
    vector_db_operations: int
    llm_calls: int
    errors_count: int

class MetricsCollector:
    """Collects and manages application metrics"""
    
    def __init__(self, max_history: int = 1000):
        self.max_history = max_history
        self.system_metrics = deque(maxlen=max_history)
        self.app_metrics = deque(maxlen=max_history)
        self.request_times = deque(maxlen=max_history)
        self.error_log = deque(maxlen=max_history)
        
        # Counters
        self.total_requests = 0
        self.total_errors = 0
        self.cache_hits = 0
        self.cache_misses = 0
        self.vector_db_ops = 0
        self.llm_calls = 0
        
        # Threading
        self.monitoring_active = False
        self.monitor_thread = None
    
    def start_monitoring(self, interval: int = 30):
        """Start background monitoring"""
        if not self.monitoring_active:
            self.monitoring_active = True
            self.monitor_thread = threading.Thread(
                target=self._monitor_loop,
                args=(interval,),
                daemon=True
            )
            self.monitor_thread.start()
            logger.info(f"Started metrics monitoring with {interval}s interval")
    
    def stop_monitoring(self):
        """Stop background monitoring"""
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        logger.info("Stopped metrics monitoring")
    
    def _monitor_loop(self, interval: int):
        """Background monitoring loop"""
        while self.monitoring_active:
            try:
                self.collect_system_metrics()
                self.collect_application_metrics()
                time.sleep(interval)
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(interval)
    
    def collect_system_metrics(self):
        """Collect system performance metrics"""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # Memory usage
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            memory_available = memory.available
            
            # Disk usage
            disk = psutil.disk_usage('/')
            disk_usage_percent = disk.percent
            
            # Network I/O
            network_io = psutil.net_io_counters()
            network_data = {
                'bytes_sent': network_io.bytes_sent,
                'bytes_recv': network_io.bytes_recv,
                'packets_sent': network_io.packets_sent,
                'packets_recv': network_io.packets_recv
            }
            
            metrics = SystemMetrics(
                timestamp=time.time(),
                cpu_percent=cpu_percent,
                memory_percent=memory_percent,
                memory_available=memory_available,
                disk_usage_percent=disk_usage_percent,
                network_io=network_data
            )
            
            self.system_metrics.append(metrics)
            
        except Exception as e:
            logger.error(f"Error collecting system metrics: {e}")
    
    def collect_application_metrics(self):
        """Collect application-specific metrics"""
        try:
            # Calculate average response time
            avg_response_time = 0
            if self.request_times:
                avg_response_time = sum(self.request_times) / len(self.request_times)
            
            # Calculate cache hit rate
            cache_hit_rate = 0
            total_cache_requests = self.cache_hits + self.cache_misses
            if total_cache_requests > 0:
                cache_hit_rate = (self.cache_hits / total_cache_requests) * 100
            
            metrics = ApplicationMetrics(
                timestamp=time.time(),
                active_requests=0,  # Would need request tracking
                total_requests=self.total_requests,
                average_response_time=avg_response_time,
                cache_hit_rate=cache_hit_rate,
                vector_db_operations=self.vector_db_ops,
                llm_calls=self.llm_calls,
                errors_count=self.total_errors
            )
            
            self.app_metrics.append(metrics)
            
        except Exception as e:
            logger.error(f"Error collecting application metrics: {e}")
    
    def record_request(self, response_time: float):
        """Record a new request"""
        self.total_requests += 1
        self.request_times.append(response_time)
    
    def record_error(self, error: Exception, context: Dict[str, Any] = None):
        """Record an error"""
        self.total_errors += 1
        error_entry = {
            'timestamp': time.time(),
            'error_type': type(error).__name__,
            'error_message': str(error),
            'context': context or {}
        }
        self.error_log.append(error_entry)
    
    def record_cache_hit(self):
        """Record a cache hit"""
        self.cache_hits += 1
    
    def record_cache_miss(self):
        """Record a cache miss"""
        self.cache_misses += 1
    
    def record_vector_db_operation(self):
        """Record a vector database operation"""
        self.vector_db_ops += 1
    
    def record_llm_call(self):
        """Record an LLM API call"""
        self.llm_calls += 1
    
    def get_system_health(self) -> Dict[str, Any]:
        """Get current system health status"""
        try:
            if not self.system_metrics:
                return {"status": "unknown", "message": "No metrics available"}
            
            latest = self.system_metrics[-1]
            
            # Health checks
            health_issues = []
            
            if latest.cpu_percent > 80:
                health_issues.append("High CPU usage")
            
            if latest.memory_percent > 85:
                health_issues.append("High memory usage")
            
            if latest.disk_usage_percent > 90:
                health_issues.append("High disk usage")
            
            status = "healthy" if not health_issues else "warning"
            
            return {
                "status": status,
                "issues": health_issues,
                "metrics": asdict(latest),
                "timestamp": datetime.fromtimestamp(latest.timestamp).isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting system health: {e}")
            return {"status": "error", "message": str(e)}
    
    def get_performance_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get performance summary for the last N hours"""
        try:
            cutoff_time = time.time() - (hours * 3600)
            
            # Filter metrics by time
            recent_system = [m for m in self.system_metrics if m.timestamp >= cutoff_time]
            recent_app = [m for m in self.app_metrics if m.timestamp >= cutoff_time]
            
            if not recent_system or not recent_app:
                return {"message": "No recent metrics available"}
            
            # System metrics summary
            avg_cpu = sum(m.cpu_percent for m in recent_system) / len(recent_system)
            avg_memory = sum(m.memory_percent for m in recent_system) / len(recent_system)
            max_cpu = max(m.cpu_percent for m in recent_system)
            max_memory = max(m.memory_percent for m in recent_system)
            
            # Application metrics summary
            total_requests = recent_app[-1].total_requests - recent_app[0].total_requests
            avg_response_time = sum(m.average_response_time for m in recent_app) / len(recent_app)
            avg_cache_hit_rate = sum(m.cache_hit_rate for m in recent_app) / len(recent_app)
            
            return {
                "period_hours": hours,
                "system_metrics": {
                    "average_cpu_percent": round(avg_cpu, 2),
                    "average_memory_percent": round(avg_memory, 2),
                    "max_cpu_percent": round(max_cpu, 2),
                    "max_memory_percent": round(max_memory, 2)
                },
                "application_metrics": {
                    "total_requests": total_requests,
                    "average_response_time_ms": round(avg_response_time * 1000, 2),
                    "average_cache_hit_rate_percent": round(avg_cache_hit_rate, 2),
                    "total_errors": recent_app[-1].errors_count - recent_app[0].errors_count
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting performance summary: {e}")
            return {"error": str(e)}
    
    def get_error_analysis(self, hours: int = 24) -> Dict[str, Any]:
        """Analyze errors from the last N hours"""
        try:
            cutoff_time = time.time() - (hours * 3600)
            recent_errors = [e for e in self.error_log if e['timestamp'] >= cutoff_time]
            
            if not recent_errors:
                return {"message": "No recent errors"}
            
            # Error type analysis
            error_types = defaultdict(int)
            for error in recent_errors:
                error_types[error['error_type']] += 1
            
            # Most common errors
            most_common = sorted(error_types.items(), key=lambda x: x[1], reverse=True)
            
            return {
                "period_hours": hours,
                "total_errors": len(recent_errors),
                "error_types": dict(error_types),
                "most_common_errors": most_common[:5],
                "recent_errors": recent_errors[-10:]  # Last 10 errors
            }
            
        except Exception as e:
            logger.error(f"Error analyzing errors: {e}")
            return {"error": str(e)}
    
    def export_metrics(self, format: str = "json") -> str:
        """Export metrics data"""
        try:
            data = {
                "system_metrics": [asdict(m) for m in self.system_metrics],
                "application_metrics": [asdict(m) for m in self.app_metrics],
                "error_log": list(self.error_log),
                "summary": {
                    "total_requests": self.total_requests,
                    "total_errors": self.total_errors,
                    "cache_hits": self.cache_hits,
                    "cache_misses": self.cache_misses,
                    "vector_db_operations": self.vector_db_ops,
                    "llm_calls": self.llm_calls
                }
            }
            
            if format.lower() == "json":
                return json.dumps(data, indent=2, default=str)
            else:
                return str(data)
                
        except Exception as e:
            logger.error(f"Error exporting metrics: {e}")
            return f"Error: {str(e)}"
    
    def clear_metrics(self):
        """Clear all collected metrics"""
        self.system_metrics.clear()
        self.app_metrics.clear()
        self.request_times.clear()
        self.error_log.clear()
        
        # Reset counters
        self.total_requests = 0
        self.total_errors = 0
        self.cache_hits = 0
        self.cache_misses = 0
        self.vector_db_ops = 0
        self.llm_calls = 0
        
        logger.info("Metrics cleared")

# Global metrics collector instance
metrics_collector = MetricsCollector()

# Decorator for automatic metrics collection
def track_metrics(operation: str):
    """Decorator to automatically track metrics for operations"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                response_time = time.time() - start_time
                metrics_collector.record_request(response_time)
                return result
            except Exception as e:
                metrics_collector.record_error(e, {"operation": operation})
                raise
        return wrapper
    return decorator