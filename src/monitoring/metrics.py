"""
Metrics collection for monitoring API performance.
"""

import psutil
import time
import GPUtil
from typing import Dict, Any, Optional
import logging
from collections import defaultdict, deque
from prometheus_client import Counter, Histogram, Gauge, Info

logger = logging.getLogger(__name__)


class APIMetrics:
    """API performance metrics for request rate, error rate, and latency."""
    
    def __init__(self):
        # Request metrics
        self.total_requests = 0
        self.total_errors = 0
        self.request_times = deque(maxlen=1000)  # Store last 1000 request times
        self.error_times = deque(maxlen=1000)    # Store last 1000 error times
        
        # Per-endpoint metrics
        self.endpoint_requests = defaultdict(int)
        self.endpoint_errors = defaultdict(int)
        self.endpoint_latencies = defaultdict(list)
        
        # Status code tracking
        self.status_codes = defaultdict(int)
        
        # Time window for rate calculations (last 60 seconds)
        self.time_window = 60
        
        # Prometheus metrics
        self.request_counter = Counter('api_requests_total', 'Total API requests', ['method', 'endpoint', 'status'])
        self.request_duration = Histogram('api_request_duration_seconds', 'Request duration', ['method', 'endpoint'])
        self.error_rate_gauge = Gauge('api_error_rate', 'Current error rate percentage')
        self.requests_per_second_gauge = Gauge('api_requests_per_second', 'Current requests per second')
    
    def record_request(self, method: str, endpoint: str, status_code: int, latency: float):
        """Record an API request."""
        current_time = time.time()
        
        # Update counters
        self.total_requests += 1
        self.request_times.append(current_time)
        
        # Track by endpoint
        endpoint_key = f"{method} {endpoint}"
        self.endpoint_requests[endpoint_key] += 1
        self.endpoint_latencies[endpoint_key].append(latency)
        
        # Track status codes
        self.status_codes[status_code] += 1
        
        # Track errors (4xx and 5xx)
        if status_code >= 400:
            self.total_errors += 1
            self.error_times.append(current_time)
            self.endpoint_errors[endpoint_key] += 1
        
        # Update Prometheus metrics
        self.request_counter.labels(method=method, endpoint=endpoint, status=str(status_code)).inc()
        self.request_duration.labels(method=method, endpoint=endpoint).observe(latency)
    
    def get_requests_per_second(self) -> float:
        """Calculate current requests per second."""
        current_time = time.time()
        cutoff_time = current_time - self.time_window
        
        # Count requests in the last time window
        recent_requests = sum(1 for req_time in self.request_times if req_time > cutoff_time)
        return recent_requests / self.time_window
    
    def get_error_rate(self) -> float:
        """Calculate current error rate percentage."""
        if self.total_requests == 0:
            return 0.0
        
        current_time = time.time()
        cutoff_time = current_time - self.time_window
        
        # Count recent requests and errors
        recent_requests = sum(1 for req_time in self.request_times if req_time > cutoff_time)
        recent_errors = sum(1 for err_time in self.error_times if err_time > cutoff_time)
        
        if recent_requests == 0:
            return 0.0
        
        return (recent_errors / recent_requests) * 100
    
    def get_average_latency(self, endpoint: str = None) -> float:
        """Get average latency for an endpoint or overall."""
        if endpoint:
            latencies = self.endpoint_latencies.get(endpoint, [])
            if not latencies:
                return 0.0
            return sum(latencies) / len(latencies)
        else:
            # Overall average latency
            all_latencies = []
            for latencies in self.endpoint_latencies.values():
                all_latencies.extend(latencies)
            
            if not all_latencies:
                return 0.0
            return sum(all_latencies) / len(all_latencies)
    
    def update_prometheus_gauges(self):
        """Update Prometheus gauge metrics."""
        self.requests_per_second_gauge.set(self.get_requests_per_second())
        self.error_rate_gauge.set(self.get_error_rate())
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current API metrics."""
        rps = self.get_requests_per_second()
        error_rate = self.get_error_rate()
        avg_latency = self.get_average_latency()
        
        # Update Prometheus gauges
        self.update_prometheus_gauges()
        
        return {
            'total_requests': self.total_requests,
            'total_errors': self.total_errors,
            'requests_per_second': rps,
            'error_rate_percent': error_rate,
            'average_latency_seconds': avg_latency,
            'status_code_distribution': dict(self.status_codes),
            'endpoint_metrics': {
                endpoint: {
                    'requests': self.endpoint_requests[endpoint],
                    'errors': self.endpoint_errors[endpoint],
                    'avg_latency': self.get_average_latency(endpoint),
                    'error_rate': (self.endpoint_errors[endpoint] / max(1, self.endpoint_requests[endpoint])) * 100
                }
                for endpoint in self.endpoint_requests.keys()
            }
        }


class ModelMetrics:
    """Model performance metrics."""
    
    def __init__(self):
        self.total_predictions = 0
        self.total_inference_time = 0.0
        self.confidence_scores = []
        
        # Separate CPU and GPU inference times
        self.cpu_inference_times = []
        self.gpu_inference_times = []
        
        # Prometheus metrics for model
        self.prediction_counter = Counter('model_predictions_total', 'Total model predictions', ['predicted_class'])
        self.inference_duration = Histogram('model_inference_duration_seconds', 'Model inference duration', ['device'])
        self.confidence_gauge = Gauge('model_confidence_score', 'Current model confidence score')
        self.avg_confidence_gauge = Gauge('model_avg_confidence_score', 'Average model confidence score')
    
    def record_prediction(self, inference_time: float, confidence: float, predicted_class: str, device: str = "cpu"):
        """Record a prediction with device-specific timing."""
        self.total_predictions += 1
        self.total_inference_time += inference_time
        self.confidence_scores.append(confidence)
        
        # Track device-specific inference times
        if "cuda" in device.lower() or "gpu" in device.lower():
            self.gpu_inference_times.append(inference_time)
        else:
            self.cpu_inference_times.append(inference_time)
        
        # Update Prometheus metrics
        self.prediction_counter.labels(predicted_class=predicted_class).inc()
        self.inference_duration.labels(device=device).observe(inference_time)
        self.confidence_gauge.set(confidence)
        
        if self.confidence_scores:
            avg_confidence = sum(self.confidence_scores) / len(self.confidence_scores)
            self.avg_confidence_gauge.set(avg_confidence)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics."""
        avg_inference_time = self.total_inference_time / max(1, self.total_predictions)
        avg_confidence = sum(self.confidence_scores) / max(1, len(self.confidence_scores))
        
        # Calculate device-specific averages
        avg_cpu_time = sum(self.cpu_inference_times) / max(1, len(self.cpu_inference_times))
        avg_gpu_time = sum(self.gpu_inference_times) / max(1, len(self.gpu_inference_times))
        
        return {
            'total_predictions': self.total_predictions,
            'avg_inference_time': avg_inference_time,
            'avg_confidence': avg_confidence,
            'cpu_inference_time_avg': avg_cpu_time,
            'gpu_inference_time_avg': avg_gpu_time,
            'cpu_predictions_count': len(self.cpu_inference_times),
            'gpu_predictions_count': len(self.gpu_inference_times)
        }


class SystemMetrics:
    """System resource metrics with enhanced monitoring."""
    
    def __init__(self):
        self.cpu_percent = 0.0
        self.memory_percent = 0.0
        self.disk_percent = 0.0
        
        # Enhanced metrics
        self.gpu_utilization = 0.0
        self.gpu_memory_used = 0.0
        self.gpu_memory_total = 0.0
        self.gpu_temperature = 0.0
        
        # Network IO metrics
        self.network_bytes_sent = 0
        self.network_bytes_recv = 0
        self.network_packets_sent = 0
        self.network_packets_recv = 0
        
        # Disk IO metrics  
        self.disk_read_bytes = 0
        self.disk_write_bytes = 0
        self.disk_read_count = 0
        self.disk_write_count = 0
        
        # Store initial values for delta calculations
        self._initial_network_io = None
        self._initial_disk_io = None
        self._last_update_time = time.time()
        
        # Prometheus gauges for system metrics
        self.cpu_usage_gauge = Gauge('system_cpu_usage_percent', 'CPU usage percentage')
        self.memory_usage_gauge = Gauge('system_memory_usage_percent', 'Memory usage percentage')
        self.disk_usage_gauge = Gauge('system_disk_usage_percent', 'Disk usage percentage')
        self.gpu_usage_gauge = Gauge('system_gpu_usage_percent', 'GPU usage percentage')
        self.gpu_memory_gauge = Gauge('system_gpu_memory_usage_percent', 'GPU memory usage percentage')
    
    def _get_gpu_metrics(self) -> Dict[str, float]:
        """Get GPU metrics if available."""
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu = gpus[0]  # Use first GPU
                return {
                    'utilization': gpu.load * 100,
                    'memory_used': gpu.memoryUsed,
                    'memory_total': gpu.memoryTotal,
                    'temperature': gpu.temperature
                }
            else:
                return {
                    'utilization': 0.0,
                    'memory_used': 0.0,
                    'memory_total': 0.0,
                    'temperature': 0.0
                }
        except Exception as e:
            logger.warning(f"GPU metrics not available: {e}")
            return {
                'utilization': 0.0,
                'memory_used': 0.0,
                'memory_total': 0.0,
                'temperature': 0.0
            }
    
    def _get_network_io(self) -> Dict[str, int]:
        """Get network IO statistics."""
        try:
            net_io = psutil.net_io_counters()
            
            if self._initial_network_io is None:
                self._initial_network_io = net_io
            
            return {
                'bytes_sent': net_io.bytes_sent - self._initial_network_io.bytes_sent,
                'bytes_recv': net_io.bytes_recv - self._initial_network_io.bytes_recv,
                'packets_sent': net_io.packets_sent - self._initial_network_io.packets_sent,
                'packets_recv': net_io.packets_recv - self._initial_network_io.packets_recv
            }
        except Exception as e:
            logger.warning(f"Network IO metrics not available: {e}")
            return {
                'bytes_sent': 0,
                'bytes_recv': 0,
                'packets_sent': 0,
                'packets_recv': 0
            }
    
    def _get_disk_io(self) -> Dict[str, int]:
        """Get disk IO statistics."""
        try:
            disk_io = psutil.disk_io_counters()
            
            if disk_io is None:
                return {
                    'read_bytes': 0,
                    'write_bytes': 0,
                    'read_count': 0,
                    'write_count': 0
                }
            
            if self._initial_disk_io is None:
                self._initial_disk_io = disk_io
            
            return {
                'read_bytes': disk_io.read_bytes - self._initial_disk_io.read_bytes,
                'write_bytes': disk_io.write_bytes - self._initial_disk_io.write_bytes,
                'read_count': disk_io.read_count - self._initial_disk_io.read_count,
                'write_count': disk_io.write_count - self._initial_disk_io.write_count
            }
        except Exception as e:
            logger.warning(f"Disk IO metrics not available: {e}")
            return {
                'read_bytes': 0,
                'write_bytes': 0,
                'read_count': 0,
                'write_count': 0
            }
    
    def update_system_metrics(self):
        """Update all system metrics."""
        # Basic system metrics
        self.cpu_percent = psutil.cpu_percent(interval=1)
        self.memory_percent = psutil.virtual_memory().percent
        self.disk_percent = psutil.disk_usage('/').percent
        
        # GPU metrics
        gpu_metrics = self._get_gpu_metrics()
        self.gpu_utilization = gpu_metrics['utilization']
        self.gpu_memory_used = gpu_metrics['memory_used']
        self.gpu_memory_total = gpu_metrics['memory_total']
        self.gpu_temperature = gpu_metrics['temperature']
        
        # Network IO metrics
        network_metrics = self._get_network_io()
        self.network_bytes_sent = network_metrics['bytes_sent']
        self.network_bytes_recv = network_metrics['bytes_recv']
        self.network_packets_sent = network_metrics['packets_sent']
        self.network_packets_recv = network_metrics['packets_recv']
        
        # Disk IO metrics
        disk_metrics = self._get_disk_io()
        self.disk_read_bytes = disk_metrics['read_bytes']
        self.disk_write_bytes = disk_metrics['write_bytes']
        self.disk_read_count = disk_metrics['read_count']
        self.disk_write_count = disk_metrics['write_count']
        
        self._last_update_time = time.time()
        
        # Update Prometheus gauges
        self.cpu_usage_gauge.set(self.cpu_percent)
        self.memory_usage_gauge.set(self.memory_percent)
        self.disk_usage_gauge.set(self.disk_percent)
        self.gpu_usage_gauge.set(self.gpu_utilization)
        
        gpu_memory_percent = (self.gpu_memory_used / max(1, self.gpu_memory_total)) * 100
        self.gpu_memory_gauge.set(gpu_memory_percent)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current system metrics."""
        return {
            # Basic system metrics
            'cpu_percent': self.cpu_percent,
            'memory_percent': self.memory_percent,
            'disk_percent': self.disk_percent,
            
            # GPU metrics
            'gpu_utilization_percent': self.gpu_utilization,
            'gpu_memory_used_mb': self.gpu_memory_used,
            'gpu_memory_total_mb': self.gpu_memory_total,
            'gpu_memory_percent': (self.gpu_memory_used / max(1, self.gpu_memory_total)) * 100,
            'gpu_temperature_celsius': self.gpu_temperature,
            
            # Network IO metrics (in bytes)
            'network_bytes_sent_total': self.network_bytes_sent,
            'network_bytes_recv_total': self.network_bytes_recv,
            'network_packets_sent_total': self.network_packets_sent,
            'network_packets_recv_total': self.network_packets_recv,
            
            # Disk IO metrics
            'disk_read_bytes_total': self.disk_read_bytes,
            'disk_write_bytes_total': self.disk_write_bytes,
            'disk_read_count_total': self.disk_read_count,
            'disk_write_count_total': self.disk_write_count
        }


class MetricsCollector:
    """Main metrics collector with comprehensive monitoring."""
    
    def __init__(self):
        self.model_metrics = ModelMetrics()
        self.system_metrics = SystemMetrics()
        self.api_metrics = APIMetrics()
        self.model_name = "unknown"
        self.num_classes = 0
        self.device = "unknown"
    
    def set_model_info(self, model_name: str, num_classes: int, device: str):
        """Set model information."""
        self.model_name = model_name
        self.num_classes = num_classes
        self.device = device
    
    def record_prediction(self, inference_time: float, confidence: float, predicted_class: str):
        """Record a model prediction."""
        self.model_metrics.record_prediction(inference_time, confidence, predicted_class, self.device)
    
    def record_api_request(self, method: str, endpoint: str, status_code: int, latency: float):
        """Record an API request."""
        self.api_metrics.record_request(method, endpoint, status_code, latency)
    
    def get_all_metrics(self) -> Dict[str, Any]:
        """Get all metrics."""
        return {
            'model': self.model_metrics.get_metrics(),
            'system': self.system_metrics.get_metrics(),
            'api': self.api_metrics.get_metrics(),
            'info': {
                'model_name': self.model_name,
                'num_classes': self.num_classes,
                'device': self.device
            }
        }