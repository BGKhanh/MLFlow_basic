"""
Comprehensive logging configuration for CIFAR-10 API.
"""

import logging
import logging.handlers
import os
import sys
import json
from datetime import datetime
from typing import Dict, Any, Optional
from pathlib import Path


class StructuredFormatter(logging.Formatter):
    """Custom formatter for structured JSON logging."""
    
    def format(self, record):
        log_entry = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }
        
        # Add exception info if present
        if record.exc_info:
            log_entry['exception'] = self.formatException(record.exc_info)
        
        # Add extra fields if present
        if hasattr(record, 'user_id'):
            log_entry['user_id'] = record.user_id
        if hasattr(record, 'request_id'):
            log_entry['request_id'] = record.request_id
        if hasattr(record, 'endpoint'):
            log_entry['endpoint'] = record.endpoint
        if hasattr(record, 'status_code'):
            log_entry['status_code'] = record.status_code
        if hasattr(record, 'latency'):
            log_entry['latency'] = record.latency
        
        return json.dumps(log_entry)


class LoggingService:
    """Comprehensive logging service for monitoring and debugging."""
    
    def __init__(self, 
                 log_dir: str = "logs",
                 app_name: str = "cifar10-api",
                 log_level: str = "INFO"):
        """
        Initialize logging service.
        
        Args:
            log_dir: Directory to store log files
            app_name: Application name for log identification
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        """
        self.log_dir = Path(log_dir)
        self.app_name = app_name
        self.log_level = getattr(logging, log_level.upper())
        
        # Create log directory
        self.log_dir.mkdir(exist_ok=True)
        
        # Initialize loggers
        self.setup_loggers()
        
        # Store loggers for easy access
        self.app_logger = logging.getLogger(f"{app_name}.app")
        self.api_logger = logging.getLogger(f"{app_name}.api")
        self.model_logger = logging.getLogger(f"{app_name}.model")
        self.system_logger = logging.getLogger(f"{app_name}.system")
        self.error_logger = logging.getLogger(f"{app_name}.error")
    
    def setup_loggers(self):
        """Setup all logging handlers and formatters."""
        
        # 1. STDOUT Handler (Console output)
        stdout_handler = logging.StreamHandler(sys.stdout)
        stdout_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        stdout_handler.setFormatter(stdout_formatter)
        stdout_handler.setLevel(logging.INFO)
        stdout_handler.addFilter(lambda record: record.levelno < logging.ERROR)
        
        # 2. STDERR Handler (Error output)
        stderr_handler = logging.StreamHandler(sys.stderr)
        stderr_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s\n%(pathname)s:%(lineno)d'
        )
        stderr_handler.setFormatter(stderr_formatter)
        stderr_handler.setLevel(logging.ERROR)
        
        # 3. Application Log File Handler
        app_file_handler = logging.handlers.RotatingFileHandler(
            self.log_dir / f"{self.app_name}_app.log",
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5
        )
        app_file_handler.setFormatter(StructuredFormatter())
        app_file_handler.setLevel(logging.DEBUG)
        
        # 4. API Access Log Handler
        api_file_handler = logging.handlers.RotatingFileHandler(
            self.log_dir / f"{self.app_name}_api_access.log",
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5
        )
        api_file_handler.setFormatter(StructuredFormatter())
        api_file_handler.setLevel(logging.INFO)
        
        # 5. Error Log Handler
        error_file_handler = logging.handlers.RotatingFileHandler(
            self.log_dir / f"{self.app_name}_errors.log",
            maxBytes=10*1024*1024,  # 10MB
            backupCount=10
        )
        error_file_handler.setFormatter(StructuredFormatter())
        error_file_handler.setLevel(logging.ERROR)
        
        # 6. Syslog Handler (if available)
        syslog_handler = None
        try:
            # Try to connect to local syslog
            syslog_handler = logging.handlers.SysLogHandler(address='/dev/log')
            syslog_formatter = logging.Formatter(
                f'{self.app_name}: %(name)s - %(levelname)s - %(message)s'
            )
            syslog_handler.setFormatter(syslog_formatter)
            syslog_handler.setLevel(logging.WARNING)
        except Exception:
            # If syslog not available, use file alternative
            syslog_handler = logging.handlers.RotatingFileHandler(
                self.log_dir / f"{self.app_name}_syslog.log",
                maxBytes=5*1024*1024,  # 5MB
                backupCount=3
            )
            syslog_handler.setFormatter(StructuredFormatter())
            syslog_handler.setLevel(logging.WARNING)
        
        # Setup specific loggers
        self._setup_logger(f"{self.app_name}.app", [stdout_handler, stderr_handler, app_file_handler, syslog_handler])
        self._setup_logger(f"{self.app_name}.api", [stdout_handler, stderr_handler, api_file_handler])
        self._setup_logger(f"{self.app_name}.model", [stdout_handler, stderr_handler, app_file_handler])
        self._setup_logger(f"{self.app_name}.system", [stdout_handler, stderr_handler, app_file_handler, syslog_handler])
        self._setup_logger(f"{self.app_name}.error", [stderr_handler, error_file_handler, syslog_handler])
        
        # Setup root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.DEBUG)
        if not root_logger.handlers:
            root_logger.addHandler(stdout_handler)
            root_logger.addHandler(stderr_handler)
    
    def _setup_logger(self, name: str, handlers: list):
        """Setup individual logger with specified handlers."""
        logger = logging.getLogger(name)
        logger.setLevel(self.log_level)
        logger.propagate = False
        
        # Clear existing handlers
        logger.handlers.clear()
        
        # Add new handlers
        for handler in handlers:
            if handler is not None:
                logger.addHandler(handler)
    
    def log_api_request(self, method: str, endpoint: str, status_code: int, 
                       latency: float, user_agent: str = None, ip: str = None):
        """Log API request with structured data."""
        extra = {
            'endpoint': endpoint,
            'status_code': status_code,
            'latency': latency
        }
        
        if user_agent:
            extra['user_agent'] = user_agent
        if ip:
            extra['client_ip'] = ip
        
        message = f"{method} {endpoint} - {status_code} - {latency:.3f}s"
        self.api_logger.info(message, extra=extra)
    
    def log_prediction(self, predicted_class: str, confidence: float, 
                      inference_time: float, device: str):
        """Log model prediction with metrics."""
        extra = {
            'predicted_class': predicted_class,
            'confidence': confidence,
            'inference_time': inference_time,
            'device': device
        }
        
        message = f"Prediction: {predicted_class} (confidence: {confidence:.3f}, time: {inference_time:.3f}s, device: {device})"
        self.model_logger.info(message, extra=extra)
    
    def log_system_event(self, event_type: str, message: str, **kwargs):
        """Log system events (startup, shutdown, errors)."""
        extra = {'event_type': event_type}
        extra.update(kwargs)
        
        self.system_logger.info(f"[{event_type}] {message}", extra=extra)
    
    def log_error(self, error: Exception, context: Dict[str, Any] = None):
        """Log errors with full context and traceback."""
        extra = {'error_type': type(error).__name__}
        if context:
            extra.update(context)
        
        self.error_logger.error(f"Error: {str(error)}", exc_info=True, extra=extra)
    
    def log_security_event(self, event_type: str, details: Dict[str, Any]):
        """Log security-related events."""
        extra = {'security_event': event_type}
        extra.update(details)
        
        message = f"Security Event: {event_type}"
        self.system_logger.warning(message, extra=extra)
    
    def get_log_stats(self) -> Dict[str, Any]:
        """Get logging statistics."""
        stats = {
            'log_directory': str(self.log_dir),
            'log_files': [],
            'total_size_mb': 0
        }
        
        for log_file in self.log_dir.glob("*.log"):
            size_mb = log_file.stat().st_size / (1024 * 1024)
            stats['log_files'].append({
                'name': log_file.name,
                'size_mb': round(size_mb, 2),
                'modified': datetime.fromtimestamp(log_file.stat().st_mtime).isoformat()
            })
            stats['total_size_mb'] += size_mb
        
        stats['total_size_mb'] = round(stats['total_size_mb'], 2)
        return stats


# Global logging service instance
_logging_service = None

def get_logging_service() -> LoggingService:
    """Get global logging service instance."""
    global _logging_service
    if _logging_service is None:
        _logging_service = LoggingService()
    return _logging_service

def setup_logging(log_dir: str = "logs", app_name: str = "cifar10-api", log_level: str = "INFO"):
    """Setup global logging service."""
    global _logging_service
    _logging_service = LoggingService(log_dir, app_name, log_level)
    return _logging_service
