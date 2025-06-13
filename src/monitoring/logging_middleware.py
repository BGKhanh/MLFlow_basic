"""
Enhanced middleware for comprehensive request logging.
"""

import time
import uuid
from fastapi import Request, Response
from fastapi.middleware.base import BaseHTTPMiddleware
from typing import Callable
import logging

from .logging_config import get_logging_service


class LoggingMiddleware(BaseHTTPMiddleware):
    """Enhanced middleware for comprehensive request/response logging."""
    
    def __init__(self, app, include_body: bool = False):
        super().__init__(app)
        self.include_body = include_body
        self.logging_service = get_logging_service()
        self.logger = logging.getLogger("cifar10-api.middleware")
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Generate unique request ID
        request_id = str(uuid.uuid4())
        
        # Extract client info
        client_ip = self._get_client_ip(request)
        user_agent = request.headers.get("user-agent", "unknown")
        
        # Start timing
        start_time = time.time()
        
        # Log request start
        self.logger.info(
            f"Request started: {request.method} {request.url.path}",
            extra={
                'request_id': request_id,
                'client_ip': client_ip,
                'user_agent': user_agent,
                'endpoint': request.url.path,
                'method': request.method
            }
        )
        
        # Store request info for potential error logging
        request.state.request_id = request_id
        request.state.start_time = start_time
        request.state.client_ip = client_ip
        
        response = None
        error_occurred = False
        
        try:
            # Process request
            response = await call_next(request)
            
        except Exception as e:
            error_occurred = True
            latency = time.time() - start_time
            
            # Log error with full context
            self.logging_service.log_error(e, {
                'request_id': request_id,
                'endpoint': request.url.path,
                'method': request.method,
                'client_ip': client_ip,
                'user_agent': user_agent,
                'latency': latency
            })
            
            # Re-raise to let FastAPI handle it
            raise
        
        finally:
            latency = time.time() - start_time
            
            if response is not None:
                # Log successful request
                self.logging_service.log_api_request(
                    method=request.method,
                    endpoint=request.url.path,
                    status_code=response.status_code,
                    latency=latency,
                    user_agent=user_agent,
                    ip=client_ip
                )
                
                # Add request ID to response headers
                response.headers["X-Request-ID"] = request_id
                
                # Log completion
                self.logger.info(
                    f"Request completed: {request.method} {request.url.path} - {response.status_code} - {latency:.3f}s",
                    extra={
                        'request_id': request_id,
                        'status_code': response.status_code,
                        'latency': latency
                    }
                )
            elif error_occurred:
                # Log error completion
                self.logger.error(
                    f"Request failed: {request.method} {request.url.path} - {latency:.3f}s",
                    extra={
                        'request_id': request_id,
                        'latency': latency
                    }
                )
        
        return response
    
    def _get_client_ip(self, request: Request) -> str:
        """Extract client IP from request."""
        # Check for forwarded headers
        x_forwarded_for = request.headers.get("X-Forwarded-For")
        if x_forwarded_for:
            return x_forwarded_for.split(",")[0].strip()
        
        x_real_ip = request.headers.get("X-Real-IP")
        if x_real_ip:
            return x_real_ip
        
        # Fallback to direct client
        if request.client:
            return request.client.host
        
        return "unknown" 