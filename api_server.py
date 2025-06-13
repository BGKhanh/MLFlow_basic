#!/usr/bin/env python3
"""
CIFAR-10 FastAPI REST API Server
Minimal REST API for model inference with monitoring and comprehensive logging.
"""

import os
import io
import time
import torch
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException, Request, Response
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from typing import Dict, Any, List
import logging
from contextlib import asynccontextmanager
from fastapi.middleware.base import BaseHTTPMiddleware

# Monitoring imports
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import make_asgi_app

from src.utils import load_config
from src.utils.model_loader import load_model, get_latest_model, list_models
from src.data import create_transforms
from src.monitoring import ModelMetrics, SystemMetrics, MetricsCollector
from src.monitoring.logging_config import setup_logging, get_logging_service
from src.monitoring.logging_middleware import LoggingMiddleware
from src.monitoring.alert_manager import AlertManager

# Setup comprehensive logging
logging_service = setup_logging(log_dir="logs", app_name="cifar10-api", log_level="INFO")
logger = logging.getLogger("cifar10-api.app")

# Global variables
model = None
device = None
class_names = [
    'airplane', 'automobile', 'bird', 'cat', 'deer', 
    'dog', 'frog', 'horse', 'ship', 'truck'
]
transform = None
config = None
metrics_collector = None

def initialize_api():
    """Initialize model and configurations."""
    global model, device, transform, config, metrics_collector
    
    try:
        logging_service.log_system_event("STARTUP", "Initializing API components")
        
        # Initialize metrics collector
        metrics_collector = MetricsCollector()
        
        # Load config
        config = load_config('config/config.yaml')
        cfg = config.get_config()
        
        # Set device
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"Using device: {device}")
        
        # Load latest model
        checkpoint_path = get_latest_model()
        if checkpoint_path is None:
            raise RuntimeError("No trained model found. Please train a model first.")
        
        model = load_model(checkpoint_path, device)
        logger.info(f"Model loaded from: {checkpoint_path}")
        logging_service.log_system_event("MODEL_LOADED", f"Model loaded successfully from {checkpoint_path}")
        
        # Set model info in metrics
        model_name = cfg['model'].get('name', 'unknown')
        num_classes = cfg['dataset'].get('num_classes', 10)
        metrics_collector.set_model_info(model_name, num_classes, device)
        
        # Create transform using config (consistent with training)
        dataset_name = cfg['dataset'].get('name', 'cifar10')
        transform = create_transforms(cfg, dataset_name, is_training=False)
        
        logger.info("API initialized successfully")
        logging_service.log_system_event("STARTUP_COMPLETE", "API initialization completed successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize API: {e}")
        logging_service.log_error(e, {"component": "api_initialization"})
        raise


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown events."""
    # Startup
    logging_service.log_system_event("APP_STARTUP", "Application starting up")
    os.makedirs("templates", exist_ok=True)
    initialize_api()
    yield
    # Shutdown
    logging_service.log_system_event("APP_SHUTDOWN", "Application shutting down")
    logger.info("Shutting down API")


class MetricsMiddleware(BaseHTTPMiddleware):
    """Middleware to track API metrics."""
    
    def __init__(self, app, metrics_collector):
        super().__init__(app)
        self.metrics_collector = metrics_collector
    
    async def dispatch(self, request: Request, call_next):
        # Start timing
        start_time = time.time()
        
        # Process request
        response = await call_next(request)
        
        # Calculate latency
        latency = time.time() - start_time
        
        # Record metrics
        if self.metrics_collector:
            self.metrics_collector.record_api_request(
                method=request.method,
                endpoint=request.url.path,
                status_code=response.status_code,
                latency=latency
            )
        
        return response


# Initialize FastAPI with lifespan
app = FastAPI(title="CIFAR-10 Model API", version="1.0.0", lifespan=lifespan)

# Initialize FastAPI metrics instrumentator
instrumentator = Instrumentator(
    should_group_status_codes=False,
    should_ignore_untemplated=True,
    should_respect_env_var=True,
    should_instrument_requests_inprogress=True,
    excluded_handlers=["/metrics"],
    env_var_name="ENABLE_METRICS",
    inprogress_name="inprogress",
    inprogress_labels=True,
)

instrumentator.instrument(app).expose(app)

# Templates for HTML pages
templates = Jinja2Templates(directory="templates")

# Add middleware in correct order
app.add_middleware(LoggingMiddleware)  # Logging first
app.add_middleware(MetricsMiddleware, metrics_collector=metrics_collector)  # Metrics second

# Initialize alert manager
alert_manager = AlertManager()

@app.get("/", response_class=HTMLResponse)
async def web_interface(request: Request):
    """Web interface for image upload and prediction."""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>CIFAR-10 Model Prediction</title>
        <style>
            body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
            .container { text-align: center; }
            .upload-area { border: 2px dashed #ccc; padding: 40px; margin: 20px 0; }
            .upload-area:hover { border-color: #007bff; }
            .result { margin: 20px 0; padding: 20px; background-color: #f8f9fa; border-radius: 5px; }
            .error { color: red; }
            .success { color: green; }
            .confidence-bar { width: 100%; background-color: #e0e0e0; height: 20px; border-radius: 10px; margin: 5px 0; }
            .confidence-fill { height: 100%; background-color: #007bff; border-radius: 10px; transition: width 0.3s; }
            .class-prob { display: flex; justify-content: space-between; align-items: center; margin: 5px 0; }
            img { max-width: 300px; max-height: 300px; margin: 10px; border: 1px solid #ddd; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>CIFAR-10 Image Classification</h1>
            <p>Upload an image to get predictions from our MobileNetV3 model</p>
            
            <div class="upload-area">
                <form id="uploadForm" enctype="multipart/form-data">
                    <input type="file" id="imageInput" name="file" accept="image/*" required>
                    <br><br>
                    <button type="submit">Predict Image</button>
                </form>
            </div>
            
            <div id="preview"></div>
            <div id="result"></div>
            
            <div>
                <h3>Available Classes:</h3>
                <p>airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck</p>
            </div>
        </div>

        <script>
            document.getElementById('imageInput').addEventListener('change', function(e) {
                const file = e.target.files[0];
                if (file) {
                    const reader = new FileReader();
                    reader.onload = function(e) {
                        document.getElementById('preview').innerHTML = 
                            '<h3>Preview:</h3><img src="' + e.target.result + '" alt="Preview">';
                    };
                    reader.readAsDataURL(file);
                }
            });

            document.getElementById('uploadForm').addEventListener('submit', async function(e) {
                e.preventDefault();
                
                const formData = new FormData();
                const fileInput = document.getElementById('imageInput');
                const file = fileInput.files[0];
                
                if (!file) {
                    document.getElementById('result').innerHTML = '<div class="error">Please select an image file.</div>';
                    return;
                }
                
                formData.append('file', file);
                
                try {
                    document.getElementById('result').innerHTML = '<div>Predicting... Please wait.</div>';
                    
                    const response = await fetch('/predict', {
                        method: 'POST',
                        body: formData
                    });
                    
                    const data = await response.json();
                    
                    if (response.ok) {
                        let resultHtml = '<div class="result success">';
                        resultHtml += '<h3>Prediction Results:</h3>';
                        resultHtml += '<p><strong>Predicted Class:</strong> ' + data.predicted_label + '</p>';
                        resultHtml += '<p><strong>Confidence:</strong> ' + (data.confidence * 100).toFixed(2) + '%</p>';
                        resultHtml += '<div class="confidence-bar"><div class="confidence-fill" style="width: ' + (data.confidence * 100) + '%"></div></div>';
                        resultHtml += '<p><strong>Inference Time:</strong> ' + data.inference_time_seconds + 's</p>';
                        
                        resultHtml += '<h4>All Class Probabilities:</h4>';
                        for (const [className, prob] of Object.entries(data.all_probabilities)) {
                            const percentage = (prob * 100).toFixed(2);
                            resultHtml += '<div class="class-prob">';
                            resultHtml += '<span>' + className + ':</span>';
                            resultHtml += '<span>' + percentage + '%</span>';
                            resultHtml += '</div>';
                        }
                        
                        resultHtml += '</div>';
                        document.getElementById('result').innerHTML = resultHtml;
                    } else {
                        document.getElementById('result').innerHTML = '<div class="error">Error: ' + data.detail + '</div>';
                    }
                } catch (error) {
                    document.getElementById('result').innerHTML = '<div class="error">Error: ' + error.message + '</div>';
                }
            });
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content, status_code=200)


@app.get("/api")
async def root():
    """API root endpoint."""
    return {"message": "CIFAR-10 Model API", "status": "running", "version": "1.0.0"}


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    global model, metrics_collector
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Update system metrics on health check
    if metrics_collector:
        metrics_collector.system_metrics.update_system_metrics()
    
    return {"status": "healthy", "model_loaded": True, "device": device}


@app.get("/models")
async def get_available_models():
    """Get list of available models."""
    try:
        models_info = list_models()
        return {"models": models_info}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list models: {str(e)}")


@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    """
    Predict image class with enhanced logging.
    """
    global model, device, transform, class_names, metrics_collector
    
    if model is None:
        error_msg = "Model not loaded"
        logging_service.log_error(RuntimeError(error_msg), {"endpoint": "/predict"})
        raise HTTPException(status_code=503, detail=error_msg)
    
    # Start timing
    start_time = time.time()
    
    try:
        # Read and validate image
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data))
        
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Apply transforms (consistent with training pipeline)
        image_array = np.array(image)
        transformed = transform(image=image_array)
        image_tensor = transformed['image'].unsqueeze(0).to(device)
        
        # Run inference
        inference_start = time.time()
        model.eval()
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            predicted_class = torch.argmax(outputs, dim=1).item()
            confidence = probabilities[0, predicted_class].item()
        
        inference_time = time.time() - inference_start
        total_time = time.time() - start_time
        
        # Record metrics
        predicted_label = class_names[predicted_class]
        if metrics_collector:
            metrics_collector.record_prediction(
                inference_time=inference_time,
                confidence=confidence,
                predicted_class=predicted_label
            )
        
        # Log prediction with detailed info
        logging_service.log_prediction(
            predicted_class=predicted_label,
            confidence=confidence,
            inference_time=inference_time,
            device=device
        )
        
        # Fix JSON serialization: Convert numpy float32 to Python float
        probabilities_numpy = probabilities[0].cpu().numpy()
        all_probabilities = {
            class_names[i]: float(prob)
            for i, prob in enumerate(probabilities_numpy)
        }
        
        # Prepare response
        result = {
            'predicted_class': predicted_class,
            'predicted_label': predicted_label,
            'confidence': float(confidence),
            'inference_time_seconds': round(float(inference_time), 4),
            'total_time_seconds': round(float(total_time), 4),
            'all_probabilities': {
                class_name: round(prob, 4)
                for class_name, prob in all_probabilities.items()
            }
        }
        
        logger.info(f"Prediction: {predicted_label} (confidence: {confidence:.3f}, time: {inference_time:.3f}s)")
        
        return JSONResponse(content=result)
        
    except Exception as e:
        logging_service.log_error(e, {
            "endpoint": "/predict",
            "file_name": file.filename if file else "unknown",
            "file_size": len(image_data) if 'image_data' in locals() else 0
        })
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.get("/classes")
async def get_classes():
    """Get available class names."""
    return {"classes": class_names}


@app.get("/metrics-info")
async def get_metrics_info():
    """Get information about available metrics."""
    return {
        "metrics_endpoint": "/metrics",
        "description": "Prometheus metrics for monitoring",
        "custom_metrics": [
            "model_inference_duration_seconds",
            "model_confidence_score", 
            "predictions_total",
            "current_model_confidence",
            "system_cpu_usage_percent",
            "system_memory_usage_percent",
            "system_disk_usage_percent"
        ]
    }


# Add Prometheus metrics endpoint
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)


# Add new endpoint for log statistics
@app.get("/logs/stats")
async def get_log_stats():
    """Get logging statistics."""
    try:
        stats = logging_service.get_log_stats()
        return JSONResponse(content=stats)
    except Exception as e:
        logging_service.log_error(e, {"endpoint": "/logs/stats"})
        raise HTTPException(status_code=500, detail=f"Failed to get log stats: {str(e)}")


@app.get("/alerts/config")
async def get_alert_config():
    """Get current alert configuration."""
    try:
        config = alert_manager.get_current_config()
        return JSONResponse(content=config)
    except Exception as e:
        logging_service.log_error(e, {"endpoint": "/alerts/config"})
        raise HTTPException(status_code=500, detail=f"Failed to get alert config: {str(e)}")

@app.post("/alerts/recipients")
async def update_alert_recipients(recipients: List[str], severity: str = "both"):
    """Update alert recipients."""
    try:
        success = alert_manager.update_alert_recipients(recipients, severity)
        if success:
            return {"message": f"Updated {severity} alert recipients", "recipients": recipients}
        else:
            raise HTTPException(status_code=500, detail="Failed to update recipients")
    except Exception as e:
        logging_service.log_error(e, {"endpoint": "/alerts/recipients"})
        raise HTTPException(status_code=500, detail=f"Failed to update recipients: {str(e)}")

@app.post("/alerts/thresholds")
async def update_alert_thresholds(thresholds: Dict[str, float]):
    """Update alert thresholds."""
    try:
        success = alert_manager.update_alert_thresholds(thresholds)
        if success:
            return {"message": "Updated alert thresholds", "thresholds": thresholds}
        else:
            raise HTTPException(status_code=500, detail="Failed to update thresholds")
    except Exception as e:
        logging_service.log_error(e, {"endpoint": "/alerts/thresholds"})
        raise HTTPException(status_code=500, detail=f"Failed to update thresholds: {str(e)}")

@app.get("/alerts/test")
async def test_alert_config():
    """Test alert configuration."""
    try:
        test_results = alert_manager.test_alert_config()
        return JSONResponse(content=test_results)
    except Exception as e:
        logging_service.log_error(e, {"endpoint": "/alerts/test"})
        raise HTTPException(status_code=500, detail=f"Failed to test alert config: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")