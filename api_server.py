#!/usr/bin/env python3
"""
CIFAR-10 FastAPI REST API Server
Minimal REST API for model inference.
"""

import os
import io
import torch
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from typing import Dict, Any, List
import logging

from src.utils import load_config
from src.utils.model_loader import load_model, get_latest_model, list_models
from src.data import create_transforms

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables
app = FastAPI(title="CIFAR-10 Model API", version="1.0.0")
model = None
device = None
class_names = [
    'airplane', 'automobile', 'bird', 'cat', 'deer', 
    'dog', 'frog', 'horse', 'ship', 'truck'
]
transform = None
config = None


def initialize_api():
    """Initialize model and configurations."""
    global model, device, transform, config
    
    try:
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
        
        # Create transform using config (consistent with training)
        dataset_name = cfg['dataset'].get('name', 'cifar10')
        transform = create_transforms(cfg, dataset_name, is_training=False)
        
        logger.info("API initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize API: {e}")
        raise


@app.on_event("startup")
async def startup_event():
    """Startup event handler."""
    initialize_api()


@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "CIFAR-10 Model API", "status": "running"}


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    global model
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "healthy", "model_loaded": True}


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
    Predict image class.
    
    Args:
        file: Uploaded image file
        
    Returns:
        Prediction results
    """
    global model, device, transform, class_names
    
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
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
        model.eval()
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            predicted_class = torch.argmax(outputs, dim=1).item()
            confidence = probabilities[0, predicted_class].item()
        
        # Prepare response
        result = {
            'predicted_class': predicted_class,
            'predicted_label': class_names[predicted_class],
            'confidence': round(confidence, 4),
            'all_probabilities': {
                class_names[i]: round(prob, 4) 
                for i, prob in enumerate(probabilities[0].cpu().numpy())
            }
        }
        
        return JSONResponse(content=result)
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.get("/classes")
async def get_classes():
    """Get available class names."""
    return {"classes": class_names}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")