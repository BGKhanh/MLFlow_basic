#!/bin/bash

# CIFAR-10 API Server Start Script

echo "Starting CIFAR-10 FastAPI Server..."

# Check if model exists
if [ ! -d "checkpoints/models" ] || [ -z "$(ls -A checkpoints/models 2>/dev/null)" ]; then
    echo "Warning: No trained models found in checkpoints/models/"
    echo "Please train a model first using: python main.py"
    exit 1
fi

# Set environment variables
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Start the API server
python api_server.py