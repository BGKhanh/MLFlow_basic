#!/bin/bash

# CIFAR-10 Docker Deployment Script
set -e

echo "🐳 Starting CIFAR-10 Docker Deployment..."

# Create necessary directories
mkdir -p checkpoints/models data mlruns

# Check if models exist
if [ ! -f "checkpoints/models/checkpoint.pth" ]; then
    echo "⚠️  No model found. You need to train a model first:"
    echo "   python main.py"
    echo ""
    echo "Or place a trained model in checkpoints/models/"
    echo ""
    read -p "Continue deployment anyway? (y/N): " -r
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Build and start services
echo "🔨 Building Docker images..."
docker-compose build

echo "🚀 Starting services..."
docker-compose up -d

# Wait for services to be healthy
echo "⏳ Waiting for services to start..."
sleep 10

# Check service status
echo "📊 Service Status:"
docker-compose ps

# Test API connectivity
echo "🧪 Testing API connectivity..."
if curl -f http://localhost:8000/health >/dev/null 2>&1; then
    echo "✅ API is healthy!"
else
    echo "❌ API health check failed"
fi

echo ""
echo "🎉 Deployment completed!"
echo "📱 API Documentation: http://localhost:8000/docs"
echo "📊 MLflow UI: http://localhost:5000 (if enabled)"
echo ""
echo "💡 Useful commands:"
echo "   docker-compose logs cifar-api    # View API logs"
echo "   docker-compose stop             # Stop services"
echo "   docker-compose down             # Stop and remove containers" 