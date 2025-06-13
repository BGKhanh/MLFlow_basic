# CIFAR-10 ML Pipeline with Complete Monitoring Stack

This repository contains a production-ready machine learning pipeline for CIFAR-10 image classification, featuring comprehensive monitoring, logging, and alerting capabilities. The pipeline includes training, inference, and a complete observability stack with Prometheus, Grafana, and Alertmanager.

## Key Features

### Core ML Pipeline
- **End-to-end Training Pipeline**: Automated workflow from data preprocessing to model evaluation
- **FastAPI REST API**: Production-ready API server for model inference
- **Experiment Tracking**: Comprehensive tracking with MLflow
- **Hyperparameter Optimization**: Automated tuning with Ray Tune
- **Distributed Training**: Parallel processing using Ray
- **Multiple Models**: Support for MobileNet, EfficientNet, DenseNet, and RegNet

### Complete Monitoring Stack
- **Prometheus Metrics**: System, API, and model performance monitoring
- **Grafana Dashboards**: Visual monitoring with real-time charts
- **Alertmanager**: Automated alerts via email for critical events
- **Comprehensive Logging**: Structured logging to files, stdout/stderr, and syslog
- **Docker Deployment**: One-command deployment with monitoring stack

## Technologies Used

- **PyTorch & torchvision**: Model development and training
- **FastAPI**: REST API server with automatic documentation
- **Prometheus**: Metrics collection and time-series database
- **Grafana**: Monitoring dashboards and visualization
- **Alertmanager**: Alert routing and notification management
- **Docker & Docker Compose**: Containerization and orchestration
- **MLflow**: Experiment tracking and model management
- **Ray & Ray Tune**: Distributed training and hyperparameter optimization

## Project Structure

```
├── config/ # Configuration files
│ └── config.yaml # Main training configuration
├── src/ # Source code
│ ├── data/ # Data processing scripts
│ ├── models/ # Model definitions
│ ├── training/ # Training scripts
│ ├── monitoring/ # Monitoring and metrics
│ │ ├── metrics.py # Custom metrics collection
│ │ ├── logging_config.py # Comprehensive logging setup
│ │ ├── logging_middleware.py # Request logging middleware
│ │ └── alert_manager.py # Alert configuration management
│ └── utils/ # Utility functions
├── monitoring/ # Monitoring stack configuration
│ ├── prometheus/ # Prometheus configuration
│ │ ├── prometheus.yml # Prometheus server config
│ │ └── alert_rules.yml # Alert rules definition
│ ├── alertmanager/ # Alertmanager configuration
│ │ └── alertmanager.yml # Alert routing and email setup
│ └── grafana/ # Grafana configuration
│ ├── provisioning/ # Auto-provisioning config
│ └── dashboards/ # Pre-built dashboards
├── tests/ # Test scripts
│ ├── test_metrics.py # Metrics functionality testing
│ └── test_monitoring_stack.py # Monitoring stack testing
├── logs/ # Application logs (auto-created)
│ ├── cifar10-api_app.log          # Application logs (JSON format)
│ ├── cifar10-api_api_access.log   # API access logs
│ ├── cifar10-api_errors.log       # Error logs with stack traces
│ └── cifar10-api_syslog.log       # System-level events
├── checkpoints/ # Model checkpoints
├── main.py # Training entry point
├── inference.py # Standalone inference
├── api_server.py # FastAPI server with monitoring
├── start_api.sh # API startup script
├── docker-compose.yml # Complete stack deployment
├── Dockerfile # API container definition
└── requirements.txt # Project dependencies
```


## Quick Start

### Prerequisites

- **Python 3.11+** (for local development)
- **Docker & Docker Compose** (for full stack deployment)
- **8GB+ RAM** recommended for full monitoring stack
- **CUDA GPU** (optional, for faster training)

### Option 1: Complete Monitoring Stack (Recommended)

Deploy the entire stack including API, Prometheus, Grafana, and Alertmanager:

```bash
# 1. Clone repository
git clone https://github.com/BGKhanh/MLFlow_basic.git
cd MLFlow_basic

# 2. Deploy complete stack
docker-compose up -d

# 3. Train a model (if no checkpoints exist)
docker-compose exec cifar-api python main.py

# 4. Access services
echo "API Documentation: http://localhost:8000/docs"
echo "Grafana Dashboard: http://localhost:3000 (admin/admin)"
echo "Prometheus Metrics: http://localhost:9090"
echo "Alertmanager: http://localhost:9093"
echo "MLflow UI: http://localhost:5000"
```

### Option 2: API Only (Lightweight)

Deploy just the API server without monitoring:

```bash
# 1. Clone and setup
git clone https://github.com/BGKhanh/MLFlow_basic.git
cd MLFlow_basic

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Train a model (required)
python main.py

# 5. Start API server
chmod +x start_api.sh
./start_api.sh
```

## Monitoring Stack Usage

### Accessing Monitoring Services

| Service | URL | Credentials | Purpose |
|---------|-----|-------------|---------|
| **API Server** | http://localhost:8000 | None | Main API and docs |
| **Grafana** | http://localhost:3000 | admin/admin | Visual dashboards |
| **Prometheus** | http://localhost:9090 | None | Metrics and alerts |
| **Alertmanager** | http://localhost:9093 | None | Alert management |
| **MLflow** | http://localhost:5000 | None | Experiment tracking |

### Pre-built Dashboards

#### System Overview Dashboard
- **CPU, Memory, Disk Usage**: Real-time system resource monitoring
- **GPU Utilization**: GPU usage and memory (if available)
- **Network & Disk I/O**: Data transfer and disk operation metrics
- **API Performance**: Request rate, error rate, response times
- **Active Alerts**: Current firing alerts count

#### Model Performance Dashboard  
- **Model Confidence**: Current and average confidence scores
- **Prediction Count**: Total predictions processed
- **Inference Time**: Performance metrics for model inference

### Alert Configuration

The system includes pre-configured alerts with email notifications:

#### Critical Alerts (Immediate notification)
- **API Error Rate > 50%**: High error rate in API responses
- **Disk Space < 50%**: Low disk space warning
- **API Service Down**: Service unavailability

#### Warning Alerts (Grouped notifications)
- **CPU Usage > 80%**: High CPU utilization
- **Memory Usage > 70%**: High memory usage
- **Response Time > 2s**: Slow API response times
- **Model Confidence < 0.6**: Low model confidence scores

### Customizing Alerts

Update alert recipients and thresholds via API:

```bash
# Update alert recipients
curl -X POST "http://localhost:8000/alerts/recipients" \
     -H "Content-Type: application/json" \
     -d '["your-email@example.com", "team@example.com"]'

# Update alert thresholds
curl -X POST "http://localhost:8000/alerts/thresholds" \
     -H "Content-Type: application/json" \
     -d '{
       "cpu_threshold": 85,
       "memory_threshold": 75,
       "error_rate_threshold": 30
     }'

# Test alert configuration
curl http://localhost:8000/alerts/test
```

## API Usage

### Core Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Web interface for image upload |
| `/docs` | GET | Interactive API documentation |
| `/health` | GET | Health check with system metrics |
| `/predict` | POST | Image classification prediction |
| `/classes` | GET | Available CIFAR-10 class names |
| `/models` | GET | List available trained models |

### Monitoring Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/metrics` | GET | Prometheus metrics for monitoring |
| `/metrics-info` | GET | Information about available metrics |
| `/logs/stats` | GET | Logging system statistics |
| `/alerts/config` | GET | Current alert configuration |
| `/alerts/recipients` | POST | Update alert email recipients |
| `/alerts/thresholds` | POST | Update alert thresholds |

### Making Predictions

#### Web Interface
Visit http://localhost:8000 for a user-friendly web interface to upload images and see predictions.

#### API Calls
```bash
# Health check with system metrics
curl http://localhost:8000/health

# Make prediction
curl -X POST "http://localhost:8000/predict" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@your_image.jpg"

# Get current metrics
curl http://localhost:8000/metrics
```

#### Python Example
```python
import requests

# Make prediction
with open("image.jpg", "rb") as f:
    files = {"file": ("image.jpg", f, "image/jpeg")}
    response = requests.post("http://localhost:8000/predict", files=files)
    result = response.json()
    
print(f"Predicted: {result['predicted_label']}")
print(f"Confidence: {result['confidence']:.3f}")
print(f"Inference time: {result['inference_time_seconds']:.3f}s")
```

## Training Pipeline

### Basic Training
```bash
# Train with default settings
python main.py

# Or in Docker
docker-compose exec cifar-api python main.py
```

### Advanced Training Options
```bash
# Hyperparameter tuning
python main.py --tune

# Distributed training
python main.py --distributed

# Combined tuning and distributed training
python main.py --tune --distributed
```

## Configuration

### Monitoring Configuration

#### Prometheus Configuration (`monitoring/prometheus/prometheus.yml`)
```yaml
global:
  scrape_interval: 15s     # Data collection frequency
  evaluation_interval: 15s # Alert evaluation frequency

scrape_configs:
  - job_name: 'cifar10-api'
    static_configs:
      - targets: ['cifar-api:8000']
    scrape_interval: 5s     # API metrics every 5 seconds
```

#### Alert Rules (`monitoring/prometheus/alert_rules.yml`)
```yaml
- alert: HighAPIErrorRate
  expr: api_error_rate > 50
  for: 2m
  labels:
    severity: critical
  annotations:
    summary: "High API Error Rate Detected"
    description: "API error rate is {{ $value }}%"
```

#### Grafana Dashboard Refresh
- **Auto-refresh**: 5 seconds
- **Time range**: Last 7 days
- **Data retention**: 7 days

### Training Configuration (`config/config.yaml`)
```yaml
model:
  name: "mobilenet"
  transfer_mode: "classifier_only"

training:
  batch_size: 128
  num_epochs: 10
  learning_rate: 0.0001

dataset:
  name: "cifar10"
  data_dir: "data/"
```

## Logging System

### Log Structure

```
├── cifar10-api_app.log          # Application logs (JSON format)
├── cifar10-api_api_access.log   # API access logs
├── cifar10-api_errors.log       # Error logs with stack traces
└── cifar10-api_syslog.log       # System-level events
```

### Log Features
- **Structured JSON logging** for easy parsing
- **Automatic log rotation** (10MB per file, 5 backups)
- **Multiple output streams**: stdout, stderr, files, syslog
- **Request tracking** with unique request IDs
- **Performance metrics** embedded in logs

### Viewing Logs
```bash
# Real-time application logs
docker-compose logs -f cifar-api

# View specific log files
tail -f logs/cifar10-api_app.log

# Search error logs
grep "ERROR" logs/cifar10-api_errors.log

# Get log statistics
curl http://localhost:8000/logs/stats
```

## Testing

### Test Monitoring Stack
```bash
# Test all services
python tests/test_monitoring_stack.py

# Test metrics collection
python tests/test_metrics.py

# Generate test traffic for metrics
python tests/test_metrics.py
# Choose 'y' when prompted
```

### Manual Testing
```bash
# Test API health
curl http://localhost:8000/health

# Verify Prometheus targets
curl http://localhost:9090/api/v1/targets

# Check Grafana health
curl http://localhost:3000/api/health

# Test alerting (triggers test alert)
curl -X POST http://localhost:8000/alerts/test
```

## Performance & Scaling

### Expected Performance
- **API Response Time**: 50-200ms per image (CPU)
- **Throughput**: 20-100 requests/second
- **Memory Usage**: ~2-4GB (full stack)
- **Model Accuracy**: ~85-95% on CIFAR-10

### Scaling Options

#### Horizontal API Scaling
```bash
# Scale API instances
docker-compose up -d --scale cifar-api=3

# Load balancer setup
# Add nginx/traefik for load balancing
```

#### Resource Optimization
```yaml
# In docker-compose.yml
services:
  cifar-api:
    deploy:
      resources:
        limits:
          memory: 2G
          cpus: '1.0'
```

## Troubleshooting

### Common Issues

#### Service Health Check
```bash
# Check all services
docker-compose ps

# View service logs
docker-compose logs [service-name]

# Restart specific service
docker-compose restart [service-name]
```

#### Monitoring Issues
```bash
# Check Prometheus targets
curl http://localhost:9090/api/v1/targets

# Verify metrics availability
curl http://localhost:8000/metrics | head -20

# Test Grafana datasource
curl -u admin:admin http://localhost:3000/api/datasources
```

#### Alert Issues
```bash
# Check alertmanager status
curl http://localhost:9093/api/v1/status

# View active alerts
curl http://localhost:9093/api/v1/alerts

# Test email configuration
curl -X POST http://localhost:8000/alerts/test
```

### Resource Requirements

| Component | CPU | Memory | Disk | Network |
|-----------|-----|--------|------|---------|
| **API Server** | 0.5-1 CPU | 512MB-2GB | 1GB | 1Mbps |
| **Prometheus** | 0.2-0.5 CPU | 512MB-1GB | 5GB | 1Mbps |
| **Grafana** | 0.1-0.3 CPU | 256MB-512MB | 1GB | 1Mbps |
| **Alertmanager** | 0.1 CPU | 128MB-256MB | 1GB | 1Mbps |

### Port Usage
- **8000**: API Server
- **3000**: Grafana Dashboard
- **9090**: Prometheus Server
- **9093**: Alertmanager
- **5000**: MLflow Tracking

## Production Deployment

### Security Considerations
```bash
# Use production compose file
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d

# Set strong passwords
export GF_SECURITY_ADMIN_PASSWORD="your-secure-password"

# Configure HTTPS
# Add SSL certificates and reverse proxy
```

### Backup Strategy
```bash
# Backup model checkpoints
cp -r checkpoints/ backup/checkpoints-$(date +%Y%m%d)/

# Backup Prometheus data
docker run --rm -v cifar_prometheus_data:/data -v $(pwd):/backup alpine \
  tar czf /backup/prometheus-backup-$(date +%Y%m%d).tar.gz -C /data .

# Backup Grafana dashboards
curl -u admin:admin http://localhost:3000/api/search > grafana-dashboards.json
```

## Development

### Development Setup
```bash
# Use development overrides
docker-compose -f docker-compose.yml -f docker-compose.override.yml up -d

# Enable hot reload
# Code changes will automatically restart the API
```

### Adding Custom Metrics
```python
# In src/monitoring/metrics.py
from prometheus_client import Counter

custom_metric = Counter('custom_events_total', 'Custom events')

# Usage in code
custom_metric.inc()
```

### Creating Custom Dashboards
1. Access Grafana: http://localhost:3000
2. Login: admin/admin
3. Create dashboard → Add panel
4. Configure Prometheus queries
5. Export JSON to `monitoring/grafana/dashboards/`

## Contributing

1. Fork the repository
2. Create feature branch: `git checkout -b feature/monitoring-enhancement`
3. Add tests for new monitoring features
4. Update documentation
5. Submit pull request

## License

MIT License - see LICENSE file for details.

## Support

- **Documentation**: Check `/docs` endpoint for API documentation
- **Monitoring**: Use Grafana dashboards for system observability
- **Logs**: Check `logs/` directory for detailed application logs
- **Issues**: Report bugs via GitHub issues with monitoring stack logs