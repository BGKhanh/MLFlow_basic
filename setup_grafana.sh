#!/bin/bash

echo "Setting up Grafana dashboard directories..."

# Create Grafana configuration directories
mkdir -p monitoring/grafana/provisioning/datasources
mkdir -p monitoring/grafana/provisioning/dashboards
mkdir -p monitoring/grafana/dashboards

echo "Grafana setup completed!"
echo ""
echo "To start the monitoring stack:"
echo "1. Run: docker-compose up -d"
echo "2. Wait for services to start (30-60 seconds)"
echo "3. Access Grafana at: http://localhost:3000"
echo "4. Login with admin/admin"
echo "5. Access Prometheus at: http://localhost:9090"
echo "6. Access Alertmanager at: http://localhost:9093"
echo ""
echo "The dashboards will be automatically provisioned!" 