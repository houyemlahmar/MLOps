#!/bin/bash

# Airflow Setup Script
# This script initializes the Airflow database and creates an admin user

echo "🚀 Setting up Apache Airflow..."

# Wait for PostgreSQL to be ready
echo "⏳ Waiting for PostgreSQL to be ready..."
sleep 10

# Initialize Airflow database
echo "📦 Initializing Airflow database..."
docker exec airflow-webserver airflow db init

# Create admin user
echo "👤 Creating admin user..."
docker exec airflow-webserver airflow users create \
    --username admin \
    --password admin \
    --firstname Admin \
    --lastname User \
    --role Admin \
    --email admin@example.com

# Install Python dependencies in Airflow containers
echo "📦 Installing Python dependencies in Airflow containers..."
docker exec airflow-webserver pip install -r /opt/airflow/project/airflow/requirements.txt
docker exec airflow-scheduler pip install -r /opt/airflow/project/airflow/requirements.txt

echo "✅ Airflow setup complete!"
echo ""
echo "🌐 Access Airflow Web UI at: http://localhost:8080"
echo "   Username: admin"
echo "   Password: admin"
echo ""
echo "📊 Available Services:"
echo "   - Airflow UI:      http://localhost:8080"
echo "   - MLflow UI:       http://localhost:5050"
echo "   - Prometheus:      http://localhost:9090"
echo "   - Grafana:         http://localhost:3000"
echo "   - Prediction API:  http://localhost:5002"
