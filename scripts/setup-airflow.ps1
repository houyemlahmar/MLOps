# Airflow Setup Script for Windows
# This script initializes the Airflow database and creates an admin user

Write-Host "[INFO] Setting up Apache Airflow..." -ForegroundColor Green

# Wait for PostgreSQL to be ready
Write-Host "[WAIT] Waiting for PostgreSQL to be ready..." -ForegroundColor Yellow
Start-Sleep -Seconds 10

# Initialize Airflow database
Write-Host "[INIT] Initializing Airflow database..." -ForegroundColor Yellow
docker exec airflow-webserver airflow db init

# Create admin user
Write-Host "[USER] Creating admin user..." -ForegroundColor Yellow
docker exec airflow-webserver airflow users create `
    --username admin `
    --password admin `
    --firstname Admin `
    --lastname User `
    --role Admin `
    --email admin@example.com

# Install Python dependencies in Airflow containers
Write-Host "[DEPS] Installing Python dependencies in Airflow containers..." -ForegroundColor Yellow
docker exec airflow-webserver pip install -r /opt/airflow/project/airflow/requirements.txt
docker exec airflow-scheduler pip install -r /opt/airflow/project/airflow/requirements.txt

Write-Host ""
Write-Host "[SUCCESS] Airflow setup complete!" -ForegroundColor Green
Write-Host ""
Write-Host "Access Airflow Web UI at: http://localhost:8080" -ForegroundColor Cyan
Write-Host "   Username: admin" -ForegroundColor White
Write-Host "   Password: admin" -ForegroundColor White
Write-Host ""
Write-Host "Available Services:" -ForegroundColor Cyan
Write-Host "   - Airflow UI:      http://localhost:8080" -ForegroundColor White
Write-Host "   - MLflow UI:       http://localhost:5050" -ForegroundColor White
Write-Host "   - Prometheus:      http://localhost:9090" -ForegroundColor White
Write-Host "   - Grafana:         http://localhost:3000" -ForegroundColor White
Write-Host "   - Prediction API:  http://localhost:5002" -ForegroundColor White
