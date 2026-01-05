# 🚀 Apache Airflow Setup Guide

This guide explains how to set up and use Apache Airflow for orchestrating the diabetes prediction ML pipeline.

---

## 📋 Overview

Apache Airflow orchestrates the complete machine learning pipeline with the following stages:

```
Start → Data Validation → Data Processing → Model Training → Evaluation → Deployment → Monitoring → End
```

### Pipeline Tasks

1. **Data Validation**: Verify data quality, schema, and completeness
2. **Data Preprocessing**: Clean and transform raw data
3. **Feature Engineering**: Select and engineer features
4. **Model Training**: Train model with MLflow tracking
5. **Model Evaluation**: Validate model performance against thresholds
6. **Deployment Decision**: Automatically decide whether to deploy based on metrics
7. **Model Registration**: Register approved models in MLflow Model Registry
8. **Drift Monitoring**: Check for data drift using statistical tests
9. **Pipeline Report**: Generate comprehensive execution report

---

## 🐳 Quick Start

### 1. Start All Services (Including Airflow)

```bash
# Start all containers
docker-compose up -d

# Wait for services to be ready (about 60 seconds)
```

### 2. Initialize Airflow

**Windows (PowerShell):**
```powershell
.\scripts\setup-airflow.ps1
```

**Linux/Mac:**
```bash
chmod +x scripts/setup-airflow.sh
./scripts/setup-airflow.sh
```

### 3. Access Airflow UI

Navigate to: **http://localhost:8080**

**Login credentials:**
- Username: `admin`
- Password: `admin`

---

## 📊 Available Services

| Service | URL | Purpose |
|---------|-----|---------|
| **Airflow UI** | http://localhost:8080 | Pipeline orchestration and monitoring |
| **MLflow UI** | http://localhost:5050 | Experiment tracking and model registry |
| **Prometheus** | http://localhost:9090 | Metrics collection |
| **Grafana** | http://localhost:3000 | Visual dashboards |
| **Prediction API** | http://localhost:5002 | Model serving |

---

## 🎯 Using the DAG

### View the Pipeline

1. Open Airflow UI at http://localhost:8080
2. Click on the **`diabetes_ml_pipeline`** DAG
3. View the graph to see all tasks and dependencies

### Run the Pipeline

**Option 1: Manual Trigger**
1. In Airflow UI, go to DAGs list
2. Toggle the DAG to "On" (unpause)
3. Click the "Play" button to trigger manually

**Option 2: Scheduled Runs**
- The DAG runs automatically **daily at 2 AM UTC**
- Adjust schedule in `airflow/dags/diabetes_ml_pipeline.py`:
  ```python
  schedule_interval='0 2 * * *'  # Daily at 2 AM
  ```

### Monitor Execution

1. Click on the DAG run to see task status
2. Green = Success, Red = Failed, Yellow = Running
3. Click any task to view logs
4. Check XCom tab to see data passed between tasks

---

## 📝 DAG Configuration

### Schedule Intervals

Common schedules you can use:

```python
# Every day at 2 AM
schedule_interval='0 2 * * *'

# Every Monday at 2 AM (weekly retraining)
schedule_interval='0 2 * * 1'

# Every hour
schedule_interval='@hourly'

# Manual only (no automatic runs)
schedule_interval=None
```

### Performance Thresholds

The pipeline checks model performance before deployment. Edit thresholds in `diabetes_ml_pipeline.py`:

```python
thresholds = {
    'test_roc_auc': 0.85,  # Minimum ROC-AUC score
}
```

---

## 🔍 Task Groups Explained

### 1. Data Validation
- **validate_data**: Checks data quality, schema, missing values, valid ranges

### 2. Data Processing
- **preprocess_data**: Cleans and encodes data
- **feature_engineering**: Selects optimal features

### 3. Model Training
- **train_model**: Trains model with MLflow tracking

### 4. Model Evaluation
- **evaluate_model**: Validates performance against thresholds

### 5. Deployment
- **decide_deployment**: Branching logic (deploy or skip)
- **register_model**: Registers approved models in MLflow
- **skip_deployment**: Placeholder if model rejected

### 6. Monitoring
- **check_data_drift**: Detects distribution shifts using KS test

### 7. Reporting
- **send_pipeline_report**: Generates JSON report with all metrics

---

## 📧 Pipeline Reports

After each run, a report is saved to:
```
logs/airflow/pipeline_report_YYYYMMDD_HHMMSS.json
```

**Report contents:**
- Pipeline execution timestamp
- Data validation status
- Model metrics (ROC-AUC, etc.)
- Model approval status
- Drift detection results

---

## 🛠️ Troubleshooting

### Airflow UI not accessible

```bash
# Check if containers are running
docker ps

# View Airflow webserver logs
docker logs airflow-webserver

# Restart Airflow services
docker-compose restart airflow-webserver airflow-scheduler
```

### DAG not appearing

```bash
# Check DAG file for syntax errors
docker exec airflow-scheduler airflow dags list

# View scheduler logs
docker logs airflow-scheduler
```

### Task failures

1. Click the failed task in Airflow UI
2. View logs for error details
3. Common issues:
   - Missing dependencies → Run setup-airflow script
   - File permissions → Check volume mounts
   - MLflow connection → Verify mlflow service is running

### Install missing packages

```bash
# Install in both Airflow containers
docker exec airflow-webserver pip install <package-name>
docker exec airflow-scheduler pip install <package-name>
```

---

## 🚀 Advanced Features

### Email Notifications

Configure email alerts in `docker-compose.yml`:

```yaml
environment:
  - AIRFLOW__SMTP__SMTP_HOST=smtp.gmail.com
  - AIRFLOW__SMTP__SMTP_PORT=587
  - AIRFLOW__SMTP__SMTP_USER=your-email@gmail.com
  - AIRFLOW__SMTP__SMTP_PASSWORD=your-app-password
  - AIRFLOW__SMTP__SMTP_MAIL_FROM=your-email@gmail.com
```

### Slack Notifications

Add Slack webhook to DAG:

```python
from airflow.providers.slack.operators.slack_webhook import SlackWebhookOperator

notify_slack = SlackWebhookOperator(
    task_id='notify_slack',
    http_conn_id='slack_webhook',
    message='Pipeline completed successfully! 🎉',
)
```

### Parallelization

Modify `docker-compose.yml` to use CeleryExecutor for parallel task execution:

```yaml
environment:
  - AIRFLOW__CORE__EXECUTOR=CeleryExecutor
```

---

## 🧹 Cleanup

### Stop all services
```bash
docker-compose down
```

### Remove all data (including Airflow DB)
```bash
docker-compose down -v
```

### Reset Airflow
```bash
docker-compose down -v
docker-compose up -d
.\scripts\setup-airflow.ps1  # Windows
./scripts/setup-airflow.sh   # Linux/Mac
```

---

## 📚 Additional Resources

- [Apache Airflow Documentation](https://airflow.apache.org/docs/)
- [Airflow Best Practices](https://airflow.apache.org/docs/apache-airflow/stable/best-practices.html)
- [DAG Writing Best Practices](https://airflow.apache.org/docs/apache-airflow/stable/howto/dag-params.html)

---

**Need Help?** Check the logs or open an issue in the repository! 🚀
