# 🏥 Diabetes Prediction MLOps Project

[![CI Tests](https://github.com/houyemlahmar/MLOps/actions/workflows/ci-tests.yml/badge.svg)](https://github.com/houyemlahmar/MLOps/actions/workflows/ci-tests.yml)
[![Docker Build](https://github.com/houyemlahmar/MLOps/actions/workflows/docker-build-push.yml/badge.svg)](https://github.com/houyemlahmar/MLOps/actions/workflows/docker-build-push.yml)
[![Deploy to GCP](https://github.com/houyemlahmar/MLOps/actions/workflows/deploy-gcp.yml/badge.svg)](https://github.com/houyemlahmar/MLOps/actions/workflows/deploy-gcp.yml)
[![Python 3.13](https://img.shields.io/badge/python-3.13-blue.svg)](https://www.python.org/downloads/)
[![MLflow](https://img.shields.io/badge/MLflow-3.6.0-blue.svg)](https://mlflow.org/)
[![Docker](https://img.shields.io/badge/docker-ready-brightgreen.svg)](https://www.docker.com/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

A complete **MLOps pipeline** for predicting diabetes using machine learning with **experiment tracking**, **data versioning**, **hyperparameter optimization**, **CI/CD automation**, and **GCP deployment**.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Technologies Used](#technologies-used)
- [Setup & Installation](#setup--installation)
- [Usage](#usage)
- [MLOps Pipeline](#mlops-pipeline)
- [Model Performance](#model-performance)
- [Next Steps](#next-steps)
- [Contributing](#contributing)

---

## 🎯 Overview

This project implements a production-ready machine learning pipeline for diabetes prediction. It follows MLOps best practices including:

- **Version-controlled data** using DVC (Data Version Control)
- **Experiment tracking** with MLflow
- **Automated training pipelines** with DVC pipelines
- **Hyperparameter optimization** with Grid Search and Random Search
- **Feature engineering** and selection
- **Model registry** for deployment readiness

**Dataset**: Diabetes prediction dataset with features including age, BMI, HbA1c levels, blood glucose, and medical history.

---

## ✨ Features

### 🔬 **Experimentation & Development**
- ✅ **Exploratory Data Analysis (EDA)**: Comprehensive data exploration and visualization
- ✅ **Data Preprocessing**: Automated encoding, missing value imputation, and data cleaning
- ✅ **Feature Engineering**: Statistical feature selection with correlation analysis
- ✅ **Model Baselines**: Multiple algorithm comparison (Logistic Regression, Random Forest)
- ✅ **Hyperparameter Tuning**: Grid Search and Random Search optimization

### 🛠️ **MLOps Infrastructure**
- ✅ **DVC Integration**: Data and model versioning with `.dvc` files
- ✅ **DVC Pipelines**: Reproducible training workflows defined in `dvc.yaml`
- ✅ **MLflow Tracking**: Experiment logging, parameter tracking, and metric visualization
- ✅ **MLflow Model Registry**: Model artifacts with signatures and metadata
- ✅ **Parameterized Training**: Configurable hyperparameters via `params.yaml`
- ✅ **REST API**: Production-ready Flask API for model serving (port 5002)
- ✅ **Web UI**: Professional responsive interface for predictions
- ✅ **Docker Containerization**: Multi-service deployment with docker-compose
- ✅ **CI/CD Pipeline**: Automated testing, building, and deployment with GitHub Actions
- 📋 **Cloud Deployment (Optional)**: GCP Cloud Run workflows configured and ready to deploy
### 🔄 **Workflow Orchestration**
- ✅ **Apache Airflow**: Complete ML pipeline orchestration and scheduling
- ✅ **Automated DAG**: Data validation → Training → Evaluation → Deployment → Monitoring
- ✅ **Task Dependencies**: Intelligent task ordering and parallel execution
- ✅ **Visual Monitoring**: DAG graph visualization and execution tracking
- ✅ **Scheduled Runs**: Daily pipeline execution at 2 AM UTC (configurable)
- ✅ **Smart Deployment**: Automatic model deployment based on performance thresholds
- ✅ **Pipeline Reports**: Comprehensive JSON reports for each run
### � **Model Monitoring & Observability**
- ✅ **Prometheus Metrics**: Real-time metrics collection and aggregation
- ✅ **Grafana Dashboards**: Visual monitoring with 7 comprehensive panels
- ✅ **Data Drift Detection**: Statistical drift analysis with Kolmogorov-Smirnov test
- ✅ **Prediction Logging**: Automatic logging of all predictions with timestamps
- ✅ **Performance Tracking**: Request rate, latency, and confidence score monitoring
- ✅ **Alert Generation**: Automated alerts for drift and anomalies
- ✅ **Metrics Endpoint**: `/metrics` endpoint for Prometheus scraping

### �🚀 **CI/CD & Automation**
- ✅ **Automated Testing**: Unit tests, data validation, and model performance checks
- ✅ **Continuous Integration**: Linting, security scanning, and coverage reports
- ✅ **Docker Build & Push**: Automated image building to GitHub Container Registry & GCP
- ✅ **Cloud Deployment**: Automatic deployment to GCP Cloud Run on merge to main
- ✅ **Scheduled Retraining**: Weekly model retraining (Mondays at 2 AM UTC)
- ✅ **Slack Notifications**: Real-time alerts for deployments and failures

### 🌐 **REST API Endpoints**
- ✅ **GET /health**: Health check and service status
- ✅ **GET /info**: Model information, features, and performance metrics
- ✅ **POST /predict**: Single patient diabetes prediction with probabilities
- ✅ **POST /predict/batch**: Batch predictions for multiple patients
- ✅ **GET /metrics**: Prometheus metrics for monitoring
- ✅ **Error Handling**: Input validation, 404, and 500 error responses

### 🎯 **Monitoring Metrics**
- **Prediction Requests**: Total requests by endpoint and prediction class
- **Request Rate**: Per-second prediction rate over time
- **Prediction Latency**: Response time histogram and percentiles
- **Model Confidence**: Average probability scores and distribution
- **Prediction Distribution**: Class balance and positive rate tracking
- **Data Drift Status**: Real-time drift detection alerts

### 📊 **Pipeline Stages (Orchestrated by Airflow)**
1. **Data Validation**: Verify data quality, schema, and completeness
2. **Data Preprocessing**: Clean and encode categorical features
3. **Feature Engineering**: Select and engineer optimal features
4. **Model Training**: Train Random Forest with MLflow tracking
5. **Model Evaluation**: Validate performance against thresholds (ROC-AUC > 0.85)
6. **Deployment Decision**: Automatically approve or reject based on metrics
7. **Model Registration**: Register approved models in MLflow Model Registry
8. **Drift Monitoring**: Check for data drift using Kolmogorov-Smirnov test
9. **Model Serving**: Deploy via REST API with real-time predictions
10. **Performance Tracking**: Monitor with Prometheus/Grafana dashboards

---

## 📁 Project Structure

```
mlops-project/
│
├── data/
│   ├── raw/
│   │   ├── diabetes.csv           # Raw dataset (tracked by DVC)
│   │   └── diabetes.csv.dvc       # DVC pointer file
│   └── processed/
│       └── diabetes_processed.csv # Preprocessed dataset
│
├── models/
│   ├── best_params.json           # Optimized hyperparameters from tuning
│   ├── model.pkl                  # Baseline model
│   └── final/
│       ├── model.pkl              # Final production model
│       └── model.pkl.dvc          # DVC tracking for final model
│
├── notebooks/
│   ├── 01_EDA_&_preprocessing_experiments.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_model_baselines.ipynb
│   └── 04_hyperparameter_search.ipynb
│
├── src/
│   ├── data.py                    # Data loading & preprocessing utilities
│   ├── train.py                   # Baseline training script
│   ├── train_final.py             # Final model training with best params
│   ├── eval.py                    # Model evaluation script
│   ├── features.py                # Feature engineering utilities
│   ├── serve.py                   # REST API + Web UI for model serving 
│   └── selected_features.json     # Optimal features list
│
├── templates/
│   └── index.html                 # Web UI interface 
│
├── static/
│   ├── css/
│   │   └── style.css              # UI styling 
│   └── js/
│       └── app.js                 # UI JavaScript 
│
├── tests/
│   └── test_api.py                # API endpoint testing suite
│
├── airflow/
│   ├── Dockerfile                 # Custom Airflow image with ML dependencies
│   ├── requirements.txt           # Airflow-specific Python packages
│   ├── dags/
│   │   └── diabetes_ml_pipeline.py # Complete ML pipeline DAG
│   ├── logs/                      # Airflow task execution logs
│   ├── plugins/                   # Custom Airflow plugins
│   └── config/                    # Airflow configuration files
│
├── grafana/
│   ├── dashboards.yml             # Dashboard provisioning config
│   └── datasources.yml            # Prometheus data source config
│
├── scripts/
│   ├── setup-airflow.ps1          # Airflow initialization (Windows)
│   ├── setup-airflow.sh           # Airflow initialization (Linux/Mac)
│   └── setup-gcp.sh               # GCP deployment setup script 
│
├── docs/
│   ├── API_DOCUMENTATION.md       # Complete API reference
│   ├── AIRFLOW_SETUP.md           # Airflow installation & usage guide
│   └── MONITORING_SETUP.md        # Prometheus & Grafana setup 
│
├── mlruns/                        # MLflow tracking directory
├── logs/
│   └── monitoring/                # Model monitoring & drift reports
├── dvc.yaml                       # DVC pipeline definition
├── dvc.lock                       # DVC pipeline lock file
├── params.yaml                    # Training hyperparameters
├── requirements.txt               # Python dependencies
├── Dockerfile                     # Docker container definition
├── docker-compose.yml             # Multi-service orchestration (7 services)
├── prometheus.yml                 # Prometheus metrics configuration
├── grafana-dashboard.json         # Pre-built Grafana dashboard
├── scheduler.py                   # Alternative lightweight scheduler
├── .dockerignore                  # Docker build exclusions
├── QUICKSTART.md                  # Quick start guide
└── README.md                      # This file
```

---

## 🛠️ Technologies Used

| **Category**            | **Technology**       | **Purpose**                          |
|-------------------------|----------------------|--------------------------------------|
| **Language**            | Python 3.13          | Core programming language            |
| **ML Framework**        | scikit-learn         | Model training & evaluation          |
| **Experiment Tracking** | MLflow               | Experiment logging & model registry  |
| **Data Versioning**     | DVC                  | Data & model version control         |
| **Orchestration**       | Apache Airflow       | Pipeline automation & scheduling     |
| **Data Processing**     | Pandas, NumPy        | Data manipulation & preprocessing    |
| **Visualization**       | Matplotlib, Seaborn  | EDA & results visualization          |
| **Monitoring**          | Prometheus, Grafana  | Metrics collection & dashboards      |
| **Notebooks**           | Jupyter              | Interactive experimentation          |
| **API Serving**         | Flask, Flask-CORS    | REST API for model predictions       |
| **Testing**             | pytest, requests     | API testing & validation             |
| **Deployment**          | Docker               | Containerization & orchestration     |

---

## 🚀 Setup & Installation

### Prerequisites
- Python 3.13+
- Git
- DVC (Data Version Control)

### 1. Clone the Repository
```bash
git clone <repository-url>
cd mlops-project
```

### 2. Create Virtual Environment
```bash
python -m venv mlops_env
```

**Activate the environment:**

**Windows (PowerShell):**
```powershell
.\mlops_env\Scripts\Activate.ps1
```

**Windows (CMD):**
```cmd
.\mlops_env\Scripts\activate.bat
```

**Linux/Mac:**
```bash
source mlops_env/bin/activate
```

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Initialize DVC (if not already initialized)
```bash
dvc init
```

### 5. Pull Data from DVC Remote (if configured)
```bash
dvc pull
```

---

## � Current Deployment Status

**Active:**
- ✅ **Local Development**: Full-featured development environment
- ✅ **Docker Compose**: 7-service containerized stack (API, MLflow, Airflow, Prometheus, Grafana)
- ✅ **CI/CD**: GitHub Actions for testing and Docker builds
- ✅ **Monitoring**: Real-time Prometheus/Grafana dashboards

**Configured (Ready to Activate):**
- 📋 **GCP Cloud Run**: Deployment workflow ready, requires GCP account setup
- 📋 **Automated Retraining**: Weekly model training workflow (can be enabled)

**Recommended Deployment**: Use Docker Compose for production-grade deployment (all services included)

---

## �💻 Usage

### 🔹 Run the Complete DVC Pipeline
```bash
dvc repro
```

This executes the full training pipeline defined in `dvc.yaml`.

### 🔹 Train the Baseline Model
```bash
python -m src.train --n_estimators 100 --test_size 0.2
```

### 🔹 Train the Final Optimized Model
```bash
python src/train_final.py
```

This uses the best hyperparameters from `models/best_params.json` and selected features from `src/selected_features.json`.

### 🔹 View MLflow Experiments
```bash
mlflow ui --port 5000
```

Then navigate to: [http://localhost:5000](http://localhost:5000)

### 🔹 Docker Deployment (Recommended) 🐳

**Start all services with docker-compose:**
```bash
docker-compose up -d
```

This will start **7 services**:
- **Diabetes Prediction API + Web UI** on port **5002**
- **MLflow Tracking Server** on port **5050**
- **Apache Airflow Webserver** on port **8080**
- **Apache Airflow Scheduler** (background)
- **PostgreSQL Database** (Airflow metadata)
- **Prometheus** on port **9090**
- **Grafana** on port **3000**

**Access the applications:**
- **Web UI:** [http://localhost:5002](http://localhost:5002)
- **MLflow UI:** [http://localhost:5050](http://localhost:5050)
- **Airflow UI:** [http://localhost:8080](http://localhost:8080) (admin/admin)
- **Prometheus:** [http://localhost:9090](http://localhost:9090)
- **Grafana:** [http://localhost:3000](http://localhost:3000) (admin/admin)

**View logs:**
```bash
docker-compose logs -f app
```

**Stop services:**
```bash
docker-compose down
```

**Rebuild after code changes:**
```bash
docker-compose up -d --build
```

---

## 🔄 Airflow Pipeline Orchestration

### Apache Airflow Setup

The project uses **Apache Airflow** to orchestrate the complete ML pipeline with automated scheduling and monitoring.

**Quick Start:**

```bash
# Start all services (including Airflow)
docker-compose up -d

# Wait for services to start (60 seconds)

# Initialize Airflow (Windows PowerShell)
.\scripts\setup-airflow.ps1

# Initialize Airflow (Linux/Mac)
chmod +x scripts/setup-airflow.sh
./scripts/setup-airflow.sh
```

**Access Airflow:**
- **Airflow UI**: http://localhost:8080 (admin/admin)
- **MLflow UI**: http://localhost:5050
- **Prometheus**: http://localhost:9090
- **Grafana**: http://localhost:3000
- **Prediction API**: http://localhost:5002

### Pipeline DAG Overview

The `diabetes_ml_pipeline` DAG orchestrates:

```
Start → Data Validation → Preprocessing → Feature Engineering → Training
  → Evaluation → Deployment Decision → Model Registration → Drift Monitoring → Report → End
```

**Features:**
- ✅ **Automated Scheduling**: Runs daily at 2 AM UTC
- ✅ **Task Dependencies**: Intelligent task ordering and error handling
- ✅ **Smart Deployment**: Only deploys models that meet ROC-AUC > 0.85 threshold
- ✅ **MLflow Integration**: Automatic experiment tracking and model registry
- ✅ **Data Validation**: Pre-training data quality checks
- ✅ **Drift Detection**: Post-training drift monitoring
- ✅ **Pipeline Reports**: JSON reports saved to `logs/airflow/`

**For detailed setup instructions, see: [docs/AIRFLOW_SETUP.md](docs/AIRFLOW_SETUP.md)**

---

## 🔄 CI/CD Pipeline

### GitHub Actions Workflows

The project includes 4 automated workflows:

#### 1. **CI Tests** (`.github/workflows/ci-tests.yml`)
Triggers on every push and PR to `main`/`develop`:
- ✅ Python linting with flake8
- ✅ Unit tests and API tests
- ✅ Data schema validation
- ✅ Model performance validation
- ✅ Security scanning with Trivy
- ✅ Code coverage reporting

#### 2. **Model Training** (`.github/workflows/model-training.yml`)
Scheduled weekly (Monday 2 AM) or manual trigger:
- ✅ Automated model retraining
- ✅ Performance evaluation
- ✅ Model artifact upload
- ✅ Metrics commit to repository

#### 3. **Docker Build & Push** (`.github/workflows/docker-build-push.yml`)
Triggers on push to `main` or version tags:
- ✅ Multi-platform Docker build
- ✅ Push to GitHub Container Registry
- ✅ Push to GCP Artifact Registry
- ✅ Image tagging and versioning

#### 4. **Deploy to GCP (Optional)** (`.github/workflows/deploy-gcp.yml`)
**Status**: Workflow configured but not yet activated. Ready for deployment when needed.
- 📋 Deploy to Cloud Run (workflow ready)
- 📋 Health check validation
- 📋 Slack deployment notification
- 📋 Automatic rollback on failure

### Setup CI/CD

1. **Active Workflows:**
   - ✅ CI Tests: Running on every push/PR
   - ✅ Docker Build: Available via manual trigger
   - 📋 GCP Deployment: Configured but not activated (optional)

2. **To Enable GCP Deployment (Optional):**
   - Follow [`.github/GCP_DEPLOYMENT.md`](.github/GCP_DEPLOYMENT.md) for complete setup
   - Configure GitHub Secrets: `GCP_PROJECT_ID`, `GCP_SA_KEY`
   - Uncomment workflow triggers in deployment files
   - Service will run on GCP Cloud Run with auto-scaling (0-10 instances)

3. **Monitor Workflows:**
   - View in GitHub Actions tab
   - Check deployment logs in GCP Console
   - Receive Slack notifications

---

### 🔹 Local Development (Alternative)

**Start the REST API Server + Web UI:**
```bash
python src/serve.py
```

**Access the application:**
- **Web UI (Recommended):** Open your browser to [http://localhost:5002](http://localhost:5002)
- **REST API:** Available at [http://localhost:5002/predict](http://localhost:5002/predict)

**Features of the Web UI:**
- 🎨 Professional, responsive design
- 📊 Interactive form for patient data input
- 📈 Real-time prediction with confidence scores
- 🎯 Visual probability distribution
- 📝 Input validation and error handling
- 💡 Sample data loading (high-risk/low-risk examples)

**Test the API directly:**
```bash
# Health check
curl http://localhost:5002/health

# Make a prediction
curl -X POST http://localhost:5002/predict \
  -H "Content-Type: application/json" \
  -d '{"age":65,"bmi":32.5,"HbA1c_level":7.5,"blood_glucose_level":180,"hypertension":1,"heart_disease":1,"gender":1,"smoking_history":2}'
```

**Run API Tests:**
```bash
python tests/test_api.py
```

### 🔹 Monitoring & Observability

**Start Prometheus + Grafana:**
```bash
docker-compose up -d prometheus grafana
```

**Access dashboards:**
- **Prometheus**: http://localhost:9090 (metrics & queries)
- **Grafana**: http://localhost:3000 (admin/admin)

**Import pre-built dashboard:**
1. Navigate to Grafana → Dashboards → Import
2. Upload `grafana-dashboard.json`
3. Select Prometheus data source

**Available metrics:**
- Prediction request rate and latency
- Model confidence distribution
- Data drift detection alerts
- Request count by endpoint and prediction class

**For detailed setup, see: [docs/MONITORING_SETUP.md](docs/MONITORING_SETUP.md)**

### 🔹 Run Jupyter Notebooks
```bash
jupyter notebook
```

Explore the notebooks in `notebooks/` for detailed experimentation steps.

---

## 🔄 MLOps Pipeline

### DVC Pipeline Stages (defined in `dvc.yaml`)

```yaml
stages:
  train:
    cmd: python -m src.train --n_estimators 100 --test_size 0.2
    deps:
      - src/train.py
      - src/data.py
      - data/raw/diabetes.csv
    outs:
      - models/model.pkl
```

### Pipeline Execution Flow

```
┌─────────────────┐
│  Raw Data       │
│  (diabetes.csv) │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Preprocessing  │
│  (data.py)      │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Feature        │
│  Selection      │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Model Training │
│  (train.py)     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Model Output   │
│  (model.pkl)    │
└─────────────────┘
```

### Tracked by DVC:
- ✅ `data/raw/diabetes.csv` (3.8 MB)
- ✅ `models/final/model.pkl` (4.9 MB)

### Tracked by MLflow:
- Hyperparameters: `n_estimators`, `max_depth`, `min_samples_split`, etc.
- Metrics: ROC-AUC, Accuracy
- Artifacts: Model files, feature lists, plots

---

## 📊 Model Performance

### 🏆 Best Model: Random Forest (Grid Search)

| **Metric**      | **Value** |
|-----------------|-----------|
| **ROC-AUC**     | 0.9707    |
| **Algorithm**   | Random Forest |
| **Hyperparameters** | `n_estimators=200`, `max_depth=10`, `min_samples_split=2` |

### Model Comparison

| **Model**                     | **ROC-AUC (CV)** | **Test ROC-AUC** |
|-------------------------------|------------------|------------------|
| Logistic Regression           | 0.9522           | 0.9502           |
| Random Forest (Grid Search)   | 0.9707           | 0.9707           |
| Random Forest (Random Search) | 0.9709           | 0.9707           |

### � Understanding ROC-AUC Score

**What is ROC-AUC?**  
ROC-AUC (Receiver Operating Characteristic - Area Under Curve) measures how well the model distinguishes between diabetic and non-diabetic patients.

**Simple Interpretation:**
- **Score: 0.9707 = 97.07%** probability that the model correctly ranks a diabetic patient as higher risk than a non-diabetic patient
- **Range:** 0.5 (random guessing) to 1.0 (perfect classification)
- **Our Score (0.9707):** Excellent! The model is highly reliable for diabetes prediction

**Why ROC-AUC instead of Accuracy?**
- Works better with imbalanced datasets (when diabetic patients are fewer than non-diabetic)
- Evaluates model performance across all decision thresholds, not just one
- More suitable for medical predictions where different risk thresholds may be needed
- Captures the trade-off between catching all diabetic cases vs. avoiding false alarms

### �📈 Selected Features (8 features)
The following features were selected based on correlation analysis and domain knowledge:

1. `age`
2. `bmi`
3. `HbA1c_level`
4. `blood_glucose_level`
5. `hypertension`
6. `heart_disease`
7. `gender`
8. `smoking_history`

### 🚀 **API Performance**

| **Metric** | **Value** |
|------------|----------|
| **Port** | 5002 |
| **Latency (Single)** | ~5-20ms |
| **Test Coverage** | 6/6 tests passing |
| **Endpoints** | 4 (health, info, predict, batch) |
| **Web UI** | ✅ Professional responsive interface |
| **Docker** | ✅ Containerized deployment |
| **Input Validation** | ✅ Enabled |
| **Error Handling** | ✅ Comprehensive |

---

## 📚 Documentation

- **[QUICKSTART.md](QUICKSTART.md)** - Quick start guide for new users
- **[docs/API_DOCUMENTATION.md](docs/API_DOCUMENTATION.md)** - Complete REST API reference
- **[docs/AIRFLOW_SETUP.md](docs/AIRFLOW_SETUP.md)** - Airflow installation & DAG configuration
- **[docs/MONITORING_SETUP.md](docs/MONITORING_SETUP.md)** - Prometheus & Grafana setup guide
- **[.github/SECRETS_SETUP.md](.github/SECRETS_SETUP.md)** - GitHub Actions secrets configuration
- **[.github/GCP_DEPLOYMENT.md](.github/GCP_DEPLOYMENT.md)** - GCP Cloud Run deployment guide

---

**Built with ❤️ by Houyem Lahmar - Software Engineer**

*Last Updated: January 06, 2026*
