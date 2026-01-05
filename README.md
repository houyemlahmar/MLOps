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
- ✅ **Cloud Deployment**: GCP Cloud Run integration for scalable serving

### 🚀 **CI/CD & Automation**
- ✅ **Automated Testing**: Unit tests, data validation, and model performance checks
- ✅ **Continuous Integration**: Linting, security scanning, and coverage reports
- ✅ **Docker Build & Push**: Automated image building to GitHub Container Registry & GCP
- ✅ **Cloud Deployment**: Automatic deployment to GCP Cloud Run on merge to main
- ✅ **Scheduled Retraining**: Weekly model retraining (Mondays at 2 AM UTC)
- ✅ **Slack Notifications**: Real-time alerts for deployments and failures

### � **REST API Endpoints**
- ✅ **GET /health**: Health check and service status
- ✅ **GET /info**: Model information, features, and performance metrics
- ✅ **POST /predict**: Single patient diabetes prediction with probabilities
- ✅ **POST /predict/batch**: Batch predictions for multiple patients
- ✅ **Error Handling**: Input validation, 404, and 500 error responses

### 📊 **Pipeline Stages**
1. **Data Loading**: Load raw diabetes dataset
2. **Preprocessing**: Clean and encode categorical features
3. **Feature Selection**: Use 8 optimal features identified through analysis
4. **Model Training**: Train Random Forest with optimized hyperparameters
5. **Model Evaluation**: Calculate ROC-AUC and accuracy metrics
6. **Model Versioning**: Save models with DVC tracking
7. **Model Serving**: Deploy via REST API with real-time predictions

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
│   ├── serve.py                   # REST API + Web UI for model serving ✅
│   └── selected_features.json     # Optimal features list
│
├── templates/
│   └── index.html                 # Web UI interface ✅
│
├── static/
│   ├── css/
│   │   └── style.css              # UI styling ✅
│   └── js/
│       └── app.js                 # UI JavaScript ✅
│
├── tests/
│   └── test_api.py                # API endpoint testing suite ✅
│
├── docs/
│   └── API_DOCUMENTATION.md       # Complete API reference ✅
│
├── mlruns/                        # MLflow tracking directory
├── dvc.yaml                       # DVC pipeline definition
├── dvc.lock                       # DVC pipeline lock file
├── params.yaml                    # Training hyperparameters
├── requirements.txt               # Python dependencies
├── Dockerfile                     # Docker container definition ✅
├── docker-compose.yml             # Docker orchestration ✅
├── .dockerignore                  # Docker build exclusions ✅
├── QUICKSTART.md                  # Quick start guide ✅
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
| **Data Processing**     | Pandas, NumPy        | Data manipulation & preprocessing    |
| **Visualization**       | Matplotlib, Seaborn  | EDA & results visualization          |
| **Notebooks**           | Jupyter              | Interactive experimentation          |
| **API Serving**         | Flask, Flask-CORS    | REST API for model predictions       |
| **Testing**             | pytest, requests     | API testing & validation             |
| **Deployment (Next)**   | Docker               | Containerization & orchestration     |

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

## 💻 Usage

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

This will start:
- **Diabetes Prediction API + Web UI** on port **5002**
- **MLflow Tracking Server** on port **5050**

**Access the applications:**
- **Web UI:** [http://localhost:5002](http://localhost:5002)
- **MLflow UI:** [http://localhost:5050](http://localhost:5050)

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

#### 4. **Deploy to GCP** (`.github/workflows/deploy-gcp.yml`)
Triggers after successful Docker build:
- ✅ Deploy to Cloud Run
- ✅ Health check validation
- ✅ Slack deployment notification
- ✅ Automatic rollback on failure

### Setup CI/CD

1. **Configure GitHub Secrets:**
   - See [`.github/SECRETS_SETUP.md`](.github/SECRETS_SETUP.md) for detailed instructions
   - Required: `GCP_PROJECT_ID`, `GCP_SA_KEY`
   - Optional: `SLACK_WEBHOOK_URL`

2. **GCP Deployment:**
   - Follow [`.github/GCP_DEPLOYMENT.md`](.github/GCP_DEPLOYMENT.md) for complete setup
   - Service runs on GCP Cloud Run
   - Auto-scaling: 0-10 instances
   - Region: `us-central1`

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

### 📈 Selected Features (8 features)
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

## 🔮 Next Steps

### 🎯 Completed Features ✅

1. **Model Serving & Deployment** ✅ COMPLETED
   - [x] Implement REST API using Flask (`src/serve.py`)
   - [x] Add prediction endpoint with input validation
   - [x] Create comprehensive API documentation
   - [x] Add health check and monitoring endpoints
   - [x] Implement batch prediction support
   - [x] Create automated test suite (6/6 tests passing)
   - [x] Build Web UI with HTML/CSS/JavaScript
   - [x] Add sample data loading and result visualization
   - [x] Docker containerization with docker-compose
   - [x] MLflow integration in containers

2. **CI/CD Pipeline** ✅ COMPLETED
   - [x] Set up GitHub Actions workflows
   - [x] Automate testing (unit tests, data validation, model checks)
   - [x] Automate model retraining on schedule (Mondays 2 AM)
   - [x] Docker build and push to registries
   - [x] Deploy to GCP Cloud Run
   - [x] Slack notifications for deployments
   - [x] Security scanning with Trivy

### 🎯 Future Enhancements

3. **Monitoring & Observability** 🎯 NEXT PRIORITY
   - [ ] Implement model drift detection
   - [ ] Add Prometheus metrics export
   - [ ] Set up Grafana dashboards
   - [ ] Log prediction requests and responses
   - [ ] A/B testing framework
   - [ ] Real-time performance monitoring

4. **Model Evaluation & Validation**
   - [ ] Complete `src/eval.py` with comprehensive metrics
   - [ ] Add cross-validation reports
   - [ ] Implement model validation checks before deployment
   - [ ] Create performance comparison visualizations

6. **Data Pipeline Enhancements**
   - [ ] Automate data ingestion from external sources
   - [ ] Add data quality checks and validation
   - [ ] Implement data preprocessing as a DVC stage
   - [ ] Set up automated feature engineering pipeline

7. **Documentation & Best Practices**
   - [ ] Add API documentation (Swagger/OpenAPI)
   - [ ] Create deployment guide
   - [ ] Document model assumptions and limitations
   - [ ] Add code quality checks (linting, type hints)

### 🚀 Advanced Features (Future)

- **A/B Testing Framework**: Compare model versions in production
- **Feature Store**: Centralized feature management
- **Online Learning**: Incremental model updates
- **Multi-model Ensemble**: Combine predictions from multiple models
- **Explainability**: SHAP values, LIME for model interpretability
- **Cloud Deployment**: AWS SageMaker, Azure ML, or GCP AI Platform

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Guidelines
- Write clean, documented code
- Add unit tests for new features
- Update documentation as needed
- Follow PEP 8 style guide

---

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 🙏 Acknowledgments

- Dataset source: [Diabetes dataset]
- MLOps tools: DVC, MLflow
- scikit-learn community

---

## 📞 Contact

For questions or feedback, please open an issue or contact the project maintainer.

---

**Built with ❤️ by the MLOps Team**

*Last Updated: November 30, 2025*
