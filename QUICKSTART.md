# 🚀 Quick Start Guide - Diabetes Prediction API

## Option 1: Docker (Recommended) 🐳

### Prerequisites
- Docker and Docker Compose installed

### Step 1: Start All Services

```bash
docker-compose up -d
```

This will start:
- **Diabetes Prediction API + Web UI** on port **5002**
- **MLflow Tracking Server** on port **5050**

### Step 2: Access the Applications

- **Web UI:** Open [http://localhost:5002](http://localhost:5002) in your browser
- **MLflow UI:** Open [http://localhost:5050](http://localhost:5050) in your browser

### Step 3: Test the API

```bash
# Health check
curl http://localhost:5002/health

# Make a prediction
curl -X POST http://localhost:5002/predict \
  -H "Content-Type: application/json" \
  -d '{"age":65,"bmi":32.5,"HbA1c_level":7.5,"blood_glucose_level":180,"hypertension":1,"heart_disease":1,"gender":1,"smoking_history":2}'
```

### View Logs
```bash
docker-compose logs -f app
```

### Stop Services
```bash
docker-compose down
```

---

## Option 2: Local Development

### Step 1: Install Dependencies

Ensure you're in the virtual environment and install the updated requirements:

```powershell
# Activate virtual environment (if not already activated)
.\mlops_env\Scripts\Activate.ps1

# Install/update dependencies
pip install flask flask-cors requests pytest

# Or install all from requirements.txt
pip install -r requirements.txt
```

### Step 2: Start the API Server

```powershell
python src/serve.py
```

You should see:
```
INFO - Loading model from models/final/model.pkl
INFO - Model loaded successfully
INFO - Loaded 8 features
INFO - Starting Flask server on 0.0.0.0:5002
```

**Keep this terminal open** - the server is running!

### Step 3: Access the Web UI

Open your browser to [http://localhost:5002](http://localhost:5002) and use the interactive form to make predictions.

### Step 4: Test the API (Open a NEW Terminal)

```powershell
# Activate virtual environment in the new terminal
.\mlops_env\Scripts\Activate.ps1

# Navigate to project directory
cd C:\Users\MSI\mlops-project

# Run the test suite
python tests/test_api.py
```

---

## Making Predictions

### Option A: Using the Web UI (Easiest)

1. Open [http://localhost:5002](http://localhost:5002)
2. Fill in patient data in the form
3. Click "Predict Diabetes Risk"
4. View results with confidence scores and probability distribution

### Option B: Using PowerShell (Invoke-WebRequest)

```powershell
$body = @{
    age = 65
    bmi = 32.5
    HbA1c_level = 7.5
    blood_glucose_level = 180
    hypertension = 1
    heart_disease = 1
    gender = 1
    smoking_history = 2
} | ConvertTo-Json

Invoke-RestMethod -Uri "http://localhost:5002/predict" -Method Post -Body $body -ContentType "application/json"
```

### Option C: Using Python

Create a file `quick_test.py`:

```python
import requests

response = requests.post(
    "http://localhost:5002/predict",
    json={
        "age": 65,
        "bmi": 32.5,
        "HbA1c_level": 7.5,
        "blood_glucose_level": 180,
        "hypertension": 1,
        "heart_disease": 1,
        "gender": 1,
        "smoking_history": 2
    }
)

print(response.json())
```

Run it:
```powershell
python quick_test.py
```

## Expected Output

```json
{
  "prediction": 1,
  "prediction_label": "Diabetic",
  "probability": {
    "non_diabetic": 0.12,
    "diabetic": 0.88
  },
  "confidence": 0.88,
  "input_features": {...}
}
```

## Troubleshooting

### Issue: "Cannot connect to API"
- **Solution**: Make sure the server is running in another terminal

### Issue: "Model file not found"
- **Solution**: Ensure `models/final/model.pkl` exists. If not, run:
  ```powershell
  python src/train_final.py
  ```

### Issue: Port 5001 already in use
- **Solution**: Change the port:
  ```powershell
  $env:FLASK_PORT=8080
  python src/serve.py
  ```

## Next Steps

1. ✅ API is running locally
2. 📝 Review full documentation: `docs/API_DOCUMENTATION.md`
3. 🐳 Containerize with Docker (next phase)
4. ☁️ Deploy to cloud (AWS/Azure/GCP)

## Stopping the Server

Press `Ctrl+C` in the terminal where the server is running.

---

**Happy Predicting! 🎯**
