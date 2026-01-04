# 🔌 Diabetes Prediction API Documentation

Complete API reference for the Diabetes Prediction Model Serving Service.

---

## 📚 Base URL

```
http://localhost:5002
```

---

## 🔗 Endpoints

### 1. Health Check
**GET** `/health`

Check if the API service is running and healthy.

**Response:**
```json
{
  "status": "healthy",
  "service": "diabetes-prediction-api",
  "model_loaded": true,
  "version": "1.0.0"
}
```

**Status Codes:**
- `200 OK`: Service is healthy

---

### 2. Model Information
**GET** `/info`

Get detailed information about the loaded model, features, and performance metrics.

**Response:**
```json
{
  "model_info": {
    "model_type": "RandomForestClassifier",
    "hyperparameters": {
      "max_depth": 10,
      "min_samples_split": 2,
      "n_estimators": 200
    },
    "performance": {
      "cv_roc_auc": 0.9707,
      "test_roc_auc": 0.9707
    },
    "features": [
      "age", "bmi", "HbA1c_level", "blood_glucose_level",
      "hypertension", "heart_disease", "gender", "smoking_history"
    ],
    "n_features": 8
  },
  "api_version": "1.0.0",
  "endpoints": {
    "/health": "GET - Health check",
    "/info": "GET - Model information",
    "/predict": "POST - Make predictions",
    "/predict/batch": "POST - Batch predictions"
  }
}
```

**Status Codes:**
- `200 OK`: Model info retrieved successfully
- `500 Internal Server Error`: Model not loaded

---

### 3. Single Prediction
**POST** `/predict`

Make a diabetes prediction for a single patient.

**Request Body:**
```json
{
  "age": 65,
  "bmi": 32.5,
  "HbA1c_level": 7.5,
  "blood_glucose_level": 180,
  "hypertension": 1,
  "heart_disease": 1,
  "gender": 1,
  "smoking_history": 2
}
```

**Feature Descriptions:**

| Feature | Type | Description | Example Values |
|---------|------|-------------|----------------|
| `age` | int/float | Patient age in years | 18-100 |
| `bmi` | float | Body Mass Index | 15.0-50.0 |
| `HbA1c_level` | float | Hemoglobin A1c level (%) | 4.0-9.0 |
| `blood_glucose_level` | int/float | Blood glucose level (mg/dL) | 80-300 |
| `hypertension` | int | Has hypertension (0=No, 1=Yes) | 0 or 1 |
| `heart_disease` | int | Has heart disease (0=No, 1=Yes) | 0 or 1 |
| `gender` | int | Gender (encoded) | 0 or 1 |
| `smoking_history` | int | Smoking history (encoded) | 0-4 |

**Response:**
```json
{
  "prediction": 1,
  "prediction_label": "Diabetic",
  "probability": {
    "non_diabetic": 0.12,
    "diabetic": 0.88
  },
  "confidence": 0.88,
  "input_features": {
    "age": 65,
    "bmi": 32.5,
    "HbA1c_level": 7.5,
    "blood_glucose_level": 180,
    "hypertension": 1,
    "heart_disease": 1,
    "gender": 1,
    "smoking_history": 2
  }
}
```

**Status Codes:**
- `200 OK`: Prediction successful
- `400 Bad Request`: Invalid input data
- `500 Internal Server Error`: Prediction error

---

### 4. Batch Prediction
**POST** `/predict/batch`

Make diabetes predictions for multiple patients in a single request.

**Request Body:**
```json
{
  "instances": [
    {
      "age": 45,
      "bmi": 25.0,
      "HbA1c_level": 5.5,
      "blood_glucose_level": 100,
      "hypertension": 0,
      "heart_disease": 0,
      "gender": 0,
      "smoking_history": 0
    },
    {
      "age": 70,
      "bmi": 35.0,
      "HbA1c_level": 8.0,
      "blood_glucose_level": 200,
      "hypertension": 1,
      "heart_disease": 1,
      "gender": 1,
      "smoking_history": 3
    }
  ]
}
```

**Response:**
```json
{
  "predictions": [
    {
      "index": 0,
      "prediction": 0,
      "prediction_label": "Non-Diabetic",
      "probability": {
        "non_diabetic": 0.92,
        "diabetic": 0.08
      },
      "confidence": 0.92
    },
    {
      "index": 1,
      "prediction": 1,
      "prediction_label": "Diabetic",
      "probability": {
        "non_diabetic": 0.15,
        "diabetic": 0.85
      },
      "confidence": 0.85
    }
  ],
  "total_instances": 2
}
```

**Status Codes:**
- `200 OK`: Batch prediction successful
- `400 Bad Request`: Invalid input data
- `500 Internal Server Error`: Batch prediction error

---

## 📝 Usage Examples

### cURL Examples

#### Health Check
```bash
curl -X GET http://localhost:5002/health
```

#### Single Prediction
```bash
curl -X POST http://localhost:5002/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age": 65,
    "bmi": 32.5,
    "HbA1c_level": 7.5,
    "blood_glucose_level": 180,
    "hypertension": 1,
    "heart_disease": 1,
    "gender": 1,
    "smoking_history": 2
  }'
```

#### Batch Prediction
```bash
curl -X POST http://localhost:5002/predict/batch \
  -H "Content-Type: application/json" \
  -d '{
    "instances": [
      {
        "age": 45,
        "bmi": 25.0,
        "HbA1c_level": 5.5,
        "blood_glucose_level": 100,
        "hypertension": 0,
        "heart_disease": 0,
        "gender": 0,
        "smoking_history": 0
      }
    ]
  }'
```

---

### Python Examples

#### Using `requests` library

```python
import requests
import json

# Base URL
API_URL = "http://localhost:5002"

# 1. Health Check
response = requests.get(f"{API_URL}/health")
print(response.json())

# 2. Single Prediction
patient_data = {
    "age": 65,
    "bmi": 32.5,
    "HbA1c_level": 7.5,
    "blood_glucose_level": 180,
    "hypertension": 1,
    "heart_disease": 1,
    "gender": 1,
    "smoking_history": 2
}

response = requests.post(
    f"{API_URL}/predict",
    json=patient_data,
    headers={'Content-Type': 'application/json'}
)
result = response.json()
print(f"Prediction: {result['prediction_label']}")
print(f"Confidence: {result['confidence']:.2%}")

# 3. Batch Prediction
batch_data = {
    "instances": [
        {
            "age": 45,
            "bmi": 25.0,
            "HbA1c_level": 5.5,
            "blood_glucose_level": 100,
            "hypertension": 0,
            "heart_disease": 0,
            "gender": 0,
            "smoking_history": 0
        },
        {
            "age": 70,
            "bmi": 35.0,
            "HbA1c_level": 8.0,
            "blood_glucose_level": 200,
            "hypertension": 1,
            "heart_disease": 1,
            "gender": 1,
            "smoking_history": 3
        }
    ]
}

response = requests.post(
    f"{API_URL}/predict/batch",
    json=batch_data
)
results = response.json()
for pred in results['predictions']:
    print(f"Patient {pred['index']}: {pred['prediction_label']} (Confidence: {pred['confidence']:.2%})")
```

---

### JavaScript/Node.js Example

```javascript
const axios = require('axios');

const API_URL = 'http://localhost:5002';

// Single Prediction
async function makePrediction() {
  const patientData = {
    age: 65,
    bmi: 32.5,
    HbA1c_level: 7.5,
    blood_glucose_level: 180,
    hypertension: 1,
    heart_disease: 1,
    gender: 1,
    smoking_history: 2
  };

  try {
    const response = await axios.post(`${API_URL}/predict`, patientData);
    console.log('Prediction:', response.data.prediction_label);
    console.log('Confidence:', (response.data.confidence * 100).toFixed(2) + '%');
  } catch (error) {
    console.error('Error:', error.response?.data || error.message);
  }
}

makePrediction();
```

---

## ⚠️ Error Responses

### 400 Bad Request
Missing or invalid input data.

```json
{
  "error": "Missing required features: ['age', 'bmi']"
}
```

### 404 Not Found
Invalid endpoint.

```json
{
  "error": "Endpoint not found",
  "message": "The requested URL was not found on the server."
}
```

### 500 Internal Server Error
Server-side error during prediction.

```json
{
  "error": "Internal server error during prediction",
  "details": "Error message details"
}
```

---

## 🚀 Quick Start

### 1. Start the API Server

```bash
# Activate virtual environment
.\mlops_env\Scripts\Activate.ps1  # Windows
source mlops_env/bin/activate      # Linux/Mac

# Start the server
python src/serve.py
```

The server will start on `http://localhost:5002`

### 2. Test the API

```bash
# Run automated tests
python tests/test_api.py
```

### 3. Make Your First Prediction

```bash
curl -X POST http://localhost:5001/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age": 50,
    "bmi": 28.0,
    "HbA1c_level": 6.5,
    "blood_glucose_level": 140,
    "hypertension": 1,
    "heart_disease": 0,
    "gender": 1,
    "smoking_history": 1
  }'
```

---

## 🔧 Configuration

The API can be configured using environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `FLASK_HOST` | `0.0.0.0` | Host to bind the server |
| `FLASK_PORT` | `5002` | Port to run the server |
| `FLASK_DEBUG` | `False` | Enable debug mode |

**Example:**
```bash
export FLASK_PORT=8080
export FLASK_DEBUG=True
python src/serve.py
```

---

## 📊 Performance

- **Latency**: ~5-20ms per prediction (single)
- **Throughput**: ~100-200 requests/second (depends on hardware)
- **Model Size**: 4.9 MB
- **Memory Usage**: ~200-300 MB

---

## 🛡️ Best Practices

1. **Always validate input data** before sending to the API
2. **Use batch predictions** for multiple instances to reduce overhead
3. **Handle errors gracefully** with proper try-catch blocks
4. **Monitor API health** using the `/health` endpoint
5. **Log prediction requests** for auditing and monitoring

---

## 📞 Support

For issues or questions:
- Open an issue on GitHub
- Check the main [README.md](../README.md) for more information
- Review the test suite in `tests/test_api.py` for more examples

---

**Last Updated:** January 4, 2026
