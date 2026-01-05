"""
REST API for Diabetes Prediction Model Serving

This module provides a Flask-based REST API for serving the trained
diabetes prediction model with endpoints for predictions, health checks,
and model information.

Author: MLOps Team
Date: January 2026
"""

import os
import json
import logging
import time
from typing import Dict, List, Any

import joblib
import pandas as pd
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__, 
            template_folder='../templates',
            static_folder='../static')
CORS(app)  # Enable CORS for all routes

# Prometheus Metrics
REQUEST_COUNT = Counter(
    'prediction_requests_total',
    'Total number of prediction requests',
    ['method', 'endpoint', 'prediction']
)

REQUEST_LATENCY = Histogram(
    'prediction_duration_seconds',
    'Prediction request latency in seconds',
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0]
)

PREDICTION_PROBABILITY = Gauge(
    'prediction_probability',
    'Last prediction probability score'
)

DATA_DRIFT = Gauge(
    'data_drift_detected',
    'Data drift detection status (1 = drift, 0 = no drift)'
)

MODEL_INFO_GAUGE = Gauge(
    'model_info',
    'Model metadata',
    ['model_type', 'version']
)

# Global variables for model and features
model = None
selected_features = None
model_info = {}


def load_model_artifacts():
    """
    Load model and associated artifacts (features, metadata).
    
    Returns:
        tuple: (model, selected_features, model_info)
    
    Raises:
        FileNotFoundError: If model files are not found
        Exception: For other loading errors
    """
    global model, selected_features, model_info
    
    try:
        # Load the trained model
        model_path = "models/final/model.pkl"
        logger.info(f"Loading model from {model_path}")
        model = joblib.load(model_path)
        logger.info("Model loaded successfully")
        
        # Load selected features
        features_path = "src/selected_features.json"
        logger.info(f"Loading features from {features_path}")
        with open(features_path, "r") as f:
            selected_features = json.load(f)
        logger.info(f"Loaded {len(selected_features)} features")
        
        # Load model hyperparameters
        params_path = "models/best_params.json"
        if os.path.exists(params_path):
            with open(params_path, "r") as f:
                best_params = json.load(f)
                model_info = {
                    "model_type": "RandomForestClassifier",
                    "hyperparameters": best_params.get("RandomForest_Grid", {}).get("best_params", {}),
                    "performance": {
                        "cv_roc_auc": best_params.get("RandomForest_Grid", {}).get("best_score", None),
                        "test_roc_auc": best_params.get("TestScores", {}).get("RF_Grid_Test", None)
                    },
                    "features": selected_features,
                    "n_features": len(selected_features)
                }
        
        return model, selected_features, model_info
        
    except FileNotFoundError as e:
        logger.error(f"Model file not found: {e}")
        raise
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise


def validate_input(data: Dict[str, Any]) -> tuple:
    """
    Validate input data for prediction.
    
    Args:
        data: Dictionary containing input features
        
    Returns:
        tuple: (is_valid, error_message, processed_data)
    """
    if not data:
        return False, "No input data provided", None
    
    # Check if required features are present
    missing_features = [f for f in selected_features if f not in data]
    if missing_features:
        return False, f"Missing required features: {missing_features}", None
    
    # Extract only the selected features in correct order
    try:
        processed_data = {feature: data[feature] for feature in selected_features}
        
        # Validate data types
        for feature, value in processed_data.items():
            if not isinstance(value, (int, float)):
                return False, f"Feature '{feature}' must be numeric, got {type(value).__name__}", None
        
        return True, None, processed_data
        
    except Exception as e:
        return False, f"Error processing input: {str(e)}", None


@app.route('/', methods=['GET'])
def home():
    """
    Serve the web UI homepage.
    
    Returns:
        HTML: Rendered template for the web interface
    """
    return render_template('index.html')


@app.route('/health', methods=['GET'])
def health_check():
    """
    Health check endpoint to verify API is running.
    
    Returns:
        JSON response with status
    """
    return jsonify({
        'status': 'healthy',
        'service': 'diabetes-prediction-api',
        'model_loaded': model is not None,
        'version': '1.0.0'
    }), 200


@app.route('/info', methods=['GET'])
def model_info_endpoint():
    """
    Get information about the loaded model.
    
    Returns:
        JSON response with model metadata
    """
    if model is None:
        return jsonify({
            'error': 'Model not loaded'
        }), 500
    
    return jsonify({
        'model_info': model_info,
        'api_version': '1.0.0',
        'endpoints': {
            '/health': 'GET - Health check',
            '/info': 'GET - Model information',
            '/predict': 'POST - Make predictions',
            '/predict/batch': 'POST - Batch predictions'
        }
    }), 200


@app.route('/predict', methods=['POST'])
def predict():
    """
    Make a prediction for a single instance.
    
    Expected JSON input:
    {
        "age": 45,
        "bmi": 28.5,
        "HbA1c_level": 6.5,
        "blood_glucose_level": 140,
        "hypertension": 1,
        "heart_disease": 0,
        "gender": 1,
        "smoking_history": 2
    }
    
    Returns:
        JSON response with prediction and probability
    """
    start_time = time.time()
    
    try:
        # Get JSON data from request
        data = request.get_json()
        
        if data is None:
            return jsonify({
                'error': 'Invalid JSON data'
            }), 400
        
        # Validate input
        is_valid, error_msg, processed_data = validate_input(data)
        if not is_valid:
            return jsonify({
                'error': error_msg
            }), 400
        
        # Create DataFrame with features in correct order
        input_df = pd.DataFrame([processed_data], columns=selected_features)
        
        # Make prediction
        prediction = model.predict(input_df)[0]
        prediction_proba = model.predict_proba(input_df)[0]
        
        # Update Prometheus metrics
        prediction_probability = float(prediction_proba[1])
        REQUEST_COUNT.labels(
            method='POST',
            endpoint='/predict',
            prediction=str(int(prediction))
        ).inc()
        PREDICTION_PROBABILITY.set(prediction_probability)
        
        # Prepare response
        response = {
            'prediction': int(prediction),
            'prediction_label': 'Diabetic' if prediction == 1 else 'Non-Diabetic',
            'probability': {
                'non_diabetic': float(prediction_proba[0]),
                'diabetic': float(prediction_proba[1])
            },
            'confidence': float(max(prediction_proba)),
            'input_features': processed_data
        }
        
        # Record latency
        REQUEST_LATENCY.observe(time.time() - start_time)
        
        logger.info(f"Prediction made: {response['prediction_label']} (confidence: {response['confidence']:.4f})")
        
        return jsonify(response), 200
        
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        return jsonify({
            'error': 'Internal server error during prediction',
            'details': str(e)
        }), 500


@app.route('/predict/batch', methods=['POST'])
def predict_batch():
    """
    Make predictions for multiple instances.
    
    Expected JSON input:
    {
        "instances": [
            {"age": 45, "bmi": 28.5, ...},
            {"age": 60, "bmi": 32.1, ...}
        ]
    }
    
    Returns:
        JSON response with predictions for all instances
    """
    try:
        # Get JSON data from request
        data = request.get_json()
        
        if data is None or 'instances' not in data:
            return jsonify({
                'error': 'Invalid JSON data. Expected format: {"instances": [...]}'
            }), 400
        
        instances = data['instances']
        
        if not isinstance(instances, list) or len(instances) == 0:
            return jsonify({
                'error': 'instances must be a non-empty list'
            }), 400
        
        # Validate and process each instance
        processed_instances = []
        for idx, instance in enumerate(instances):
            is_valid, error_msg, processed_data = validate_input(instance)
            if not is_valid:
                return jsonify({
                    'error': f'Invalid data at instance {idx}: {error_msg}'
                }), 400
            processed_instances.append(processed_data)
        
        # Create DataFrame
        input_df = pd.DataFrame(processed_instances, columns=selected_features)
        
        # Make predictions
        predictions = model.predict(input_df)
        predictions_proba = model.predict_proba(input_df)
        
        # Prepare response
        results = []
        for idx, (pred, proba) in enumerate(zip(predictions, predictions_proba)):
            results.append({
                'index': idx,
                'prediction': int(pred),
                'prediction_label': 'Diabetic' if pred == 1 else 'Non-Diabetic',
                'probability': {
                    'non_diabetic': float(proba[0]),
                    'diabetic': float(proba[1])
                },
                'confidence': float(max(proba))
            })
        
        response = {
            'predictions': results,
            'total_instances': len(results)
        }
        
        logger.info(f"Batch prediction made for {len(results)} instances")
        
        return jsonify(response), 200
        
    except Exception as e:
        logger.error(f"Error during batch prediction: {str(e)}")
        return jsonify({
            'error': 'Internal server error during batch prediction',
            'details': str(e)
        }), 500


@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors."""
    return jsonify({
        'error': 'Endpoint not found',
        'message': 'The requested URL was not found on the server.'
    }), 404


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors."""
    return jsonify({
        'error': 'Internal server error',
        'message': 'An unexpected error occurred.'
    }), 500


def main():
    """
    Main function to start the Flask application.
    """
    # Load model artifacts
    try:
        load_model_artifacts()
        logger.info("Model artifacts loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model artifacts: {e}")
        logger.error("Server will start but predictions will fail until model is loaded")
    
    # Get configuration from environment variables
    host = os.getenv('FLASK_HOST', '0.0.0.0')
    port = int(os.getenv('FLASK_PORT', 5002))
    debug = os.getenv('FLASK_DEBUG', 'False').lower() == 'true'
    
    # Start the server
    logger.info(f"Starting Flask server on {host}:{port}")
    logger.info(f"API Documentation available at http://{host}:{port}/info")
    app.run(host=host, port=port, debug=debug)


@app.route('/metrics')
def metrics():
    """
    Prometheus metrics endpoint
    
    Returns:
        Prometheus-formatted metrics
    """
    return generate_latest(), 200, {'Content-Type': CONTENT_TYPE_LATEST}


if __name__ == '__main__':
    main()
