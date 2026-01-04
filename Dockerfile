# Multi-stage Dockerfile for MLOps Diabetes Prediction System
# Base image: Python 3.13 slim for smaller image size
FROM python:3.13-slim as base

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    FLASK_HOST=0.0.0.0 \
    FLASK_PORT=5002

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better layer caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/
COPY models/ ./models/
COPY data/ ./data/
COPY templates/ ./templates/
COPY static/ ./static/
COPY params.yaml .
COPY dvc.yaml .

# Create directories for MLflow and logs
RUN mkdir -p /app/mlruns /app/logs

# Expose port for Flask API
EXPOSE 5002

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:5002/health || exit 1

# Run the Flask application
CMD ["python", "src/serve.py"]
