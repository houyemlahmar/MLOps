# Custom Airflow Image with ML Dependencies
FROM apache/airflow:2.8.1-python3.11

USER root

# Install system dependencies if needed
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    && apt-get clean

USER airflow

# Install Python ML dependencies
COPY requirements.txt /requirements.txt
RUN pip install --no-cache-dir -r /requirements.txt

# Install specific compatible versions to avoid conflicts
RUN pip install --no-cache-dir \
    numpy==1.26.4 \
    scipy==1.11.4 \
    scikit-learn==1.3.2 \
    pandas==2.2.3 \
    mlflow==3.6.0 \
    pyyaml==6.0.2 \
    prometheus-client>=0.19.0
