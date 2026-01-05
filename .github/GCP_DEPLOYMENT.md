# 🚀 GCP Deployment Guide

Complete guide to deploying the Diabetes Prediction MLOps system to Google Cloud Platform (GCP) using Cloud Run.

## Prerequisites

- Google Cloud Platform account with billing enabled
- `gcloud` CLI installed and configured
- Docker installed (for local testing)
- GitHub repository with Actions enabled

## 1. GCP Project Setup

### Create or select a GCP project:

```bash
# Create new project
gcloud projects create diabetes-prediction-mlops --name="Diabetes Prediction MLOps"

# Set as active project
gcloud config set project diabetes-prediction-mlops

# Enable required APIs
gcloud services enable run.googleapis.com
gcloud services enable artifactregistry.googleapis.com
gcloud services enable containerregistry.googleapis.com
gcloud services enable cloudbuild.googleapis.com
```

## 2. Create Service Account for GitHub Actions

```bash
# Set environment variables
export GCP_PROJECT_ID=$(gcloud config get-value project)
export SA_NAME="github-actions"
export SA_EMAIL="${SA_NAME}@${GCP_PROJECT_ID}.iam.gserviceaccount.com"

# Create service account
gcloud iam service-accounts create $SA_NAME \
  --display-name="GitHub Actions Deployment"

# Grant IAM roles
gcloud projects add-iam-policy-binding $GCP_PROJECT_ID \
  --member="serviceAccount:${SA_EMAIL}" \
  --role="roles/run.admin"

gcloud projects add-iam-policy-binding $GCP_PROJECT_ID \
  --member="serviceAccount:${SA_EMAIL}" \
  --role="roles/artifactregistry.admin"

gcloud projects add-iam-policy-binding $GCP_PROJECT_ID \
  --member="serviceAccount:${SA_EMAIL}" \
  --role="roles/iam.serviceAccountUser"

gcloud projects add-iam-policy-binding $GCP_PROJECT_ID \
  --member="serviceAccount:${SA_EMAIL}" \
  --role="roles/storage.admin"

# Create and download service account key
gcloud iam service-accounts keys create ~/gcp-key.json \
  --iam-account=$SA_EMAIL

echo "Service account key created at: ~/gcp-key.json"
echo "Add this as GCP_SA_KEY secret in GitHub"
```

## 3. Create Artifact Registry Repository

```bash
gcloud artifacts repositories create diabetes-prediction \
  --repository-format=docker \
  --location=us-central1 \
  --description="Docker images for diabetes prediction API"
```

## 4. Configure GitHub Secrets

Add these secrets to your GitHub repository (Settings → Secrets → Actions):

1. **GCP_PROJECT_ID**: Your GCP project ID
   ```bash
   echo $GCP_PROJECT_ID
   ```

2. **GCP_SA_KEY**: Contents of service account JSON key
   ```bash
   cat ~/gcp-key.json
   # Copy the entire JSON content
   ```

3. **SLACK_WEBHOOK_URL** (optional): For deployment notifications

## 5. Manual Deployment (Optional)

Test deployment manually before automation:

```bash
# Build Docker image
docker build -t diabetes-prediction-api .

# Tag for GCP Artifact Registry
docker tag diabetes-prediction-api \
  us-central1-docker.pkg.dev/$GCP_PROJECT_ID/diabetes-prediction/api:latest

# Configure Docker authentication
gcloud auth configure-docker us-central1-docker.pkg.dev

# Push image
docker push us-central1-docker.pkg.dev/$GCP_PROJECT_ID/diabetes-prediction/api:latest

# Deploy to Cloud Run
gcloud run deploy diabetes-prediction-api \
  --image=us-central1-docker.pkg.dev/$GCP_PROJECT_ID/diabetes-prediction/api:latest \
  --platform=managed \
  --region=us-central1 \
  --allow-unauthenticated \
  --port=5002 \
  --memory=2Gi \
  --cpu=2 \
  --min-instances=0 \
  --max-instances=10 \
  --timeout=300 \
  --set-env-vars="FLASK_HOST=0.0.0.0,FLASK_PORT=5002"

# Get service URL
gcloud run services describe diabetes-prediction-api \
  --region=us-central1 \
  --format='value(status.url)'
```

## 6. Automated CI/CD Pipeline

Once GitHub secrets are configured, the pipeline automatically:

### On Push to Main:
1. **CI Tests** (`ci-tests.yml`):
   - Runs unit tests
   - Validates data schema
   - Checks model performance
   - Security scanning

2. **Docker Build** (`docker-build-push.yml`):
   - Builds Docker image
   - Pushes to GitHub Container Registry
   - Pushes to GCP Artifact Registry

3. **Deploy to GCP** (`deploy-gcp.yml`):
   - Deploys to Cloud Run
   - Runs health checks
   - Sends Slack notification

### On Schedule (Monday 2 AM):
- **Model Retraining** (`model-training.yml`):
  - Trains model with latest data
  - Validates performance
  - Commits updated metrics

## 7. Access Your Deployment

After successful deployment:

```bash
# Get service URL
export SERVICE_URL=$(gcloud run services describe diabetes-prediction-api \
  --region=us-central1 \
  --format='value(status.url)')

# Test endpoints
curl $SERVICE_URL/health
curl $SERVICE_URL/info

# Make prediction
curl -X POST $SERVICE_URL/predict \
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

# Open Web UI in browser
echo "Web UI: $SERVICE_URL"
```

## 8. Monitoring & Logs

### View Cloud Run logs:
```bash
gcloud run services logs read diabetes-prediction-api \
  --region=us-central1 \
  --limit=50
```

### Monitor in Console:
- **Cloud Run Dashboard**: https://console.cloud.google.com/run
- **Logs Explorer**: https://console.cloud.google.com/logs
- **Monitoring**: https://console.cloud.google.com/monitoring

## 9. Cost Optimization

### Cloud Run Pricing:
- **Free tier**: 2 million requests/month, 360,000 GB-seconds
- **Beyond free tier**: ~$0.24 per million requests

### Optimization tips:
```bash
# Set aggressive scaling
gcloud run services update diabetes-prediction-api \
  --region=us-central1 \
  --min-instances=0 \
  --max-instances=5 \
  --concurrency=80

# Reduce memory if possible
gcloud run services update diabetes-prediction-api \
  --region=us-central1 \
  --memory=1Gi

# Set budget alerts
gcloud billing budgets create \
  --billing-account=YOUR_BILLING_ACCOUNT \
  --display-name="MLOps Budget Alert" \
  --budget-amount=50USD \
  --threshold-rule=percent=80
```

## 10. Cleanup

To avoid charges, delete resources when done:

```bash
# Delete Cloud Run service
gcloud run services delete diabetes-prediction-api \
  --region=us-central1

# Delete Artifact Registry repository
gcloud artifacts repositories delete diabetes-prediction \
  --location=us-central1

# Delete service account
gcloud iam service-accounts delete $SA_EMAIL

# Delete project (if no longer needed)
gcloud projects delete $GCP_PROJECT_ID
```

## Troubleshooting

### Error: "Permission denied"
- Check service account IAM roles
- Verify `GCP_SA_KEY` secret is correctly formatted JSON

### Error: "Container failed to start"
- Check Cloud Run logs: `gcloud run services logs read ...`
- Verify Dockerfile exposes port 5002
- Test image locally: `docker run -p 5002:5002 <image>`

### Error: "Out of memory"
- Increase memory allocation
- Optimize model size (reduce n_estimators)
- Use model compression techniques

### Deployment takes too long
- Enable concurrent builds in Cloud Build
- Use Docker layer caching
- Reduce Docker image size

## Security Considerations

1. **Authentication**: Enable Cloud Run authentication for production
   ```bash
   gcloud run services update diabetes-prediction-api \
     --region=us-central1 \
     --no-allow-unauthenticated
   ```

2. **VPC Connector**: Deploy in private VPC for sensitive data

3. **Secret Manager**: Store API keys and credentials securely
   ```bash
   gcloud secrets create mlflow-tracking-uri \
     --data-file=-
   ```

4. **IAM**: Use least-privilege principle for service accounts

5. **Audit Logs**: Enable Cloud Audit Logs for compliance

## Next Steps

- [ ] Set up custom domain with Cloud DNS
- [ ] Configure Cloud CDN for static assets
- [ ] Implement Cloud Armor for DDoS protection
- [ ] Add Cloud Monitoring alerts
- [ ] Set up Cloud Trace for performance monitoring
- [ ] Implement Blue/Green deployments
- [ ] Add A/B testing with Traffic Splitting
