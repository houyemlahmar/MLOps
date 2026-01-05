# 🚀 CI/CD Pipeline - Quick Start

## What's Been Implemented

Your Diabetes Prediction MLOps project now has a **complete CI/CD pipeline** with:

### ✅ 4 GitHub Actions Workflows

1. **`.github/workflows/ci-tests.yml`** - Continuous Integration
   - Automated testing on every push/PR
   - Data validation
   - Model performance checks
   - Security scanning
   - Slack notifications

2. **`.github/workflows/model-training.yml`** - Automated Retraining
   - Scheduled weekly retraining (Mondays 2 AM)
   - Manual trigger option
   - Performance validation
   - Artifact storage

3. **`.github/workflows/docker-build-push.yml`** - Container Build
   - Multi-platform Docker builds
   - Push to GitHub Container Registry
   - Push to GCP Artifact Registry
   - Image versioning

4. **`.github/workflows/deploy-gcp.yml`** - Cloud Deployment
   - Deploy to GCP Cloud Run
   - Health checks
   - Automatic rollback
   - Deployment notifications

---

## 🎯 Getting Started (3 Steps)

### Step 1: Setup GCP (10 minutes)

**Option A: Automated Setup (Recommended)**
```bash
# Make script executable
chmod +x scripts/setup-gcp.sh

# Run setup (replace with your project ID)
./scripts/setup-gcp.sh your-gcp-project-id
```

**Option B: Manual Setup**
Follow the detailed guide: `.github/GCP_DEPLOYMENT.md`

### Step 2: Configure GitHub Secrets (5 minutes)

Go to your GitHub repository:
**Settings → Secrets and variables → Actions → New repository secret**

Add these secrets:
1. **GCP_PROJECT_ID**: Your GCP project ID
2. **GCP_SA_KEY**: Service account JSON key (from Step 1)
3. **SLACK_WEBHOOK_URL** (optional): For deployment notifications

See `.github/SECRETS_SETUP.md` for detailed instructions.

### Step 3: Push to Trigger Pipeline (1 minute)

```bash
# Stage all changes
git add .

# Commit with descriptive message
git commit -m "feat: Add CI/CD pipeline with GCP deployment"

# Push to trigger workflows
git push origin main
```

**Monitor your pipeline:**
- GitHub Actions tab: Check workflow runs
- GCP Console: View Cloud Run deployments
- Slack: Receive notifications (if configured)

---

## 📊 CI/CD Workflow Triggers

| Workflow | Trigger | Purpose |
|----------|---------|---------|
| **CI Tests** | Push/PR to main/develop | Validate code quality |
| **Model Training** | Mondays 2 AM UTC / Manual | Retrain model weekly |
| **Docker Build** | Push to main / Tags | Build & push images |
| **GCP Deploy** | After Docker build | Deploy to Cloud Run |

---

## 🔧 Customization

### Change Retraining Schedule

Edit `.github/workflows/model-training.yml`:
```yaml
schedule:
  # Change to daily at midnight
  - cron: '0 0 * * *'
```

### Change GCP Region

Edit `.github/workflows/deploy-gcp.yml`:
```yaml
env:
  GCP_REGION: europe-west1  # Change from us-central1
```

### Adjust Cloud Run Resources

In `deploy-gcp.yml`, modify:
```yaml
--memory=2Gi      # Increase to 4Gi
--cpu=2           # Increase to 4
--max-instances=10  # Increase to 20
```

### Add More Tests

Create new test files in `tests/` directory:
```python
# tests/test_model.py
def test_model_accuracy():
    # Your test code
    pass
```

---

## 📈 Monitoring & Logs

### View GitHub Actions Logs
```
https://github.com/YOUR_USERNAME/YOUR_REPO/actions
```

### View Cloud Run Logs
```bash
gcloud run services logs read diabetes-prediction-api \
  --region=us-central1 \
  --limit=100
```

### View in GCP Console
- **Cloud Run**: https://console.cloud.google.com/run
- **Logs**: https://console.cloud.google.com/logs
- **Monitoring**: https://console.cloud.google.com/monitoring

---

## 🐛 Troubleshooting

### Workflow Fails: "Permission denied"
- Check GitHub secrets are correctly set
- Verify service account has required IAM roles
- Ensure GCP APIs are enabled

### Docker Build Fails
- Check Dockerfile syntax
- Verify all dependencies in requirements.txt
- Test build locally: `docker build -t test .`

### Deployment Fails: "Container failed to start"
- Check Cloud Run logs for errors
- Verify port 5002 is exposed in Dockerfile
- Test container locally: `docker run -p 5002:5002 <image>`

### Model Performance Below Threshold
- Check data quality in `data/raw/diabetes.csv`
- Adjust hyperparameters in `params.yaml`
- Review feature engineering in notebooks

---

## 💰 Cost Estimate

### GCP Cloud Run (Free Tier Includes):
- 2 million requests/month
- 360,000 GB-seconds memory
- 180,000 vCPU-seconds

**Beyond free tier:**
- ~$0.24 per million requests
- ~$0.00002400 per GB-second
- ~$0.00001000 per vCPU-second

**Estimated monthly cost (1000 req/day):**
- API requests: **Free** (within free tier)
- Compute: **< $5/month**
- Storage: **< $1/month**
- **Total: < $6/month**

### GitHub Actions (Free Tier):
- Public repos: **Unlimited minutes**
- Private repos: **2000 minutes/month free**

---

## 🎉 What You've Achieved

✅ **Professional MLOps Pipeline**
- Automated testing and validation
- Continuous integration and deployment
- Cloud-native architecture
- Production-ready monitoring

✅ **Enterprise Best Practices**
- Infrastructure as Code
- GitOps workflow
- Automated retraining
- Security scanning

✅ **Portfolio-Ready Project**
- Complete documentation
- CI/CD badges
- Cloud deployment
- Professional structure

---

## 📚 Additional Resources

- **GitHub Actions Docs**: https://docs.github.com/en/actions
- **GCP Cloud Run Docs**: https://cloud.google.com/run/docs
- **MLflow Docs**: https://mlflow.org/docs/latest
- **DVC Docs**: https://dvc.org/doc

---

## 🚀 Next Steps

1. ✅ **Push code** to trigger first deployment
2. ✅ **Test deployed API** at Cloud Run URL
3. ✅ **Monitor workflows** in GitHub Actions
4. 📊 **Add monitoring** (Prometheus/Grafana)
5. 🔄 **Implement A/B testing**
6. 📈 **Add model drift detection**
7. 🎯 **Set up Airflow** for complex workflows (when needed)

---

**Need Help?**
- Check `.github/GCP_DEPLOYMENT.md` for detailed instructions
- Review `.github/SECRETS_SETUP.md` for secret configuration
- See workflow YAML files for pipeline details

**🎉 Congratulations! Your MLOps pipeline is ready for production!**
