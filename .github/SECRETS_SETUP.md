# 🔐 GitHub Secrets Configuration

To enable CI/CD with GCP deployment, configure these secrets in your GitHub repository:

## Required Secrets

### 1. GCP Authentication
Go to **Settings → Secrets and variables → Actions → New repository secret**

#### `GCP_PROJECT_ID`
- **Value:** Your GCP Project ID (e.g., `my-mlops-project-12345`)
- **How to find:** 
  ```bash
  gcloud config get-value project
  ```

#### `GCP_SA_KEY`
- **Value:** Service Account JSON key
- **How to create:**
  
  ```bash
  # 1. Create service account
  gcloud iam service-accounts create github-actions \
    --display-name="GitHub Actions Service Account"
  
  # 2. Grant necessary roles
  gcloud projects add-iam-policy-binding $GCP_PROJECT_ID \
    --member="serviceAccount:github-actions@$GCP_PROJECT_ID.iam.gserviceaccount.com" \
    --role="roles/run.admin"
  
  gcloud projects add-iam-policy-binding $GCP_PROJECT_ID \
    --member="serviceAccount:github-actions@$GCP_PROJECT_ID.iam.gserviceaccount.com" \
    --role="roles/artifactregistry.admin"
  
  gcloud projects add-iam-policy-binding $GCP_PROJECT_ID \
    --member="serviceAccount:github-actions@$GCP_PROJECT_ID.iam.gserviceaccount.com" \
    --role="roles/iam.serviceAccountUser"
  
  # 3. Create and download key
  gcloud iam service-accounts keys create key.json \
    --iam-account=github-actions@$GCP_PROJECT_ID.iam.gserviceaccount.com
  
  # 4. Copy the contents of key.json and paste as secret value
  cat key.json
  
  # 5. Delete the local key file for security
  rm key.json
  ```

### 2. Slack Notifications (Optional)

#### `SLACK_WEBHOOK_URL`
- **Value:** Slack Incoming Webhook URL
- **How to create:**
  1. Go to https://api.slack.com/apps
  2. Create new app → "From scratch"
  3. Enable "Incoming Webhooks"
  4. Add New Webhook to Workspace
  5. Copy Webhook URL (e.g., `https://hooks.slack.com/services/T00000000/B00000000/XXXXXXXXXXXX`)

## Verification

After adding secrets, verify they are set correctly:

```bash
# Check if GCP authentication works
gcloud auth activate-service-account --key-file=key.json
gcloud projects list

# Test Cloud Run permissions
gcloud run services list --region=us-central1
```

## Security Best Practices

1. **Never commit secrets** to version control
2. **Use least privilege** - grant only necessary IAM roles
3. **Rotate service account keys** regularly (every 90 days)
4. **Enable Secret Scanning** in GitHub repository settings
5. **Use environment-specific secrets** for dev/staging/prod

## Troubleshooting

### Error: "Permission denied"
- Verify service account has required roles
- Check IAM policy bindings with: `gcloud projects get-iam-policy $GCP_PROJECT_ID`

### Error: "Service account key not found"
- Ensure JSON key is properly formatted
- Check for extra spaces or newlines in secret value

### Error: "Cloud Run API not enabled"
```bash
gcloud services enable run.googleapis.com
gcloud services enable artifactregistry.googleapis.com
gcloud services enable containerregistry.googleapis.com
```

## Next Steps

Once secrets are configured:
1. Push code to trigger CI pipeline
2. Monitor workflow in GitHub Actions tab
3. Check deployment at Cloud Run URL
4. Review Slack notifications (if configured)
