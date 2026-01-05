#!/bin/bash

# 🚀 Quick GCP Setup Script for Diabetes Prediction MLOps
# This script automates the initial GCP setup for CI/CD deployment

set -e  # Exit on error

echo "🚀 Diabetes Prediction MLOps - GCP Setup"
echo "=========================================="
echo ""

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if gcloud is installed
if ! command -v gcloud &> /dev/null; then
    echo -e "${RED}❌ Error: gcloud CLI not found${NC}"
    echo "Install from: https://cloud.google.com/sdk/docs/install"
    exit 1
fi

echo -e "${GREEN}✅ gcloud CLI found${NC}"

# Get or prompt for project ID
if [ -z "$1" ]; then
    echo -e "${YELLOW}Enter your GCP Project ID (or press Enter to use current):${NC}"
    read PROJECT_ID
    if [ -z "$PROJECT_ID" ]; then
        PROJECT_ID=$(gcloud config get-value project 2>/dev/null)
    fi
else
    PROJECT_ID=$1
fi

if [ -z "$PROJECT_ID" ]; then
    echo -e "${RED}❌ Error: No project ID provided${NC}"
    exit 1
fi

echo ""
echo "Using Project ID: $PROJECT_ID"
echo ""

# Set project
gcloud config set project $PROJECT_ID

# Enable required APIs
echo "📦 Enabling required GCP APIs..."
gcloud services enable run.googleapis.com --quiet
gcloud services enable artifactregistry.googleapis.com --quiet
gcloud services enable containerregistry.googleapis.com --quiet
gcloud services enable cloudbuild.googleapis.com --quiet
echo -e "${GREEN}✅ APIs enabled${NC}"
echo ""

# Create service account
SA_NAME="github-actions"
SA_EMAIL="${SA_NAME}@${PROJECT_ID}.iam.gserviceaccount.com"

echo "👤 Creating service account: $SA_NAME"
gcloud iam service-accounts create $SA_NAME \
  --display-name="GitHub Actions Deployment" \
  --quiet || echo "Service account already exists"

# Grant IAM roles
echo "🔐 Granting IAM roles..."
roles=(
  "roles/run.admin"
  "roles/artifactregistry.admin"
  "roles/iam.serviceAccountUser"
  "roles/storage.admin"
)

for role in "${roles[@]}"; do
  gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:${SA_EMAIL}" \
    --role="$role" \
    --quiet
done
echo -e "${GREEN}✅ IAM roles granted${NC}"
echo ""

# Create Artifact Registry repository
echo "📦 Creating Artifact Registry repository..."
gcloud artifacts repositories create diabetes-prediction \
  --repository-format=docker \
  --location=us-central1 \
  --description="Docker images for diabetes prediction API" \
  --quiet || echo "Repository already exists"
echo -e "${GREEN}✅ Artifact Registry created${NC}"
echo ""

# Create service account key
KEY_FILE="gcp-sa-key-${PROJECT_ID}.json"
echo "🔑 Creating service account key..."
gcloud iam service-accounts keys create $KEY_FILE \
  --iam-account=$SA_EMAIL \
  --quiet

echo -e "${GREEN}✅ Service account key created: $KEY_FILE${NC}"
echo ""

# Display setup summary
echo "=========================================="
echo "✅ GCP Setup Complete!"
echo "=========================================="
echo ""
echo "📋 Next Steps:"
echo ""
echo "1. Add these secrets to GitHub repository:"
echo "   Settings → Secrets and variables → Actions → New repository secret"
echo ""
echo "   Secret Name: GCP_PROJECT_ID"
echo "   Value: $PROJECT_ID"
echo ""
echo "   Secret Name: GCP_SA_KEY"
echo "   Value: <contents of $KEY_FILE>"
echo ""
echo "2. View service account key:"
echo "   cat $KEY_FILE"
echo ""
echo "3. (Optional) Add Slack webhook URL:"
echo "   Secret Name: SLACK_WEBHOOK_URL"
echo "   Value: <your Slack webhook URL>"
echo ""
echo "4. Push code to trigger CI/CD pipeline:"
echo "   git add ."
echo "   git commit -m 'feat: Add CI/CD pipeline'"
echo "   git push origin main"
echo ""
echo "5. Monitor deployment:"
echo "   - GitHub Actions: https://github.com/YOUR_USERNAME/YOUR_REPO/actions"
echo "   - GCP Console: https://console.cloud.google.com/run?project=$PROJECT_ID"
echo ""
echo "⚠️  IMPORTANT: Delete $KEY_FILE after adding to GitHub secrets!"
echo "   rm $KEY_FILE"
echo ""
echo "📖 Full documentation: .github/GCP_DEPLOYMENT.md"
echo ""
