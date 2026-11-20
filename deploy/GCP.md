# One-time Google Cloud setup (about 20 minutes, fits the free tier / trial credit)

Replace `PROJECT` and `BUCKET`. Region `asia-south1` (Mumbai).

```bash
gcloud config set project PROJECT
gcloud services enable run.googleapis.com artifactregistry.googleapis.com iamcredentials.googleapis.com logging.googleapis.com

# 1. Container registry + bucket for models, data and prediction logs
gcloud artifacts repositories create ml --repository-format=docker --location=asia-south1
gcloud storage buckets create gs://BUCKET --location=asia-south1

# 2. Deploy service account
gcloud iam service-accounts create gh-deployer
SA=gh-deployer@PROJECT.iam.gserviceaccount.com
for role in roles/run.admin roles/artifactregistry.writer roles/storage.objectAdmin roles/iam.serviceAccountUser; do
  gcloud projects add-iam-policy-binding PROJECT --member="serviceAccount:$SA" --role="$role"
done

# 3. Workload Identity Federation: GitHub Actions gets short-lived tokens, no JSON keys in secrets
gcloud iam workload-identity-pools create github --location=global
gcloud iam workload-identity-pools providers create-oidc gh --location=global --workload-identity-pool=github \
  --issuer-uri="https://token.actions.githubusercontent.com" \
  --attribute-mapping="google.subject=assertion.sub,attribute.repository=assertion.repository" \
  --attribute-condition="assertion.repository=='SharbeshS8199/mlops-churn-pipeline'"
POOL=$(gcloud iam workload-identity-pools describe github --location=global --format='value(name)')
gcloud iam service-accounts add-iam-policy-binding $SA --role=roles/iam.workloadIdentityUser \
  --member="principalSet://iam.googleapis.com/$POOL/attribute.repository/SharbeshS8199/mlops-churn-pipeline"

# 4. Prediction logs: Cloud Run stdout -> Cloud Logging -> hourly files in GCS (read by drift-monitor.yml)
gcloud logging sinks create churn-predictions storage.googleapis.com/BUCKET \
  --log-filter='resource.type="cloud_run_revision" AND resource.labels.service_name="churn-api" AND jsonPayload.prediction:*'
```

The sink writes hourly JSON files under `gs://BUCKET/run.googleapis.com/stdout/...`. A small
scheduled job (or `gcloud storage cat ... | jq -c .jsonPayload`) concatenates the last N days into
`gs://BUCKET/logs/predictions.jsonl`, which is the file `drift-monitor.yml` reads.

**GitHub → Settings → Secrets and variables → Actions → Variables:**

| Variable | Value |
|---|---|
| `GCP_PROJECT` | PROJECT |
| `GCS_BUCKET` | BUCKET |
| `GCP_DEPLOY_SA` | gh-deployer@PROJECT.iam.gserviceaccount.com |
| `GCP_WIF_PROVIDER` | output of `gcloud iam workload-identity-pools providers describe gh --location=global --workload-identity-pool=github --format='value(name)'` |

**AWS instead?** The same container runs on AWS App Runner. Swap the `auth`/`deploy-cloudrun` steps
for `aws-actions/configure-aws-credentials` (OIDC role) + push to ECR + `aws apprunner start-deployment`,
and use S3 instead of GCS.
