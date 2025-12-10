# ☁️ Deployment Guide: AWS App Runner (Serverless)

This guide explains how to deploy the FinGraph AML Dashboard to **AWS App Runner**.
App Runner is the best choice because:
1.  **It's Serverless:** You don't manage servers (EC2).
2.  **It scales to zero:** It pauses when no one is using it (saving money).
3.  **It supports WebSockets:** Required for Streamlit.

---

## ✅ Prerequisites
1.  **AWS Account** (You have $100 credits).
2.  **AWS CLI** installed and configured (`aws configure`).
3.  **Docker** installed and running.

---

## 🚀 Step 1: Create a Repository (ECR)
We need a place to store your Docker image.

> **Tip:** If you are the project owner, check `docs/INTERNAL_DEPLOYMENT_NOTES.md` (git-ignored) for your specific account IDs and quick-run commands.

```bash
# 1. Create the repo
aws ecr create-repository --repository-name fingraph-aml

# 2. Login Docker to AWS (Replace region if needed, e.g., us-east-1)
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin <YOUR_ACCOUNT_ID>.dkr.ecr.us-east-1.amazonaws.com
```

---

## 📦 Step 2: Build & Push Image
We will build the image on your laptop and upload it to AWS.

```bash
# 1. Build the image (Make sure you are in the project root)
docker build -t fingraph-aml -f docker/Dockerfile.app .

# 2. Tag it for AWS
docker tag fingraph-aml:latest <YOUR_ACCOUNT_ID>.dkr.ecr.us-east-1.amazonaws.com/fingraph-aml:latest

# 3. Push it
docker push <YOUR_ACCOUNT_ID>.dkr.ecr.us-east-1.amazonaws.com/fingraph-aml:latest
```

---

## 🌐 Step 3: Deploy to App Runner
1.  Go to the **AWS Console** -> Search for **"App Runner"**.
2.  Click **"Create Service"**.
3.  **Source:** Select "Container Registry" -> "Amazon ECR".
4.  **Image:** Browse and select `fingraph-aml:latest`.
5.  **Deployment settings:** "Manual" (easiest for now).
6.  **Configuration:**
    *   **Port:** `8501` (Important! Streamlit runs on this port).
    *   **CPU/Memory:** 1 vCPU / 2 GB (Cheapest option).
7.  **Environment Variables:**
    *   Key: `PYTHONPATH` | Value: `/app/src`
8.  Click **"Create & Deploy"**.

---

## 💰 Cost Saving Tips
*   **App Runner** charges per second of active use.
*   **Pause it:** If you are not showing it to a recruiter, you can "Pause" the service in the console to stop billing.
*   **Auto-Pause:** It automatically scales down to the minimum provisioned instances when idle.

---

## 🔗 Your Live URL
Once deployed (takes ~5 mins), AWS will give you a URL like:
`https://fingraph-xyz.awsapprunner.com`

You can send this link to anyone!
