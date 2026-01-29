# Deployment Guide: AWS ECR & ECS

This guide explains how to deploy the Spotify Recommender System to AWS using Elastic Container Registry (ECR) and Elastic Container Service (ECS).

## Prerequisites
- AWS CLI installed and configured (`aws configure`).
- Docker installed and running.

## 1. Create ECR Repositories
You need two repositories: one for the backend, one for the frontend.

```bash
aws ecr create-repository --repository-name spotify-backend
aws ecr create-repository --repository-name spotify-frontend
```

## 2. Authenticate Docker to ECR
Replace `us-east-1` and `123456789012` with your region and Account ID.

```bash
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 123456789012.dkr.ecr.us-east-1.amazonaws.com
```

## 3. Build and Tag Images

### Backend
```bash
# Build
docker build -t spotify-backend -f Dockerfile.backend .

# Tag
docker tag spotify-backend:latest 123456789012.dkr.ecr.us-east-1.amazonaws.com/spotify-backend:latest

# Push
docker push 123456789012.dkr.ecr.us-east-1.amazonaws.com/spotify-backend:latest
```

### Frontend
**Important**: When building the frontend for production, you must set the `VITE_API_URL` to your production backend load balancer URL (ALB) or public IP.

```bash
# Build (Replace URL with your future ALB URL)
docker build -t spotify-frontend -f Dockerfile.frontend --build-arg VITE_API_URL=http://your-alb-url.com .

# Tag
docker tag spotify-frontend:latest 123456789012.dkr.ecr.us-east-1.amazonaws.com/spotify-frontend:latest

# Push
docker push 123456789012.dkr.ecr.us-east-1.amazonaws.com/spotify-frontend:latest
```

## 4. Deploy to ECS (Fargate)

### Task Definition
1.  Go to **ECS > Task Definitions > Create new Task Definition**.
2.  Select **Fargate**.
3.  Add Container 1 (Backend): Use the ECR URI for `spotify-backend`. Port Map: `8000`.
4.  Add Container 2 (Frontend): Use the ECR URI for `spotify-frontend`. Port Map: `5173`.
5.  **Environment Variables**: Add `SPOTIFY_CLIENT_ID` and `SPOTIFY_CLIENT_SECRET` to the Backend container.

### Service
1.  Create a **Cluster**.
2.  Create a **Service** using the Task Definition.
3.  **Load Balancer**: Create an Application Load Balancer (ALB).
    - Listener 80 -> Target Group (Frontend Port 5173).
    - Listener 8000 -> Target Group (Backend Port 8000).

## 5. MLOps Considerations
- **Data Persistence**: With Fargate, the filesystem is ephemeral. If you restart the task, the user feedback (`feedback_data.csv`) will be lost.
- **Solution**: Use **AWS EFS (Elastic File System)** mounted to `/app/data` in the Backend container to persist feedback and model files across restarts.
