# F1 ML API - Deployment Guide

## 🚀 Production Deployment Options

### Option A: Docker Container Deployment

#### 1. Local Docker Deployment
```bash
# Build Docker image
docker build -t f1-ml-api:latest .

# Run container
docker run -d -p 8080:8080 --name f1-ml-api f1-ml-api:latest

# Test API
curl http://localhost:8080/health
```

#### 2. Docker Compose Deployment
```bash
# Start services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

#### 3. Cloud Container Services

##### AWS ECS Deployment
```bash
# Create ECR repository
aws ecr create-repository --repository-name f1-ml-api

# Build and push image
docker build -t f1-ml-api:latest .
docker tag f1-ml-api:latest <account-id>.dkr.ecr.<region>.amazonaws.com/f1-ml-api:latest
docker push <account-id>.dkr.ecr.<region>.amazonaws.com/f1-ml-api:latest

# Deploy to ECS using AWS CLI or Console
```

##### Google Cloud Run Deployment
```bash
# Build and push to Google Container Registry
docker build -t gcr.io/<project-id>/f1-ml-api:latest .
docker push gcr.io/<project-id>/f1-ml-api:latest

# Deploy to Cloud Run
gcloud run deploy f1-ml-api \
    --image gcr.io/<project-id>/f1-ml-api:latest \
    --platform managed \
    --region us-central1 \
    --allow-unauthenticated
```

##### Azure Container Instances
```bash
# Build and push to Azure Container Registry
az acr build --registry <registry-name> --image f1-ml-api:latest .

# Deploy to Container Instances
az container create \
    --resource-group <resource-group> \
    --name f1-ml-api \
    --image <registry-name>.azurecr.io/f1-ml-api:latest \
    --cpu 1 --memory 2 \
    --ports 8080 \
    --dns-name-label f1-ml-api
```

### Option B: Kubernetes Deployment

#### 1. Local Kubernetes (minikube/Docker Desktop)
```bash
# Apply deployments
kubectl apply -f k8s-deployment.yaml
kubectl apply -f k8s-service.yaml

# Check status
kubectl get pods
kubectl get services

# Port forward for testing
kubectl port-forward service/f1-ml-api-service 8080:80
```

#### 2. Cloud Kubernetes Services

##### AWS EKS
```bash
# Create EKS cluster
eksctl create cluster --name f1-ml-cluster --region us-west-2

# Deploy application
kubectl apply -f k8s-deployment.yaml
kubectl apply -f k8s-service.yaml
```

##### Google GKE
```bash
# Create GKE cluster
gcloud container clusters create f1-ml-cluster --zone us-central1-a

# Deploy application
kubectl apply -f k8s-deployment.yaml
kubectl apply -f k8s-service.yaml
```

### Option C: Serverless Deployment

#### AWS Lambda with API Gateway
- Package API as Lambda function
- Use AWS SAM or Serverless framework
- Configure API Gateway for HTTP endpoints
- Handle cold starts and memory limits

#### Google Cloud Functions
- Deploy individual endpoints as functions
- Use Cloud Functions Framework
- Configure HTTP triggers
- Manage function-to-function communication

## 🔧 Configuration

### Environment Variables
```bash
PYTHONPATH=/app
PYTHONDONTWRITEBYTECODE=1
PYTHONUNBUFFERED=1
```

### Resource Requirements
- **Memory**: 1GB recommended (512MB minimum)
- **CPU**: 1 vCPU recommended (0.5 vCPU minimum)
- **Disk**: 2GB for models and dependencies
- **Network**: HTTP/HTTPS on port 8080

### Health Checks
- **Endpoint**: `/health`
- **Interval**: 30 seconds
- **Timeout**: 10 seconds
- **Retries**: 3

## 📊 Monitoring and Logging

### API Endpoints for Monitoring
- `GET /health` - Health status
- `GET /model_info` - Model information
- `GET /` - Basic service info

### Logging Configuration
- Application logs: INFO level
- Access logs: Enabled
- Error logs: ERROR level with stack traces

### Metrics to Monitor
- Request rate (requests/second)
- Response time (95th percentile)
- Error rate (4xx/5xx responses)
- Model prediction latency
- Memory and CPU usage

## 🔒 Security Considerations

### API Security
- Implement authentication (API keys, JWT tokens)
- Add rate limiting
- Input validation and sanitization
- HTTPS only in production

### Container Security
- Use minimal base images
- Scan for vulnerabilities
- Non-root user execution
- Read-only file system where possible

## 🚀 Scaling and Performance

### Horizontal Scaling
- Use load balancers
- Configure auto-scaling groups
- Monitor resource utilization

### Performance Optimization
- Model caching strategies
- Connection pooling
- Request batching
- Asynchronous processing

## 🧪 Testing in Production

### Smoke Tests
```bash
# Health check
curl -f http://your-api-url/health

# Model info
curl http://your-api-url/model_info

# Sample prediction
curl -X POST http://your-api-url/predict_full \
  -H "Content-Type: application/json" \
  -d @sample_request.json
```

### Load Testing
```bash
# Using Apache Bench
ab -n 1000 -c 10 http://your-api-url/health

# Using Artillery
artillery run load-test-config.yml
```

## 🔄 CI/CD Pipeline

### GitHub Actions Example
```yaml
name: Deploy F1 ML API
on:
  push:
    branches: [main]
jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Build and Deploy
        run: |
          docker build -t f1-ml-api:latest .
          # Add deployment commands
```

## 📋 Troubleshooting

### Common Issues
1. **Models not loading**: Check file paths and permissions
2. **Memory errors**: Increase container memory limits
3. **Slow predictions**: Optimize model serialization
4. **Connection timeouts**: Adjust health check intervals

### Debug Commands
```bash
# Check container logs
docker logs f1-ml-api-container

# Debug inside container
docker exec -it f1-ml-api-container /bin/bash

# Check model files
ls -la /app/models/
```

---
*F1 ML API Deployment Guide - Generated for Phase 10 Production Deployment*
