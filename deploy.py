#!/usr/bin/env python3
"""
PHASE 10.4: Deployment Guide and Scripts
Production deployment utilities for F1 ML API
"""

import subprocess
import sys
import json
from pathlib import Path

def build_docker_image():
    """Build Docker image for F1 ML API"""
    
    print("🔨 Building Docker image for F1 ML API...")
    
    try:
        # Build Docker image
        result = subprocess.run([
            'docker', 'build', 
            '-t', 'f1-ml-api:latest',
            '-t', 'f1-ml-api:1.0.0',
            '.'
        ], capture_output=True, text=True, cwd='.')
        
        if result.returncode == 0:
            print("✅ Docker image built successfully!")
            print("   Tags: f1-ml-api:latest, f1-ml-api:1.0.0")
            return True
        else:
            print(f"❌ Docker build failed: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Docker build error: {e}")
        return False

def run_docker_container():
    """Run Docker container locally for testing"""
    
    print("🚀 Running Docker container locally...")
    
    try:
        # Run container
        result = subprocess.run([
            'docker', 'run',
            '-d',  # Detached mode
            '-p', '8080:8080',  # Port mapping
            '--name', 'f1-ml-api-container',
            'f1-ml-api:latest'
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            container_id = result.stdout.strip()
            print(f"✅ Container started successfully!")
            print(f"   Container ID: {container_id[:12]}")
            print(f"   API available at: http://localhost:8080")
            print(f"   API docs at: http://localhost:8080/docs")
            return container_id
        else:
            print(f"❌ Container start failed: {result.stderr}")
            return None
            
    except Exception as e:
        print(f"❌ Container run error: {e}")
        return None

def test_docker_deployment():
    """Test the deployed Docker container"""
    
    print("🧪 Testing Docker deployment...")
    
    import time
    import requests
    
    # Wait for container to be ready
    print("   Waiting for container to start...")
    time.sleep(5)
    
    try:
        # Test health endpoint
        response = requests.get("http://localhost:8080/health", timeout=10)
        if response.status_code == 200:
            health_data = response.json()
            print("✅ Health check passed!")
            print(f"   Models loaded: {health_data.get('models_loaded', False)}")
            print(f"   Model version: {health_data.get('model_version', 'unknown')}")
            return True
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Docker test error: {e}")
        return False

def stop_docker_container():
    """Stop and remove Docker container"""
    
    print("🛑 Stopping Docker container...")
    
    try:
        # Stop container
        subprocess.run(['docker', 'stop', 'f1-ml-api-container'], 
                      capture_output=True, text=True)
        
        # Remove container
        subprocess.run(['docker', 'rm', 'f1-ml-api-container'], 
                      capture_output=True, text=True)
        
        print("✅ Container stopped and removed")
        return True
        
    except Exception as e:
        print(f"❌ Container stop error: {e}")
        return False

def generate_deployment_configs():
    """Generate deployment configuration files"""
    
    print("📋 Generating deployment configurations...")
    
    # Docker Compose file
    docker_compose = {
        'version': '3.8',
        'services': {
            'f1-ml-api': {
                'build': '.',
                'ports': ['8080:8080'],
                'environment': [
                    'PYTHONPATH=/app',
                    'PYTHONDONTWRITEBYTECODE=1',
                    'PYTHONUNBUFFERED=1'
                ],
                'volumes': [
                    './models:/app/models:ro',
                    './data:/app/data:ro'
                ],
                'healthcheck': {
                    'test': ['CMD', 'curl', '-f', 'http://localhost:8080/health'],
                    'interval': '30s',
                    'timeout': '10s',
                    'retries': 3,
                    'start_period': '5s'
                },
                'restart': 'unless-stopped'
            }
        }
    }
    
    # Save Docker Compose
    with open('docker-compose.yml', 'w') as f:
        import yaml
        yaml.dump(docker_compose, f, default_flow_style=False)
    
    print("✅ docker-compose.yml created")
    
    # Kubernetes deployment
    k8s_deployment = {
        'apiVersion': 'apps/v1',
        'kind': 'Deployment',
        'metadata': {
            'name': 'f1-ml-api',
            'labels': {'app': 'f1-ml-api'}
        },
        'spec': {
            'replicas': 3,
            'selector': {'matchLabels': {'app': 'f1-ml-api'}},
            'template': {
                'metadata': {'labels': {'app': 'f1-ml-api'}},
                'spec': {
                    'containers': [{
                        'name': 'f1-ml-api',
                        'image': 'f1-ml-api:1.0.0',
                        'ports': [{'containerPort': 8080}],
                        'env': [
                            {'name': 'PYTHONPATH', 'value': '/app'},
                            {'name': 'PYTHONDONTWRITEBYTECODE', 'value': '1'},
                            {'name': 'PYTHONUNBUFFERED', 'value': '1'}
                        ],
                        'livenessProbe': {
                            'httpGet': {'path': '/health', 'port': 8080},
                            'initialDelaySeconds': 30,
                            'periodSeconds': 10
                        },
                        'readinessProbe': {
                            'httpGet': {'path': '/health', 'port': 8080},
                            'initialDelaySeconds': 5,
                            'periodSeconds': 5
                        },
                        'resources': {
                            'requests': {'memory': '512Mi', 'cpu': '250m'},
                            'limits': {'memory': '1Gi', 'cpu': '500m'}
                        }
                    }]
                }
            }
        }
    }
    
    # Save Kubernetes deployment
    with open('k8s-deployment.yaml', 'w') as f:
        import yaml
        yaml.dump(k8s_deployment, f, default_flow_style=False)
    
    print("✅ k8s-deployment.yaml created")
    
    # Kubernetes service
    k8s_service = {
        'apiVersion': 'v1',
        'kind': 'Service',
        'metadata': {
            'name': 'f1-ml-api-service',
            'labels': {'app': 'f1-ml-api'}
        },
        'spec': {
            'selector': {'app': 'f1-ml-api'},
            'ports': [{
                'protocol': 'TCP',
                'port': 80,
                'targetPort': 8080
            }],
            'type': 'LoadBalancer'
        }
    }
    
    # Save Kubernetes service
    with open('k8s-service.yaml', 'w') as f:
        import yaml
        yaml.dump(k8s_service, f, default_flow_style=False)
    
    print("✅ k8s-service.yaml created")

def generate_deployment_guide():
    """Generate comprehensive deployment guide"""
    
    print("📖 Generating deployment guide...")
    
    guide_content = """# F1 ML API - Deployment Guide

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
gcloud run deploy f1-ml-api \\
    --image gcr.io/<project-id>/f1-ml-api:latest \\
    --platform managed \\
    --region us-central1 \\
    --allow-unauthenticated
```

##### Azure Container Instances
```bash
# Build and push to Azure Container Registry
az acr build --registry <registry-name> --image f1-ml-api:latest .

# Deploy to Container Instances
az container create \\
    --resource-group <resource-group> \\
    --name f1-ml-api \\
    --image <registry-name>.azurecr.io/f1-ml-api:latest \\
    --cpu 1 --memory 2 \\
    --ports 8080 \\
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
curl -X POST http://your-api-url/predict_full \\
  -H "Content-Type: application/json" \\
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
"""
    
    with open('DEPLOYMENT_GUIDE.md', 'w') as f:
        f.write(guide_content)
    
    print("✅ DEPLOYMENT_GUIDE.md created")

def main():
    """Main deployment script"""
    
    print("🏁 F1 ML API - PHASE 10.4 DEPLOYMENT")
    print("=" * 60)
    
    if len(sys.argv) < 2:
        print("Usage: python deploy.py [command]")
        print("Commands:")
        print("  build     - Build Docker image")
        print("  run       - Run Docker container")
        print("  test      - Test deployment")
        print("  stop      - Stop container")
        print("  configs   - Generate deployment configs")
        print("  guide     - Generate deployment guide")
        print("  full      - Complete deployment process")
        return
    
    command = sys.argv[1]
    
    try:
        if command == "build":
            build_docker_image()
        elif command == "run":
            run_docker_container()
        elif command == "test":
            test_docker_deployment()
        elif command == "stop":
            stop_docker_container()
        elif command == "configs":
            generate_deployment_configs()
        elif command == "guide":
            generate_deployment_guide()
        elif command == "full":
            print("🚀 Running complete deployment process...")
            if build_docker_image():
                container_id = run_docker_container()
                if container_id and test_docker_deployment():
                    print("\n✅ DEPLOYMENT SUCCESSFUL!")
                    print("   🌐 API URL: http://localhost:8080")
                    print("   📚 Docs: http://localhost:8080/docs")
                    print("   🔧 Health: http://localhost:8080/health")
                else:
                    print("\n❌ Deployment failed during testing")
            else:
                print("\n❌ Deployment failed during build")
        else:
            print(f"❌ Unknown command: {command}")
    
    except KeyboardInterrupt:
        print("\n⏹️  Deployment interrupted by user")
    except Exception as e:
        print(f"\n❌ Deployment error: {e}")

if __name__ == "__main__":
    main()