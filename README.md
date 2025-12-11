ICT Solution Artefact

Student ID: 23377348
Name: Akash Venkatesan
Project: A Kubernetes Operator for Adaptive Deployment and Real Time Model Evaluation in MLOps Workflows

Overview

This archive contains all the source code, configurations, and deployment files for my MSc research project. The project implements a Kubernetes operator that uses epsilon-greedy algorithm to route traffic between three ML models based on their performance.

What's Included

The archive has the following structure:

- logistic-regression-service/ - Flask service for logistic regression model
- svm-service/ - Flask service for SVM model
- random-forest-service/ - Flask service for random forest model
- models/ - Training scripts and model artifacts
- operator/ - Kubernetes operator code
- traffic-generator/ - Scripts to generate traffic
- k8s-manifests/ - Kubernetes YAML files for deployment
- monitoring/ - Grafana dashboard configuration
- Configuration_Manual.pdf - Detailed deployment instructions

Getting Started

Please refer to the Configuration_Manual.pdf for complete deployment instructions. The manual covers:

1. Prerequisites and setup
2. Azure infrastructure setup (resource group, AKS cluster, container registry)
3. Building and pushing Docker images
4. Deploying all components to Kubernetes
5. Setting up Prometheus and Grafana monitoring
6. Testing and validation

Quick Overview of Components

Model Services
Each service folder (logistic-regression-service, svm-service, random-forest-service) contains:
- app.py - Flask application code
- Dockerfile - Docker image definition
- requirements.txt - Python dependencies
- model.joblib - Pre-trained model file

Operator
The operator monitors model performance from Prometheus and adjusts traffic routing based on epsilon-greedy algorithm (epsilon = 0.2).

Traffic Generator
Generates prediction requests to the model services. Can be configured to use different traffic patterns.

K8s Manifests
Contains deployment YAMLs for all services:
- logistics-regression.yaml
- svm.yaml
- random-forest.yaml
- operator-rbac.yaml
- operator.yaml
- traffic-generator.yaml

Deployment Notes

The Configuration_Manual.pdf has step-by-step commands for deploying everything. Main steps are:

1. Create Azure resources (resource group, AKS cluster, ACR)
2. Build Docker images for each service
3. Push images to Azure Container Registry
4. Deploy model services using kubectl
5. Deploy operator with RBAC
6. Deploy traffic generator
7. Install Prometheus and Grafana for monitoring
8. Import Grafana dashboard

Monitoring

The system uses Prometheus to collect metrics from model services and Grafana for visualization. The dashboard shows:
- Traffic distribution across models
- Model latency
- Total requests per model

Configuration Parameters

Key configuration values used in the project:
- Epsilon (exploration rate): 0.2
- Operator reconciliation interval: 30 seconds
- Prometheus scrape interval: 15 seconds
- Metrics time window: 5 minutes
- EMA alpha: 0.3
- Grafana refresh rate: 5 seconds

Testing

After deployment, you can verify everything is working by:

```
kubectl get deployments
kubectl get pods
kubectl logs -l app=adaptive-router-operator
kubectl logs -l app=traffic-generator
```

Access Grafana dashboard:
```
kubectl port-forward -n prometheus svc/grafana 3000:3000
```
Then open http://localhost:3000 (login: admin/admin)

Cleanup

To stop the cluster:
```
az aks stop --resource-group mlops-rg --name mlops-cluster
```

To delete everything:
```
az group delete --name mlops-rg --yes
```

Notes

- Make sure to follow the Configuration_Manual.pdf for proper setup
- The project was tested on Azure Kubernetes Service with Kubernetes 1.32
- All model services expose Prometheus metrics on /metrics endpoint
- The operator implements epsilon-greedy multi-armed bandit algorithm
