#!/bin/bash
# Build Docker images for each model

set -e

# Models to build
MODELS=("logistic_regression" "random_forest" "svm")

# Docker registry (change this to your registry)
REGISTRY="your-registry"  # e.g., docker.io/username or your ACR

echo "Building model images..."

for model in "${MODELS[@]}"; do
    echo "Building image for $model..."

    docker build \
        --build-arg MODEL_FILE="${model}.joblib" \
        -t "${REGISTRY}/iris-${model}:latest" \
        -f Dockerfile \
        .

    echo "Successfully built ${REGISTRY}/iris-${model}:latest"
done

echo ""
echo "All images built successfully!"
echo ""
echo "To push images to registry:"
for model in "${MODELS[@]}"; do
    echo "  docker push ${REGISTRY}/iris-${model}:latest"
done
