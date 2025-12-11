FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements_service.txt .
RUN pip install --no-cache-dir -r requirements_service.txt

# Copy service code
COPY model_service.py .

# Copy model artifact (will be overridden by ConfigMap in Kubernetes)
# This is just for local testing
COPY artifacts/*.joblib /app/model.joblib 2>/dev/null || true

# Expose ports
EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=40s \
  CMD python -c "import requests; requests.get('http://localhost:5000/health')" || exit 1

# Run as non-root user
RUN useradd -m -u 1000 modeluser && chown -R modeluser:modeluser /app
USER modeluser

# Start service
CMD ["python", "model_service.py"]
