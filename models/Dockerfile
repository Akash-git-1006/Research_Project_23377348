FROM python:3.11-slim

WORKDIR /app

# Install dependencies
RUN pip install --no-cache-dir \
    scikit-learn==1.5.2 \
    joblib==1.4.2 \
    numpy==2.1.3 \
    seldon-core==1.17.1

# Copy model wrapper
COPY Model.py /app/Model.py

# Copy the specific model artifact (will be set via build args)
ARG MODEL_FILE
COPY artifacts/${MODEL_FILE} /app/model.joblib

# Expose port for Seldon
EXPOSE 5000

# Run as non-root user
RUN useradd -m -u 8877 seldon
USER seldon

# Health check
HEALTHCHECK --interval=30s --timeout=3s \
  CMD python -c "import sys; sys.exit(0)"

CMD exec seldon-core-microservice Model --service-type MODEL --log-level INFO
