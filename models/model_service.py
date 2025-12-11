"""
Flask service for ML model inference with Prometheus metrics
"""
import os
import time
import joblib
import numpy as np
from flask import Flask, request, jsonify
from prometheus_client import Counter, Histogram, Gauge, generate_latest, REGISTRY
from werkzeug.middleware.dispatcher import DispatcherMiddleware
from prometheus_client import make_wsgi_app
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Get model name from environment variable
MODEL_NAME = os.environ.get('MODEL_NAME', 'unknown')
MODEL_PATH = os.environ.get('MODEL_PATH', '/app/model.joblib')

# Initialize Flask app
app = Flask(__name__)

# Prometheus metrics
REQUEST_COUNT = Counter(
    f'model_requests_total',
    'Total number of prediction requests',
    ['model_name', 'status']
)

REQUEST_LATENCY = Histogram(
    f'model_request_duration_seconds',
    'Request latency in seconds',
    ['model_name']
)

PREDICTION_COUNT = Counter(
    f'model_predictions_total',
    'Total number of predictions made',
    ['model_name', 'predicted_class']
)

ERROR_COUNT = Counter(
    f'model_errors_total',
    'Total number of errors',
    ['model_name', 'error_type']
)

MODEL_HEALTH = Gauge(
    f'model_health_status',
    'Model health status (1=healthy, 0=unhealthy)',
    ['model_name']
)

# Load model at startup
model = None

def load_model():
    """Load the ML model"""
    global model
    try:
        logger.info(f"Loading model from {MODEL_PATH}")
        model = joblib.load(MODEL_PATH)
        MODEL_HEALTH.labels(model_name=MODEL_NAME).set(1)
        logger.info(f"Model {MODEL_NAME} loaded successfully")
        return True
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        MODEL_HEALTH.labels(model_name=MODEL_NAME).set(0)
        ERROR_COUNT.labels(model_name=MODEL_NAME, error_type='model_load_error').inc()
        return False

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    if model is None:
        return jsonify({'status': 'unhealthy', 'model': MODEL_NAME}), 503
    return jsonify({'status': 'healthy', 'model': MODEL_NAME}), 200

@app.route('/predict', methods=['POST'])
def predict():
    """Prediction endpoint"""
    start_time = time.time()

    try:
        # Get input data
        data = request.get_json()

        if 'instances' not in data:
            REQUEST_COUNT.labels(model_name=MODEL_NAME, status='error').inc()
            ERROR_COUNT.labels(model_name=MODEL_NAME, error_type='invalid_input').inc()
            return jsonify({'error': 'Missing "instances" in request body'}), 400

        # Convert to numpy array
        X = np.array(data['instances'])

        if model is None:
            load_model()

        if model is None:
            REQUEST_COUNT.labels(model_name=MODEL_NAME, status='error').inc()
            ERROR_COUNT.labels(model_name=MODEL_NAME, error_type='model_not_loaded').inc()
            return jsonify({'error': 'Model not loaded'}), 503

        # Make prediction
        predictions = model.predict(X)
        probabilities = model.predict_proba(X) if hasattr(model, 'predict_proba') else None

        # Track predictions
        for pred in predictions:
            PREDICTION_COUNT.labels(model_name=MODEL_NAME, predicted_class=str(pred)).inc()

        # Prepare response
        response = {
            'predictions': predictions.tolist(),
            'model': MODEL_NAME
        }

        if probabilities is not None:
            response['probabilities'] = probabilities.tolist()

        # Record metrics
        REQUEST_COUNT.labels(model_name=MODEL_NAME, status='success').inc()
        latency = time.time() - start_time
        REQUEST_LATENCY.labels(model_name=MODEL_NAME).observe(latency)

        logger.info(f"Prediction successful for {MODEL_NAME}, latency: {latency:.4f}s")

        return jsonify(response), 200

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        REQUEST_COUNT.labels(model_name=MODEL_NAME, status='error').inc()
        ERROR_COUNT.labels(model_name=MODEL_NAME, error_type='prediction_error').inc()

        latency = time.time() - start_time
        REQUEST_LATENCY.labels(model_name=MODEL_NAME).observe(latency)

        return jsonify({'error': str(e), 'model': MODEL_NAME}), 500

@app.route('/metrics', methods=['GET'])
def metrics():
    """Prometheus metrics endpoint"""
    return generate_latest(REGISTRY), 200, {'Content-Type': 'text/plain; charset=utf-8'}

@app.route('/', methods=['GET'])
def root():
    """Root endpoint"""
    return jsonify({
        'service': 'ML Model Inference Service',
        'model': MODEL_NAME,
        'endpoints': {
            '/health': 'Health check',
            '/predict': 'Make predictions (POST)',
            '/metrics': 'Prometheus metrics'
        }
    }), 200

if __name__ == '__main__':
    # Load model at startup
    load_model()

    # Run Flask app
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
