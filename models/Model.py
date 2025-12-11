"""
Seldon model wrapper for sklearn models
"""
import joblib
import numpy as np
from pathlib import Path


class Model:
    """
    Model wrapper compatible with Seldon Core
    """

    def __init__(self, model_path="model.joblib"):
        """Initialize the model"""
        self.model = None
        self.model_path = model_path
        self.ready = False

    def load(self):
        """Load the model from disk"""
        print(f"Loading model from {self.model_path}")
        try:
            self.model = joblib.load(self.model_path)
            self.ready = True
            print("Model loaded successfully")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise

    def predict(self, X, features_names=None):
        """
        Make predictions

        Args:
            X: Input features (numpy array or list)
            features_names: Feature names (optional)

        Returns:
            Predictions as numpy array
        """
        if not self.ready:
            self.load()

        # Convert to numpy array if needed
        if not isinstance(X, np.ndarray):
            X = np.array(X)

        print(f"Predicting for input shape: {X.shape}")
        predictions = self.model.predict(X)
        print(f"Predictions: {predictions}")

        return predictions

    def predict_proba(self, X, features_names=None):
        """
        Predict class probabilities

        Args:
            X: Input features
            features_names: Feature names (optional)

        Returns:
            Class probabilities
        """
        if not self.ready:
            self.load()

        if not isinstance(X, np.ndarray):
            X = np.array(X)

        if hasattr(self.model, 'predict_proba'):
            probas = self.model.predict_proba(X)
            print(f"Predicted probabilities shape: {probas.shape}")
            return probas
        else:
            # If model doesn't support predict_proba, return one-hot encoded predictions
            predictions = self.predict(X, features_names)
            n_classes = len(np.unique(predictions))
            probas = np.zeros((len(predictions), n_classes))
            probas[np.arange(len(predictions)), predictions] = 1.0
            return probas

    def health_status(self):
        """Return health status"""
        return self.ready
