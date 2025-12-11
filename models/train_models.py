"""
Train multiple ML models for adaptive deployment testing
"""
import joblib
import json
from pathlib import Path
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np


def train_and_save_models():
    """Train three different models on Iris dataset"""
    print("Loading Iris dataset...")
    iris = load_iris()
    X, y = iris.data, iris.target

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print(f"Training data shape: {X_train.shape}")
    print(f"Test data shape: {X_test.shape}")

    # Define models with different characteristics
    models = {
        'logistic_regression': {
            'model': LogisticRegression(max_iter=200, random_state=42),
            'description': 'Fast, simple linear model'
        },
        'random_forest': {
            'model': RandomForestClassifier(n_estimators=100, random_state=42),
            'description': 'Ensemble model with higher accuracy'
        },
        'svm': {
            'model': SVC(kernel='rbf', gamma='scale', random_state=42),
            'description': 'Support Vector Machine with RBF kernel'
        }
    }

    # Create output directory
    output_dir = Path(__file__).parent
    artifacts_dir = output_dir / 'artifacts'
    artifacts_dir.mkdir(exist_ok=True)

    results = {}

    # Train and evaluate each model
    for model_name, model_config in models.items():
        print(f"\n{'='*60}")
        print(f"Training {model_name}...")
        print(f"Description: {model_config['description']}")
        print('='*60)

        model = model_config['model']

        # Train
        model.fit(X_train, y_train)

        # Predict
        y_pred = model.predict(X_test)

        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')

        print(f"Accuracy:  {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1 Score:  {f1:.4f}")

        # Save model
        model_path = artifacts_dir / f'{model_name}.joblib'
        joblib.dump(model, model_path)
        print(f"Model saved to: {model_path}")

        # Save metadata
        metadata = {
            'model_name': model_name,
            'model_type': type(model).__name__,
            'description': model_config['description'],
            'metrics': {
                'accuracy': float(accuracy),
                'precision': float(precision),
                'recall': float(recall),
                'f1_score': float(f1)
            },
            'training_data': {
                'n_samples_train': len(X_train),
                'n_samples_test': len(X_test),
                'n_features': X_train.shape[1],
                'n_classes': len(np.unique(y))
            },
            'model_path': str(model_path),
            'framework': 'sklearn'
        }

        metadata_path = artifacts_dir / f'{model_name}_metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"Metadata saved to: {metadata_path}")

        results[model_name] = metadata

    # Save summary
    summary_path = artifacts_dir / 'models_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*60}")
    print("Training complete!")
    print(f"All models saved to: {artifacts_dir}")
    print(f"Summary saved to: {summary_path}")
    print('='*60)

    # Print comparison
    print("\nModel Comparison:")
    print(f"{'Model':<25} {'Accuracy':<12} {'F1 Score':<12}")
    print('-'*50)
    for model_name, data in results.items():
        acc = data['metrics']['accuracy']
        f1 = data['metrics']['f1_score']
        print(f"{model_name:<25} {acc:<12.4f} {f1:<12.4f}")

    return results


if __name__ == '__main__':
    results = train_and_save_models()
