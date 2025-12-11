# ML Models

This directory contains the machine learning models used for adaptive deployment testing.

## Models

We train three different models on the Iris dataset:

1. **Logistic Regression**: Fast, simple linear model
2. **Random Forest**: Ensemble model with higher accuracy
3. **SVM**: Support Vector Machine with RBF kernel

## Training

To train all models:

```bash
cd models
python train_models.py
```

This will create an `artifacts/` directory containing:
- `{model_name}.joblib` - Serialized model files
- `{model_name}_metadata.json` - Model metadata and metrics
- `models_summary.json` - Summary of all models

## Model Artifacts

The artifacts directory structure:
```
artifacts/
├── logistic_regression.joblib
├── logistic_regression_metadata.json
├── random_forest.joblib
├── random_forest_metadata.json
├── svm.joblib
├── svm_metadata.json
└── models_summary.json
```

## Using Models

Models can be loaded using joblib:

```python
import joblib

model = joblib.load('artifacts/logistic_regression.joblib')
predictions = model.predict(X)
```
