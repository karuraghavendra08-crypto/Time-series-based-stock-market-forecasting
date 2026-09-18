import os
import joblib
import json
import pytest

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, 'mini_projects_website', 'saved_models')

def test_model_metrics_json_exists():
    metrics_path = os.path.join(MODELS_DIR, 'model_metrics.json')
    assert os.path.exists(metrics_path), 'model_metrics.json must exist'
    
    with open(metrics_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    assert 'models' in data
    assert len(data['models']) >= 5

def test_regime_classifier_loads_and_predicts():
    regime_path = os.path.join(MODELS_DIR, 'regime_classifier.joblib')
    assert os.path.exists(regime_path), 'regime_classifier.joblib must exist'
    
    model = joblib.load(regime_path)
    n_feats = getattr(model, 'n_features_in_', 8)
    dummy_input = [[0.0] * n_feats]
    pred = model.predict(dummy_input)
    assert pred[0] in [0, 1], 'Regime classifier output should be binary (0 or 1)'

def test_random_forest_loads_and_predicts():
    rf_path = os.path.join(MODELS_DIR, 'random_forest_lag.joblib')
    assert os.path.exists(rf_path), 'random_forest_lag.joblib must exist'
    
    model = joblib.load(rf_path)
    n_feats = getattr(model, 'n_features_in_', 24)
    dummy_input = [[0.0] * n_feats]
    pred = model.predict(dummy_input)
    assert isinstance(float(pred[0]), float)

def test_dl_scaler_exists():
    scaler_path = os.path.join(MODELS_DIR, 'dl_scaler.joblib')
    assert os.path.exists(scaler_path), 'dl_scaler.joblib must exist'
