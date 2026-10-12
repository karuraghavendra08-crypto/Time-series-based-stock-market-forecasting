import os
import sys
import pytest

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

WEBSITE_DIR = os.path.join(BASE_DIR, 'mini_projects_website')
if WEBSITE_DIR not in sys.path:
    sys.path.insert(0, WEBSITE_DIR)

from app import app

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_home_page(client):
    res = client.get('/')
    assert res.status_code == 200
    assert b'Read the Market' in res.data or b'Time Series' in res.data

def test_stock_market_page(client):
    res = client.get('/project/stock-market')
    assert res.status_code == 200
    assert b'Model Laboratory' in res.data
    assert b'Prediction Breakdown' in res.data
    assert b'Financial-Risk Disclaimer' in res.data

def test_api_dataset_metadata_modern(client):
    res = client.get('/api/dataset-metadata?dataset=modern')
    assert res.status_code == 200
    data = res.get_json()
    assert data['valid_observations'] == 2765
    assert data['duplicate_records'] == 0
    assert data['first_valid_date'] == '2015-01-02'
    assert data['last_valid_date'] == '2025-12-30'
    assert data['unit'] == 'pts'

def test_api_dataset_metadata_historical(client):
    res = client.get('/api/dataset-metadata?dataset=historical')
    assert res.status_code == 200
    data = res.get_json()
    assert data['valid_observations'] == 27824
    assert data['duplicate_records'] == 0
    assert data['first_valid_date'] == '1901-01-04'
    assert data['last_valid_date'] == '2012-12-31'
    assert data['unit'] == 'pts'

def test_api_stock_data(client):
    res = client.get('/api/stock-data?dataset=modern&n=100')
    assert res.status_code == 200
    data = res.get_json()
    assert 'dates' in data
    assert 'closes' in data
    assert data['count'] == 100
    assert data['unit'] == 'pts'

def test_api_model_metrics(client):
    res = client.get('/api/model-metrics')
    assert res.status_code == 200
    data = res.get_json()
    assert 'models' in data
    assert len(data['models']) >= 7

def test_api_model_status(client):
    res = client.get('/api/model-status')
    assert res.status_code == 200
    data = res.get_json()
    assert 'files' in data

def test_api_predict_regime(client):
    res = client.get('/api/predict?model=bull_bear_regime&ticker=^DJI')
    assert res.status_code == 200
    data = res.get_json()
    assert data['success'] is True
    assert 'predicted_target_price' in data
    assert 'direction' in data
    assert data['unit'] == 'pts'
    assert 'interpretation' in data
    assert 'feature_importance' in data

def test_api_predict_equity_unit(client):
    res = client.get('/api/predict?model=rf&ticker=AAPL')
    assert res.status_code == 200
    data = res.get_json()
    assert data['success'] is True
    assert data['unit'] == '$'
    assert data['currency'] == 'USD'

def test_api_predict_invalid_model(client):
    res = client.get('/api/predict?model=non_existent_model')
    assert res.status_code == 400
    data = res.get_json()
    assert data['success'] is False
