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
    assert b'Portfolio' in res.data or b'Dow Jones' in res.data

def test_stock_market_page(client):
    res = client.get('/project/stock-market')
    assert res.status_code == 200
    assert b'Dow Jones' in res.data

def test_api_stock_data(client):
    res = client.get('/api/stock-data?n=100')
    assert res.status_code == 200
    data = res.get_json()
    assert 'dates' in data
    assert 'closes' in data
    assert data['count'] == 100

def test_api_model_metrics(client):
    res = client.get('/api/model-metrics')
    assert res.status_code == 200
    data = res.get_json()
    assert 'metrics' in data
    assert len(data['metrics']) >= 5

def test_api_model_status(client):
    res = client.get('/api/model-status')
    assert res.status_code == 200
    data = res.get_json()
    assert 'files' in data

def test_api_predict(client):
    res = client.get('/api/predict?model=bull_bear_regime')
    assert res.status_code == 200
    data = res.get_json()
    assert data['success'] is True
    assert 'predicted_close_price' in data
    assert 'direction' in data
