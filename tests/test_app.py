import os
import sys
import json
import pytest
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

WEBSITE_DIR = os.path.join(BASE_DIR, 'mini_projects_website')
if WEBSITE_DIR not in sys.path:
    sys.path.insert(0, WEBSITE_DIR)

from app import (
    app,
    fetch_live_market_data,
    compute_model_features,
    compute_regime_features,
    FEATURE_COLS_32,
    _LIVE_MARKET_CACHE
)

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

# ── 1. Page Routes ─────────────────────────────────────────────────────────────
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

# ── 2. Dataset Metadata & Historical CSV ───────────────────────────────────────
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

def test_api_stock_data_csv(client):
    res = client.get('/api/stock-data?dataset=modern&n=100')
    assert res.status_code == 200
    data = res.get_json()
    assert 'dates' in data
    assert 'closes' in data
    assert data['count'] == 100
    assert data['unit'] == 'pts'
    assert "Yahoo" not in data.get('source', '')  # Must not claim CSV is Yahoo

# ── 3. Model Laboratory Metrics & Status ───────────────────────────────────────
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
    assert data['models_loaded_count'] >= 5

# ── 4. Live Market Data Endpoint ───────────────────────────────────────────────
def test_api_live_market_data_schema(client):
    res = client.get('/api/live-market-data?ticker=^DJI')
    # If live network succeeds: 200 with full schema
    # If network/Yahoo unavailable in CI: 503 structured error
    if res.status_code == 200:
        data = res.get_json()
        assert data['success'] is True
        assert data['ticker'] == '^DJI'
        assert 'source' in data
        assert 'last_close' in data
        assert 'history_dates' in data
        assert 'history_closes' in data
        assert 'features' in data
        assert data['unit'] == 'pts'
    else:
        assert res.status_code == 503
        data = res.get_json()
        assert data['success'] is False
        assert 'error' in data

def test_api_live_market_data_equity(client):
    res = client.get('/api/live-market-data?ticker=AAPL')
    if res.status_code == 200:
        data = res.get_json()
        assert data['success'] is True
        assert data['ticker'] == 'AAPL'
        assert data['unit'] == '$'
        assert data['currency'] == 'USD'
    else:
        assert res.status_code == 503
        data = res.get_json()
        assert data['success'] is False

# ── 5. Yahoo Fetching Failure & Cache Retention (Mocked) ───────────────────────
def test_yahoo_fetching_mock_success():
    # Clear any live cache before this mock test to prevent cross-test contamination
    _LIVE_MARKET_CACHE.clear()

    # Construct a synthetic DataFrame simulating genuine 100-day daily OHLCV from yfinance
    dates = pd.date_range(end='2025-01-01', periods=100, freq='D')
    np.random.seed(42)
    closes = 40000.0 + np.cumsum(np.random.randn(100) * 100)
    highs = closes + np.abs(np.random.randn(100) * 50)
    lows = closes - np.abs(np.random.randn(100) * 50)
    opens = closes + np.random.randn(100) * 20
    volumes = np.random.randint(1000000, 5000000, size=100)

    mock_df = pd.DataFrame({
        'Open': opens,
        'High': highs,
        'Low': lows,
        'Close': closes,
        'Volume': volumes
    }, index=dates)

    mock_ticker = MagicMock()
    mock_ticker.history.return_value = mock_df

    with patch('yfinance.Ticker', return_value=mock_ticker):
        payload = fetch_live_market_data('^DJI')
        assert payload['success'] is True
        assert payload['ticker'] == '^DJI'
        assert payload['last_close'] == round(float(closes[-1]), 2)
        assert len(payload['history_closes']) == 100
        assert 'features' in payload
        assert len(payload['features']) == 32

def test_yahoo_fetching_mock_total_failure_returns_honest_error():
    # When both yfinance and direct Yahoo API fail and cache is empty
    _LIVE_MARKET_CACHE.clear()

    with patch('yfinance.Ticker', side_effect=Exception("Upstream Yahoo Down")), \
         patch('urllib.request.urlopen', side_effect=Exception("Network Timeout")):
        payload = fetch_live_market_data('^DJI')
        assert payload['success'] is False
        assert "error" in payload
        # Check the error message describes the failure
        error_msg = payload["error"].lower()
        assert any(kw in error_msg for kw in ["failed", "unable", "retrieve", "error", "no data"]), \
            f"Expected failure description in error, got: {payload['error']}"
        # Ensure NO fake prices or synthetic candle arrays are returned
        assert "default_price" not in payload
        assert payload.get("success") is False

def test_yahoo_fetching_preserves_cached_data_and_marks_stale():
    # Pre-populate cache using the correct key structure used by app.py ("timestamp" not "ts")
    # Set timestamp far in the past so the 30-second hot-cache window is bypassed
    _LIVE_MARKET_CACHE['^DJI'] = {
        'timestamp': 1.0,   # epoch second 1 = very old, will not be served as hot cache
        'data': {
            "success": True,
            "ticker": "^DJI",
            "name": "Dow Jones Industrial Average",
            "last_close": 42000.50,
            "day_change": 150.0,
            "day_pct_change": 0.36,
            "last_data_timestamp": "2025-01-01 16:00:00 EST",
            "fetched_at_utc": "2025-01-01 21:00:00 UTC",
            "data_interval": "1d",
            "source": "Yahoo Finance (Cached)",
            "source_method": "yfinance",
            "unit": "pts",
            "currency": "Points",
            "recent_closes": [41900.0, 42000.50],
            "history_closes": [41900.0, 42000.50],
            "history_dates": ["2024-12-31", "2025-01-01"],
            "recent_dates": ["2024-12-31", "2025-01-01"],
            "features": {},
            "regime_features": {}
        }
    }

    # Now simulate upstream failure — app must fall back to the stale cache
    with patch('yfinance.Ticker', side_effect=Exception("Rate Limited")), \
         patch('urllib.request.urlopen', side_effect=Exception("Rate Limited")):
        payload = fetch_live_market_data('^DJI')
        assert payload['success'] is True
        assert payload['is_stale'] is True
        assert payload['last_data_timestamp'] == "2025-01-01 16:00:00 EST"
        assert payload['last_close'] == 42000.50

# ── 6. Prediction Endpoint & Model Inference ───────────────────────────────────
def test_api_predict_regime_classification_no_fake_price(client):
    # Classification models must NOT fabricate a numerical target price
    res = client.get('/api/predict?model=bull_bear_regime&ticker=^DJI')
    assert res.status_code == 200
    data = res.get_json()
    assert data['success'] is True
    assert data['is_numerical'] is False
    assert data['predicted_target_price'] is None
    assert data['price_change'] is None
    assert data['percent_change'] is None
    assert 'direction' in data
    assert 'prob_up' in data
    assert 0.0 <= data['prob_up'] <= 1.0
    assert data['unit'] == 'pts'
    assert 'interpretation' in data

def test_api_predict_arma_numerical_regression(client):
    # ARMA model is a continuous regression model that genuinely forecasts next-day log-return
    res = client.get('/api/predict?model=arma&ticker=^DJI')
    assert res.status_code == 200
    data = res.get_json()
    assert data['success'] is True
    assert data['is_numerical'] is True
    assert data['predicted_target_price'] is not None
    assert isinstance(data['predicted_target_price'], (int, float))
    assert data['predicted_target_price'] > 0
    assert data['price_change'] is not None
    assert data['percent_change'] is not None

def test_api_predict_unsupported_instrument_handled_honestly(client):
    # Models trained strictly on Dow Jones must explicitly reject AAPL with 400 Bad Request
    res = client.get('/api/predict?model=rf&ticker=AAPL')
    assert res.status_code == 400
    data = res.get_json()
    assert data['success'] is False
    assert data.get('unsupported_instrument') is True
    assert data['ticker'] == 'AAPL'
    assert 'Dow Jones' in data['error']

def test_api_predict_invalid_model(client):
    res = client.get('/api/predict?model=non_existent_model')
    assert res.status_code == 400
    data = res.get_json()
    assert data['success'] is False

# ── 7. Feature Engineering Validation ──────────────────────────────────────────
def test_feature_parity_32_columns():
    # Use 250 rows so 200-period rolling MA converges at the last row
    n = 250
    df = pd.DataFrame({
        'DATE': pd.date_range(end='2025-01-01', periods=n, freq='D'),
        'Open': [100.0 + i + (i % 7) * 0.5 for i in range(n)],
        'High': [102.0 + i + (i % 5) * 0.3 for i in range(n)],
        'Low': [99.0 + i - (i % 4) * 0.2 for i in range(n)],
        'Close': [101.0 + i + (i % 11) * 0.4 for i in range(n)],
        'Volume': [1000000 + i * 500 for i in range(n)]
    })

    feats_df = compute_model_features(df)
    assert len(feats_df) == n
    for col in FEATURE_COLS_32:
        assert col in feats_df.columns, f"Missing feature column: {col}"
    # Verify last row has finite values after fillna in app.py
    last_row = feats_df.fillna(0.0).iloc[-1]
    for col in FEATURE_COLS_32:
        val = last_row[col]
        assert not np.isnan(val) and not np.isinf(val), f"Non-finite value for {col}: {val}"
