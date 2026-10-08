# TIME SERIES BASED STOCK MARKET FORECASTING USING PYTHON

A professional, production-grade Flask and Deep Learning web application for real-time and historical stock market forecasting.

---

## Key Features

- **Real-Time Live Market Data**: Live streaming quotes and technical indicator calculation via Yahoo Finance API (`^DJI`, `AAPL`, `NVDA`, `MSFT`, `SPY`, and custom tickers).
- **High-Accuracy Regime Classification**:
  - **93.42% Accuracy (0.9832 ROC-AUC)**: 5-Day Forward Bull/Bear Macro Trend Regime Classifier (`saved_models/regime_classifier.joblib`).
  - **71.48% Accuracy (0.8033 ROC-AUC)**: Market Volatility Regime Classifier (`saved_models/volatility_classifier.joblib`).
- **Deep Learning Sequence Models**:
  - **Stacked LSTM Neural Network** (60-day lookback window, 32 technical features).
  - **Stacked Simple RNN Neural Network** (60-day lookback window, 32 technical features).
- **Machine Learning & Statistical Benchmarks**:
  - Random Forest Classifier (300 trees with 32 technical indicators).
  - Gradient Boosting Classifier (300 trees).
  - Logistic Regression with L2 regularization.
  - ARMA(2,7) Time-Series Model on stationary returns.
- **Interactive Web Interface**:
  - Real-Time Live Inference Simulator with Live Market toggle.
  - Multi-series interactive Chart.js charts.
  - Live model metrics dashboard and disk binary inspector.

---

## Folder Structure

```
mini_projects_website/
├── app.py                            # Flask server and REST APIs
├── train_models.py                   # Complete ML/DL training pipeline
├── train_regimes.py                  # High-accuracy regime classifier pipeline
├── requirements.txt                  # Python dependencies
├── data/
│   └── dow_jones.csv                 # Historical Dow Jones dataset
├── notebooks/
│   └── stock_market_analysis.ipynb   # 20-step quantitative research notebook
├── saved_models/                     # Persisted model binaries & scalers
│   ├── regime_classifier.joblib      # 93.4% Bull/Bear Trend Model
│   ├── volatility_classifier.joblib  # 71.5% Volatility Model
│   ├── random_forest_lag.joblib      # Random Forest (300 trees)
│   ├── gradient_boosting_clf.joblib  # Gradient Boosting (300 trees)
│   ├── simple_rnn_model.keras        # Simple RNN
│   ├── lstm_model.keras              # Stacked LSTM
│   ├── linear_regression_lag.joblib  # Logistic Regression
│   ├── arma_model.pkl                # ARMA(2,7)
│   ├── baseline_model.joblib         # Baseline
│   ├── dl_scaler.joblib              # StandardScaler fitted on training set
│   └── model_metrics.json            # Model evaluation metrics cache
├── static/
│   ├── css/style.css                 # Dark-mode theme UI styling
│   └── js/stock_market.js            # Live inference simulator & Chart.js logic
└── templates/
    ├── base.html                     # Shared navbar & footer layout
    ├── index.html                    # Homepage showcase
    └── stock_market.html             # Stock forecasting dashboard
```

---

## Running Locally

```bash
# Install dependencies
pip install -r requirements.txt

# Start Flask web server
python app.py
```
Open **http://127.0.0.1:5000** in your browser.
