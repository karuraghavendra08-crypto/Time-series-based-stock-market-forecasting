# TIME SERIES BASED STOCK MARKET FORECASTING USING PYTHON

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Flask](https://img.shields.io/badge/framework-Flask-lightgrey.svg)](https://flask.palletsprojects.com/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15+-orange.svg)](https://www.tensorflow.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-latest-green.svg)](https://scikit-learn.org/)
[![Yahoo Finance](https://img.shields.io/badge/Live_Data-yfinance-purple.svg)](https://pypi.org/project/yfinance/)
[![Deployment](https://img.shields.io/badge/Render-Live-brightgreen.svg)](https://stock-market-forecasting-6cw5.onrender.com/project/stock-market)

An end-to-end quantitative financial engineering, time-series forecasting, and machine learning web platform built with **Python 3.11, Flask, TensorFlow/Keras, Scikit-Learn, Statsmodels, Yahoo Finance, and Chart.js**.

---

## 📈 Key Features & Model Highlights

* **Real-Time Live Market Streaming**: Instant live quotes and real-time 32-feature technical indicator extraction via Yahoo Finance for `^DJI`, `AAPL`, `NVDA`, `MSFT`, `SPY`, and custom user tickers.
* **High-Accuracy Market Regime Classifiers**:
  * **93.42% Accuracy (0.9832 ROC-AUC)**: 5-Day Forward Bull/Bear Macro Trend Regime Classifier (`saved_models/regime_classifier.joblib`).
  * **71.48% Accuracy (0.8033 ROC-AUC)**: Market Volatility Regime Classifier (`saved_models/volatility_classifier.joblib`).
* **Deep Learning Sequence Models**:
  * **Stacked LSTM Neural Network** (60-day sequence lookback, 32 technical indicators, batch norm, dropout).
  * **Stacked Simple RNN Neural Network** (60-day sequence lookback, 32 technical indicators).
* **Ensemble ML & Statistical Models**:
  * **Random Forest Classifier (300 Trees)**: 80.38% Directional Accuracy (0.8840 ROC-AUC).
  * **Gradient Boosting Classifier (300 Trees)**: 79.84% Directional Accuracy (0.8757 ROC-AUC).
  * **Logistic Regression Classifier**: 79.03% Directional Accuracy (0.8576 ROC-AUC).
  * **ARMA(2,7) Time Series**: Statistical benchmark on stationary log-returns.
* **Interactive Web Platform**:
  * Real-Time Inference Simulator with Live Market Data toggle and custom lag parameters.
  * Multi-Series Chart.js Visualizer with individual model toggling.
  * Live Model Metrics scorecard and disk binary storage inspector.

---

## 📊 Model Evaluation Summary

| Model | Architecture / Type | Directional / Class Accuracy | ROC-AUC | F1-Score | Persisted Binary |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Bull/Bear Regime Classifier** | Gradient Boosting (8 Features) | **93.42%** | **0.9832** | **0.9421** | `regime_classifier.joblib` |
| **Random Forest Classifier** | Ensemble ML (300 Trees) | **80.38%** | **0.8840** | **0.8450** | `random_forest_lag.joblib` |
| **Gradient Boosting Classifier** | Boosting ML (300 Trees) | **79.84%** | **0.8757** | **0.8460** | `gradient_boosting_clf.joblib` |
| **Logistic Regression** | Supervised ML + L2 Reg | **79.03%** | **0.8576** | **0.8382** | `linear_regression_lag.joblib` |
| **Stacked Simple RNN** | Deep Learning (60-day sequence) | **73.39%** | **0.7426** | **0.7843** | `simple_rnn_model.keras` |
| **Market Volatility Classifier** | Random Forest (8 Features) | **71.48%** | **0.8033** | **0.7512** | `volatility_classifier.joblib` |
| **ARMA(2,7) Time Series** | Statsmodels | **68.55%** | — | — | `arma_model.pkl` |
| **Stacked LSTM Neural Net** | Deep Learning (60-day sequence) | **63.44%** | **0.6255** | **0.7405** | `lstm_model.keras` |
| **Most-Frequent Baseline** | Naive Benchmark | 60.75% | 0.5000 | — | `baseline_model.joblib` |

---

## 🚀 Live Demo & Deployment

* **Live Web App**: [https://stock-market-forecasting-6cw5.onrender.com/project/stock-market](https://stock-market-forecasting-6cw5.onrender.com/project/stock-market)
* **GitHub Repository**: [https://github.com/karuraghavendra08-crypto/Time-series-based-stock-market-forecasting](https://github.com/karuraghavendra08-crypto/Time-series-based-stock-market-forecasting)

---

## 💻 Running Locally

```bash
# 1. Clone repository
git clone https://github.com/karuraghavendra08-crypto/Time-series-based-stock-market-forecasting.git
cd Time-series-based-stock-market-forecasting

# 2. Install dependencies
pip install -r requirements.txt

# 3. Start local server
python app.py
```

Access the dashboard at **http://127.0.0.1:5000**.
