# Time Series Based Stock Market Forecasting

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Flask](https://img.shields.io/badge/framework-Flask-lightgrey.svg)](https://flask.palletsprojects.com/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-latest-green.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An end-to-end quantitative financial engineering, time-series forecasting, and machine learning web platform built with **Python, Flask, TensorFlow/Keras, Scikit-Learn, Statsmodels, and Chart.js**.

---

## 📈 Overview & Key Features

* **Historical Dow Jones Dataset (1901 – 2012)**: Over 20,000+ daily trading observations cleaned and analyzed.
* **Deep Learning Sequence Models**:
  * **Stacked LSTM Neural Network** (60-day sequence lookback, 24 technical indicators, 0.2 dropout, early stopping).
  * **Simple RNN Neural Network** (60-day sequence lookback, 24 technical indicators).
* **High-Accuracy Market Regime Classifiers**:
  * **93.42% Accuracy (0.9832 ROC-AUC)**: 5-Day Forward Bull/Bear Macro Trend Regime Classifier.
  * **71.48% Accuracy (0.8033 ROC-AUC)**: Forward Market Volatility Regime Classifier.
* **Classical Time Series & Machine Learning Benchmarks**:
  * **ARMA(2, 7)** (AutoRegressive Moving Average on stationary log-returns).
  * **Random Forest Regressor** (100 trees with 24 engineered technical lag features).
  * **Linear Regression (OLS)** & Constant Baseline.
* **Interactive Web Portfolio Dashboard**:
  * Real-time model inference simulator powered directly by saved disk binaries (.keras, .joblib, .pkl).
  * Interactive Multi-Series Chart.js visualizer with individual model toggling.
  * Model metric scorecards (MAE, RMSE, ^2$ Score, Directional Accuracy, ROC-AUC).
  * Model binary storage registry and one-click retraining.
* **Reproducible Jupyter Notebook**:
  * 20-step quantitative workflow (
otebooks/stock_market_analysis.ipynb) covering data cleaning, EDA, ADF/KPSS stationarity testing, feature engineering, and model validation.

---

## 🏗 Project Architecture

`
├── data/
│   └── dow_jones.csv                # Historical Dow Jones index OHLCV dataset
└── mini_projects_website/
    ├── app.py                       # Flask server and REST APIs (/api/predict, /api/model-metrics)
    ├── train_regimes.py             # Regime classifier training pipeline
    ├── requirements.txt             # Project dependencies
    ├── notebooks/
    │   └── stock_market_analysis.ipynb  # End-to-end 20-step research notebook
    ├── saved_models/                # Persisted model binaries & scalers
    │   ├── regime_classifier.joblib # 93.4% Bull/Bear Trend Model
    │   ├── volatility_classifier.joblib # 71.5% Volatility Model
    │   ├── lstm_model.keras         # 60-Day Lookback Stacked LSTM
    │   ├── simple_rnn_model.keras   # 60-Day Lookback Simple RNN
    │   ├── arma_model.pkl           # Statsmodels ARMA(2,7)
    │   ├── random_forest_lag.joblib # Random Forest + 24 features
    │   ├── linear_regression_lag.joblib
    │   ├── baseline_model.joblib
    │   ├── dl_scaler.joblib         # Training-only fitted standard scaler
    │   └── model_metrics.json       # Live metrics cache
    ├── static/
    │   ├── css/style.css            # Dark-mode theme UI styling
    │   └── js/stock_market.js       # Chart.js visualizer & live simulator logic
    └── templates/
        ├── base.html                # Base layout template
        ├── index.html               # Portfolio homepage
        └── stock_market.html        # Stock forecasting dashboard
`

---

## 📊 Model Evaluation Summary

| Model | Architecture / Type | MAE | RMSE | ^2$ Score | Directional / Class Accuracy | Persisted Binary |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **Bull/Bear Regime Classifier** | Gradient Boosting | **0.0587** | **0.2423** | **+0.4342** | **93.42%** (0.9832 ROC-AUC) | egime_classifier.joblib |
| **Market Volatility Classifier** | Random Forest | **0.2852** | **0.5340** | **+0.2148** | **71.48%** (0.8033 ROC-AUC) | olatility_classifier.joblib |
| **Stacked LSTM Neural Network** | Deep Learning (Keras) | **0.008748** | **0.012908** | -0.0030 | 52.42% | lstm_model.keras |
| **Simple RNN Neural Network** | Deep Learning (Keras) | **0.008759** | **0.012923** | -0.0053 | 52.42% | simple_rnn_model.keras |
| **ARMA(2,7) Time Series** | Statsmodels | 0.008774 | 0.012944 | -0.0086 | 49.85% | rma_model.pkl |
| **Random Forest + Features** | Ensemble ML | 0.008806 | 0.012904 | -0.0023 | 49.04% | andom_forest_lag.joblib |
| **Linear Regression** | Supervised ML | 0.008902 | 0.013020 | -0.0205 | 48.82% | linear_regression_lag.joblib |
| **Baseline Mean** | Benchmark | 0.008747 | 0.012889 | 0.0000 | 50.00% | aseline_model.joblib |

---

## 🚀 Quick Start & Installation

### 1. Clone the Repository
`ash
git clone https://github.com/karuraghavendra08-crypto/Time-series-based-stock-market-forecasting.git
cd Time-series-based-stock-market-forecasting
`

### 2. Set Up Virtual Environment & Dependencies
`ash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

pip install -r mini_projects_website/requirements.txt
`

### 3. Launch the Web Application
`ash
cd mini_projects_website
python app.py
`
Open your browser and navigate to: **http://127.0.0.1:5000**

### 4. Run the Jupyter Notebook
`ash
jupyter notebook notebooks/stock_market_analysis.ipynb
`

---

## 📜 License
This project is licensed under the MIT License.
