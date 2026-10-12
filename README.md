# 📈 Time Series Based Stock Market Forecasting Using Python

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-Web_App-lightgrey.svg)](https://flask.palletsprojects.com/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15+-orange.svg)](https://www.tensorflow.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-latest-green.svg)](https://scikit-learn.org/)
[![Live Demo](https://img.shields.io/badge/🚀_Live_Demo-Render-brightgreen.svg)](https://stock-market-forecasting-6cw5.onrender.com/project/stock-market)
[![GitHub](https://img.shields.io/badge/GitHub-Repository-black.svg)](https://github.com/karuraghavendra08-crypto/Time-series-based-stock-market-forecasting)

---

## 🔬 Project Overview (Executive Summary)

**Time Series Based Stock Market Forecasting Using Python** is an audited, end-to-end quantitative financial analytics and machine learning platform. It compares classical econometrics, machine learning ensembles, and deep sequence neural networks against historical benchmarks and live real-time market streams from Yahoo Finance.

The platform provides:
1. **Interactive Real-Time Forecaster**: Live forward predictions across Dow Jones Industrial Average (`^DJI`) and major global equities (`AAPL`, `MSFT`, `NVDA`, `SPY`, `QQQ`, `TSLA`, `AMZN`, `GOOGL`).
2. **Dedicated Model Laboratory**: Side-by-side out-of-sample benchmark comparing 8 models with rigorous separation of classification (Accuracy, ROC-AUC, F1) and regression metrics (Return MAE/RMSE).
3. **Prediction Breakdown Panel ("Understand This Forecast")**: Plain-English explanations of model predictions, key technical feature drivers (RSI, SMA ratios, volatility), and verified return-to-price mathematical conversions.
4. **Verified Dynamic Dataset Metadata**: Complete transparency distinguishing the **Historical Century Dow Jones Archive (1901–2012)** from the **Modern Era Benchmark (2015–2025)**.

---

## 📂 Dataset Verification & Audited Observations

| Dataset | File | Coverage Period | Valid Trading Sessions | Unique Dates | Frequency | Price Range | Units |
|---|---|---|---|---|---|---|---|
| **Historical Century Archive** | `dow_jones.csv` | **04 Jan 1901 – 31 Dec 2012** | **27,824** | 27,824 (0 duplicates) | Daily Trading Days | 49.17 – 14,164.50 | Index Points (pts) |
| **Modern Era Benchmark** | `dow_jones_modern_2015_2025.csv` | **02 Jan 2015 – 30 Dec 2025** | **2,765** | 2,765 (0 duplicates) | Daily Trading Days | 15,660.18 – 48,731.16 | Index Points (pts) |
| **Live Stream** | Yahoo Finance API (`yfinance`) | 1-Year Rolling Daily Candles | Live Rolling Sessions | 100% Unique | Real-Time Live Feed | Real-time quote | pts (Index) / $ USD (Equities) |

> **Audit Finding**: The 27,824 observation count belongs to the authentic 111-year historical Dow Jones archive (1901–2012). The modern era covers 2,765 trading sessions (2015–2025). The platform dynamically exposes both datasets via the `/api/dataset-metadata` endpoint.

---

## 📊 Model Comparison Laboratory (Verified Test Benchmark)

All models are evaluated on a **strict chronological forward test split (70% Train / 15% Val / 15% Test)** with zero look-ahead bias and scalers fitted exclusively on the training split.

### A. Trend Regime & Directional Classification Models

| Model | Architecture | Target Definition | Direction Accuracy | ROC-AUC | Precision | Recall | F1-Score | Status |
|---|---|---|---|---|---|---|---|---|
| 🥇 **Bull/Bear Trend Regime** | Gradient Boosting | 5-Day Trend State ($\text{SMA}_{20} > \text{SMA}_{50}$) | **93.42%** | **0.9832** | 94.18% | 95.83% | 0.9500 | Out-of-Sample Verified |
| 🥈 **Random Forest Classifier** | 200 Trees, Balanced Weights | 5-Day Forward Direction | **80.38%** | **0.8840** | 81.22% | 88.05% | 0.8450 | Out-of-Sample Verified |
| 🥉 **Gradient Boosting** | 150 Trees, depth=3 | 5-Day Forward Direction | **79.84%** | **0.8757** | 78.93% | 91.15% | 0.8460 | Out-of-Sample Verified |
| **Logistic Regression** | L2 Regularized Linear Classifier | 5-Day Forward Direction | **79.03%** | **0.8576** | 78.91% | 89.38% | 0.8382 | Out-of-Sample Verified |
| **Simple RNN** | Stacked SimpleRNN(64 $\to$ 32) | 5-Day Forward Direction (30d Lookback) | **73.39%** | **0.7426** | 77.25% | 79.65% | 0.7843 | Out-of-Sample Verified |
| **Market Volatility Classifier** | Random Forest (100 Trees) | 5-Day Volatility Regime (High vs Calm) | **71.48%** | **0.8033** | 72.10% | 70.80% | 0.7144 | Out-of-Sample Verified |
| **Stacked LSTM** | Stacked LSTM(64 $\to$ 32) + BatchNorm | 5-Day Forward Direction (30d Lookback) | **63.44%** | **0.6255** | 65.10% | 85.84% | 0.7405 | Out-of-Sample Verified |
| *Naive Majority Baseline* | Dummy Most-Frequent Class | Benchmark Reference (Always Predict Up) | *60.75%* | *0.5000* | *60.75%* | *100.0%* | *0.7558* | Naive Reference |

### B. Econometric & Continuous Return Models

| Model | Architecture | Prediction Target | Direction Accuracy | Return MAE | Return RMSE | $R^2$ Score | Status |
|---|---|---|---|---|---|---|---|
| **ARMA(2,7)** | Statsmodels ARIMA | Next-Day Log-Return ($r_{t+1}$) | **68.55%** | **0.0078** (0.78%) | 0.0112 | 0.0215 | Out-of-Sample Verified |

> **Scientific Disclosure on the 93.42% Accuracy**: The 93.42% score is the out-of-sample accuracy of the **Macro Trend Regime Classifier**, which predicts whether the 20-day moving average remains above the 50-day moving average 5 days into the future. Because moving averages are smooth and autocorrelated, macro trend states persist over short horizons. It does *not* claim 93% accuracy on predicting noisy next-day random-walk prices.

---

## 🧮 Mathematical Price & Return Conversions

When models forecast a return or implied regime drift $r_{\text{log}}$, price transitions are computed using exact continuous compounding identities:

$$\text{Predicted Target Price} = P_{\text{baseline}} \times \exp(r_{\text{log}})$$
$$\text{Simple Return} = \exp(r_{\text{log}}) - 1.0$$
$$\text{Price Delta } (\Delta P) = P_{\text{target}} - P_{\text{baseline}}$$
$$\text{Percentage Change } (\% \Delta) = 100 \times \text{Simple Return}$$

All four quantities are strictly synchronized with floating-point precision, ensuring $P_{\text{target}}$, $\Delta P$, and $\% \Delta$ agree mathematically without rounding discrepancies.

---

## 🛠️ Technology Stack

| Domain | Tools & Libraries |
|---|---|
| **Programming Language** | Python 3.11 |
| **Web Server & REST API** | Flask 3.0+ |
| **Machine Learning Ensembles** | Scikit-Learn (Random Forest, Gradient Boosting, Logistic Regression) |
| **Deep Sequence Modeling** | TensorFlow 2.15+ / Keras (Stacked LSTM, Simple RNN) |
| **Time Series Econometrics** | Statsmodels (ARMA 2,7) |
| **Data Processing & Feature Engine** | Pandas, NumPy (32 Lookahead-Free Indicators) |
| **Real-Time Market Streaming** | Yahoo Finance API (`yfinance`) |
| **Frontend UI & Visualizations** | Modern Glass-Free Editorial CSS3, Vanilla JS, Chart.js 4.4 |
| **Automated Testing** | Pytest (18 Automated Unit & Integration Tests) |
| **Production Deployment** | Render Cloud Platform + GitHub CI/CD |

---

## 32 Lookahead-Free Technical Indicators

All technical indicators use backward-looking causal rolling windows:
1. **Returns & Momentum**: `return_1`, `return_5`, `return_10`, `return_20`, `momentum_10`, `momentum_20`, `return_lag_1`, `return_lag_2`.
2. **Moving Average Trend Distance**: `ma_5_ratio`, `ma_10_ratio`, `ma_20_ratio`, `ma_50_ratio`, `ma_200_ratio`.
3. **Crossover & Trend Strength**: `sma5_cross_sma20`, `sma20_cross_sma50`, `trend_strength`.
4. **Volatility & Intraday**: `volatility_5`, `volatility_20`, `high_low_ratio`, `open_close_ratio`, `bb_pct`, `bb_width`, `atr_norm`.
5. **Momentum Oscillators**: `rsi_14`, `stoch_k`, `stoch_d`, `williams_r`, `macd`, `macd_signal`, `macd_hist`.
6. **Volume & Flow**: `volume_change`, `obv_change`.

---

## ⚠️ Financial-Risk Disclaimer

> **ABOUT THESE FORECASTS**  
> This application is an educational and experimental platform for financial time-series analysis. Predictions are generated using statistical or machine learning models and historical market observations.  
> Financial markets are uncertain. Historical patterns and past model performance do not guarantee future outcomes. Forecasts may be inaccurate, incomplete, or affected by changes in market conditions and data availability.  
> This application does not guarantee investment returns and should not be treated as personalized financial advice or as a standalone basis for investment decisions.

---

## 💻 Local Setup & Execution Guide

### 1. Clone the Repository
```bash
git clone https://github.com/karuraghavendra08-crypto/Time-series-based-stock-market-forecasting.git
cd Time-series-based-stock-market-forecasting
```

### 2. Create Virtual Environment & Install Dependencies
```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

pip install -r requirements.txt
```

### 3. Run Automated Pytest Suite
```bash
python -m pytest
```
*Expected: 18 passed tests validating data ranges, metadata, model loading, return-to-price mathematics, and REST endpoints.*

### 4. Start the Application Locally
```bash
python app.py
```
Open [http://localhost:5000](http://localhost:5000) or [http://localhost:5000/project/stock-market](http://localhost:5000/project/stock-market) in your browser.

---

## 📁 Project Structure

```
📦 Project Root
├── 📂 mini_projects_website/
│   ├── app.py              ← Flask web server (main backend & REST API)
│   ├── train_models.py     ← ML model training pipeline (zero-leakage)
│   ├── 📂 saved_models/    ← Trained .joblib / .keras / .pkl files & metrics
│   ├── 📂 templates/       ← Jinja2 HTML templates (Editorial black-and-white)
│   ├── 📂 static/          ← High-craft CSS, JavaScript & Chart.js logic
│   └── 📂 data/            ← Historical (1901-2012) & Modern (2015-2025) datasets
├── 📂 tests/               ← Automated test suite (18 unit & integration tests)
├── README.md               ← Comprehensive documentation & audit report
├── requirements.txt        ← Python dependencies
├── render.yaml             ← Render cloud deployment config
└── Procfile                ← Gunicorn production startup command
```

---

## 🎤 Interview Q&A — Quick Reference

**Q: What is the goal of this project?**
> To predict stock market trend direction and macro market regimes using historical data, 32 lookahead-free technical indicators, and machine learning.

**Q: What datasets did you use?**
> The authentic Historical Century Dow Jones Industrial Average (^DJI) daily archive (1901–2012, 27,824 trading days), the Modern Era Benchmark (2015–2025, 2,765 trading days), and real-time live market streaming via Yahoo Finance.

**Q: What is the difference between classification and price regression?**
> Direction classifiers predict market state (Bullish/Bearish, $Y \in \{0, 1\}$) with calibrated class probabilities. Numerical models (such as ARMA) forecast continuous log-returns. The platform strictly separates classification metrics (Accuracy, ROC-AUC, F1) from regression metrics (Return MAE/RMSE).

**Q: What is the source of the 93.42% accuracy?**
> It is the out-of-sample accuracy of the Bull/Bear Macro Trend Regime Classifier, which predicts whether the 20-day moving average remains above the 50-day moving average 5 days into the future. Moving averages exhibit natural trend persistence.

**Q: How did you prevent data leakage?**
> We used strict chronological 70% Train / 15% Val / 15% Test forward splitting without random shuffling, calculated rolling indicators using backward-looking causal windows only, and fitted preprocessing scalers exclusively on the training split.

---

## 🌐 Live Production Links

- **🚀 Live Application on Render**: [https://stock-market-forecasting-6cw5.onrender.com/project/stock-market](https://stock-market-forecasting-6cw5.onrender.com/project/stock-market)
- **💻 GitHub Repository**: [https://github.com/karuraghavendra08-crypto/Time-series-based-stock-market-forecasting](https://github.com/karuraghavendra08-crypto/Time-series-based-stock-market-forecasting)

