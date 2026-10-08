# 📈 Time Series Based Stock Market Forecasting Using Python

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-Web_App-lightgrey.svg)](https://flask.palletsprojects.com/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15+-orange.svg)](https://www.tensorflow.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-latest-green.svg)](https://scikit-learn.org/)
[![Live Demo](https://img.shields.io/badge/🚀_Live_Demo-Render-brightgreen.svg)](https://stock-market-forecasting-6cw5.onrender.com/project/stock-market)
[![GitHub](https://img.shields.io/badge/GitHub-Repository-black.svg)](https://github.com/karuraghavendra08-crypto/Time-series-based-stock-market-forecasting)

---

## 🧠 What Is This Project? (Simple Explanation)

This project **predicts the stock market's next-day direction (Up ▲ or Down ▼)** using historical price data and machine learning.

Think of it like a weather forecast — instead of predicting rain or sunshine, it predicts whether the stock market will **go up (Bullish 🟢) or go down (Bearish 🔴)** the next day.

It uses **7 different AI/ML models**, compares their accuracy, and presents everything on a **live interactive website** that anyone can visit online.

---

## 🎯 What Problem Does It Solve?

> *"Can a computer learn from past stock prices and predict whether tomorrow's market will rise or fall?"*

Yes — and this project proves it. It applies multiple machine learning and deep learning techniques to **Dow Jones Industrial Average (^DJI)** historical data and live real-time market feeds to make predictions.

---

## 🔑 Key Features (What It Can Do)

| Feature | Description |
|---|---|
| 📡 **Live Market Data** | Fetches real-time stock prices from Yahoo Finance |
| 🤖 **7 ML/AI Models** | Random Forest, LSTM, RNN, Gradient Boosting, ARMA & more |
| 📊 **Interactive Charts** | Visual comparison of all model predictions on a web dashboard |
| 🌐 **Live Website** | Deployed on Render — accessible from anywhere in the world |
| 🔮 **Instant Predictions** | Enter any stock ticker and get an immediate Up/Down forecast |

---

## 🛠️ Technology Stack (Tools Used)

| Category | Tool / Library |
|---|---|
| **Language** | Python 3.11 |
| **Web Framework** | Flask |
| **Deep Learning** | TensorFlow / Keras (LSTM, RNN) |
| **Machine Learning** | Scikit-Learn (Random Forest, Gradient Boosting, Logistic Regression) |
| **Time Series** | Statsmodels (ARMA) |
| **Live Data** | yfinance (Yahoo Finance API) |
| **Frontend Charts** | Chart.js |
| **Deployment** | Render + GitHub CI/CD |

---

## 📊 Model Performance Summary

> All models predict **market direction** — Up (1) or Down (0). This is a binary classification problem.

| Model | Accuracy | ROC-AUC | Notes |
|---|---|---|---|
| 🥇 **Bull/Bear Regime Classifier** | **93.42%** | 0.9832 | Best model — identifies overall market mood |
| 🥈 **Random Forest** | 80.38% | 0.8840 | 300 decision trees voting together |
| 🥉 **Gradient Boosting** | 79.84% | 0.8757 | Trees that learn from each other's mistakes |
| **Logistic Regression** | 79.03% | 0.8576 | Simple but effective linear classifier |
| **Simple RNN** | 73.39% | 0.7426 | Neural network with memory of recent days |
| **Volatility Classifier** | 71.48% | 0.8033 | Predicts calm vs. turbulent market periods |
| **ARMA(2,7)** | 68.55% | — | Traditional statistics time-series model |
| **LSTM Neural Network** | 63.44% | 0.6255 | Deep learning with long-term memory |
| *Baseline (always predict Up)* | *60.75%* | *—* | *Random-guess benchmark* |

---

## 🏗️ How It Works — Step by Step

```
1. DATA COLLECTION
   └─ Historical Dow Jones CSV + Live Yahoo Finance API (yfinance)

2. FEATURE ENGINEERING
   └─ 32 technical indicators computed: RSI, MACD, Bollinger Bands, EMA, SMA...

3. MODEL TRAINING
   └─ 7 models trained & saved to disk (.joblib, .keras, .pkl)

4. WEB DASHBOARD (Flask)
   └─ User picks a stock ticker → gets an instant Up/Down prediction

5. LIVE INFERENCE
   └─ App fetches live data → computes 32 features → runs models → shows result
```

---

## 🌐 Live Links

- **🚀 Live Website**: [https://stock-market-forecasting-6cw5.onrender.com/project/stock-market](https://stock-market-forecasting-6cw5.onrender.com/project/stock-market)
- **💻 GitHub (View Only)**: [https://github.com/karuraghavendra08-crypto/Time-series-based-stock-market-forecasting/blob/main/README.md](https://github.com/karuraghavendra08-crypto/Time-series-based-stock-market-forecasting/blob/main/README.md)

---

## 💻 Run It Locally (3 Steps)

```bash
# 1. Clone the project
git clone https://github.com/karuraghavendra08-crypto/Time-series-based-stock-market-forecasting.git
cd Time-series-based-stock-market-forecasting

# 2. Install all libraries
pip install -r requirements.txt

# 3. Start the website
python app.py
```
Open **http://127.0.0.1:5000** in your browser. 🎉

---

## 📁 Project Structure

```
📦 Project Root
├── 📂 mini_projects_website/
│   ├── app.py              ← Flask web server (main backend)
│   ├── train_models.py     ← ML model training pipeline
│   ├── 📂 saved_models/    ← Trained .joblib / .keras / .pkl files
│   ├── 📂 templates/       ← HTML pages (Jinja2)
│   ├── 📂 static/          ← CSS, JavaScript, images
│   └── 📂 data/            ← dow_jones.csv dataset
├── README.md               ← This file
├── requirements.txt        ← Python dependencies
├── render.yaml             ← Render deployment config
└── Procfile                ← Gunicorn startup command
```

---

## 🎤 Interview Q&A — Quick Reference

**Q: What is the goal of this project?**
> To predict whether the Dow Jones stock market will go up or down the next trading day using historical data and machine learning.

**Q: What dataset did you use?**
> Historical Dow Jones Industrial Average (^DJI) OHLCV data stored in `dow_jones.csv`, plus real-time data pulled from Yahoo Finance via the `yfinance` Python library.

**Q: What is a time series?**
> A sequence of data points collected at regular time intervals — like daily stock closing prices. We use patterns in past values to predict future ones.

**Q: What are technical indicators and why are they important?**
> Mathematical calculations derived from price and volume data — e.g., RSI (momentum), MACD (trend direction), Bollinger Bands (volatility ranges). We computed 32 of these as features to give our models richer signals than raw price alone.

**Q: Why did you use 7 different models instead of just one?**
> To rigorously compare traditional ML (Random Forest, Logistic Regression), boosting methods (Gradient Boosting), deep learning (LSTM, RNN), and statistical approaches (ARMA). Each model family has different strengths, and comparing them is itself a key research contribution.

**Q: What is LSTM and why is it relevant here?**
> Long Short-Term Memory — a recurrent neural network that maintains a "memory" of past time steps. It's relevant because stock prices depend heavily on recent history (we feed it 60 days of past data).

**Q: What does ROC-AUC mean?**
> Area Under the ROC Curve — measures how well a model separates two classes (Up vs. Down). 1.0 = perfect, 0.5 = random guessing. Our best model scores 0.9832.

**Q: How did you deploy the project?**
> Using Render (a cloud hosting platform) connected to GitHub. Every `git push` automatically redeploys the website via CI/CD pipeline.

**Q: What was the biggest challenge?**
> Preventing **data leakage** — accidentally using future data during training, which gives falsely high accuracy. In time-series, you must always train on past data and test on strictly future data.

**Q: What is your best model and why?**
> The Bull/Bear Regime Classifier at 93.42% accuracy. It uses 8 macroeconomic/trend features to classify the overall market regime (bull vs. bear), which is more stable than predicting individual daily wiggles.

---

## 💼 LinkedIn Post Template

> **Instructions for ChatGPT**: Copy the text in the box below and give ChatGPT this prompt:
> *"Polish this LinkedIn post to sound professional and engaging. Keep the technical terms but make it easy for non-technical readers too. Do not change the links."*

---

```
🚀 Excited to share my latest project: Time Series Based Stock Market Forecasting Using Python!

As part of my journey in Data Science and Machine Learning, I built an end-to-end stock market prediction platform that:

📊 Trains 7 ML/AI models — LSTM, RNN, Random Forest, Gradient Boosting, ARMA & more
📡 Fetches LIVE stock prices in real time using Yahoo Finance API
🌐 Serves predictions through an interactive Flask web dashboard — deployed online!
🔮 Predicts next-day market direction (Bullish/Bearish) with up to 93.42% accuracy

✅ 32 engineered technical indicators (RSI, MACD, Bollinger Bands, EMA, SMA...)
✅ Deep learning with 60-day lookback windows (LSTM & RNN)
✅ Ensemble models outperforming deep learning baselines
✅ CI/CD auto-deployment via GitHub + Render
✅ Interactive frontend charts built with Chart.js

🔗 Live Demo: https://stock-market-forecasting-6cw5.onrender.com/project/stock-market
💻 GitHub: https://github.com/karuraghavendra08-crypto/Time-series-based-stock-market-forecasting/blob/main/README.md

This project taught me how to handle real-world financial data, prevent data leakage in time-series models, and build production-ready ML web applications from scratch.

Would love to hear your feedback! 🙌

#MachineLearning #DataScience #Python #StockMarket #DeepLearning #Flask
#TimeSeries #LSTM #MLProject #AI #FinTech #OpenToWork
```
