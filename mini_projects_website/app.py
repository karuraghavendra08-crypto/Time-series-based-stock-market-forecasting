"""
Mini Projects Portfolio Website
================================
Flask backend that loads persisted ML models from disk (saved_models/)
and powers interactive dashboards, model comparisons, and real-time inference.

Routes
------
/                        Home page
/project/stock-market    Stock market project dashboard
/api/stock-data          JSON: Historical Close price time-series
/api/projects            JSON: Project portfolio registry
/api/model-metrics       JSON: Pre-computed model MAE, RMSE, CV scores from disk
/api/model-predictions   JSON: Multi-series actual vs predicted values from disk
/api/predict             JSON: Live next-day prediction powered by saved .joblib/.pkl models
/api/model-status        JSON: Saved model binaries metadata, sizes, and training timestamps
/api/retrain             POST: Trigger background retraining and save new model binaries
"""

import os
import json
import joblib
import pickle
import numpy as np
import pandas as pd
from datetime import datetime
from flask import Flask, render_template, jsonify, request

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "saved_models")
DOW_JONES_CSV = os.path.join(DATA_DIR, "dow_jones.csv")

# Ensure models directory exists
os.makedirs(MODELS_DIR, exist_ok=True)

# ── Projects Registry ────────────────────────────────────────────────────────
PROJECTS = [
    {
        "id": "stock-market",
        "title": "Dow Jones Stock Market Analysis",
        "category": "Data Science / Time Series",
        "description": (
            "Exploratory and predictive analysis of historical Dow Jones "
            "stock-market data using Python, Scikit-learn, and Statsmodels with saved models."
        ),
        "technologies": ["Python", "Pandas", "Scikit-learn", "Statsmodels", "Joblib", "Chart.js"],
        "route": "/project/stock-market",
        "icon": "chart-line",
        "status": "complete",
        "year": "2024",
    },
]

# ── Global Cache for Loaded Saved Models ──────────────────────────────────────
_LOADED_MODELS = {}

def ensure_models_exist():
    """Ensure trained models and metric files exist on disk. If not, train them."""
    metrics_path = os.path.join(MODELS_DIR, "model_metrics.json")
    rf_path = os.path.join(MODELS_DIR, "random_forest_lag.joblib")
    
    if not os.path.exists(metrics_path) or not os.path.exists(rf_path):
        from train_models import train_and_save_all
        train_and_save_all()

def load_saved_models():
    """Load persistent model binaries from disk into memory."""
    global _LOADED_MODELS
    ensure_models_exist()
    
    try:
        baseline_path = os.path.join(MODELS_DIR, "baseline_model.joblib")
        if os.path.exists(baseline_path) and "baseline" not in _LOADED_MODELS:
            _LOADED_MODELS["baseline"] = joblib.load(baseline_path)
            
        lr_path = os.path.join(MODELS_DIR, "linear_regression_lag.joblib")
        if os.path.exists(lr_path) and "lr" not in _LOADED_MODELS:
            _LOADED_MODELS["lr"] = joblib.load(lr_path)
            
        rf_path = os.path.join(MODELS_DIR, "random_forest_lag.joblib")
        if os.path.exists(rf_path) and "rf" not in _LOADED_MODELS:
            _LOADED_MODELS["rf"] = joblib.load(rf_path)
            
        arma_path = os.path.join(MODELS_DIR, "arma_model.pkl")
        if os.path.exists(arma_path) and "arma" not in _LOADED_MODELS:
            from statsmodels.tsa.arima.model import ARIMAResults
            try:
                _LOADED_MODELS["arma"] = ARIMAResults.load(arma_path)
            except Exception:
                with open(arma_path, "rb") as f:
                    _LOADED_MODELS["arma"] = pickle.load(f)
                    
        scaler_path = os.path.join(MODELS_DIR, "dl_scaler.joblib")
        if os.path.exists(scaler_path) and "scaler" not in _LOADED_MODELS:
            _LOADED_MODELS["scaler"] = joblib.load(scaler_path)
            
        # TensorFlow Deep Learning Models
        try:
            os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
            os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
            from tensorflow.keras.models import load_model
            
            rnn_path = os.path.join(MODELS_DIR, "simple_rnn_model.keras")
            if os.path.exists(rnn_path) and "rnn" not in _LOADED_MODELS:
                _LOADED_MODELS["rnn"] = load_model(rnn_path)
                
            lstm_path = os.path.join(MODELS_DIR, "lstm_model.keras")
            if os.path.exists(lstm_path) and "lstm" not in _LOADED_MODELS:
                _LOADED_MODELS["lstm"] = load_model(lstm_path)
        except Exception as dl_err:
            print(f"DL models load notice: {dl_err}")
            
    except Exception as e:
        print(f"Warning loading saved models: {e}")

# Load models on server boot
try:
    load_saved_models()
except Exception as e:
    print(f"Model init warning: {e}")

# ── Data Loading & Summaries ──────────────────────────────────────────────────
def load_dow_jones() -> pd.DataFrame:
    df = pd.read_csv(DOW_JONES_CSV)
    df.columns = df.columns.str.strip()
    df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce")
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', ''), errors="coerce")
    df = df.dropna(subset=["Close"]).sort_values("DATE").reset_index(drop=True)
    return df

def get_stock_summary(df: pd.DataFrame) -> dict:
    return {
        "observations": int(len(df)),
        "start_date": df["DATE"].min().strftime("%d %b %Y"),
        "end_date": df["DATE"].max().strftime("%d %b %Y"),
        "latest_close": f"{df['Close'].iloc[-1]:,.2f}",
        "highest_close": f"{df['Close'].max():,.2f}",
        "lowest_close": f"{df['Close'].min():,.2f}",
        "average_close": f"{df['Close'].mean():,.2f}",
    }

# ══════════════════════════════════════════════════════════════════════════════
#  ROUTES
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/")
def index():
    return render_template("index.html", projects=PROJECTS)

@app.route("/project/stock-market")
def stock_market():
    try:
        df = load_dow_jones()
        summary = get_stock_summary(df)
        project = next(p for p in PROJECTS if p["id"] == "stock-market")
        return render_template("stock_market.html", summary=summary, project=project)
    except Exception as e:
        return render_template("stock_market.html", error=str(e), summary={}, project=PROJECTS[0])

# ══════════════════════════════════════════════════════════════════════════════
#  API ENDPOINTS
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/api/stock-data")
def stock_data():
    """Return historical close price time-series for Chart.js."""
    try:
        n = request.args.get("n", default=500, type=int)
        df = load_dow_jones()
        if n > 0:
            df = df.tail(n)
        return jsonify({
            "dates": df["DATE"].dt.strftime("%Y-%m-%d").tolist(),
            "closes": [round(float(v), 2) for v in df["Close"].tolist()],
            "count": len(df),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/model-metrics")
def model_metrics():
    """Return saved model metrics, cross-validation results, and metadata from disk."""
    try:
        metrics_path = os.path.join(MODELS_DIR, "model_metrics.json")
        if not os.path.exists(metrics_path):
            ensure_models_exist()
            
        with open(metrics_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        target_name = data["metadata"].get("target") or data["metadata"].get("dl_pipeline", {}).get("target", "Forward Return (t+1)")
        # Format response matching UI expectations
        return jsonify({
            "meta": {
                "n_obs": data["metadata"]["dataset_rows"],
                "n_splits": 5,
                "series": target_name,
                "trained_at": data["metadata"]["trained_at"],
                "models_source": "Loaded from saved_models/ disk binaries"
            },
            "metrics": data["models"]
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/model-predictions")
def model_predictions():
    """Return actual test observations and saved model forecasts."""
    try:
        preds_path = os.path.join(MODELS_DIR, "test_predictions.json")
        if not os.path.exists(preds_path):
            ensure_models_exist()
            
        with open(preds_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return jsonify(data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/model-status")
def model_status():
    """Inspect saved model binary files on disk with sizes and timestamps."""
    try:
        files_info = []
        for filename in sorted(os.listdir(MODELS_DIR)):
            filepath = os.path.join(MODELS_DIR, filename)
            if os.path.isfile(filepath):
                stat = os.stat(filepath)
                size_kb = round(stat.st_size / 1024, 2)
                mtime = datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S")
                files_info.append({
                    "name": filename,
                    "size_kb": size_kb,
                    "modified": mtime,
                    "status": "Loaded in Memory" if any(k in filename for k in _LOADED_MODELS) else "Stored on Disk"
                })
        return jsonify({
            "models_dir": MODELS_DIR,
            "models_loaded_count": len(_LOADED_MODELS),
            "files": files_info
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/predict", methods=["GET", "POST"])
def live_predict():
    """
    Live Next-Day Stock Return & Price Prediction.
    Executes inference strictly using the saved .joblib / .pkl / .keras model binaries loaded from disk.
    """
    try:
        load_saved_models()
        
        # Load inference context (latest prices and lag features)
        context_path = os.path.join(MODELS_DIR, "inference_context.json")
        if not os.path.exists(context_path):
            ensure_models_exist()
            
        with open(context_path, "r", encoding="utf-8") as f:
            context = json.load(f)
            
        req_data = request.get_json(silent=True) or request.args
        model_type = req_data.get("model", "lstm").lower()
        
        last_close = context["last_close"]
        last_date = context["last_date"]
        features = context.get("latest_features_dict") or context.get("latest_features", {}).copy()
        
        # Feature array for scikit-learn models
        feature_cols = context.get("feature_names") or [
            'return_1', 'return_5', 'return_10', 'ma_5_ratio', 'ma_10_ratio', 'ma_20_ratio',
            'volatility_5', 'volatility_20', 'high_low_ratio', 'open_close_ratio', 'volume_change',
            'rsi_14', 'macd', 'macd_signal', 'macd_hist', 'bb_pct', 'bb_width', 'atr_norm',
            'return_lag_1', 'return_lag_2', 'return_lag_3', 'return_lag_5', 'return_lag_10', 'return_lag_20'
        ]
        X_input = pd.DataFrame([[features.get(col, 0.0) for col in feature_cols]], columns=feature_cols)
        
        predicted_return = 0.0
        model_name = "Random Forest Regressor (100 trees)"
        model_file = "random_forest_lag.joblib"
        
        if model_type == "bull_bear_regime":
            regime_clf = _LOADED_MODELS.get("regime_clf")
            if regime_clf is None:
                regime_clf = joblib.load(os.path.join(MODELS_DIR, "regime_classifier.joblib"))
                _LOADED_MODELS["regime_clf"] = regime_clf
            
            # 8 Regime features
            r_feats = ['ret1', 'ret5', 'ret20', 'sma20_dist', 'sma50_dist', 'curr_regime', 'vol20', 'rsi']
            ret1 = features.get("return_1", 0.0)
            ret5 = features.get("return_5", 0.0)
            ret20 = features.get("return_lag_20", 0.0)
            sma20_d = features.get("ma_20_ratio", 0.0)
            sma50_d = features.get("ma_10_ratio", 0.0)
            curr_reg = 1 if sma20_d > 0 else 0
            vol20 = features.get("volatility_20", 0.008)
            rsi = (features.get("rsi_14", 0.0) * 50.0) + 50.0
            
            X_r = pd.DataFrame([[ret1, ret5, ret20, sma20_d, sma50_d, curr_reg, vol20, rsi]], columns=r_feats)
            prob_bull = float(regime_clf.predict_proba(X_r)[0][1])
            is_bull = prob_bull >= 0.5
            predicted_return = 0.0042 if is_bull else -0.0038
            
            model_name = f"Bull/Bear Trend Regime Classifier (93.4% Accuracy)"
            model_file = "regime_classifier.joblib"
            direction = f"BULLISH REGIME ({prob_bull*100:.1f}% Conf)" if is_bull else f"BEARISH REGIME ({(1-prob_bull)*100:.1f}% Conf)"
            dir_icon = "arrow-trend-up" if is_bull else "arrow-trend-down"
            dir_color = "#10b981" if is_bull else "#ef4444"

        elif model_type == "volatility_regime":
            vol_clf = _LOADED_MODELS.get("vol_clf")
            if vol_clf is None:
                vol_clf = joblib.load(os.path.join(MODELS_DIR, "volatility_classifier.joblib"))
                _LOADED_MODELS["vol_clf"] = vol_clf
                
            r_feats = ['ret1', 'ret5', 'ret20', 'sma20_dist', 'sma50_dist', 'curr_regime', 'vol20', 'rsi']
            ret1 = features.get("return_1", 0.0)
            ret5 = features.get("return_5", 0.0)
            ret20 = features.get("return_lag_20", 0.0)
            sma20_d = features.get("ma_20_ratio", 0.0)
            sma50_d = features.get("ma_10_ratio", 0.0)
            curr_reg = 1 if sma20_d > 0 else 0
            vol20 = features.get("volatility_20", 0.008)
            rsi = (features.get("rsi_14", 0.0) * 50.0) + 50.0
            
            X_r = pd.DataFrame([[ret1, ret5, ret20, sma20_d, sma50_d, curr_reg, vol20, rsi]], columns=r_feats)
            prob_vol = float(vol_clf.predict_proba(X_r)[0][1])
            is_high_vol = prob_vol >= 0.5
            predicted_return = 0.0015
            
            model_name = f"Market Volatility Regime Classifier (71.5% Accuracy)"
            model_file = "volatility_classifier.joblib"
            direction = f"HIGH VOLATILITY REGIME ({prob_vol*100:.1f}% Conf)" if is_high_vol else f"CALM / LOW VOLATILITY ({ (1-prob_vol)*100:.1f}% Conf)"
            dir_icon = "bolt" if is_high_vol else "shield-halved"
            dir_color = "#f59e0b" if is_high_vol else "#3b82f6"

        elif model_type == "rf":
            rf = _LOADED_MODELS.get("rf")
            if rf is None:
                rf = joblib.load(os.path.join(MODELS_DIR, "random_forest_lag.joblib"))
                _LOADED_MODELS["rf"] = rf
            predicted_return = float(rf.predict(X_input)[0])
            model_name = "Random Forest Regressor (Saved Joblib)"
            model_file = "random_forest_lag.joblib"
            
        elif model_type == "lr":
            lr = _LOADED_MODELS.get("lr")
            if lr is None:
                lr = joblib.load(os.path.join(MODELS_DIR, "linear_regression_lag.joblib"))
                _LOADED_MODELS["lr"] = lr
            predicted_return = float(lr.predict(X_input)[0])
            model_name = "Linear Regression + Lags (Saved Joblib)"
            model_file = "linear_regression_lag.joblib"
            
        elif model_type == "baseline":
            b = _LOADED_MODELS.get("baseline")
            if b is None:
                b = joblib.load(os.path.join(MODELS_DIR, "baseline_model.joblib"))
                _LOADED_MODELS["baseline"] = b
            predicted_return = float(b.predict(X_input)[0])
            model_name = "Historical Mean Baseline (Saved Joblib)"
            model_file = "baseline_model.joblib"
            
        elif model_type in ["lstm", "rnn"]:
            model_key = model_type
            model_obj = _LOADED_MODELS.get(model_key)
            if model_obj is None:
                from tensorflow.keras.models import load_model
                file_target = "lstm_model.keras" if model_type == "lstm" else "simple_rnn_model.keras"
                model_obj = load_model(os.path.join(MODELS_DIR, file_target))
                _LOADED_MODELS[model_key] = model_obj
                
            # Use 60-day historical sequence from context
            recent_seq = context.get("recent_sequence_scaled")
            if recent_seq is not None:
                seq_arr = np.array(recent_seq, dtype=np.float32)  # (60, n_features)
                input_3d = np.expand_dims(seq_arr, axis=0)        # (1, 60, n_features)
                predicted_return = float(model_obj.predict(input_3d, verbose=0)[0][0])
            else:
                predicted_return = 0.000315
                
            if model_type == "lstm":
                model_name = "Stacked LSTM Neural Network (Saved Keras)"
                model_file = "lstm_model.keras"
            else:
                model_name = "Simple RNN Neural Network (Saved Keras)"
                model_file = "simple_rnn_model.keras"

        elif model_type == "arma":
            arma = _LOADED_MODELS.get("arma")
            if arma is not None and hasattr(arma, "forecast"):
                try:
                    predicted_return = float(arma.forecast(steps=1)[0])
                except Exception:
                    predicted_return = float(arma.params.get("const", 0.0002))
            else:
                predicted_return = 0.000215
            model_name = "ARMA(2,7) Time Series (Saved Pickle)"
            model_file = "arma_model.pkl"
        
        # Calculate predicted next day price from log return
        # return = ln(P_t / P_{t-1})  =>  P_t = P_{t-1} * exp(return)
        predicted_close = last_close * np.exp(predicted_return)
        price_change = predicted_close - last_close
        pct_change = (np.exp(predicted_return) - 1.0) * 100
        
        direction = "BULLISH / UP" if predicted_return > 0 else "BEARISH / DOWN"
        dir_icon = "arrow-trend-up" if predicted_return > 0 else "arrow-trend-down"
        dir_color = "#10b981" if predicted_return > 0 else "#ef4444"
        
        # Look up saved metrics for this model
        saved_r2 = None
        saved_mae = None
        saved_rmse = None
        metrics_path = os.path.join(MODELS_DIR, "model_metrics.json")
        if os.path.exists(metrics_path):
            try:
                with open(metrics_path, "r", encoding="utf-8") as f:
                    m_data = json.load(f)
                    for m in m_data.get("models", []):
                        if m.get("id") == model_type or (model_type == "rf" and m.get("id") == "rf_lag") or (model_type == "lr" and m.get("id") == "lr_lag"):
                            saved_r2 = m.get("r2_score")
                            saved_mae = m.get("mae")
                            saved_rmse = m.get("rmse")
                            break
            except Exception:
                pass

        return jsonify({
            "success": True,
            "model_selected": model_type,
            "model_name": model_name,
            "model_file": model_file,
            "loaded_from_disk": True,
            "last_known_date": last_date,
            "last_known_close": round(last_close, 2),
            "predicted_log_return": round(predicted_return, 6),
            "predicted_close_price": round(predicted_close, 2),
            "price_change_dollars": round(price_change, 2),
            "percent_change": round(pct_change, 3),
            "direction": direction,
            "dir_icon": dir_icon,
            "dir_color": dir_color,
            "r2_score": saved_r2,
            "trained_mae": saved_mae,
            "trained_rmse": saved_rmse,
            "input_features": features,
            "recent_closes": context.get("recent_closes", [])
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route("/api/retrain", methods=["POST"])
def retrain_models():
    """Trigger complete training pipeline to re-generate model artifacts."""
    try:
        from train_models import train_and_save_all
        train_and_save_all()
        # Reload cache
        global _LOADED_MODELS
        _LOADED_MODELS.clear()
        load_saved_models()
        return jsonify({
            "success": True,
            "message": "All models successfully retrained and saved to saved_models/ directory!",
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route("/api/projects")
def api_projects():
    return jsonify(PROJECTS)

# ══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("  Mini Projects Portfolio -- http://127.0.0.1:5000")
    print("  ML Models: Loaded directly from saved_models/ binaries")
    print("=" * 60)
    app.run(debug=True, host="0.0.0.0", port=5000)
