import urllib.request
import json
import time

time.sleep(2)
base = "http://127.0.0.1:5000"

print("--- Testing /api/live-market-data ---")
r = urllib.request.urlopen(f"{base}/api/live-market-data?ticker=^DJI", timeout=15)
data = json.loads(r.read())
print("Ticker:", data.get("ticker"), "| Name:", data.get("name"))
print("Date:", data.get("last_date"), "| Close: $", data.get("last_close"))
print("Day Change:", data.get("day_change"), f"({data.get('day_pct_change')}%)")
print("Lags (1d, lag1, lag2, 5d, 10d):", data.get("lags"))
print("RSI:", data.get("rsi_actual"), "| Volatility 20d:", data.get("volatility_20"))
print("Source:", data.get("source"))

print("\n--- Testing Live Predictions for Multiple Models ---")
models = ["bull_bear_regime", "volatility_regime", "rf", "gb", "lr", "rnn", "lstm", "arma"]
for m in models:
    url = f"{base}/api/predict?model={m}&use_live_market=true&ticker=^DJI"
    res = urllib.request.urlopen(url, timeout=15)
    p = json.loads(res.read())
    print(f"[{m}] {p.get('model_name')}")
    print(f"     => {p.get('direction')} (Prob UP: {p.get('prob_up')}) | Live Close: ${p.get('last_known_close')} -> Pred: ${p.get('predicted_close_price')} | {p.get('market_source')}")

print("\n--- Testing Tech Stock Tickers (AAPL, NVDA, MSFT) ---")
for sym in ["AAPL", "NVDA", "MSFT", "SPY"]:
    url = f"{base}/api/predict?model=bull_bear_regime&use_live_market=true&ticker={sym}"
    res = urllib.request.urlopen(url, timeout=15)
    p = json.loads(res.read())
    print(f"[{sym}] Live Close: ${p.get('last_known_close')} => {p.get('direction')} ({p.get('percent_change'):+.2f}%)")

print("\nAll live market API tests passed successfully!")
