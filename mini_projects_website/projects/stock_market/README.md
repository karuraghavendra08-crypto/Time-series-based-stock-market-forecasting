# Dow Jones Stock Market Analysis

**Category:** Data Science / Time Series
**Status:** Complete
**Dataset:** `data/dow_jones.csv`
**Notebook:** `notebooks/stock_market_analysis.ipynb`

---

## Overview

This project performs a comprehensive time-series analysis of the Dow Jones Industrial Average
using 100+ years of daily market data (1901–2012).

The analysis follows a rigorous workflow:

1. Data loading and cleaning
2. Exploratory data analysis
3. Log transformation and first differencing
4. Stationarity testing (ADF and KPSS)
5. Time-series cross-validation
6. Multiple forecasting models including ARMA, ARMAX, and Random Forest

---

## Key Findings

- The raw Close price is non-stationary; log-differencing produces a near-stationary series.
- ARMA(2,7) is fitted on the first-differenced log Close price.
- Random Forest with lagged prices provides a competitive baseline without weather data.
- Weather variables (from LaGuardia Airport) were investigated in the original analysis
  but are not available in this repository.

---

## Files

| File | Description |
|------|-------------|
| `data/dow_jones.csv` | Raw Dow Jones dataset |
| `notebooks/stock_market_analysis.ipynb` | Full analysis notebook |
| `templates/stock_market.html` | Flask project page |
| `static/js/stock_market.js` | Interactive chart logic |

---

## Running the Analysis

```bash
# From mini_projects_website/
jupyter notebook notebooks/stock_market_analysis.ipynb
```
