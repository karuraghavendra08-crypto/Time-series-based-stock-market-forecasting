# Mini Projects Portfolio Website

A professional, dark-themed Flask web application for showcasing Python, Data Science,
Machine Learning, and Web Development mini projects.

---

## Purpose

This website is a personal portfolio demonstrating practical skills in:

- **Python 3** programming
- **Pandas & NumPy** for data manipulation
- **Scikit-learn & Statsmodels** for machine learning and statistical modelling
- **Time-Series Analysis** (ADF/KPSS stationarity tests, ARMA, ARMAX)
- **Flask** for web backend and REST APIs
- **Chart.js** for interactive browser visualisations
- **Jupyter Notebooks** for reproducible research

---

## Current Projects

| # | Project | Category | Status |
|---|---------|----------|--------|
| 1 | Dow Jones Stock Market Analysis | Data Science / Time Series | ✅ Live |

---

## Folder Structure

```
mini_projects_website/
│
├── app.py                   ← Flask application (routes + API)
├── requirements.txt         ← Python dependencies
├── README.md                ← This file
│
├── data/
│   └── dow_jones.csv        ← Historical Dow Jones dataset
│
├── notebooks/
│   └── stock_market_analysis.ipynb   ← Jupyter notebook
│
├── templates/
│   ├── base.html            ← Shared layout (navbar + footer)
│   ├── index.html           ← Home / project gallery page
│   └── stock_market.html    ← Stock market analysis page
│
├── static/
│   ├── css/
│   │   └── style.css        ← All styles (dark theme)
│   └── js/
│       └── stock_market.js  ← Chart.js logic (fetches /api/stock-data)
│
└── projects/
    └── stock_market/
        └── README.md        ← Project-specific documentation
```

---

## Installation

### 1. Clone / navigate to the project

```bash
cd mini_projects_website
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate it:

**Windows:**
```bash
venv\Scripts\activate
```

**Mac / Linux:**
```bash
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## Running the Flask Website

```bash
python app.py
```

Then open your browser and visit:

```
http://127.0.0.1:5000
```

---

## Running the Jupyter Notebook

```bash
jupyter notebook notebooks/stock_market_analysis.ipynb
```

The notebook uses:

```
../data/dow_jones.csv
```

It will read from the `data/` folder one level up — no external downloads needed.

---

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Home page — project gallery |
| `/project/stock-market` | GET | Dow Jones analysis dashboard |
| `/api/stock-data?n=250` | GET | JSON: last N Close prices |
| `/api/projects` | GET | JSON: projects registry |

---

## Adding a New Project

Adding a new project requires only two steps:

### Step 1 — Register the project in `app.py`

Open `app.py` and add a new entry to the `PROJECTS` list:

```python
{
    "id": "house-price",
    "title": "House Price Prediction",
    "category": "Machine Learning / Regression",
    "description": "Predict house prices using regression and feature engineering.",
    "technologies": ["Python", "Scikit-learn", "XGBoost", "Pandas"],
    "route": "/project/house-price",
    "icon": "home",          # Font Awesome icon name (without fa-)
    "status": "complete",    # or "coming-soon"
    "year": "2024",
},
```

The homepage will automatically render a card for it.

### Step 2 — Create the project page

Add a route to `app.py`:

```python
@app.route("/project/house-price")
def house_price():
    return render_template("house_price.html")
```

Create `templates/house_price.html` extending `base.html`:

```html
{% extends "base.html" %}
{% block content %}
  <!-- your project content here -->
{% endblock %}
```

Optionally add:
- `data/house_prices.csv`
- `notebooks/house_price_analysis.ipynb`
- `static/js/house_price.js`
- `projects/house_price/README.md`

---

## Technologies Used

| Technology | Purpose |
|-----------|---------|
| Python 3 | Backend logic & data analysis |
| Flask | Web framework & REST API |
| Pandas | CSV loading & data manipulation |
| NumPy | Numerical operations |
| Scikit-learn | Machine learning models |
| Statsmodels | ARMA / ARMAX & statistical tests |
| Matplotlib | Static charts in notebooks |
| Chart.js | Interactive browser charts |
| Jupyter | Reproducible research notebooks |
| HTML5 / CSS3 | Frontend structure & styling |
| Vanilla JS | Dynamic chart interactions |

---

## Dataset

`data/dow_jones.csv`

| Column | Type | Description |
|--------|------|-------------|
| DATE | datetime | Trading date |
| Open | float | Opening price |
| High | float | Day high |
| Low | float | Day low |
| Close | float | Closing price (primary target) |
| Volume | float | Trading volume |

Source: Historical Dow Jones Industrial Average data.
Non-trading days are represented as `na` and removed during preprocessing.

---

## Notes

- The weather-dependent models (ARMAX + weather, RF + weather) from the original
  notebook require a LaGuardia Airport weather dataset that is NOT included here.
  Only models using `dow_jones.csv` alone can be run locally.

- The website is designed to be beginner-friendly. Read `app.py` from top to bottom
  and you will understand the entire backend in a few minutes.
