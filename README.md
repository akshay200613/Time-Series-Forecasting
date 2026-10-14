# Time Series Forecasting

A collection of quantitative and statistical time series forecasting projects exploring financial stock prices and large-scale retail demand.

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Statsmodels](https://img.shields.io/badge/Statsmodels-ARIMA%20%2F%20SARIMAX-green.svg)](https://www.statsmodels.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-orange.svg)](https://scikit-learn.org/)

---

## 📂 Projects in this Repository

| Project | Domain | Key Techniques | Best Model | Highlight Metric |
| :--- | :--- | :--- | :--- | :--- |
| [**AAPL Stock Price Forecasting**](./AAPL_Stock) | Financial Markets (Equities) | ADF Stationarity, Rolling Walk-Forward ARIMA, Holt-Winters, Gradient Boosting + Technical Indicators | **Rolling Walk-Forward ARIMA(1,1,1)** | **MAE: $2.69** (86.6% error reduction vs baseline) |
| [**Walmart Retail Sales Forecasting**](./WalmartProject) | Retail Demand & Supply Chain | 52-Week Seasonality, Holiday Exogenous Shocks, MarkDown Analysis, SARIMAX, Damped Exponential Smoothing | **Damped Trend Exponential Smoothing** | Captures $2.4M holiday peaks without explosive trend drift |

---

## 🏛️ Architecture & Repository Structure

```
Time-Series-Forecasting/
├── .gitignore                         # Comprehensive git ignore rules (checkpoints, caches, envs)
├── requirements.txt                   # Universal Python dependency manifest
├── README.md                          # Repository overview & navigation guide
│
├── AAPL_Stock/                        # Apple Inc. Stock Price Forecasting Project
│   ├── README.md                      # In-depth project documentation & benchmark results
│   ├── apple Stock price ARIMA.ipynb  # Interactive Jupyter Notebook with end-to-end pipeline
│   ├── forecast_aapl_enhanced.py     # Production CLI script running benchmark suite
│   ├── AAPL_2020_2026.csv             # Historical daily trading data (1,508 trading days)
│   └── aapl_model_comparison.png      # Benchmark visualization chart
│
└── WalmartProject/                    # Walmart Store Sales Forecasting Project
    ├── README.md                      # In-depth case study documentation
    ├── Walmart Forecastin b final.ipynb # Store 1 aggregated forecasting (ARIMA, SARIMAX, Exp Smoothing)
    ├── Walmart.ipynb                  # Department-level forecasting (Store 1, Dept 1), auto_arima, EDA
    ├── Project Documentation.docx     # Case study technical documentation
    ├── Walmart Sales Forecasting Case Study.pptx # Stakeholder presentation deck
    ├── stores.csv                     # Metadata for 45 stores (Type, Size)
    ├── train.csv/train.csv            # Historical weekly sales training data
    ├── test.csv/test.csv              # Test set
    └── features.csv/features.csv      # Weather, economic indicators (CPI, Unemployment), MarkDowns
```

---

## 🔬 Comparative Analysis of Time Series Domains

This repository contrasts two distinct time-series challenges:

| Dimension | Financial Assets (AAPL Stock) | Retail Sales Demand (Walmart) |
| :--- | :--- | :--- |
| **Trend Behavior** | Stochastic, random walk with drift | Deterministic baseline with annual cycle |
| **Seasonality** | None (calendar effects are negligible) | **Strong 52-week annual cycle** (Thanksgiving, Christmas) |
| **Exogenous Drivers** | Market sentiment, interest rates, earnings | Weather, fuel prices, **promotional markdowns**, holidays |
| **Optimal Modeling Strategy** | **Rolling Walk-Forward Validation** (continuous belief update) | **Damped Seasonal Smoothing & SARIMAX** with holiday exogenous inputs |
| **Key Failure Mode** | Static long-horizon flatline | Over-projecting long-term trends past holiday peaks |

---

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/akshay200613/Time-Series-Forecasting.git
cd Time-Series-Forecasting
```

### 2. Environment Setup & Installation
Using standard `pip`:
```bash
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

pip install -r requirements.txt
```

*(Or using ultra-fast `uv`)*:
```bash
uv pip install -r requirements.txt
```

### 3. Running the Projects

#### Apple Stock Price Benchmark:
```bash
python AAPL_Stock/forecast_aapl_enhanced.py
```
Or launch Jupyter:
```bash
jupyter notebook "AAPL_Stock/apple Stock price ARIMA.ipynb"
```

#### Walmart Sales Forecasting:
```bash
jupyter notebook "WalmartProject/Walmart Forecastin b final.ipynb"
```

---

## 📜 License
Distributed under the MIT License. See `LICENSE` for more information.
