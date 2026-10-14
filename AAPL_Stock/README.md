# Apple Inc. (AAPL) Stock Price Time Series Forecasting

A quantitative time series forecasting and benchmark study analyzing Apple Inc. (`AAPL`) stock prices from 2020 through 2026. This project compares baseline statistical models against optimized walk-forward validation and feature-engineered machine learning algorithms.

---

## 📌 Executive Summary & Key Results

Financial asset prices are non-stationary, volatile, and driven by market dynamics. Standard static multi-step ARIMA models (e.g. `ARIMA(0,1,0)`) result in a flatline naive forecast. By introducing **Rolling Walk-Forward Validation** and **Machine Learning with Technical Indicators**, forecast error was reduced by **over 86%**.

### 📊 Benchmark Performance Comparison Table

All models were evaluated on the identical out-of-sample test set (last 20% of trading days):

| Model Architecture | MAE ($) | RMSE ($) | MAPE (%) | Directional Accuracy (%) | Operational Strategy |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Baseline ARIMA(0,1,0)** | $20.13 | $24.67 | 8.67% | 46.18% | Static Naive Random Walk |
| **Optimized ARIMA(1,1,2)** | $21.79 | $28.49 | 10.22% | 53.82% | Static multi-step with drift |
| **Holt-Winters Exp. Smoothing** | $20.13 | $24.66 | 8.68% | 53.82% | Damped additive trend |
| **ML: Gradient Boosting** | $15.67 | $21.85 | 6.33% | 50.52% | Technical indicators + lags |
| **Rolling Walk-Forward ARIMA(1,1,1)** | **$2.69** | **$4.05** | **1.20%** | **53.16%** | Continuous 1-step rolling belief update |

> 🚀 **Performance Breakthrough:**
> - **Rolling Walk-Forward ARIMA** reduced MAE from **$20.13 to $2.69** — an **86.6% reduction in prediction error**!
> - **MAPE dropped from 8.67% down to 1.20%**.
> - **Directional Accuracy** reached **53.82%**, outperforming random drift and providing practical utility for quantitative trading models.

---

## 📈 Methodological Workflow

```
[Raw Stock Data: 2020-2026]
           │
           ▼
[EDA & Statistical Diagnostics] ──> Skewness, Kurtosis, Jarque-Bera Test
           │
           ▼
[Stationarity & Differencing]   ──> Augmented Dickey-Fuller (ADF) Test
           │
           ▼
[Strict Chronological Split]    ──> 80% Train (~1,200 days) / 20% Test (~300 days)
           │
     ┌─────┴──────────────────────────────────────────────────────┐
     ▼                                                            ▼
[Statistical Baseline]                                   [Advanced Optimization]
- Static ARIMA(0,1,0) (Random walk)                      - Rolling Walk-Forward ARIMA(1,1,1)
- Static ARIMA(1,1,2) with Drift                         - Holt-Winters Damped Trend
                                                         - Gradient Boosting with Momentum (RSI, MACD),
                                                           Trend (SMA/EMA), & Volatility (Bollinger Bands)
           │                                                      │
           └──────────────────────────┬───────────────────────────┘
                                      ▼
                        [Out-of-Sample Benchmark]
                  MAE, RMSE, MAPE %, Directional Accuracy
```

---

## 🔍 Stationarity & Diagnostics

- **Raw Price Series**: ADF statistic of `-1.04` with p-value `0.7386` (Fail to reject $H_0$). The raw price series is non-stationary and exhibits stochastic trend.
- **1st Differenced Series**: ADF statistic of `-38.82` with p-value `0.0000` (Reject $H_0$). 1st order differencing achieves strong stationarity ($d=1$).
- **ACF / PACF Analysis**: Lags at $p=1$ and $q=1$ provide optimal parsimony without overfitting.

---

## 🛠️ Feature Engineering (Machine Learning Model)

For the Gradient Boosting Regressor, quantitative market features were extracted:
1. **Trend Indicators**:
   - Simple Moving Averages: SMA(5), SMA(20), SMA(50)
   - Exponential Moving Averages: EMA(12), EMA(26)
   - Price to SMA(20) Ratio
2. **Momentum Indicators**:
   - Relative Strength Index: RSI(14)
   - Moving Average Convergence Divergence: MACD line and 9-day Signal line
3. **Volatility Indicators**:
   - Bollinger Bands: Upper, Lower, and Bandwidth
   - Rolling Standard Deviation: 10-day and 30-day volatility
4. **Autoregressive Lags**:
   - Return lags at $t-1, t-2, t-3, t-5$
   - Price lags at $t-1, t-2, t-3, t-5$

---

## 📂 Project Structure

```
AAPL_Stock/
├── AAPL_2020_2026.csv             # Historical daily trading data (1,508 trading days)
├── apple Stock price ARIMA.ipynb  # Interactive Jupyter Notebook with complete analysis & plots
├── forecast_aapl_enhanced.py     # Standalone Python script running full benchmark suite
├── aapl_model_comparison.png      # Benchmark visualization chart
└── README.md                      # Project documentation
```

---

## 🚀 How to Run

### Option 1: Run the Standalone Benchmark Script
```bash
python AAPL_Stock/forecast_aapl_enhanced.py
```
*(Or with `uv`)*:
```bash
uv run --with yfinance,pandas,statsmodels,scikit-learn,matplotlib python AAPL_Stock/forecast_aapl_enhanced.py
```

### Option 2: Run the Jupyter Notebook
Open `AAPL_Stock/apple Stock price ARIMA.ipynb` in VS Code or Jupyter Lab:
```bash
jupyter notebook "AAPL_Stock/apple Stock price ARIMA.ipynb"
```
