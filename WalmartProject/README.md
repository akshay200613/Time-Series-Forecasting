# Walmart Store Sales Time Series Forecasting

A data science case study and time series forecasting implementation based on the **Kaggle Walmart Recruiting: Store Sales Forecasting** challenge. This project models historical sales data across Walmart retail stores, evaluating seasonal volatility, macroeconomic factors, promotional markdowns, and holiday effects.

---

## 📌 Executive Summary

Retail inventory management requires accurate demand forecasting to balance stockout risks during holiday surges against costly post-holiday overstock. 

Weekly sales data exhibits **strong annual seasonality (52-week cycle)** and dramatic holiday spikes (Thanksgiving / Black Friday and Christmas) that can reach up to **2.4M USD** compared to the baseline **1.4M - 1.7M USD**.

### 🏆 Key Model Benchmark

Models evaluated on weekly sales forecasting (Store 1 aggregated sales):

| Model Architecture | Methodology | Seasonal Component | Exogenous Features | Performance & Observations |
| :--- | :--- | :---: | :---: | :--- |
| **ARIMA(1,1,1)** | Autoregressive Integrated Moving Average | ❌ None | None | Baseline model; suffers from flatline decay over long forecast horizons. |
| **SARIMAX(1,1,1)(1,1,1)₅₂** | Seasonal ARIMA with Exogenous Regressors | ✅ 52-week period | `IsHoliday` flag | Captures periodic annual cycle; holiday flag acts as a shock trigger for demand surges. |
| **Damped Trend Exponential Smoothing** | Holt-Winters with Additive Seasonality & Damped Trend | ✅ 52-week period | Implicit | **Recommended model:** Damped parameter prevents runaway linear trends while accurately following seasonal peaks and post-holiday dips. |

> 💡 **Core Finding:**
> Damped Trend Exponential Smoothing offers the most realistic projection horizon. Traditional linear trend models over-project long-term sales, whereas the damping parameter ($\phi$) stabilizes baseline demand while respecting 52-week seasonal peaks.

---

## 📊 Dataset Architecture

The project utilizes data across 45 Walmart stores located in different regions:

1. **`stores.csv`**:
   - `Store`: Store number (1 to 45)
   - `Type`: Store classification (`Type A`, `Type B`, `Type C` based on volume/size)
   - `Size`: Total square footage
2. **`train.csv/train.csv`**:
   - Historical sales records from **2010-02-05 to 2012-11-01**
   - Columns: `Store`, `Dept`, `Date`, `Weekly_Sales`, `IsHoliday`
3. **`features.csv/features.csv`**:
   - Regional economic and promotional factors:
     - `Temperature`: Average regional temperature
     - `Fuel_Price`: Regional cost of fuel
     - `MarkDown1` to `MarkDown5`: Anonymized promotional discount data (introduced after Nov 2011)
     - `CPI`: Consumer Price Index
     - `Unemployment`: Regional unemployment rate
     - `IsHoliday`: Binary indicator of statutory holidays
4. **`test.csv/test.csv`**:
   - Test period records from **2012-11-02 to 2013-07-26** for out-of-sample submission

---

## 🔍 Key Exploratory Insights

1. **Stable Baseline Demand with Extreme Peaks**:
   - Average weekly sales hover predictably between **$1.4M and $1.7M**.
   - Massive spikes occur during **Super Bowl, Labor Day, Thanksgiving (Black Friday), and Christmas**, reaching upwards of **$2.4M**.
2. **Post-Holiday Dropoff**:
   - Immediately following holiday peaks, sales experience a sharp contraction as consumer purchasing pulls forward.
3. **Store Type Disparities**:
   - `Type A` stores dominate sales volume due to physical footprint and inventory capacity, followed by `Type B`, with `Type C` representing smaller format convenience locations.
4. **MarkDown Impact**:
   - Promotional markdowns (MarkDown 1-5) correlate heavily with high-traffic promotional periods like Thanksgiving and pre-Christmas clearance.

---

## 📁 Repository Structure

```
WalmartProject/
├── Walmart Forecastin b final.ipynb       # Store 1 aggregated forecasting: ARIMA, SARIMAX, Exponential Smoothing
├── Walmart.ipynb                          # Department-level forecasting (Store 1, Dept 1), auto_arima, EDA
├── Project Documentation.docx             # Comprehensive case study documentation
├── Walmart Sales Forecasting Case Study.pptx # Executive stakeholder presentation deck
├── stores.csv                             # Store metadata (45 stores, types, square footage)
├── train.csv/train.csv                    # Historical training sales data
├── test.csv/test.csv                      # Test set
├── features.csv/features.csv              # Economic, weather, and promotional markdown data
└── README.md                              # Project documentation
```

---

## 🎯 Business Recommendations

1. **Holiday Inventory Staging**:
   - Supply chains should stage high-margin inventory 2 to 3 weeks prior to Thanksgiving week to accommodate the 40–70% demand surge without stockouts.
2. **Damped Baseline Forecasts for Purchasing**:
   - Prevent over-purchasing in Q1 by utilizing damped trend parameters, as post-holiday sales normalize sharply in January and February.
3. **Targeted Promotional Timing**:
   - MarkDowns 1 and 2 demonstrate highest elastic lift when synchronized with holiday announcements rather than post-event liquidation.

---

## 🚀 How to Run

Open either notebook in Jupyter or VS Code:
```bash
jupyter notebook "WalmartProject/Walmart Forecastin b final.ipynb"
```
Or view the department-level pipeline:
```bash
jupyter notebook "WalmartProject/Walmart.ipynb"
```
