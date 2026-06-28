# 🚗 Tesla EV Sales — End-to-End ML Pipeline
### Week 2 Assignment | Data Science Internship
**Author:** Dhanraj Deshnukh

---

## 📌 Overview

This notebook builds a **complete, production-style Machine Learning pipeline** on a Tesla worldwide deliveries and pricing dataset spanning **2015–2025**. It covers every stage from raw data ingestion to time series forecasting, with a focus on clean code, proper validation, and interpretable results.

---

## 📁 Project Structure

```
Week2_Dhanraj_Deshnukh.ipynb   ← Main notebook (fully self-contained)
plots/                          ← All saved chart outputs
  ├── eda_price_dist.png
  ├── eda_time.png
  ├── eda_corr.png
  ├── eda_pie.png
  ├── feature_importance.png
  ├── residuals.png
  ├── ts_raw.png
  ├── acf_pacf.png
  ├── ts_forecast.png
  └── model_comparison.png
README.md                       ← This file
```

---

## 🗂️ Dataset

The dataset is reconstructed inline (no external CSV required), making the notebook **fully self-contained and reproducible** on any machine.

| Column | Description |
|---|---|
| `Year` | Year of record (2015–2025) |
| `Month` | Month of record (1–12) |
| `Region` | Sales region (Asia, Europe, Middle East, North America) |
| `Model` | Tesla model (Model 3, Model S, Model X, Model Y, Cybertruck) |
| `Estimated_Deliveries` | Number of vehicles delivered |
| `Production_Units` | Number of vehicles produced |
| `Avg_Price_USD` | Average sale price in USD (**regression target**) |
| `Battery_Capacity_kWh` | Battery capacity in kWh |
| `Range_km` | Vehicle range in kilometres |
| `CO2_Saved_tons` | Estimated CO₂ savings |

- **Shape:** 200 rows × 10 columns
- **No missing values**
- **4 regions, 5 models**

---

## 🔄 Pipeline Stages

| # | Stage | Description |
|---|---|---|
| 1 | Data Ingestion & Inspection | Load dataset, check dtypes, shape, null audit |
| 2 | Exploratory Data Analysis | Distributions, trends over time, correlations, model/region share |
| 3 | Preprocessing | IQR-based outlier capping (Winsorization), Label Encoding, RobustScaler |
| 4 | Feature Engineering | Lag features, rolling stats, interaction terms, cyclic encoding |
| 5 | Regression Modeling | 5 models trained and evaluated on 80/20 split |
| 6 | Hyperparameter Tuning | RandomizedSearchCV with 5-fold CV on best model |
| 7 | Time Series Forecasting | SARIMA on aggregated monthly global deliveries |
| 8 | Model Evaluation & Summary | Metrics table, residual analysis, final comparison chart |

---

## 🔧 Feature Engineering

Six new features were engineered to improve signal:

| Feature | Formula / Logic | Rationale |
|---|---|---|
| `Delivery_Efficiency` | `Deliveries / Production_Units` | Logistics quality proxy |
| `Price_per_km` | `Avg_Price_USD / Range_km` | Cost efficiency per km of range |
| `Battery_x_Range` | `Battery_kWh × Range_km` | Interaction term for powertrain quality |
| `Quarter` | Derived from `Month` | Captures end-of-quarter business patterns |
| `Month_sin / Month_cos` | Cyclic encoding of month | Preserves circular nature of calendar months |
| `Is_Premium` | 1 if Model S or Model X | Business flag for high-end vehicles |
| `Delivery_lag1` | Previous row's deliveries | Short-term temporal signal |
| `Delivery_roll3` | 3-month rolling mean | Smoothed trend signal |

---

## 🤖 Models Used

| Model | Type | Notes |
|---|---|---|
| Linear Regression | Baseline | No regularisation |
| Ridge Regression | L2 regularised | `alpha=1.0` |
| Lasso Regression | L1 regularised | Feature selection via sparsity |
| Random Forest | Ensemble (bagging) | 200 estimators |
| XGBoost | Ensemble (boosting) | `lr=0.05`, `max_depth=4` |
| **Tuned Random Forest** | Best model (tuned) | RandomizedSearchCV, 40 iterations, 5-fold CV |

All models evaluated on: **MAE, RMSE, R², CV-R²**

---

## 📈 Time Series Forecasting

- Series: **Global monthly deliveries** aggregated across all regions and models
- Stationarity tested with **Augmented Dickey-Fuller (ADF)** test
- Order selection guided by **ACF / PACF** plots
- Model: **SARIMA(1,1,1)(1,1,1,12)** — handles trend and 12-month seasonal cycle
- 80/20 temporal train/test split
- Forecast evaluated with MAE, RMSE, R² and plotted with 95% confidence intervals

---

## 📊 Key Findings

| Stage | Finding |
|---|---|
| **EDA** | `Avg_Price_USD` is roughly uniform ($50k–$120k); deliveries grew consistently 2015–2025 |
| **Correlation** | `CO2_Saved_tons` highly correlated with deliveries; price has low linear correlation with most features |
| **Feature Eng.** | `Delivery_Efficiency`, `Battery_x_Range`, and lag features added meaningful signal |
| **Best Model** | Tuned Random Forest achieved the lowest RMSE on the held-out test set |
| **Tuning** | RandomizedSearchCV improved CV R² over the baseline Random Forest |
| **Time Series** | SARIMA captured the global delivery trend; 95% CI intervals are well-calibrated |

---

## 🛠️ Tech Stack

| Library | Purpose |
|---|---|
| `numpy`, `pandas` | Data manipulation |
| `matplotlib`, `seaborn` | Visualisation |
| `scikit-learn` | ML models, preprocessing, tuning |
| `xgboost` | Gradient boosted trees |
| `statsmodels` | SARIMA, ADF test, ACF/PACF |
| `scipy` | RandomizedSearchCV distributions |

---

## ▶️ How to Run

1. Open the notebook in **Google Colab** or any Jupyter environment
2. Run all cells top to bottom — `xgboost` and `statsmodels` are auto-installed if missing
3. All plots save automatically to the `./plots/` directory
4. No external dataset file needed — data is embedded in the notebook

```bash
# If running locally
pip install xgboost statsmodels
jupyter notebook Week2_Dhanraj_Deshnukh.ipynb
```

---

## 📋 Evaluation Metrics

| Metric | Formula | Interpretation |
|---|---|---|
| **MAE** | Mean of \|actual − predicted\| | Average error in USD |
| **RMSE** | √(Mean of squared errors) | Penalises large errors more |
| **R²** | 1 − SS_res / SS_tot | Proportion of variance explained (higher = better) |
| **CV-R²** | Mean R² across 5 folds | Generalisation estimate |

---

*Week 2 Assignment — Data Science Internship*
