# Multi-Series Forecasting for Retail Demand Optimization

**Author:** Dhanraj Deshmukh — Final Project, Data Science Internship
**Dataset:** Kaggle "Store Item Demand Forecasting Challenge"
**Scored on:** SMAPE (Symmetric Mean Absolute Percentage Error)

---

## Problem Statement

Forecast daily sales for **10 stores x 50 items = 500 related time series** over a 3-month horizon (Jan-Mar 2018), using 5 years of historical daily sales data (2013-2017, ~913,000 rows).

The challenge is framed as a **retail demand system**, not merely a competition entry - meaning the output must support real supply chain decisions: inventory reorder points, service-level trade-offs, and coherent forecasts across aggregation levels.

---

## Architecture

```
Data Ingestion
    +-- Feature Engineering (lags, rolling stats, calendar, Fourier)
            +-- Tier 1: Global LightGBM (all 500 series)
            +-- Tier 2: N-HiTS / Quantile Baseline (50-series subset)
            +-- Tier 3: Chronos / ETS zero-shot (cold-start benchmark)
                        |
                        v
            Hierarchical Reconciliation (MinT)
                        |
                        v
            Ensemble / Routing Layer
            +-- Strategy A: Inverse-error weighted average
            +-- Strategy B: Per-series routing (Tier 3 for cold-start)
                        |
                        v
            Inventory Decision Layer (Newsvendor safety-stock formula)
                        |
                        v
            Monitoring Module (rolling SMAPE + PSI drift detection)
```

---

## Repository Structure

```
final-project/
  Demand_Forecasting_System.ipynb   <- Main notebook: Phases 0-10
  README.md                          <- This file
  system_design.md                   <- Full system design document
  results/
    tier_comparison.csv              <- SMAPE/WAPE/pinball per tier per fold
    reconciliation_check.csv         <- Coherence verification results
    inventory_recommendations.csv    <- Reorder points at 90/95/99% SL
    submission.csv                   <- Kaggle submission (LightGBM predictions)
    final_dashboard.html             <- Interactive Plotly dashboard
    *.png                            <- EDA and result plots
```

---

## Notebook Phases

| Phase | Description |
|-------|-------------|
| 0 | Environment setup, data loading, schema validation |
| 1 | EDA: aggregate trends, ACF/PACF, stationarity (ADF), seasonality heatmap, store comparison |
| 2 | Feature engineering: lag (1/7/14/28d), rolling stats, calendar, Fourier terms |
| 3 | Tier 1 - Global LightGBM: 3-fold rolling-origin CV, SMAPE/WAPE, feature importance |
| 4 | Tier 2 - N-HiTS / Seasonal Naive: quantile forecasts (P10/P50/P90), pinball loss |
| 5 | Tier 3 - Chronos / ETS: zero-shot benchmark + cold-start simulation (30-day truncation) |
| 6 | Hierarchical reconciliation: MinT / bottom-up, coherence check, SMAPE impact |
| 7 | Ensembling: inverse-error weighting vs. per-series routing; full tier comparison |
| 8 | Inventory layer: reorder points at 90/95/99% service levels for 20 sample series |
| 9 | Monitoring: rolling SMAPE drift + PSI input distribution shift detection |
| 10 | Final dashboard, Kaggle submission, written production recommendation |

---

## Key Findings

### Why Three Tiers?
- **Tier 1** is fast and interpretable - the production default for established series
- **Tier 2** produces calibrated uncertainty bands, essential for safety-stock calculations
- **Tier 3** solves cold-start: new store-item pairs with <90 days of history

### Reconciliation
Hierarchical MinT reconciliation ensures that item-level forecasts sum coherently to
store-total forecasts - a prerequisite for coordinated supply chain planning.

### Inventory Decision Layer
reorder_point = lead_time_demand(P50) + z(service_level) x sigma_forecast

At a 7-day lead time, moving from 90% to 99% service level approximately doubles
the safety stock buffer.

---

## How to Run

1. Install dependencies:
   pip install pandas numpy lightgbm scikit-learn matplotlib seaborn plotly statsmodels scipy
   pip install neuralforecast hierarchicalforecast
   # Optional (Tier 3): pip install chronos-forecasting

2. Ensure the data directory is at: ../demand-forecasting-kernels-only/ relative to final-project/

3. Open and run Demand_Forecasting_System.ipynb top-to-bottom.

---

## Limitations

- No external covariates (no price, promotion, or holiday data)
- Tier 2 & 3 evaluated on 50-series subset for compute reasons
- SMAPE distorts near zero - WAPE included as a complementary metric
- Chronos-2 requires GPU for efficient inference on all 500 series
