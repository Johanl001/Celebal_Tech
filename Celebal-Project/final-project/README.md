# Multi-Series Retail Demand Forecasting & Inventory Optimization

**Author:** Dhanraj Deshmukh\
**Project:** Final Project --- Data Science Internship\
**Dataset:** Kaggle --- *Store Item Demand Forecasting Challenge*\
**Forecasting Task:** Daily demand forecasting for 500 related
store-item time series\
**Primary Evaluation Metric:** SMAPE (Symmetric Mean Absolute Percentage
Error)

------------------------------------------------------------------------

## 1. Project Overview

This project develops an end-to-end **multi-series retail demand
forecasting system** for predicting daily sales across:

-   **10 stores**
-   **50 items**
-   **500 related store-item time series**
-   Approximately **5 years of historical daily sales (2013--2017)**
-   A **3-month forecasting horizon (Jan--Mar 2018)**

The system is designed as a **retail decision-support pipeline**, rather
than only as a Kaggle forecasting solution. In addition to generating
demand predictions, it connects forecasting with:

-   time-series exploratory analysis,
-   feature engineering,
-   global machine-learning forecasting,
-   neural/quantile forecasting baselines,
-   hierarchical reconciliation,
-   ensemble and model-routing strategies,
-   inventory reorder-point calculations,
-   and post-deployment model monitoring.

The final pipeline therefore links **forecast → uncertainty → inventory
decision → monitoring**.

------------------------------------------------------------------------

## 2. Business Problem

Retail demand varies across stores, products, weekdays, months, and
seasonal cycles. A useful forecasting system must capture both:

1.  **Temporal dependence** within each store-item series.
2.  **Shared patterns** across hundreds of related series.

Accurate forecasts can support:

-   inventory replenishment,
-   safety-stock planning,
-   service-level management,
-   stockout-risk reduction,
-   and coordinated planning at store and item aggregation levels.

The project treats the forecasting task as a **multi-series demand
system** where model accuracy and operational usefulness are considered
together.

------------------------------------------------------------------------

## 3. System Architecture

``` text
                         Historical Sales Data
                                  |
                                  v
                    Data Ingestion & Validation
                                  |
                                  v
                       Exploratory Data Analysis
                                  |
                                  v
                         Feature Engineering
             +--------------------+--------------------+
             |                    |                    |
             v                    v                    v
        Lag Features        Rolling Statistics     Calendar/Fourier
             |                    |                    |
             +--------------------+--------------------+
                                  |
                                  v
                       Multi-Tier Forecasting
                                  |
          +-----------------------+-----------------------+
          |                       |                       |
          v                       v                       v
   Tier 1: LightGBM       Tier 2: N-HiTS /        Tier 3: Chronos /
   Global Model           Quantile Baseline       ETS Benchmark
          |                       |                       |
          +-----------------------+-----------------------+
                                  |
                                  v
                    Hierarchical Reconciliation
                         (MinT / Bottom-Up)
                                  |
                                  v
                       Ensemble / Model Routing
                                  |
                                  v
                    Forecast + Uncertainty Estimates
                                  |
                                  v
                     Inventory Decision Layer
                  (Newsvendor / Safety Stock)
                                  |
                                  v
                       Model Monitoring
                    (SMAPE + PSI Drift Detection)
```

------------------------------------------------------------------------

## 4. Data & Forecasting Scope

  Component                 Description
  ------------------------- ------------------------------
  Stores                    10
  Items                     50
  Total series              500
  Historical period         2013--2017
  Historical observations   \~913,000
  Forecast horizon          Jan--Mar 2018
  Primary metric            SMAPE
  Complementary metric      WAPE
  Inventory evaluation      90%, 95%, 99% service levels

The supplied final submission contains **45,000 prediction rows** with
two columns:

``` text
id
sales
```

This corresponds to the 500-series forecasting setup over a 90-day
horizon.

------------------------------------------------------------------------

# 5. Exploratory Data Analysis

## 5.1 Aggregate Demand Trend

The aggregate daily-sales analysis shows strong recurring seasonality
and changing demand levels across the 2013--2017 history.

![Aggregate Daily Sales](eda_aggregate_sales.png)

### Observations

-   Daily sales exhibit substantial short-term fluctuations around the
    longer-term trend.
-   The 28-day rolling mean makes the broader seasonal structure easier
    to observe.
-   Demand generally rises through the year and repeatedly reaches
    higher levels during the middle of the year.
-   The yearly pattern is not purely stationary: the overall level
    changes between years.
-   This supports the use of both **lagged demand features** and
    **rolling statistics**.

------------------------------------------------------------------------

## 5.2 Day-of-Week × Month Seasonality

The pooled day-of-week/month heatmap highlights a strong calendar effect
across the 500 series.

![Day-of-Week Month Heatmap](eda_heatmap_dow_month.png)

### Key observations

-   Sales increase from the beginning of the week toward the weekend.
-   **Sunday** has the highest average sales across the displayed
    combinations.
-   **July** is the strongest month across almost every day of the week.
-   The highest displayed average is approximately **80.3 sales per
    store-item on Sunday in July**.
-   January and December show comparatively lower average demand.
-   Calendar features such as `day_of_week`, `month`, and `is_weekend`
    are therefore important predictive variables.

------------------------------------------------------------------------

## 5.3 ACF & PACF Analysis

Autocorrelation and partial autocorrelation were examined for
representative store-item series.

![ACF PACF](eda_acf_pacf.png)

### Key observations

The ACF plots show persistent positive autocorrelation over many lags
rather than a rapid decay to zero.

Notable recurring peaks appear around:

-   lag 7,
-   lag 14,
-   lag 21,
-   lag 28,
-   and subsequent weekly intervals.

The PACF plots also show strong short-lag and weekly-related structure.

### Modeling implication

The diagnostics support the inclusion of:

-   `lag_1`
-   `lag_7`
-   `lag_14`
-   `lag_28`
-   rolling 7-day statistics
-   rolling 28-day statistics
-   calendar variables
-   Fourier seasonal terms

This provides the global model with both **recent demand information**
and **weekly/longer seasonal context**.

------------------------------------------------------------------------

# 6. Feature Engineering

The forecasting feature set combines autoregressive, statistical,
calendar, and seasonal representations.

### Lag features

``` text
lag_1
lag_7
lag_14
lag_28
```

These capture recent demand and recurring weekly patterns.

### Rolling features

``` text
rolling_mean_7
rolling_mean_28
rolling_std_7
rolling_std_28
```

Rolling means capture local demand level while rolling standard
deviations provide information about recent volatility.

### Calendar features

``` text
day_of_week
day_of_month
month
quarter
year
is_weekend
is_month_start
is_month_end
```

### Fourier features

``` text
fourier_sin_1
fourier_cos_1
fourier_sin_2
fourier_cos_2
```

These provide smooth representations of recurring seasonal behavior.

### Series identifiers

``` text
store
item
```

These allow the global model to learn differences between stores and
products.

------------------------------------------------------------------------

# 7. LightGBM Global Forecasting Model

The primary production-oriented forecasting model is a **global LightGBM
model** trained across the store-item series.

Instead of maintaining 500 completely independent models, the global
approach allows the learner to exploit common demand patterns while
retaining store/item identity through features.

![LightGBM Feature Importance](lgbm_feature_importance.png)

## Feature Importance Findings

The gain-based feature-importance plot shows a clear hierarchy:

1.  `rolling_mean_7`
2.  `rolling_mean_28`
3.  `lag_7`
4.  `day_of_week`
5.  `lag_14`
6.  `item`
7.  `lag_28`

The remaining calendar and Fourier features contribute comparatively
less gain in the displayed model.

### Interpretation

The strongest signal comes from **recent local demand level**, followed
by longer-window demand level and weekly seasonality.

In particular:

-   `rolling_mean_7` is the dominant feature.
-   `rolling_mean_28` provides a broader demand baseline.
-   `lag_7` confirms the importance of weekly recurrence.
-   `day_of_week` adds explicit calendar information.
-   `lag_14` and `lag_28` extend the weekly seasonal memory.

This is consistent with the ACF/PACF diagnostics.

------------------------------------------------------------------------

# 8. Multi-Tier Forecasting Strategy

The system is organized into three forecasting tiers.

## Tier 1 --- Global LightGBM

**Purpose:** Primary production model for established series.

Characteristics:

-   Fast inference.
-   Strong tabular forecasting performance.
-   Interpretable feature importance.
-   Trains across all 500 series.
-   Uses lag, rolling, calendar, Fourier, store, and item features.

------------------------------------------------------------------------

## Tier 2 --- N-HiTS / Quantile Baseline

**Purpose:** Provide an alternative forecasting family and uncertainty
estimates.

The representative actual-vs-predicted plots show the model tracking the
overall demand range while attempting to follow short-term fluctuations.

![N-HiTS Actual vs Predicted](nhits_actual_vs_pred.png)

The forecasting comparison includes:

-   historical demand,
-   actual observations during the evaluation window,
-   predicted P50 demand.

This tier is particularly useful when the downstream inventory layer
requires a forecast distribution or prediction quantiles rather than
only a point estimate.

------------------------------------------------------------------------

## Tier 3 --- Chronos / ETS Benchmark

**Purpose:** Zero-shot and cold-start benchmarking.

This tier is intended for cases where a series has limited historical
information and a feature-rich global model may have insufficient data.

The project design uses this tier as a fallback/reference approach
rather than the default production model.

------------------------------------------------------------------------

# 9. Hierarchical Reconciliation

Retail forecasts can be evaluated at multiple aggregation levels:

``` text
All Stores
   |
   +-- Store 1
   |     +-- Item 1
   |     +-- Item 2
   |     +-- ...
   |
   +-- Store 2
   |     +-- Item 1
   |     +-- Item 2
   |
   +-- ...
```

The project includes **hierarchical reconciliation using MinT /
bottom-up approaches**.

The purpose is to make forecasts coherent across aggregation levels so
that:

``` text
Store Total = Sum of its Item Forecasts
```

This is important for supply-chain planning because independently
generated forecasts can otherwise disagree when aggregated.

------------------------------------------------------------------------

# 10. Inventory Decision Layer

Forecasting accuracy alone is not sufficient for inventory planning.

The system converts demand forecasts into operational reorder-point
recommendations.

The core formulation is:

``` text
Reorder Point
    =
Lead-Time Demand (P50)
    +
z(Service Level) × Forecast Uncertainty
```

The project evaluates reorder points at:

-   **90% service level**
-   **95% service level**
-   **99% service level**

![Inventory Reorder Points](inventory_reorder_points.png)

## Service-Level Trade-off

As the target service level increases, the required reorder point also
increases.

The plotted sample series show a consistent pattern:

``` text
90% Service Level < 95% Service Level < 99% Service Level
```

This represents the fundamental inventory trade-off:

> Higher service levels reduce stockout risk but require more inventory
> buffer.

At a 7-day lead time, the project analysis indicates that moving from
90% toward 99% service can substantially increase the safety-stock
requirement.

------------------------------------------------------------------------

# 11. Model Monitoring

A production forecasting system must monitor performance after
deployment.

The project uses two complementary monitoring signals:

1.  **Rolling SMAPE** --- performance degradation.
2.  **PSI (Population Stability Index)** --- input/distribution drift.

![Model Monitoring](monitoring_smape_psi.png)

## 11.1 Rolling SMAPE

The monitoring dashboard compares series-level rolling SMAPE against:

-   **Backtest baseline:** 11.9%
-   **Retraining threshold:** 14.3%

Series above the retraining threshold can be treated as candidates for
investigation or model refresh.

The plot shows that several series exceed the threshold, indicating
heterogeneous forecasting performance across the portfolio.

------------------------------------------------------------------------

## 11.2 PSI Drift Detection

The PSI plot uses:

``` text
Minor drift threshold = 0.10
Major drift threshold = 0.20
```

Several series exceed the major-drift threshold.

One displayed series has a particularly large PSI value of approximately
**1.57**, indicating a substantial distribution shift relative to the
reference distribution.

### Operational response

A monitoring workflow can therefore be:

``` text
PSI / SMAPE Alert
       |
       v
Identify affected series
       |
       v
Investigate demand/distribution change
       |
       +---- Stable -> Continue monitoring
       |
       +---- Degraded -> Retrain / reroute model
```

------------------------------------------------------------------------

# 12. End-to-End Notebook Phases

  Phase   Description
  ------- ------------------------------------------------------------------
  0       Environment setup, data loading, schema validation
  1       EDA: aggregate trends, ACF/PACF, seasonality, store comparison
  2       Feature engineering: lags, rolling statistics, calendar, Fourier
  3       Tier 1 --- Global LightGBM with rolling-origin validation
  4       Tier 2 --- N-HiTS / Seasonal Naive quantile forecasting
  5       Tier 3 --- Chronos / ETS zero-shot and cold-start benchmark
  6       Hierarchical reconciliation using MinT / bottom-up
  7       Ensemble and per-series model routing
  8       Inventory reorder points at 90/95/99% service levels
  9       Rolling SMAPE and PSI drift monitoring
  10      Dashboard, final submission, and production recommendations

------------------------------------------------------------------------

# 13. Repository Structure

``` text
final-project/
│
├── Demand_Forecasting_System.ipynb
├── README.md
├── system_design.md
│
├── results/
│   ├── tier_comparison.csv
│   ├── reconciliation_check.csv
│   ├── inventory_recommendations.csv
│   ├── submission.csv
│   └── final_dashboard.html
│
└── *.png
    ├── eda_acf_pacf.png
    ├── eda_aggregate_sales.png
    ├── eda_heatmap_dow_month.png
    ├── inventory_reorder_points.png
    ├── lgbm_feature_importance.png
    ├── monitoring_smape_psi.png
    └── nhits_actual_vs_pred.png
```

------------------------------------------------------------------------

# 14. How to Run

## 1. Install dependencies

``` bash
pip install pandas numpy lightgbm scikit-learn matplotlib seaborn plotly statsmodels scipy
pip install neuralforecast hierarchicalforecast
```

Optional Tier 3 dependency:

``` bash
pip install chronos-forecasting
```

## 2. Configure the data directory

Ensure the dataset is available at:

``` text
../demand-forecasting-kernels-only/
```

relative to the project directory.

## 3. Run the notebook

Open:

``` text
Demand_Forecasting_System.ipynb
```

and execute the notebook from top to bottom.

------------------------------------------------------------------------

# 15. Key Findings

### 1. Demand is strongly seasonal

The aggregate trend and day-of-week/month heatmap demonstrate recurring
seasonal structure across the 500 series.

### 2. Weekly demand is a major signal

ACF/PACF diagnostics show strong persistence and recurring
weekly-related peaks. This is reflected in the importance of `lag_7`.

### 3. Recent demand level is the strongest predictive signal

The LightGBM gain plot ranks `rolling_mean_7` substantially above the
other features, followed by `rolling_mean_28`.

### 4. Calendar effects matter

`day_of_week` is among the most useful predictive features, while the
heatmap shows systematically higher weekend demand.

### 5. Forecasting and inventory planning should be connected

The reorder-point layer translates demand forecasts into
service-level-aware inventory decisions.

### 6. Model performance is heterogeneous

The monitoring results show that some series perform comfortably below
the retraining threshold while others exceed it.

### 7. Distribution drift can be substantial

PSI identifies multiple series with major drift, including one displayed
series with PSI around 1.57.

------------------------------------------------------------------------

# 16. Production Recommendation

The project supports a practical routing strategy:

``` text
Established Series
       |
       v
Global LightGBM
       |
       +---- Good performance ----> Production Forecast
       |
       +---- Degraded performance
                    |
                    v
             Alternative Tier
                    |
             +------+------+
             |             |
          N-HiTS        Chronos/ETS
             |             |
             +------+------+
                    |
                    v
             Best Available Forecast
                    |
                    v
          Hierarchical Reconciliation
                    |
                    v
            Inventory Decision Layer
                    |
                    v
             Monitoring & Retraining
```

The recommended default is **Global LightGBM** because it combines
speed, scalability, and interpretability across all 500 series.
Alternative forecasting tiers provide additional coverage for
uncertainty estimation, degraded series, and cold-start scenarios.

------------------------------------------------------------------------

# 17. Limitations

-   No external covariates such as price, promotion, weather, or holiday
    information are included.
-   Tier 2 and Tier 3 evaluations are restricted to a subset of series
    for computational reasons.
-   SMAPE can become unstable or less informative for very low-demand
    observations; WAPE is therefore used as a complementary metric.
-   Large-scale Chronos inference can require GPU resources for
    efficient execution.
-   Feature importance from LightGBM represents model gain and should
    not be interpreted as causal importance.
-   PSI indicates distributional change but does not by itself identify
    the business cause of the drift.

------------------------------------------------------------------------

# 18. Deliverables

The project produces:

-   A complete forecasting notebook.
-   Global LightGBM demand forecasts.
-   Alternative N-HiTS / quantile forecasts.
-   Cold-start benchmarking strategy.
-   Hierarchical reconciliation.
-   Ensemble/model-routing framework.
-   Service-level-based reorder points.
-   SMAPE and PSI monitoring.
-   EDA and diagnostic visualizations.
-   Final forecast submission.

------------------------------------------------------------------------

## 19. Conclusion

This project presents a complete **retail demand forecasting and
inventory optimization pipeline** rather than an isolated predictive
model.

The analysis shows that the demand system contains strong weekly and
calendar seasonality, persistent temporal dependence, and meaningful
variation between series. A global LightGBM model can exploit these
signals efficiently, while N-HiTS/quantile and zero-shot approaches
provide complementary capabilities for uncertainty and cold-start
scenarios.

The final architecture extends beyond prediction by adding
**hierarchical reconciliation, inventory service-level decisions, and
production monitoring**. This makes the system suitable as a foundation
for practical retail demand planning where forecasts must ultimately
support replenishment and risk decisions.
