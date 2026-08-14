# Multi-Series Forecasting for Retail Demand Optimization
## Complete System Design & AI-Assisted Build Plan

Author: Dhanraj Deshmukh — Final Project, Data Science Internship
Reference dataset: Kaggle "Store Item Demand Forecasting Challenge" (`demand-forecasting-kernels-only`) — 10 stores × 50 items = 500 related daily time series, ~913,000 rows, 5 years of history (2013–2017), 3-month forecast horizon, scored on SMAPE.

A note before diving in: this final project does not have its own row on the DS Instructions sheet (which stops at Week 8) — this document is structured the way the weekly ones are (a build plan mapped to explicit success criteria), so it should read as a natural continuation, not a departure from the pattern.

---

## 1. Reframing the problem like a practitioner, not a competitor

The Kaggle leaderboard version of this problem rewards squeezing out the last 0.5% of SMAPE. A senior DS approaching this as a *retail demand system*, not a competition entry, asks a different first question: **what decision does this forecast actually drive, and what does that decision need from the forecast that a single point estimate doesn't give you?**

Inventory and supply chain decisions need three things a plain point forecast doesn't provide on its own:
1. **A distribution, not a point** — you don't order to the mean demand, you order to cover a target service level, which means you need quantiles (P50, P90, P95), not just a single number.
2. **Coherence across aggregation levels** — a supply chain plans at multiple levels simultaneously (per SKU-per-store for shelf replenishment, per-store for delivery truck loading, per-item-across-all-stores for supplier purchase orders). If your 500 individual forecasts don't sum up consistently to your store-level and item-level totals, different teams making decisions off different aggregations will contradict each other.
3. **Graceful degradation for new or sparse series** — a real retail catalog constantly adds new SKUs and opens new stores with little to no history. A system that only works for series with 5 years of data isn't a system, it's a benchmark result.

That reframing is what separates a "trained three models and picked the best SMAPE" submission from a system design — and it's what the rest of this document builds toward.

---

## 2. Research grounding — what actually works here, and why

**Why per-series models (one ARIMA per store-item pair) don't scale or perform well:** 500 series means 500 independently-tuned models, no ability to borrow statistical strength across similar series (e.g. two items with similar seasonal patterns), and no way to generalize to a new store-item pair that just started selling. This class of problem is exactly what "global forecasting models" were developed for — a single model trained across all series simultaneously, using the series identifier (store, item) as a categorical feature or embedding, so the model learns shared temporal structure (weekly/yearly seasonality patterns common across the whole catalog) while still being able to specialize per series.

**Three tiers of global models, each with a different tradeoff, worth building as an ensemble rather than picking one:**

- **Tier 1 — Gradient boosting on engineered features (LightGBM/XGBoost).** Fast to train even on 500×5-years of data, handles categorical store/item IDs natively, strong baseline that typically gets within a few percent of far more complex approaches on this kind of tabular-friendly retail data. This is the tier that should exist first — it's the benchmark everything else has to beat to justify its added complexity.
- **Tier 2 — A global deep learning forecaster (N-HiTS or a Temporal Fusion Transformer via Nixtla's `neuralforecast` or `pytorch-forecasting`).** These learn shared temporal representations across all 500 series jointly and tend to outperform gradient boosting specifically on multi-horizon forecasts with strong seasonality, at the cost of longer training time and more careful tuning.
- **Tier 3 — A zero-shot time series foundation model (Chronos-2, TimesFM, or Moirai-2).** As of 2026, these pretrained transformer models deliver strong zero-shot forecasts with no task-specific training at all, and recent benchmarks show them matching or exceeding tuned domain-specific models on standard evaluation suites (GIFT-Eval, LOTSA). The practical value here isn't replacing Tiers 1–2 — it's **handling cold-start series** (a brand new store-item combination with days, not years, of history) where a model trained from scratch has nothing to learn from, but a foundation model pretrained on millions of other series can still produce a reasonable forecast out of the box.

**Why this three-tier structure is the "advanced and futuristic" answer, not just "use three models":** the real design insight is that each tier covers a failure mode of the others. Gradient boosting is fast and interpretable but needs hand-engineered lag features and doesn't naturally model uncertainty. The deep global model captures richer temporal patterns but needs enough history per series to be worth its cost. The foundation model needs no training data at all but is the least specialized to *this* catalog's specific patterns. An ensemble/routing layer that uses Tier 1 or 2 for established series and falls back to Tier 3 for sparse/new ones is a genuinely production-shaped design, not a toy comparison.

**Why hierarchical reconciliation matters here specifically:** the store-item series naturally roll up two ways — sum across items within a store (store-level totals, useful for delivery truck loading) and sum across stores for an item (item-level totals, useful for supplier purchase orders). If each of the 500 series is forecast independently, these aggregates won't be coherent (the sum of the parts won't equal the independently-forecast whole). Methods like MinT (trace minimization) or bottom-up reconciliation fix this by adjusting the base forecasts so different aggregation levels agree — a genuinely necessary step for a forecast that's meant to drive coordinated decisions across a supply chain, not just win a metric.

---

## 3. System architecture

The diagram above shows the six-stage pipeline. Walking through the design reasoning for each stage:

### 3.1 Data ingestion
Loads the 913K-row dataset, validates schema (date, store, item, sales), and reshapes it into a long-format panel (one row per store-item-date) suitable for global model training. This stage also establishes the **train/validation/test split using rolling-origin (time series) cross-validation** — never a random split, since that would leak future information backward, exactly as flagged in the Week 2 assignment's chronological-split requirement.

### 3.2 Feature engineering
Two categories of features, matched to what each model tier actually needs:
- **For the gradient boosting tier:** lag features (1, 7, 14, 28 days), rolling statistics (7/28-day mean and std), calendar features (day-of-week, month, is-weekend, days-to-month-end), and Fourier terms to capture yearly seasonality smoothly instead of via one-hot month dummies.
- **For the deep learning tier:** raw sequences plus static covariates (store ID, item ID, and any store/item metadata) fed as embeddings the model learns itself — deep global models generally need far less manual feature engineering than gradient boosting, since the model learns temporal representations directly.

### 3.3 Model ensemble (the three tiers described in Section 2)
Each tier trained and backtested independently first, then combined — either via simple weighted averaging, or (more advanced) via a stacking meta-model that learns per-series which tier to trust more, based on that series' history length and volatility.

### 3.4 Reconciliation
Takes the 500 base forecasts and enforces coherence across the store-level and item-level aggregates using MinT reconciliation (`hierarchicalforecast` from Nixtla implements this directly on top of any base forecast).

### 3.5 Inventory decision layer
This is the step that closes the loop back to the actual business objective. Converts the reconciled probabilistic forecast into a concrete recommendation using a **newsvendor-style safety stock formula**: reorder point = lead-time demand (from the forecast) + safety stock (from the forecast's uncertainty and a target service level, e.g. z-score × forecast std-dev for a 95% service level). This is the deliverable a supply chain team would actually consume — not a SMAPE number.

### 3.6 Monitoring
Tracks forecast error drift over time (rolling SMAPE/WAPE by series) and input feature drift (e.g. PSI on recent sales distributions vs. training distribution — directly extending the Week 1 "model monitoring basics" and "distribution/stationarity testing" concepts), triggering retraining when either crosses a threshold. This closes the loop back into the model ensemble, shown as the feedback note in the diagram.

---

## 4. Build plan — phased, with ready-to-use AI prompts

Each phase below includes a prompt written to be pasted directly into an AI coding assistant (Claude Code, Cursor, or a Colab AI cell) to generate that piece of the system. They're written assuming you paste them in order, since each phase's prompt references the artifacts (dataframes, functions) the previous phase produces.

### Phase 0 — Environment and data setup

> Set up a Python environment for a multi-series time series forecasting project. Install pandas, numpy, lightgbm, scikit-learn, neuralforecast, hierarchicalforecast, chronos-forecasting (or the appropriate Hugging Face package for Chronos-2), matplotlib, and plotly. Load the Kaggle "Store Item Demand Forecasting Challenge" train.csv (columns: date, store, item, sales) into a pandas DataFrame. Print shape, dtypes, date range, and confirm there are exactly 500 unique (store, item) combinations with no gaps in the daily date range per series. Convert the date column to datetime and set a MultiIndex on (store, item, date).

### Phase 1 — EDA and seasonality diagnostics

> Using the loaded store-item sales DataFrame, produce: (1) an aggregate daily sales line plot with a 28-day rolling mean overlay, (2) an ACF/PACF plot for a sample of 5 representative series to identify weekly and yearly seasonality lags, (3) an ADF stationarity test on both raw sales and first-differenced sales for the same sample series, printing and interpreting the p-values, (4) a heatmap of average sales by day-of-week and month to visualize the seasonality structure, and (5) a boxplot comparing sales distributions across the 10 stores to check for store-level scale differences that might need per-store normalization.

### Phase 2 — Feature engineering pipeline

> Build a feature engineering function that takes the long-format store-item-date sales DataFrame and returns it augmented with: lag features at 1, 7, 14, and 28 days; rolling mean and rolling std at 7 and 28-day windows; calendar features (day of week, month, day of month, is_weekend, is_month_start, is_month_end); and two Fourier feature pairs (sin/cos) for yearly seasonality using period=365.25. Ensure all lag/rolling features are computed per (store, item) group so no leakage occurs across series. Handle the resulting NaN values from the lag/rolling windows appropriately for a chronological split (do not fill with global means — instead drop or forward-fill only within each series' own early history).

### Phase 3 — Tier 1: global LightGBM baseline

> Using the engineered feature set, train a single global LightGBM regressor across all 500 store-item series simultaneously, using store and item as categorical features. Split the data chronologically using rolling-origin cross-validation with at least 3 folds, each fold training on all data up to a cutoff date and validating on the following 90 days (matching the competition's 3-month horizon). Report per-fold SMAPE and WAPE, plus the mean and std across folds. Plot feature importances. Save the best model.

### Phase 4 — Tier 2: global deep learning forecaster

> Using Nixtla's neuralforecast library, train an N-HiTS model (or a Temporal Fusion Transformer if compute allows) on the same 500-series panel, using store and item IDs as static categorical covariates. Use the same rolling-origin cross-validation folds as Phase 3 so results are directly comparable. Configure the model to output quantile forecasts (at minimum P10, P50, P90) rather than a single point estimate, using a pinball/quantile loss. Report the same SMAPE/WAPE metrics as Phase 3 for the P50 forecast, plus the pinball loss for the quantile forecasts, and plot predicted vs actual for a handful of representative series (one high-volume, one low-volume, one with a clear trend).

### Phase 5 — Tier 3: zero-shot foundation model benchmark

> Using Amazon's Chronos-2 (or TimesFM as an alternative), run zero-shot forecasting on: (a) the full set of 500 series for direct comparison against Tiers 1 and 2 on the same rolling-origin folds, and (b) a simulated cold-start scenario where you truncate 20 series to only their most recent 30 days of history, to specifically test how the foundation model performs versus Tiers 1–2 when there isn't enough history for those models to have learned that series' pattern well. Report SMAPE for both scenarios and specifically highlight the cold-start comparison, since that's the scenario this tier is meant to solve.

### Phase 6 — Hierarchical reconciliation

> Using Nixtla's hierarchicalforecast library, define two aggregation hierarchies over the 500 base series: total-by-store (summing across all items within each store) and total-by-item (summing across all stores for each item). Apply MinT (trace minimization) reconciliation to the ensemble's base forecasts from Phases 3–5 so that the reconciled store-level and item-level aggregates are coherent with the sum of their constituent series. Verify coherence numerically (sum of reconciled item-level series forecasts should exactly equal the reconciled store-total forecast) and report whether reconciliation improved or degraded SMAPE at each aggregation level compared to the unreconciled base forecasts.

### Phase 7 — Ensembling and model selection

> Build an ensemble/routing layer that combines the Tier 1 (LightGBM), Tier 2 (N-HiTS), and Tier 3 (Chronos-2) forecasts. Implement two combination strategies to compare: (1) simple inverse-error weighted averaging based on each tier's validation SMAPE, and (2) a per-series routing rule that uses Tier 3 (foundation model) for any series with fewer than 90 days of history and a weighted blend of Tier 1/Tier 2 otherwise. Backtest both strategies against each individual tier on the same rolling-origin folds and report which wins on aggregate SMAPE, and whether the routing strategy specifically improves cold-start series performance versus using Tier 1/2 alone.

### Phase 8 — Inventory decision layer

> Using the final ensemble's quantile forecasts (P50, P90, P95), implement a safety-stock and reorder-point calculation per store-item series: reorder_point = lead_time_demand (sum of P50 forecast over the lead time window, e.g. 7 days) + safety_stock (z-score for target service level × forecast standard deviation over the lead time window, derived from the P90/P50 spread). Make the target service level configurable (e.g. 90%, 95%, 99%) and produce a summary table showing, for a sample of 20 series, the recommended reorder point at each service level alongside the historical average demand, so the tradeoff between service level and inventory cost is visible.

### Phase 9 — Monitoring and retraining triggers

> Build a monitoring module that computes, on a rolling basis: (1) per-series and aggregate SMAPE/WAPE on the most recent completed forecast window compared to actuals, (2) a population stability index (PSI) comparing the distribution of recent actual sales against the training distribution, per series, to detect demand pattern drift. Flag any series where PSI exceeds 0.2 (moderate drift) or where rolling SMAPE has increased by more than 20% relative to its backtest baseline, and produce a simple report table of flagged series as the trigger condition for retraining.

### Phase 10 — Final report and results dashboard

> Build a summary notebook section (or a lightweight Plotly Dash / Streamlit dashboard if time allows) presenting: the SMAPE/WAPE comparison table across all three tiers and the ensemble, the reconciliation coherence check results, the cold-start comparison from Phase 5, a sample of actual-vs-predicted plots for representative series, the inventory decision table from Phase 8, and a written summary of which tier performed best in which scenario (established high-volume series vs. sparse/new series) with a recommendation for which combination should go into production.

---

## 5. Evaluation criteria and success metrics

| Metric | Definition | Why it's here |
|---|---|---|
| SMAPE | Symmetric Mean Absolute Percentage Error | The competition's own scoring metric — needed for direct benchmarking against the Kaggle leaderboard context |
| WAPE | Weighted Absolute Percentage Error | More robust than SMAPE for low-volume series where percentage errors get noisy; a better metric for the business-facing report |
| Pinball loss | Quantile loss at P10/P50/P90 | Measures whether the *uncertainty* estimate is well-calibrated, not just the point forecast — directly relevant to the inventory decision layer |
| Reconciliation coherence | Exact equality check between aggregated base forecasts and reconciled totals | Confirms the hierarchical reconciliation step actually did its job |
| Cold-start SMAPE gap | SMAPE on truncated-history series, Tier 3 vs Tiers 1–2 | The specific number that justifies including the foundation model tier at all |
| Service-level vs. inventory cost tradeoff | Reorder point at 90/95/99% service levels | Translates the whole system's output into a business-legible result |

---

## 6. Submission structure (mirroring the established weekly pattern)

Since this final project has no dedicated row on the instructions sheet, structuring the submission the same way as Weeks 1–8 keeps it consistent and easy for an evaluator to navigate:

```
final-project/
  Demand_Forecasting_System.ipynb   ← Phases 0-10, in order, each as its own section
  README.md                          ← Problem statement, architecture summary, key findings
  system_design.md                   ← This document
  results/
    tier_comparison.csv              ← SMAPE/WAPE/pinball loss per tier per fold
    reconciliation_check.csv
    inventory_recommendations.csv
```

---

## 7. Risks, limitations, and honest caveats

- **Compute cost**: Tier 2 and Tier 3 both need meaningfully more compute than Tier 1 — if Colab's free-tier GPU quota is a constraint, it's reasonable (and worth stating explicitly in the final report) to run Tiers 2–3 on a representative subset of series rather than all 500, while keeping Tier 1 on the full set.
- **This dataset has no external covariates** — no price, promotion, or holiday data is included in the raw Kaggle files, which caps how much the feature engineering and foundation-model covariate-handling can actually demonstrate. Worth stating this limitation directly rather than implying the system was tested against a richer real-world data setting.
- **Foundation model licensing/access**: confirm Chronos-2's current license and any API cost before committing to it as a core dependency — it was open-source/Hugging-Face-hosted as of early 2026 per public documentation, but this is exactly the kind of detail worth a quick check right before you actually build, not assumed from this document.
- **SMAPE's known distortion at near-zero actuals**: some store-item series legitimately have days with zero or near-zero sales, where SMAPE can behave erratically. WAPE is included specifically as a more stable secondary metric for this reason, and it's worth explaining that reasoning if asked in the evaluation meeting.
