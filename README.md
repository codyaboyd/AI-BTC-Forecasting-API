# BTC Internal Price Prediction Service

## Overview

This service provides a **compute-heavy Bitcoin (BTC) price forecasting API** using historical market data from CoinGecko.

It is explicitly optimized for **accuracy, robustness, and research value over latency**, combining:

- CoinGecko historical **BTC price, volume, and market cap**
- Extensive **feature engineering** (returns, momentum, volatility, ratios, calendar effects)
- **Optuna-tuned XGBoost** with walk-forward cross-validation
- **SARIMAX** on BTC returns
- **Ensemble averaging** (XGB + SARIMAX)
- **Robust iterative multi-day forecasting**
  - Features recomputed each step
  - SARIMAX state updated per predicted return
  - Market cap evolves with price
  - Volume carried forward conservatively

This service is designed for **internal research, analytics, and decision support**, not trading automation.

---

## Key Endpoints

### Health Check
```
GET /health
```

---

### Live Spot Price (Reference Only)
```
GET /price/live?vs_currency=usd
```

- Pulls the current BTC spot price from CoinGecko
- **Not a model output**
- Intended for dashboards and sanity checks

---

### Forecast
```
GET /predict?days=7&vs_currency=usd
```

**Query Parameters**

| Param | Description | Notes |
|-----|------------|------|
| `days` | Days forward to forecast | 1–365 |
| `vs_currency` | Quote currency | default: `usd` |
| `optuna_trials` | Override tuning iterations | higher = slower, potentially better |

---

## Environment Variables

```bash
# Core
VS_CURRENCY=usd

# CoinGecko
COINGECKO_API_KEY=your_key_here
COINGECKO_KEY_HEADER=x-cg-demo-api-key   # or x-cg-pro-api-key

# Modeling / Accuracy Controls
OPTUNA_TRIALS=80        # Increase for better tuning (slow)
HISTORY_DAYS=max        # Use full historical BTC data
```

### Accuracy vs Compute Guidance

| Setting | Effect |
|------|-------|
| `OPTUNA_TRIALS ↑` | Better hyperparameter fit, exponential runtime |
| `HISTORY_DAYS=max` | More stable long-term structure |
| Lower trials | Faster iteration, noisier fits |

---

## Architecture Summary

```
CoinGecko (BTC)
        ↓
Daily Market Frame
(date, price, mcap, volume)
        ↓
Feature Engineering
(returns, momentum, volatility, ratios, calendar)
        ↓
┌──────────────────────────────────────┐
│  XGBoost (Optuna + Walk-Forward CV)  │
│  SARIMAX (returns, evolving state)   │
└──────────────┬───────────────────────┘
               ↓
        Ensemble Return Forecast
               ↓
   Robust Iterative Price Reconstruction
```

---

## What the Model Predicts (Important)

- The model **predicts next-day log returns**, not absolute prices
- Prices are reconstructed iteratively
- SARIMAX state and engineered features evolve step-by-step
- Forecast error **compounds with horizon length**

### Reliability Rule of Thumb

| Horizon | Reliability |
|------|-------------|
| 1–3 days | Relatively high |
| 7–14 days | Medium |
| 30+ days | Low (trend-biased, exploratory only) |

---

## Internal Usage Recommendations

### ✅ Good Uses
- Internal dashboards
- Regime / volatility analysis
- Scenario comparison
- Research & experimentation
- Risk modeling inputs
- Narrative stress-testing

### ❌ Not Recommended
- Automated trading without additional signals
- User-facing predictions
- Marketing claims
- Legal, compliance, or financial disclosures

---

## Performance Expectations

| OPTUNA_TRIALS | Typical Runtime |
|--------------|----------------|
| ~40 | 30–60 seconds |
| ~80 | 2–4 minutes |
| 200+ | 5–15+ minutes |

**Strong Recommendation**

Run forecasts behind:
- A **job queue**
- Or a **cron-based cache** (hourly / daily)

Avoid synchronous, user-triggered execution.

---

## Running Locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

uvicorn app:app --host 0.0.0.0 --port 8080
```

---

## Internal Disclaimer

This system provides **statistical forecasts derived from historical BTC market data only**.

It does **not account for**:
- News or macro events
- Exchange outages
- Regulatory actions
- Market manipulation
- Liquidity shocks
- Black-swan events

Outputs should be interpreted as **research signals**, not predictions of future reality.
