from __future__ import annotations

import numpy as np
import pandas as pd
import optuna
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

import xgboost as xgb
from statsmodels.tsa.statespace.sarimax import SARIMAX


# -----------------------------
# Result schema
# -----------------------------
@dataclass
class ForecastResult:
    coin_id: str
    vs_currency: str
    horizon_days: int
    last_date: str
    last_price: float
    predictions: List[Dict[str, float]]
    diagnostics: Dict[str, float]


# -----------------------------
# Utilities
# -----------------------------
def _rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def _safe_log(arr: np.ndarray | pd.Series) -> np.ndarray:
    x = np.asarray(arr, dtype=float)
    return np.log(np.clip(x, 1e-18, None))


def _ensure_datetime(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"])
    return out


# -----------------------------
# Feature engineering (robust)
# -----------------------------
def _compute_features(
    daily_df: pd.DataFrame,
    include_target: bool,
) -> pd.DataFrame:
    """
    Compute features from a *daily frame* that contains:
      date, price, mcap, volume

    If include_target=True:
      adds y = next-day log return (ret1 shifted -1), and drops NaNs.

    If include_target=False:
      does NOT create y, and keeps the last row (needed for forecasting).
    """
    df = _ensure_datetime(daily_df)

    # Sort and enforce daily frequency (safety; your build_daily_frame already does this)
    df = df.sort_values("date").reset_index(drop=True)

    # Core transforms
    df["log_price"] = _safe_log(df["price"].values)
    df["ret1"] = df["log_price"].diff()

    # Rolling stats (momentum + volatility)
    for w in (3, 7, 14, 30, 60):
        df[f"ret_mean_{w}"] = df["ret1"].rolling(w).mean()
        df[f"ret_std_{w}"] = df["ret1"].rolling(w).std()
        df[f"mom_{w}"] = df["log_price"] - df["log_price"].shift(w)

    # Volume/mcap derived features
    # Note: volume/mcap can be 0 or missing; safe log handles small values.
    df["log_vol"] = _safe_log((df["volume"].fillna(0).astype(float).values + 1.0))
    df["log_mcap"] = _safe_log((df["mcap"].fillna(0).astype(float).values + 1.0))
    df["vol_mcap_ratio"] = (df["volume"].astype(float) / df["mcap"].astype(float)).replace(
        [np.inf, -np.inf], np.nan
    )

    # Calendar effects
    df["dow"] = df["date"].dt.dayofweek
    df["dom"] = df["date"].dt.day
    df["month"] = df["date"].dt.month

    if include_target:
        df["y"] = df["ret1"].shift(-1)
        df = df.dropna().reset_index(drop=True)
    else:
        # For forecasting, we still need a clean last row.
        # We must drop rows that are invalid due to rolling windows,
        # but we should keep the last row (no y required).
        df = df.dropna(subset=["ret1", "ret_mean_7", "ret_std_14", "mom_14"]).reset_index(drop=True)

    return df


# -----------------------------
# Modeling
# -----------------------------
def _build_xgb_pipeline(cat_cols: List[str], num_cols: List[str], params: dict) -> Pipeline:
    pre = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
            ("num", "passthrough", num_cols),
        ],
        remainder="drop",
    )
    model = xgb.XGBRegressor(
        **params,
        n_estimators=params.get("n_estimators", 2000),
        random_state=42,
        tree_method="hist",
        verbosity=0,
    )
    return Pipeline([("pre", pre), ("model", model)])


def tune_xgb_heavy(
    df: pd.DataFrame,
    feature_cols: List[str],
    y_col: str = "y",
    trials: int = 60,
    cv_splits: int = 5,
) -> Tuple[Pipeline, Dict[str, float]]:
    """
    Compute-heavy tuning using Optuna + walk-forward CV.
    Objective: minimize RMSE on next-day returns.
    """
    X = df[feature_cols].copy()
    y = df[y_col].astype(float).values

    cat_cols = [c for c in feature_cols if c in ("dow", "dom", "month")]
    num_cols = [c for c in feature_cols if c not in cat_cols]

    tscv = TimeSeriesSplit(n_splits=cv_splits)

    def objective(trial: optuna.Trial) -> float:
        params = {
            "max_depth": trial.suggest_int("max_depth", 2, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.15, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_float("min_child_weight", 0.5, 10.0, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 1.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "gamma": trial.suggest_float("gamma", 0.0, 2.0),
            "n_estimators": trial.suggest_int("n_estimators", 800, 3500),
        }

        pipe = _build_xgb_pipeline(cat_cols, num_cols, params)
        rmses = []
        for train_idx, test_idx in tscv.split(X):
            Xtr, Xte = X.iloc[train_idx], X.iloc[test_idx]
            ytr, yte = y[train_idx], y[test_idx]
            pipe.fit(Xtr, ytr)
            pred = pipe.predict(Xte)
            rmses.append(_rmse(yte, pred))
        return float(np.mean(rmses))

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=trials, show_progress_bar=False)

    best_params = study.best_params
    pipe = _build_xgb_pipeline(cat_cols, num_cols, best_params)
    pipe.fit(X, y)

    return pipe, {"xgb_cv_rmse": float(study.best_value)}


def fit_sarimax(
    df: pd.DataFrame,
    y_col: str = "y",
    exog_cols: Optional[List[str]] = None,
) -> Tuple[object, Dict[str, float]]:
    """
    SARIMAX on returns with optional exogenous regressors.
    """
    y = df[y_col].astype(float).values
    exog = df[exog_cols].astype(float).values if exog_cols else None

    model = SARIMAX(
        y,
        exog=exog,
        order=(2, 0, 2),
        enforce_stationarity=False,
        enforce_invertibility=False,
    ).fit(disp=False)

    return model, {"sarimax_aic": float(model.aic)}


# -----------------------------
# Robust iterative forecast
# -----------------------------
def iterative_forecast_robust(
    daily_df: pd.DataFrame,
    feature_cols: List[str],
    xgb_pipe: Pipeline,
    sarimax_res: object,
    exog_cols_for_sarimax: Optional[List[str]],
    horizon_days: int,
) -> List[Dict[str, float]]:
    """
    Robust multi-step:
      - Maintain a daily frame (date, price, mcap, volume)
      - At each step, recompute engineered features for the latest row
      - Predict next-day return with XGB and SARIMAX
      - Ensemble them
      - Update price, and extend SARIMAX state using predicted return
    """
    base_daily = _ensure_datetime(daily_df)[["date", "price", "mcap", "volume"]].copy()
    base_daily = base_daily.sort_values("date").reset_index(drop=True)

    last_real_date = pd.to_datetime(base_daily.iloc[-1]["date"])
    preds: List[Dict[str, float]] = []

    # For "future" mcap/volume, we need a rule.
    # Reasonable default:
    # - mcap scales with price (assumes supply roughly constant): mcap_{t+1} = mcap_t * exp(ret)
    # - volume carry-forward last observed volume (can be swapped to EMA if you want)
    last_mcap = float(base_daily.iloc[-1]["mcap"]) if pd.notna(base_daily.iloc[-1]["mcap"]) else float("nan")
    last_vol = float(base_daily.iloc[-1]["volume"]) if pd.notna(base_daily.iloc[-1]["volume"]) else float("nan")

    # Keep a SARIMAX "live" results object we append to
    sar_state = sarimax_res

    for i in range(1, horizon_days + 1):
        next_date = last_real_date + pd.Timedelta(days=i)

        # Compute features on current daily history (no target; keep last row)
        feat_df = _compute_features(base_daily, include_target=False)
        if len(feat_df) == 0:
            raise RuntimeError("Not enough history to compute features for forecasting.")

        last_feat_row = feat_df.iloc[-1]

        # Build XGB features for "next_date"
        # Approach: clone last_feat_row but set calendar vars to next_date
        feat_row = last_feat_row.copy()
        feat_row["date"] = next_date
        feat_row["dow"] = next_date.dayofweek
        feat_row["dom"] = next_date.day
        feat_row["month"] = next_date.month

        # IMPORTANT:
        # ret_mean/std/mom/log_vol/log_mcap/ratio in feat_row reflect the *current* state up to today.
        # That's appropriate for a next-day forecast.

        # XGB predicts next-day return
        X_feat = pd.DataFrame([feat_row[feature_cols].to_dict()])
        xgb_ret = float(xgb_pipe.predict(X_feat)[0])

        # SARIMAX predicts next-day return
        if exog_cols_for_sarimax:
            ex = np.array([float(feat_row[c]) for c in exog_cols_for_sarimax], dtype=float).reshape(1, -1)
            sar_ret = float(sar_state.forecast(steps=1, exog=ex)[0])
        else:
            sar_ret = float(sar_state.forecast(steps=1)[0])

        ens_ret = 0.5 * xgb_ret + 0.5 * sar_ret

        # Update price
        last_price = float(base_daily.iloc[-1]["price"])
        next_price = float(np.exp(np.log(max(last_price, 1e-18)) + ens_ret))

        # Update synthetic mcap/volume
        if np.isfinite(last_mcap):
            next_mcap = float(last_mcap * np.exp(ens_ret))
        else:
            next_mcap = float("nan")

        # Carry-forward volume if we have it; otherwise NaN
        next_vol = last_vol if np.isfinite(last_vol) else float("nan")

        preds.append(
            {
                "date": next_date.strftime("%Y-%m-%d"),
                "predicted_price": next_price,
                "predicted_return": ens_ret,
            }
        )

        # Extend daily history with synthetic next day
        base_daily = pd.concat(
            [
                base_daily,
                pd.DataFrame(
                    [{"date": next_date, "price": next_price, "mcap": next_mcap, "volume": next_vol}]
                ),
            ],
            ignore_index=True,
        )

        # Update sarimax state with the predicted return so multi-step isn't static
        if exog_cols_for_sarimax:
            sar_state = sar_state.append([ens_ret], exog=ex, refit=False)
        else:
            sar_state = sar_state.append([ens_ret], refit=False)

        # update cached mcap/volume
        last_mcap = next_mcap
        last_vol = next_vol

    return preds


# -----------------------------
# Public entry point
# -----------------------------
def run_forecast(
    df_feat: pd.DataFrame,
    coin_id: str,
    vs_currency: str,
    horizon_days: int,
    optuna_trials: int = 60,
) -> ForecastResult:
    """
    df_feat is expected to be the output of your BTC-only features.py add_features(),
    BUT we recompute a training frame internally to ensure we can also build a
    consistent daily_df for robust iterative forecasting.

    We rely only on df_feat having at least: date, price, mcap, volume.
    """
    if not all(c in df_feat.columns for c in ("date", "price", "mcap", "volume")):
        raise ValueError("df_feat must contain at least: date, price, mcap, volume")

    # Build daily frame from df_feat (these are raw columns carried through)
    daily_df = _ensure_datetime(df_feat)[["date", "price", "mcap", "volume"]].copy()
    daily_df = daily_df.sort_values("date").reset_index(drop=True)

    # Recompute training features WITH target from daily_df (robust / consistent)
    train_df = _compute_features(daily_df, include_target=True)

    # Select feature columns (exclude target + obvious non-features)
    drop = {"y", "date", "price", "mcap", "volume"}
    feature_cols = [c for c in train_df.columns if c not in drop]

    # Train XGB (heavy tuning)
    xgb_pipe, diag_xgb = tune_xgb_heavy(train_df, feature_cols, trials=optuna_trials, cv_splits=5)

    # SARIMAX exog: BTC-only, and robust (these exist and are consistent)
    # You can broaden this list if you also forecast mcap/volume dynamically.
    exog_cols = [c for c in ("ret_mean_7", "ret_std_14", "mom_14") if c in train_df.columns]
    sar_model, diag_sar = fit_sarimax(train_df, exog_cols=exog_cols if exog_cols else None)

    # Predict forward robustly
    preds = iterative_forecast_robust(
        daily_df=daily_df,
        feature_cols=feature_cols,
        xgb_pipe=xgb_pipe,
        sarimax_res=sar_model,
        exog_cols_for_sarimax=exog_cols if exog_cols else None,
        horizon_days=horizon_days,
    )

    last = daily_df.iloc[-1]
    return ForecastResult(
        coin_id=coin_id,
        vs_currency=vs_currency,
        horizon_days=horizon_days,
        last_date=str(pd.to_datetime(last["date"]).date()),
        last_price=float(last["price"]),
        predictions=preds,
        diagnostics={**diag_xgb, **diag_sar, "rows_used": float(len(train_df))},
    )
