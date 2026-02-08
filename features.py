import numpy as np
import pandas as pd


def _safe_log(x: pd.Series) -> pd.Series:
    return np.log(np.clip(x.astype(float), 1e-18, None))


def build_daily_frame(market_chart_json: dict) -> pd.DataFrame:
    """
    Convert CoinGecko market_chart into a clean daily dataframe:
    date, price, mcap, volume
    """
    prices = pd.DataFrame(market_chart_json["prices"], columns=["ts_ms", "price"])
    mcaps  = pd.DataFrame(market_chart_json.get("market_caps", []), columns=["ts_ms", "mcap"])
    vols   = pd.DataFrame(market_chart_json.get("total_volumes", []), columns=["ts_ms", "volume"])

    df = prices.merge(mcaps, on="ts_ms", how="left").merge(vols, on="ts_ms", how="left")
    df["date"] = pd.to_datetime(df["ts_ms"], unit="ms", utc=True).dt.tz_convert(None)
    df = df.drop(columns=["ts_ms"]).sort_values("date")

    # Make daily: keep last observation per day
    df["day"] = df["date"].dt.floor("D")
    df = (
        df.groupby("day", as_index=False)
          .last()
          .rename(columns={"day": "date"})
          .drop(columns=["date_y"], errors="ignore")
    )
    df = df[["date", "price", "mcap", "volume"]]

    # Fill gaps (rare, but happens). Forward fill market data.
    df = df.set_index("date").asfreq("D")
    df[["price", "mcap", "volume"]] = df[["price", "mcap", "volume"]].ffill()
    df = df.dropna().reset_index()

    return df


def add_features(df_in: pd.DataFrame) -> pd.DataFrame:
    """
    Heavy feature engineering for a single asset (BTC):
    - log returns, rolling volatility/momentum
    - volume/mcap transforms
    - calendar effects
    Target y is next-day log return.
    """
    df = df_in.copy()

    # Core transforms
    df["log_price"] = _safe_log(df["price"])
    df["ret1"] = df["log_price"].diff()

    # Rolling stats (momentum + volatility)
    for w in (3, 7, 14, 30, 60):
        df[f"ret_mean_{w}"] = df["ret1"].rolling(w).mean()
        df[f"ret_std_{w}"] = df["ret1"].rolling(w).std()
        df[f"mom_{w}"] = df["log_price"] - df["log_price"].shift(w)

    # Volume/mcap derived features
    df["log_vol"] = _safe_log(df["volume"].fillna(0) + 1.0)
    df["log_mcap"] = _safe_log(df["mcap"].fillna(0) + 1.0)
    df["vol_mcap_ratio"] = (df["volume"] / df["mcap"]).replace([np.inf, -np.inf], np.nan)

    # Calendar effects
    df["dow"] = df["date"].dt.dayofweek
    df["dom"] = df["date"].dt.day
    df["month"] = df["date"].dt.month

    # Target: next-day return
    df["y"] = df["ret1"].shift(-1)

    # Clean
    df = df.dropna().reset_index(drop=True)
    return df
