"""Price features and the 5-day realized-volatility target.

Convention: a row dated t is the forecast origin, i.e. the close of day t.
* Features use data up to and including day t (tests/test_features.py checks this).
* The target is RV(t, t+5) = sqrt(sum_{i=1..5} r_{t+i}^2), built from the next 5 log returns.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

LN2 = np.log(2.0)


def log_returns(close: pd.Series) -> pd.Series:
    return np.log(close).diff()


def realized_vol_target(ret: pd.Series, horizon: int = 5) -> pd.Series:
    """sqrt(sum of squared returns over t+1..t+h), aligned to row t."""
    sq = ret.pow(2)
    fwd = sq[::-1].rolling(horizon, min_periods=horizon).sum()[::-1].shift(-1)
    return np.sqrt(fwd)


def parkinson_var(high: pd.Series, low: pd.Series) -> pd.Series:
    return np.log(high / low).pow(2) / (4.0 * LN2)


def garman_klass_var(o: pd.Series, h: pd.Series, l: pd.Series, c: pd.Series) -> pd.Series:
    gk = 0.5 * np.log(h / l).pow(2) - (2.0 * LN2 - 1.0) * np.log(c / o).pow(2)
    return gk.clip(lower=0.0)


def price_features(px: pd.DataFrame, vix: pd.Series, horizon: int = 5, var_floor: float = 1e-8) -> pd.DataFrame:
    """Build features + target for one ticker. `px` has Open/High/Low/Close/Volume."""
    out = pd.DataFrame(index=px.index)
    r = log_returns(px["Close"])
    r2 = r.pow(2)
    out["ret"] = r
    out["abs_ret"] = r.abs()

    # HAR components, scaled to the 5-day horizon so they share units with the target.
    # Scaling only shifts the log by a constant, which the regression intercept absorbs.
    rv_d = horizon * r2
    rv_w = r2.rolling(5, min_periods=5).sum() * (horizon / 5)
    rv_m = r2.rolling(22, min_periods=22).mean() * horizon
    out["log_rv_d"] = 0.5 * np.log(rv_d.clip(lower=var_floor))
    out["log_rv_w"] = 0.5 * np.log(rv_w.clip(lower=var_floor))
    out["log_rv_m"] = 0.5 * np.log(rv_m.clip(lower=var_floor))

    pk = parkinson_var(px["High"], px["Low"])
    gk = garman_klass_var(px["Open"], px["High"], px["Low"], px["Close"])
    out["log_parkinson"] = 0.5 * np.log((horizon * pk).clip(lower=var_floor))
    out["log_gk"] = 0.5 * np.log((horizon * gk).clip(lower=var_floor))
    out["log_parkinson_w"] = 0.5 * np.log((horizon * pk.rolling(5, min_periods=5).mean()).clip(lower=var_floor))
    out["log_gk_w"] = 0.5 * np.log((horizon * gk.rolling(5, min_periods=5).mean()).clip(lower=var_floor))

    lv = np.log(px["Volume"].replace(0, np.nan))
    mu = lv.rolling(60, min_periods=20).mean()
    sd = lv.rolling(60, min_periods=20).std()
    out["volume_z"] = ((lv - mu) / sd).clip(-6, 6)

    # VIX closes at 16:15 ET and stocks at 16:00. That 15-minute overlap is a small,
    # documented look-ahead.
    v = vix.reindex(px.index)
    out["vix"] = v
    out["log_vix"] = np.log(v)
    out["dlog_vix"] = np.log(v).diff()

    dow = px.index.dayofweek
    for d in range(5):
        out[f"dow_{d}"] = (dow == d).astype(np.float32)

    rv5 = realized_vol_target(r, horizon)
    out["rv5"] = rv5
    out["y"] = 0.5 * np.log((rv5.pow(2)).clip(lower=var_floor))  # log RV
    out["rv2_true"] = rv5.pow(2)
    return out


def build_price_panel(prices: dict[str, pd.DataFrame], vix: pd.DataFrame, calendar: pd.DatetimeIndex,
                      horizon: int = 5, var_floor: float = 1e-8) -> pd.DataFrame:
    """Long panel indexed by (date, ticker) on a common trading calendar."""
    frames = []
    vix_close = vix["Close"].reindex(calendar).ffill()
    for t, px in prices.items():
        px = px.reindex(calendar)
        f = price_features(px, vix_close, horizon, var_floor)
        f["ticker"] = t
        frames.append(f)
    panel = pd.concat(frames)
    panel.index.name = "date"
    return panel.reset_index().set_index(["date", "ticker"]).sort_index()
