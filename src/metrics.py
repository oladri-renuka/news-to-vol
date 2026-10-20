"""Forecast losses and tests of equal predictive accuracy.

Conventions:
    y      = log realized vol (target), so the realized variance is RV^2 = exp(2y)
    yhat   = forecast of log vol
    h      = forecast of the 5-day variance (needed for QLIKE)

QLIKE (Patton 2011) in variance space:  L(RV^2, h) = RV^2/h - log(RV^2/h) - 1 >= 0.
It is robust to noise in the volatility proxy and penalizes under-prediction more than over-prediction.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def mse(y: np.ndarray, yhat: np.ndarray) -> float:
    return float(np.mean((np.asarray(y) - np.asarray(yhat)) ** 2))


def qlike_vec(rv2: np.ndarray, h: np.ndarray, floor: float = 1e-10) -> np.ndarray:
    rv2 = np.maximum(np.asarray(rv2, float), floor)
    h = np.maximum(np.asarray(h, float), floor)
    r = rv2 / h
    return r - np.log(r) - 1.0


def qlike(rv2, h) -> float:
    return float(np.mean(qlike_vec(rv2, h)))


def smearing_factor(y: np.ndarray, yhat: np.ndarray) -> float:
    """Duan (1983) smearing: E[exp(2(y - yhat))].

    It turns a log-vol forecast into an unbiased variance forecast:
    h = exp(2 * yhat) * smear. Estimate it on the fit window only.
    """
    e = np.asarray(y) - np.asarray(yhat)
    return float(np.mean(np.exp(2.0 * e)))


def newey_west_var(d: np.ndarray, lags: int) -> float:
    """HAC long-run variance of a mean-zero-tested series, with Bartlett weights."""
    d = np.asarray(d, float) - np.mean(d)
    T = len(d)
    v = d @ d / T
    for k in range(1, lags + 1):
        w = 1.0 - k / (lags + 1.0)
        v += 2.0 * w * (d[k:] @ d[:-k]) / T
    return v


def diebold_mariano(loss_a: np.ndarray, loss_b: np.ndarray, h: int = 5) -> dict:
    """DM test of H0: E[loss_a - loss_b] = 0, with the HLN (1997) small-sample correction.

    The losses must be a time series, e.g. the cross-sectional mean per date. Overlapping
    h-step targets make d_t MA(h-1), so the HAC variance uses h-1 lags.
    A negative statistic means model A has lower loss.
    """
    d = np.asarray(loss_a, float) - np.asarray(loss_b, float)
    d = d[np.isfinite(d)]
    T = len(d)
    if T < 10:
        return {"dm_stat": np.nan, "p_value": np.nan, "mean_diff": np.nan, "T": T}
    lrv = newey_west_var(d, max(h - 1, 0))
    if lrv <= 0:
        lrv = np.var(d) + 1e-18
    dm = d.mean() / np.sqrt(lrv / T)
    hln = np.sqrt((T + 1 - 2 * h + h * (h - 1) / T) / T)
    stat = dm * hln
    p = 2 * stats.t.sf(abs(stat), df=T - 1)
    return {"dm_stat": float(stat), "p_value": float(p), "mean_diff": float(d.mean()), "T": T}


def per_date_losses(df: pd.DataFrame, pred_col: str = "y_pred", var_col: str = "var_pred") -> pd.DataFrame:
    """Cross-sectional mean loss per date. `df` has date, y, rv2_true, and the prediction columns."""
    e2 = (df["y"] - df[pred_col]) ** 2
    ql = qlike_vec(df["rv2_true"], df[var_col])
    return pd.DataFrame({"date": df["date"], "mse": e2, "qlike": ql}).groupby("date").mean()


def model_confidence_set(losses: pd.DataFrame, size: float = 0.1, reps: int = 1000, block: int = 10,
                         seed: int = 0) -> pd.DataFrame:
    """Hansen-Lunde-Nason MCS via `arch`. `losses` is a T x M frame with one column per model."""
    from arch.bootstrap import MCS

    mcs = MCS(losses.dropna(), size=size, reps=reps, block_size=block, method="R", seed=seed)
    mcs.compute()
    pv = mcs.pvalues.rename(columns={"Pvalue": "mcs_pvalue"})
    pv["in_mcs"] = pv.index.isin(mcs.included)
    return pv
