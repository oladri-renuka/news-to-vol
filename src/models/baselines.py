"""Econometric baselines: EWMA (RiskMetrics), GARCH(1,1)-t, HAR-RV, HAR-RV + sentiment.

Each `predict_*` returns a DataFrame with columns [date, ticker, y_pred, var_pred], where
y_pred is the log 5-day vol forecast and var_pred is the 5-day variance forecast.
"""
from __future__ import annotations

import logging
import warnings

import numpy as np
import pandas as pd

from src.metrics import smearing_factor

log = logging.getLogger(__name__)


# ---------------------------------------------------------------- EWMA
def ewma_variance(ret: np.ndarray, lam: float = 0.94, init_n: int = 22) -> np.ndarray:
    """One-step-ahead variance sigma^2_{t+1|t}, computed with data through t (NaN until init)."""
    r = np.asarray(ret, float)
    out = np.full(len(r), np.nan)
    first = np.flatnonzero(np.isfinite(r))
    if len(first) < init_n:
        return out
    s0 = first[0]
    var = np.nanvar(r[s0:s0 + init_n])
    for t in range(s0, len(r)):
        rt = r[t] if np.isfinite(r[t]) else 0.0
        var = lam * var + (1 - lam) * rt * rt
        if t >= s0 + init_n - 1:
            out[t] = var
    return out


def predict_ewma(panel: pd.DataFrame, test_dates: pd.DatetimeIndex, lam: float = 0.94, horizon: int = 5) -> pd.DataFrame:
    rows = []
    for t, g in panel.groupby(level="ticker"):
        g = g.droplevel("ticker")
        v1 = pd.Series(ewma_variance(g["ret"].to_numpy(), lam), index=g.index)
        v = horizon * v1.reindex(test_dates)  # RiskMetrics: flat term structure
        rows.append(pd.DataFrame({"date": test_dates, "ticker": t, "var_pred": v.to_numpy()}))
    out = pd.concat(rows, ignore_index=True)
    out["y_pred"] = 0.5 * np.log(out["var_pred"])
    return out


# ---------------------------------------------------------------- GARCH
def predict_garch(panel: pd.DataFrame, fit_start: pd.Timestamp, fit_end: pd.Timestamp,
                  test_dates: pd.DatetimeIndex, horizon: int = 5, dist: str = "t") -> pd.DataFrame:
    """GARCH(1,1) with Student-t errors, one fit per ticker on [fit_start, fit_end].

    The test forecasts keep the fitted parameters fixed and filter the variance through each
    origin t, so each forecast uses only returns up to t.
    """
    from arch import arch_model

    rows = []
    test_end = test_dates.max()
    for t, g in panel.groupby(level="ticker"):
        r = g.droplevel("ticker")["ret"].loc[fit_start:test_end].dropna() * 100.0
        am = arch_model(r, mean="Constant", vol="GARCH", p=1, q=1, dist=dist, rescale=False)
        next_after_fit = r.index[r.index > fit_end]
        if len(next_after_fit) == 0:
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                res = am.fit(last_obs=next_after_fit[0], disp="off", options={"maxiter": 500})
            except Exception as e:  # pragma: no cover
                log.warning("GARCH fit failed for %s: %s", t, e)
                continue
            first_test = test_dates[test_dates.isin(r.index)].min()
            fc = res.forecast(horizon=horizon, start=first_test, reindex=False)
        var5 = fc.variance.sum(axis=1) / 1e4
        var5 = var5.reindex(test_dates)
        rows.append(pd.DataFrame({"date": test_dates, "ticker": t, "var_pred": var5.to_numpy()}))
    out = pd.concat(rows, ignore_index=True)
    out["y_pred"] = 0.5 * np.log(out["var_pred"])
    return out


# ---------------------------------------------------------------- HAR
def _ols(X: np.ndarray, y: np.ndarray, ridge: float = 0.0) -> np.ndarray:
    """OLS with intercept. Constant columns get a zero coefficient. That happens, e.g., to the
    sentiment features of a ticker with no headlines in the fit window."""
    beta = np.zeros(X.shape[1] + 1)
    keep = np.flatnonzero(X.std(0) > 1e-12)
    X1 = np.column_stack([np.ones(len(X)), X[:, keep]])
    A = X1.T @ X1
    if ridge:
        A = A + ridge * np.diag([0.0] + [1.0] * len(keep))
    b = np.linalg.lstsq(A, X1.T @ y, rcond=None)[0]
    beta[0], beta[1 + keep] = b[0], b[1:]
    return beta


def fit_har(panel: pd.DataFrame, fit_dates: pd.DatetimeIndex, cols: list[str], pooled: bool = False,
            ridge: float = 0.0) -> dict:
    """Return {ticker: (coef, smear)}. Pooled models share a single coef under the key '__all__'."""
    fit = panel[panel.index.get_level_values("date").isin(fit_dates)].dropna(subset=cols + ["y"])
    models = {}
    groups = [("__all__", fit)] if pooled else list(fit.groupby(level="ticker"))
    for t, g in groups:
        X, y = g[cols].to_numpy(float), g["y"].to_numpy(float)
        beta = _ols(X, y, ridge)
        yhat = beta[0] + X @ beta[1:]
        models[t] = (beta, smearing_factor(y, yhat))
    return models


def predict_har(panel: pd.DataFrame, models: dict, test_dates: pd.DatetimeIndex, cols: list[str]) -> pd.DataFrame:
    te = panel[panel.index.get_level_values("date").isin(test_dates)].dropna(subset=cols)
    rows = []
    for t, g in te.groupby(level="ticker"):
        beta, smear = models.get(t, models.get("__all__"))
        yhat = beta[0] + g[cols].to_numpy(float) @ beta[1:]
        rows.append(pd.DataFrame({"date": g.index.get_level_values("date"), "ticker": t,
                                  "y_pred": yhat, "var_pred": np.exp(2 * yhat) * smear}))
    return pd.concat(rows, ignore_index=True)


def har_coef_table(models: dict, cols: list[str]) -> pd.DataFrame:
    return pd.DataFrame({t: np.r_[b, s] for t, (b, s) in models.items()},
                        index=["const"] + cols + ["smear"]).T
