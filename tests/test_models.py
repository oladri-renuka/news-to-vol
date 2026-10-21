import numpy as np
import pandas as pd
import pytest
import torch

from src.models.baselines import ewma_variance, fit_har, predict_ewma, predict_garch, predict_har
from src.models.deep import LSTMForecaster, TransformerForecaster, n_params


def test_ewma_recursion():
    r = np.array([0.01, -0.02, 0.015, 0.0, 0.03])
    v = ewma_variance(r, lam=0.94, init_n=2)
    var = np.var(r[:2])
    for t in range(len(r)):
        var = 0.94 * var + 0.06 * r[t] ** 2
        if t >= 1:
            assert np.isclose(v[t], var)


def test_ewma_forecast_uses_only_past(small_panel):
    dates = pd.DatetimeIndex(sorted(small_panel.index.get_level_values("date").unique()))[300:320]
    a = predict_ewma(small_panel, dates)
    p2 = small_panel.copy()
    future = p2.index.get_level_values("date") > dates.max()
    p2.loc[future, "ret"] = 0.5
    b = predict_ewma(p2, dates)
    np.testing.assert_allclose(a.var_pred, b.var_pred)


def test_har_recovers_coefficients():
    rng = np.random.default_rng(0)
    n = 3000
    dates = pd.bdate_range("2010-01-01", periods=n)
    X = rng.normal(size=(n, 3))
    y = 0.1 + X @ np.array([0.3, 0.4, 0.2]) + rng.normal(0, 0.1, n)
    df = pd.DataFrame(X, columns=["a", "b", "c"], index=pd.MultiIndex.from_product([dates, ["T"]], names=["date", "ticker"]))
    df["y"] = y
    m = fit_har(df, dates[:2000], ["a", "b", "c"])
    beta, smear = m["T"]
    np.testing.assert_allclose(beta, [0.1, 0.3, 0.4, 0.2], atol=0.02)
    assert 1.0 < smear < 1.1
    p = predict_har(df, m, dates[2000:], ["a", "b", "c"])
    assert len(p) == 1000 and np.corrcoef(p.y_pred, y[2000:])[0, 1] > 0.95


def test_garch_runs_and_is_positive(small_panel):
    dates = pd.DatetimeIndex(sorted(small_panel.index.get_level_values("date").unique()))
    p = predict_garch(small_panel, dates[30], dates[600], dates[601:650])
    assert len(p) == 3 * 49
    assert (p.var_pred > 0).all() and np.isfinite(p.y_pred).all()


@pytest.mark.parametrize("cls", [LSTMForecaster, TransformerForecaster])
def test_deep_shapes_and_overfit(cls):
    torch.manual_seed(0)
    m = cls(n_features=6, n_tickers=4)
    x = torch.randn(32, 60, 6)
    tk = torch.randint(0, 4, (32,))
    y = x[:, -5:, 0].mean(1)  # a learnable function of the recent past
    assert m(x, tk).shape == (32,)
    opt = torch.optim.Adam(m.parameters(), 1e-2)
    l0 = None
    for _ in range(150):
        loss = torch.nn.functional.mse_loss(m(x, tk), y)
        opt.zero_grad(); loss.backward(); opt.step()
        l0 = l0 or loss.item()
    assert loss.item() < 0.2 * l0
    assert n_params(m) < 500_000


def test_pipeline_drops_partial_bar():
    from src.pipeline.daily import drop_incomplete_bar

    df = pd.DataFrame({"Close": [1.0, 2.0]}, index=pd.to_datetime(["2026-10-19", "2026-10-20"]))
    intraday = pd.Timestamp("2026-10-20 11:00", tz="America/New_York")
    after = pd.Timestamp("2026-10-20 17:00", tz="America/New_York")
    assert len(drop_incomplete_bar(df, intraday)) == 1
    assert len(drop_incomplete_bar(df, after)) == 2


def test_har_handles_constant_sentiment_column():
    rng = np.random.default_rng(1)
    dates = pd.bdate_range("2010-01-01", periods=500)
    df = pd.DataFrame({"a": rng.normal(size=500), "sent": 0.0},
                      index=pd.MultiIndex.from_product([dates, ["T"]], names=["date", "ticker"]))
    df["y"] = 0.5 * df.a + rng.normal(0, 0.1, 500)
    beta, _ = fit_har(df, dates, ["a", "sent"])["T"]
    assert beta[2] == 0.0 and abs(beta[1] - 0.5) < 0.05


def test_har_pooled_fallback_for_ticker_without_fit_rows():
    """Ticker covered in the test year but not in the fit window (e.g. MSFT for HAR+sentiment)."""
    rng = np.random.default_rng(2)
    dates = pd.bdate_range("2015-01-01", periods=600)
    idx = pd.MultiIndex.from_product([dates, ["A", "B"]], names=["date", "ticker"])
    df = pd.DataFrame({"a": rng.normal(size=len(idx))}, index=idx)
    df["y"] = 0.5 * df.a + rng.normal(0, 0.1, len(idx))
    df.loc[(df.index.get_level_values("ticker") == "B") & (df.index.get_level_values("date") < dates[500]), "a"] = np.nan
    m = fit_har(df, dates[:500], ["a"], min_rows=250)
    assert "A" in m and "B" not in m and "__all__" in m
    p = predict_har(df, m, dates[500:], ["a"])
    assert set(p.ticker) == {"A", "B"}
    assert p[p.ticker == "B"].pooled_fallback.all() and not p[p.ticker == "A"].pooled_fallback.any()
