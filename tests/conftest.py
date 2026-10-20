import numpy as np
import pandas as pd
import pytest


def make_ohlcv(n: int = 400, seed: int = 0, start: str = "2015-01-01") -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start, periods=n)
    # GARCH-ish vol clustering so the volatility models have something to fit.
    sig = np.empty(n); r = np.empty(n); s2 = 1e-4
    for t in range(n):
        s2 = 2e-6 + 0.08 * (r[t - 1] ** 2 if t else 1e-4) + 0.9 * s2
        sig[t] = np.sqrt(s2); r[t] = sig[t] * rng.standard_t(6) / np.sqrt(1.5)
    close = 100 * np.exp(np.cumsum(r))
    open_ = close * np.exp(rng.normal(0, 0.003, n))
    high = np.maximum(open_, close) * np.exp(np.abs(rng.normal(0, sig)))
    low = np.minimum(open_, close) * np.exp(-np.abs(rng.normal(0, sig)))
    vol = rng.lognormal(15, 0.3, n)
    return pd.DataFrame({"Open": open_, "High": high, "Low": low, "Close": close, "Volume": vol}, index=idx)


@pytest.fixture
def ohlcv():
    return make_ohlcv()


@pytest.fixture
def vix(ohlcv):
    rng = np.random.default_rng(1)
    return pd.Series(15 + np.cumsum(rng.normal(0, 0.5, len(ohlcv))).clip(-5, 40), index=ohlcv.index)


@pytest.fixture
def small_panel():
    from src.data.features import build_price_panel

    prices = {t: make_ohlcv(900, seed=i, start="2009-01-01") for i, t in enumerate(["AAA", "BBB", "CCC"])}
    cal = prices["AAA"].index
    vix = pd.DataFrame({"Close": 18 + np.sin(np.arange(len(cal)) / 30) * 5}, index=cal)
    panel = build_price_panel(prices, vix, cal)
    for c in ["sent_mean", "log_news_count", "max_neg", "has_news", "mkt_sent_mean", "mkt_log_news_count"]:
        panel[c] = 0.0
    return panel
