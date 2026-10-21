import numpy as np
import pandas as pd

from src.data.sentiment_features import build_sentiment_features

CAL = pd.bdate_range("2021-01-04", periods=5)


def _scored(rows):
    return pd.DataFrame(rows, columns=["ticker", "trading_day", "p_pos", "p_neg"]).assign(
        trading_day=lambda d: pd.to_datetime(d.trading_day))


def test_decay_and_counts():
    s = _scored([("A", "2021-01-04", 0.9, 0.05), ("A", "2021-01-04", 0.7, 0.1), ("A", "2021-01-06", 0.0, 0.8)])
    f = build_sentiment_features(s, CAL, ["A", "B"], decay=0.5).xs("A", level="ticker")
    # Day 0: two headlines, mean score ((0.85)+(0.6))/2
    assert np.isclose(f.sent_mean.iloc[0], 0.725)
    assert f.has_news.tolist() == [1, 0, 1, 0, 0]
    # Day 1: no news, but the decayed sentiment carries over (S and N both halve, so the mean is the same)
    assert np.isclose(f.sent_mean.iloc[1], 0.725)
    assert np.isclose(f.log_news_count.iloc[1], np.log1p(1.0))
    # Day 2: S = -0.8 + 0.25*1.45, N = 1 + 0.5
    assert np.isclose(f.sent_mean.iloc[2], (-0.8 + 0.25 * 1.45) / 1.5)
    assert np.isclose(f.max_neg.iloc[2], 0.8) and np.isclose(f.max_neg.iloc[3], 0.4)


def test_ticker_without_news_is_zero_but_has_market():
    s = _scored([("A", "2021-01-04", 0.9, 0.05)])
    f = build_sentiment_features(s, CAL, ["A", "B"], decay=0.5).xs("B", level="ticker")
    assert (f.has_news == 0).all() and (f.sent_mean == 0).all()
    assert np.isclose(f.mkt_sent_mean.iloc[0], 0.85)


def test_no_future_leakage():
    s = _scored([("A", "2021-01-04", 0.9, 0.05), ("A", "2021-01-07", 0.0, 0.9)])
    f = build_sentiment_features(s, CAL, ["A"], decay=0.5).xs("A", level="ticker")
    s2 = _scored([("A", "2021-01-04", 0.9, 0.05)])
    f2 = build_sentiment_features(s2, CAL, ["A"], decay=0.5).xs("A", level="ticker")
    pd.testing.assert_frame_equal(f.iloc[:3], f2.iloc[:3])


# ---- two-tier coverage design
def test_coverage_set_threshold():
    from src.data.sentiment_features import covered_ticker_years

    rows = [("A", "2021-01-04", 0.5, 0.1)] * 3 + [("B", "2021-01-04", 0.5, 0.1)]
    cov = covered_ticker_years(_scored(rows), min_per_year=2)
    assert cov == {("A", 2021)}


def test_gap_years_are_nan_not_zero():
    """A dataset gap (uncovered ticker-year) must never read as has_news = 0."""
    s = _scored([("A", "2021-01-04", 0.9, 0.05)])
    f = build_sentiment_features(s, CAL, ["A", "B"], decay=0.5, covered={("A", 2021)})
    a, b = f.xs("A", level="ticker"), f.xs("B", level="ticker")
    assert (a.news_covered == 1).all() and (b.news_covered == 0).all()
    # Covered year: quiet days are a genuine has_news = 0
    assert a.has_news.tolist() == [1, 0, 0, 0, 0]
    # Uncovered year: every sentiment feature is unknown, including has_news and the market aggregates
    for c in ["sent_mean", "log_news_count", "max_neg", "has_news", "mkt_sent_mean", "mkt_log_news_count"]:
        assert b[c].isna().all(), c


def test_market_aggregate_ignores_uncovered_tickers():
    s = _scored([("A", "2021-01-04", 0.9, 0.05), ("B", "2021-01-04", 0.0, 0.9)])
    f = build_sentiment_features(s, CAL, ["A", "B"], decay=0.5, covered={("A", 2021)})
    assert np.isclose(f.xs("A", level="ticker").mkt_sent_mean.iloc[0], 0.85)  # B's headline excluded
