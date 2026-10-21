"""Aggregate headline-level sentiment into daily per-ticker and market-wide features.

Raw daily quantities for ticker i on trading day t, over headlines h assigned to t:
    score_sum = sum_h (p_pos - p_neg),   n = #headlines,   max_neg = max_h p_neg

Exponential decay, so yesterday's news still counts a little today (decay d in [0, 1)):
    S_t = score_sum_t + d * S_{t-1}
    N_t = n_t + d * N_{t-1}
    sent_mean_t = S_t / N_t            (0 when N_t is ~0)
    log_news_count_t = log1p(N_t)
    max_neg_t = max(max_neg_t, d * max_neg_{t-1})
    has_news_t = 1 if n_t > 0 else 0   (raw, not decayed)

The market-wide features use the same recursion on headlines pooled across all tickers.

Coverage (two-tier design): FNSPID has whole ticker-years with no headlines at all. Those are
dataset gaps, not quiet periods. A ticker-year is "covered" if it has at least
`min_headlines_per_year` headlines. In uncovered ticker-years every sentiment feature,
has_news included, is NaN (unknown), never 0, and `news_covered` = 0. has_news = 0 means
"no headline that day" only inside a covered year. Market-wide aggregates use covered
ticker-days only, so a dataset gap does not look like a market-wide lull.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

SENT_COLS = ["sent_mean", "log_news_count", "max_neg", "has_news", "mkt_sent_mean", "mkt_log_news_count"]
MASKED_COLS = SENT_COLS  # set to NaN in uncovered ticker-years
COVER_COL = "news_covered"


def covered_ticker_years(scored: pd.DataFrame, min_per_year: int) -> set[tuple[str, int]]:
    """{(ticker, year)} with at least `min_per_year` headlines, by trading-day year."""
    s = scored.dropna(subset=["trading_day"])
    cnt = s.groupby([s.ticker, s.trading_day.dt.year]).size()
    return {(t, int(y)) for (t, y), n in cnt.items() if n >= min_per_year}


def _daily_raw(scored: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    s = scored.assign(score=scored.p_pos - scored.p_neg)
    return s.groupby(by).agg(score_sum=("score", "sum"), n=("score", "size"), max_neg=("p_neg", "max"))


def _decay(raw: pd.DataFrame, calendar: pd.DatetimeIndex, d: float) -> pd.DataFrame:
    raw = raw.reindex(calendar).fillna({"score_sum": 0.0, "n": 0.0, "max_neg": 0.0})
    S = np.zeros(len(raw)); N = np.zeros(len(raw)); M = np.zeros(len(raw))
    ss, nn, mm = raw.score_sum.to_numpy(), raw.n.to_numpy(float), raw.max_neg.to_numpy()
    s_prev = n_prev = m_prev = 0.0
    for i in range(len(raw)):
        s_prev = ss[i] + d * s_prev
        n_prev = nn[i] + d * n_prev
        m_prev = max(mm[i], d * m_prev)
        S[i], N[i], M[i] = s_prev, n_prev, m_prev
    out = pd.DataFrame(index=calendar)
    out["sent_mean"] = np.where(N > 1e-6, S / np.maximum(N, 1e-6), 0.0)
    out["log_news_count"] = np.log1p(N)
    out["max_neg"] = M
    out["has_news"] = (nn > 0).astype(np.float32)
    return out


def build_sentiment_features(scored: pd.DataFrame, calendar: pd.DatetimeIndex, tickers: list[str],
                             decay: float = 0.5, covered: set[tuple[str, int]] | None = None) -> pd.DataFrame:
    """`scored` needs columns: ticker, trading_day, p_pos, p_neg. Returns a (date, ticker) panel.

    If `covered` is given, uncovered ticker-years are masked to NaN, and the market
    aggregates use covered ticker-days only. If it is None, every ticker-year counts as covered.
    """
    calendar = pd.DatetimeIndex(calendar)
    scored = scored.dropna(subset=["trading_day", "p_pos", "p_neg"])
    per = _daily_raw(scored, ["ticker", "trading_day"])
    if covered is not None:
        keys = list(zip(scored.ticker, scored.trading_day.dt.year))
        mkt_src = scored[[k in covered for k in keys]]
    else:
        mkt_src = scored
    mkt = _decay(_daily_raw(mkt_src, ["trading_day"]), calendar, decay)
    mkt = mkt.rename(columns={"sent_mean": "mkt_sent_mean", "log_news_count": "mkt_log_news_count"})[
        ["mkt_sent_mean", "mkt_log_news_count"]]
    frames = []
    for t in tickers:
        raw = per.xs(t, level="ticker") if t in per.index.get_level_values(0) else pd.DataFrame(
            columns=["score_sum", "n", "max_neg"], dtype=float)
        f = _decay(raw, calendar, decay).join(mkt)
        f["ticker"] = t
        frames.append(f)
    out = pd.concat(frames)
    out.index.name = "date"
    out = out.reset_index().set_index(["date", "ticker"]).sort_index()
    if covered is None:
        out[COVER_COL] = 1.0
    else:
        years = out.index.get_level_values("date").year
        tks = out.index.get_level_values("ticker")
        out[COVER_COL] = np.array([(t, int(y)) in covered for t, y in zip(tks, years)], dtype=np.float32)
        out.loc[out[COVER_COL] == 0, MASKED_COLS] = np.nan
    return out


def empty_sentiment(calendar: pd.DatetimeIndex, tickers: list[str]) -> pd.DataFrame:
    """No scored headlines: everything is unknown (NaN) and nothing is covered."""
    idx = pd.MultiIndex.from_product([pd.DatetimeIndex(calendar), tickers], names=["date", "ticker"])
    out = pd.DataFrame(np.nan, index=idx, columns=SENT_COLS)
    out[COVER_COL] = 0.0
    return out
