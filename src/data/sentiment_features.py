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
"""
from __future__ import annotations

import numpy as np
import pandas as pd

SENT_COLS = ["sent_mean", "log_news_count", "max_neg", "has_news", "mkt_sent_mean", "mkt_log_news_count"]


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
                             decay: float = 0.5) -> pd.DataFrame:
    """`scored` needs columns: ticker, trading_day, p_pos, p_neg. Returns a (date, ticker) panel."""
    calendar = pd.DatetimeIndex(calendar)
    scored = scored.dropna(subset=["trading_day", "p_pos", "p_neg"])
    per = _daily_raw(scored, ["ticker", "trading_day"])
    mkt = _decay(_daily_raw(scored, ["trading_day"]), calendar, decay)
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
    return out.reset_index().set_index(["date", "ticker"]).sort_index()


def empty_sentiment(calendar: pd.DatetimeIndex, tickers: list[str]) -> pd.DataFrame:
    idx = pd.MultiIndex.from_product([pd.DatetimeIndex(calendar), tickers], names=["date", "ticker"])
    return pd.DataFrame(0.0, index=idx, columns=SENT_COLS)
