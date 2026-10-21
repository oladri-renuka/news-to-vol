"""Streamlit dashboard: forecast vs realized vol, rolling error vs HAR, and the day's most negative headlines.

    streamlit run dashboard/app.py
"""
from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.config import load_config, resolve  # noqa: E402
from src.metrics import qlike_vec  # noqa: E402

st.set_page_config(page_title="Volatility forecasts", layout="wide")
CFG = load_config("configs/eval.yaml")
# Show price-only (tier 1) and sentiment (tier 2) models together.
CFG["eval"]["models"].update(load_config("configs/eval_sentiment.yaml")["eval"]["models"])
ANN = np.sqrt(252 / 5)


@st.cache_data
def load_backtest() -> pd.DataFrame:
    root = resolve(CFG["paths"]["artifacts"]) / "predictions"
    frames = []
    for label, rel in CFG["eval"]["models"].items():
        p = root / rel
        if p.exists():
            df = pd.read_parquet(p)
            df = df.groupby(["date", "ticker"], as_index=False).agg(
                y=("y", "first"), rv2_true=("rv2_true", "first"), y_pred=("y_pred", "mean"), var_pred=("var_pred", "mean"))
            frames.append(df.assign(model=label))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


@st.cache_data
def load_store() -> tuple[pd.DataFrame, pd.DataFrame]:
    path = resolve("artifacts/store/forecasts.sqlite")
    if not path.exists():
        return pd.DataFrame(), pd.DataFrame()
    con = sqlite3.connect(path)
    fc = pd.read_sql("SELECT * FROM forecasts", con)
    hl = pd.read_sql("SELECT * FROM headlines", con)
    con.close()
    return fc, hl


@st.cache_data
def load_headlines() -> pd.DataFrame:
    p = resolve(CFG["paths"]["interim"]) / "headlines_scored.parquet"
    return pd.read_parquet(p, columns=["trading_day", "ticker", "headline", "p_neg", "p_pos"]) if p.exists() else pd.DataFrame()


bt = load_backtest()
fc, live_hl = load_store()
st.title("Next-week volatility: deep models vs HAR-RV")
if bt.empty:
    st.warning("No backtest predictions found. Run `make baselines` first.")
    st.stop()

tickers = sorted(bt.ticker.unique())
c1, c2, c3 = st.columns([1, 2, 1])
ticker = c1.selectbox("Ticker", tickers, index=tickers.index("AAPL") if "AAPL" in tickers else 0)
models = sorted(bt.model.unique())
default = [m for m in ["HAR", "LSTM-price", "LSTM-sent", "GARCH"] if m in models]
chosen = c2.multiselect("Models", models, default=default or models[:3])
window = c3.slider("Rolling window (days)", 21, 252, 63)

# --- latest live forecasts
if not fc.empty:
    last = fc[fc["asof"] == fc["asof"].max()]
    st.subheader(f"Latest forecasts (as of {last['asof'].iloc[0]}), annualized 5-day vol")
    st.dataframe(last.pivot(index="ticker", columns="model", values="vol_ann").style.format("{:.1%}"),
                 width="stretch", height=240)

# --- forecast vs realized
g = bt[(bt.ticker == ticker) & bt.model.isin(chosen)]
fig = go.Figure()
real = g.drop_duplicates("date").sort_values("date")
fig.add_trace(go.Scatter(x=real.date, y=np.exp(real.y) * ANN, name="realized", line=dict(color="gray", width=1)))
for m in chosen:
    s = g[g.model == m].sort_values("date")
    fig.add_trace(go.Scatter(x=s.date, y=np.exp(s.y_pred) * ANN, name=m, line=dict(width=1.3)))
fig.update_layout(title=f"{ticker}: forecast vs realized (annualized)", yaxis_tickformat=".0%", height=380,
                  margin=dict(l=10, r=10, t=40, b=10), legend=dict(orientation="h"))
st.plotly_chart(fig, width="stretch")

# --- rolling error vs HAR
if "HAR" in models:
    har = bt[(bt.ticker == ticker) & (bt.model == "HAR")].set_index("date").sort_index()
    har_ql = pd.Series(qlike_vec(har.rv2_true, har.var_pred), index=har.index).rolling(window).mean()
    fig2 = go.Figure()
    for m in chosen:
        if m == "HAR":
            continue
        s = bt[(bt.ticker == ticker) & (bt.model == m)].set_index("date").sort_index()
        ql = pd.Series(qlike_vec(s.rv2_true, s.var_pred), index=s.index).rolling(window).mean()
        fig2.add_trace(go.Scatter(x=ql.index, y=(ql / har_ql.reindex(ql.index)), name=m))
    fig2.add_hline(y=1.0, line_dash="dash")
    fig2.update_layout(title=f"Rolling {window}d QLIKE relative to HAR (below 1 = better)", height=320,
                       margin=dict(l=10, r=10, t=40, b=10), legend=dict(orientation="h"))
    st.plotly_chart(fig2, width="stretch")

# --- error table
tab = bt[bt.ticker == ticker].assign(se=lambda d: (d.y - d.y_pred) ** 2,
                                     ql=lambda d: qlike_vec(d.rv2_true, d.var_pred))
tab = tab.assign(year=tab.date.dt.year).groupby(["model", "year"]).ql.mean().unstack()
st.subheader(f"{ticker}: QLIKE by test year")
st.dataframe(tab.style.format("{:.3f}").highlight_min(axis=0, color="#cfe8cf"), width="stretch")

# --- most negative headlines
st.subheader("Most negative headlines")
hl = live_hl if not live_hl.empty else load_headlines()
if hl.empty:
    st.info("No scored headlines yet. Run `make sentiment` on the GPU box, or drop live headlines into data/live/headlines.csv.")
else:
    hl = hl.assign(trading_day=pd.to_datetime(hl.trading_day))
    days = sorted(hl.trading_day.dt.date.unique())
    day = st.select_slider("Trading day", options=days, value=days[-1])
    scope = st.radio("Scope", ["selected ticker", "all tickers"], horizontal=True)
    d = hl[hl.trading_day.dt.date == day]
    if scope == "selected ticker":
        d = d[d.ticker == ticker]
    st.dataframe(d.sort_values("p_neg", ascending=False).head(15)[["ticker", "headline", "p_neg", "p_pos"]],
                 width="stretch", hide_index=True)
