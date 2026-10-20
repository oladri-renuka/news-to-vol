"""Daily batch pipeline: latest prices -> features -> next-week vol forecasts -> SQLite store.

    python -m src.pipeline.daily --config configs/pipeline.yaml
    python -m src.pipeline.daily --config configs/pipeline.yaml --asof 2023-12-29 --offline   # cached prices only

Live headlines after 2023 need a separate news source, because FNSPID ends in 2023. If
`live_headlines_csv` exists (columns: published_at_utc, ticker, headline), the pipeline scores those
headlines and stores them for the dashboard. The deployed models use price features only.
"""
from __future__ import annotations

import argparse
import logging
import sqlite3
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import torch

from src.config import load_config, resolve
from src.data.dataset import WindowedPanel, panel_meta
from src.data.features import build_price_panel
from src.data.prices import download_ticker, load_prices
from src.models.baselines import fit_har, predict_ewma, predict_har
from src.models.deep import build_model
from src.utils import setup_logging

log = logging.getLogger("volsent.daily")

SCHEMA = """
CREATE TABLE IF NOT EXISTS forecasts (
    asof TEXT, ticker TEXT, model TEXT, horizon INTEGER,
    log_vol REAL, vol_5d REAL, vol_ann REAL, created_at TEXT,
    PRIMARY KEY (asof, ticker, model)
);
CREATE TABLE IF NOT EXISTS realized (
    date TEXT, ticker TEXT, rv5 REAL, PRIMARY KEY (date, ticker)
);
CREATE TABLE IF NOT EXISTS headlines (
    trading_day TEXT, ticker TEXT, headline TEXT, p_neg REAL, p_pos REAL,
    PRIMARY KEY (trading_day, ticker, headline)
);
"""


def connect(cfg: dict) -> sqlite3.Connection:
    path = resolve(cfg["pipeline"]["store"])
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path)
    con.executescript(SCHEMA)
    return con


def fetch_live_prices(cfg: dict, tickers: list[str], offline: bool) -> dict[str, pd.DataFrame]:
    start = (pd.Timestamp.today() - pd.DateOffset(years=cfg["pipeline"]["history_years"])).strftime("%Y-%m-%d")
    out = {}
    for t in tickers:
        df = load_prices(cfg, t) if offline else download_ticker(t, start, None)
        if df.empty:
            log.warning("no prices for %s, falling back to the cache", t)
            df = load_prices(cfg, t)
        out[t] = drop_incomplete_bar(df) if not offline else df
    return out


def drop_incomplete_bar(df: pd.DataFrame, now: pd.Timestamp | None = None, settle: str = "16:30") -> pd.DataFrame:
    """Before the close has settled, yfinance returns a partial bar for today. Drop it."""
    now = now or pd.Timestamp.now(tz="America/New_York")
    today = now.tz_localize(None).normalize()
    if len(df) and df.index.max() == today and now.strftime("%H:%M") < settle:
        return df.iloc[:-1]
    return df


def load_deep(path) -> tuple[torch.nn.Module, dict]:
    state = torch.load(path, map_location="cpu", weights_only=False)  # our own artifact
    model = build_model(state["model_cfg"], len(state["feature_cols"]), len(state["tickers"]))
    model.load_state_dict(state["model_state"])
    model.eval()
    return model, state


def deep_forecast(model, state: dict, panel: pd.DataFrame, asof: pd.Timestamp) -> pd.DataFrame:
    wp = WindowedPanel(panel, state["feature_cols"], state["window"])
    t = wp.date_pos[asof]
    rows = []
    for i, tk in enumerate(wp.tickers):
        if tk not in state["tickers"] or not wp.valid_window[i, t]:
            continue
        x = (wp.X[i, t - state["window"] + 1:t + 1] - state["x_mu"]) / state["x_sd"]
        xt = torch.as_tensor(np.nan_to_num(x, nan=0.0)[None], dtype=torch.float32)
        tid = torch.tensor([state["tickers"].index(tk)])
        with torch.inference_mode():
            y = model(xt, tid).item() * state["y_sd"] + state["y_mu"]
        rows.append({"ticker": tk, "log_vol": y})
    return pd.DataFrame(rows)


def run(cfg: dict, asof: str | None, offline: bool) -> pd.DataFrame:
    pcfg = cfg["pipeline"]
    ckpt = resolve(pcfg["deep_checkpoint"])
    try:
        tickers = panel_meta(cfg)["tickers"]
    except FileNotFoundError:  # e.g. in CI, where only the checkpoint is shipped
        tickers = torch.load(ckpt, map_location="cpu", weights_only=False)["tickers"]
    prices = fetch_live_prices(cfg, tickers, offline)
    vix = fetch_live_prices(cfg, ["^VIX"], offline)["^VIX"]
    gspc = fetch_live_prices(cfg, ["^GSPC"], offline)["^GSPC"]
    calendar = gspc.index
    if asof:
        calendar = calendar[calendar <= pd.Timestamp(asof)]
    asof_ts = calendar.max()
    panel = build_price_panel({t: p.loc[:asof_ts] for t, p in prices.items()}, vix.loc[:asof_ts], calendar,
                              cfg["target"]["horizon"], cfg["target"]["variance_floor"])
    h = cfg["target"]["horizon"]
    log.info("forecast origin %s, %d tickers", asof_ts.date(), len(tickers))

    out = []
    # HAR: refit on the trailing window, using only rows whose 5-day target is already realized.
    fit_dates = calendar[(calendar >= asof_ts - pd.DateOffset(years=pcfg["har_fit_years"]))][:-h]
    har_cols = cfg["features"]["har"]
    har = predict_har(panel, fit_har(panel, fit_dates, har_cols), pd.DatetimeIndex([asof_ts]), har_cols)
    out.append(har.assign(model="HAR", log_vol=har.y_pred)[["ticker", "model", "log_vol"]])
    ew = predict_ewma(panel, pd.DatetimeIndex([asof_ts]), cfg["baselines"]["ewma_lambda"], h)
    out.append(ew.assign(model="EWMA", log_vol=ew.y_pred)[["ticker", "model", "log_vol"]])
    if ckpt.exists():
        model, state = load_deep(ckpt)
        if set(state["feature_cols"]) - set(panel.columns):
            log.warning("checkpoint needs features the live panel lacks (sentiment?); skipping the deep model")
        else:
            d = deep_forecast(model, state, panel, asof_ts)
            out.append(d.assign(model=pcfg["deep_label"]))
    else:
        log.warning("no deep checkpoint at %s; run V2 training first", ckpt)

    fc = pd.concat(out, ignore_index=True)
    fc["asof"] = asof_ts.strftime("%Y-%m-%d")
    fc["horizon"] = h
    fc["vol_5d"] = np.exp(fc.log_vol)
    fc["vol_ann"] = fc.vol_5d * np.sqrt(252 / h)
    fc["created_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")

    con = connect(cfg)
    con.executemany("INSERT OR REPLACE INTO forecasts VALUES (?,?,?,?,?,?,?,?)",
                    fc[["asof", "ticker", "model", "horizon", "log_vol", "vol_5d", "vol_ann", "created_at"]]
                    .itertuples(index=False, name=None))
    real = panel["rv5"].dropna().reset_index()
    real = real[real.date >= asof_ts - pd.DateOffset(days=pcfg["realized_backfill_days"])]
    con.executemany("INSERT OR REPLACE INTO realized VALUES (?,?,?)",
                    [(d.strftime("%Y-%m-%d"), t, float(v)) for d, t, v in real.itertuples(index=False, name=None)])
    _score_live_headlines(cfg, con, calendar)
    con.commit(); con.close()
    log.info("wrote %d forecasts to %s", len(fc), resolve(pcfg["store"]))
    return fc


def _score_live_headlines(cfg: dict, con: sqlite3.Connection, calendar: pd.DatetimeIndex) -> None:
    path = resolve(cfg["pipeline"].get("live_headlines_csv", "data/live/headlines.csv"))
    if not path.exists():
        return
    from src.data.news import align_to_trading_day
    from src.models.sentiment import score_headlines

    raw = pd.read_csv(path)
    al = align_to_trading_day(pd.to_datetime(raw.published_at_utc), calendar, "UTC",
                              cfg["news"]["market_tz"], cfg["news"]["cutoff"], cfg["news"]["date_only_policy"])
    df = pd.concat([raw, al], axis=1).dropna(subset=["trading_day"])
    scored = score_headlines(cfg, df[["ticker", "headline", "trading_day"]],
                             resolve(cfg["paths"]["interim"]) / "live_headline_scores.parquet")
    con.executemany("INSERT OR REPLACE INTO headlines VALUES (?,?,?,?,?)",
                    [(d.strftime("%Y-%m-%d"), t, h, float(n), float(p)) for t, h, d, n, p in
                     scored[["ticker", "headline", "trading_day", "p_neg", "p_pos"]].itertuples(index=False, name=None)])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/pipeline.yaml")
    ap.add_argument("--asof", default=None, help="forecast origin (default: latest trading day)")
    ap.add_argument("--offline", action="store_true", help="use cached study prices instead of downloading")
    args = ap.parse_args()
    setup_logging()
    fc = run(load_config(args.config), args.asof, args.offline)
    print(fc.pivot(index="ticker", columns="model", values="vol_ann").round(3).to_string())


if __name__ == "__main__":
    main()
