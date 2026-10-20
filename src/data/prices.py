"""Download daily OHLCV from yfinance once and cache it as Parquet (one file per ticker).

    python -m src.data.prices --config configs/base.yaml [--refresh]
"""
from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import pandas as pd
import yaml

from src.config import load_config, resolve
from src.utils import setup_logging

log = logging.getLogger(__name__)

OHLCV = ["Open", "High", "Low", "Close", "Volume"]


def _cache_path(cfg: dict, ticker: str) -> Path:
    safe = ticker.replace("^", "IDX_")
    return resolve(cfg["paths"]["raw"]) / "prices" / f"{safe}.parquet"


def download_ticker(ticker: str, start: str, end: str | None, retries: int = 3) -> pd.DataFrame:
    import yfinance as yf

    for attempt in range(retries):
        try:
            # auto_adjust=True adjusts OHLC consistently for splits and dividends.
            # Without it, a 7:1 split shows up as a -195% log return.
            df = yf.Ticker(ticker).history(start=start, end=end, auto_adjust=True, actions=False)
            if len(df):
                df = df[OHLCV].copy()
                df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
                df.index.name = "date"
                return df[~df.index.duplicated(keep="last")]
        except Exception as e:  # network hiccups
            log.warning("%s attempt %d failed: %s", ticker, attempt + 1, e)
        time.sleep(2 * (attempt + 1))
    return pd.DataFrame(columns=OHLCV)


def load_prices(cfg: dict, ticker: str, refresh: bool = False, end: str | None = None) -> pd.DataFrame:
    path = _cache_path(cfg, ticker)
    if path.exists() and not refresh:
        return pd.read_parquet(path)
    if end is None:
        # yfinance's end is exclusive.
        end = (pd.Timestamp(cfg["study"]["end"]) + pd.DateOffset(days=1)).strftime("%Y-%m-%d")
    df = download_ticker(ticker, cfg["study"]["download_start"], end)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path)
    return df


def all_tickers(cfg: dict) -> tuple[list[str], dict]:
    with open(resolve(cfg["universe_config"])) as f:
        ucfg = yaml.safe_load(f)
    mk = ucfg["market_tickers"]
    return ucfg["ranked_candidates"] + [mk["index"], mk["vix"]], ucfg


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args()
    setup_logging()
    cfg = load_config(args.config)
    tickers, _ = all_tickers(cfg)
    for t in tickers:
        df = load_prices(cfg, t, refresh=args.refresh)
        rng = f"{df.index.min().date()}..{df.index.max().date()}" if len(df) else "EMPTY"
        log.info("%-6s %5d rows %s", t, len(df), rng)


if __name__ == "__main__":
    main()
