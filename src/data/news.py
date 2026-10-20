"""FNSPID headlines: filter to the universe and align to trading days with no look-ahead.

    python -m src.data.news --config configs/base.yaml            # filter + align + coverage
    python -m src.data.news --config configs/base.yaml --audit    # check the timestamp timezone

Alignment rule: a forecast made at the close of trading day t may use only
headlines published before 16:00 ET on day t. Headlines at or after the cutoff,
or on non-trading days, move to the next trading day. A row stamped exactly
00:00:00 has no reliable time, so under the conservative policy it is assumed
to come after the close.
"""
from __future__ import annotations

import argparse
import logging
import re

import numpy as np
import pandas as pd
import yaml

from src.config import load_config, resolve
from src.utils import setup_logging

log = logging.getLogger(__name__)


def universe_symbols(cfg: dict) -> tuple[dict[str, str], list[str]]:
    with open(resolve(cfg["universe_config"])) as f:
        u = yaml.safe_load(f)
    alias = {t: t for t in u["ranked_candidates"]}
    for canon, syms in (u.get("news_aliases") or {}).items():
        for s in syms:
            alias[s] = canon
    return alias, u["ranked_candidates"]


def read_fnspid(cfg: dict, chunksize: int = 500_000) -> pd.DataFrame:
    """Stream the large CSV and keep only universe tickers in the study window."""
    alias, _ = universe_symbols(cfg)
    paths = cfg["news"]["fnspid_csvs"]
    keep = []
    lo = pd.Timestamp(cfg["study"]["start"]) - pd.DateOffset(days=10)
    hi = pd.Timestamp(cfg["study"]["end"]) + pd.DateOffset(days=1)
    n_total = 0
    for path in map(resolve, paths):
        if not path.exists():
            log.warning("missing %s, skipping", path)
            continue
        log.info("reading %s", path)
        reader = pd.read_csv(path, usecols=["Date", "Article_title", "Stock_symbol"],
                             chunksize=chunksize, dtype=str, on_bad_lines="skip")
        for chunk in reader:
            n_total += len(chunk)
            chunk = chunk[chunk["Stock_symbol"].isin(alias.keys())].dropna(subset=["Date", "Article_title"])
            if chunk.empty:
                continue
            ts = pd.to_datetime(chunk["Date"].str.replace(" UTC", "", regex=False), errors="coerce", format="mixed")
            chunk = chunk.assign(ts_raw=ts).dropna(subset=["ts_raw"])
            chunk = chunk[(chunk.ts_raw >= lo) & (chunk.ts_raw < hi)]
            chunk["ticker"] = chunk["Stock_symbol"].map(alias)
            keep.append(chunk[["ts_raw", "ticker", "Article_title"]].rename(columns={"Article_title": "headline"}))
    log.info("scanned %d FNSPID rows", n_total)
    df = pd.concat(keep, ignore_index=True)
    df["headline"] = df["headline"].str.strip()
    df = df.drop_duplicates(subset=["ticker", "ts_raw", "headline"])
    return df.sort_values("ts_raw").reset_index(drop=True)


def align_to_trading_day(ts_raw: pd.Series, calendar: pd.DatetimeIndex, src_tz: str = "UTC",
                         market_tz: str = "America/New_York", cutoff: str = "16:00",
                         date_only_policy: str = "next_day") -> pd.DataFrame:
    """Map raw timestamps to the trading day whose close may first use them.

    Returns columns: ts_et, date_only, trading_day (NaT if beyond the calendar).
    """
    ts_raw = pd.to_datetime(ts_raw)
    date_only = (ts_raw.dt.hour == 0) & (ts_raw.dt.minute == 0) & (ts_raw.dt.second == 0)
    ts_et = ts_raw.dt.tz_localize(src_tz).dt.tz_convert(market_tz)
    ch, cm = (int(x) for x in cutoff.split(":"))

    local_date = ts_et.dt.tz_localize(None).dt.normalize()
    after_close = (ts_et.dt.hour > ch) | ((ts_et.dt.hour == ch) & (ts_et.dt.minute >= cm))
    if date_only_policy == "next_day":
        # The stamp is a calendar date in the source's own convention, so take the raw
        # date and assume "after the close".
        local_date = local_date.where(~date_only, ts_raw.dt.normalize())
        after_close = after_close | date_only
    elif date_only_policy == "same_day":
        local_date = local_date.where(~date_only, ts_raw.dt.normalize())
        after_close = after_close & ~date_only
    else:
        raise ValueError(date_only_policy)

    cal = pd.DatetimeIndex(calendar).sort_values()
    # First trading day >= local_date, or > local_date if the headline is after the close.
    pos = np.where(after_close.to_numpy(),
                   cal.searchsorted(local_date.to_numpy(), side="right"),
                   cal.searchsorted(local_date.to_numpy(), side="left"))
    td = pd.Series(pd.NaT, index=ts_raw.index, dtype="datetime64[ns]")
    ok = pos < len(cal)
    td[ok] = cal[pos[ok]]
    return pd.DataFrame({"ts_et": ts_et, "date_only": date_only, "trading_day": td})


INTRADAY_PATTERNS = {
    "pre_market": re.compile(r"pre-?market", re.I),     # true time ~ 04:00-09:30 ET
    "mid_day": re.compile(r"mid-?day", re.I),           # true time ~ 11:00-14:00 ET
    "after_hours": re.compile(r"after-?hours", re.I),   # true time ~ 16:00-20:00 ET (or next morning)
}


def audit_timezone(df: pd.DataFrame, candidates=("UTC", "America/New_York")) -> pd.DataFrame:
    """Median ET hour of session-specific headlines under each source-tz hypothesis.

    If FNSPID really is UTC, "Mid-Day" headlines should land around 11:00-14:00 ET.
    """
    timed = df[~((df.ts_raw.dt.hour == 0) & (df.ts_raw.dt.minute == 0) & (df.ts_raw.dt.second == 0))]
    rows = []
    for name, pat in INTRADAY_PATTERNS.items():
        sub = timed[timed.headline.str.contains(pat)]
        for tz in candidates:
            h = sub.ts_raw.dt.tz_localize(tz).dt.tz_convert("America/New_York")
            hours = h.dt.hour + h.dt.minute / 60
            rows.append({"pattern": name, "assumed_tz": tz, "n": len(sub),
                         "median_hour_et": hours.median(), "p10": hours.quantile(.1), "p90": hours.quantile(.9)})
    return pd.DataFrame(rows)


def coverage_table(news: pd.DataFrame) -> pd.DataFrame:
    news = news.dropna(subset=["trading_day"])
    return (news.assign(year=news.trading_day.dt.year)
            .groupby(["ticker", "year"]).size().unstack(fill_value=0))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--audit", action="store_true")
    ap.add_argument("--set", nargs="*", default=[])
    args = ap.parse_args()
    setup_logging()
    cfg = load_config(args.config, args.set)
    interim = resolve(cfg["paths"]["interim"])
    interim.mkdir(parents=True, exist_ok=True)
    raw_path = interim / "headlines_raw.parquet"

    if raw_path.exists():
        df = pd.read_parquet(raw_path)
    else:
        df = read_fnspid(cfg)
        df.to_parquet(raw_path)
    log.info("%d universe headlines, %s..%s", len(df), df.ts_raw.min(), df.ts_raw.max())

    if args.audit:
        frac_date_only = ((df.ts_raw.dt.hour == 0) & (df.ts_raw.dt.minute == 0)).mean()
        print(f"share of rows stamped 00:00:00 (date only): {frac_date_only:.1%}")
        print(audit_timezone(df).to_string(index=False))
        return

    from src.data.prices import load_prices
    cal = load_prices(cfg, "^GSPC").index
    ncfg = cfg["news"]
    al = align_to_trading_day(df.ts_raw, cal, ncfg["timestamp_tz"], ncfg["market_tz"],
                              ncfg["cutoff"], ncfg["date_only_policy"])
    df = pd.concat([df, al], axis=1).dropna(subset=["trading_day"])
    lo, hi = pd.Timestamp(cfg["study"]["start"]), pd.Timestamp(cfg["study"]["end"])
    df = df[(df.trading_day >= lo) & (df.trading_day <= hi)].reset_index(drop=True)
    df["ts_et"] = df["ts_et"].astype(str)
    df.to_parquet(interim / "headlines_aligned.parquet")
    cov = coverage_table(df)
    reports = resolve(cfg["paths"]["reports"]) / "tables"
    reports.mkdir(parents=True, exist_ok=True)
    cov.to_csv(reports / "news_coverage.csv")
    print(cov.to_string())


if __name__ == "__main__":
    main()
