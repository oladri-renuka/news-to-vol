"""Panel assembly, universe selection, walk-forward splits and sequence windows.

    python -m src.data.dataset --config configs/base.yaml     # build and cache data/processed/panel.parquet
"""
from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
import yaml

from src.config import load_config, resolve
from src.data.features import build_price_panel
from src.data.prices import load_prices
from src.data.sentiment_features import (COVER_COL, SENT_COLS, build_sentiment_features, covered_ticker_years,
                                         empty_sentiment)
from src.utils import setup_logging

log = logging.getLogger(__name__)


# ---------------------------------------------------------------- universe
def select_universe(cfg: dict, coverage: pd.DataFrame | None) -> tuple[list[str], pd.DataFrame]:
    """Apply the rule in configs/universe.yaml. Returns the tickers and a decision log."""
    with open(resolve(cfg["universe_config"])) as f:
        u = yaml.safe_load(f)
    lo, hi = pd.Timestamp(cfg["study"]["start"]), pd.Timestamp(cfg["study"]["end"])
    cal = load_prices(cfg, u["market_tickers"]["index"]).loc[lo:hi].index
    years = range(lo.year, hi.year + 1)
    ncfg = cfg["news"]
    rows, chosen = [], []
    for t in u["ranked_candidates"]:
        px = load_prices(cfg, t).loc[lo:hi]
        missing = len(cal.difference(px.index))
        ok_price = len(px) > 0 and missing <= 5 and px.index.max() >= cal.max() - pd.DateOffset(days=7)
        if coverage is not None:
            cnt = coverage.reindex(index=[t], columns=list(years), fill_value=0).iloc[0] if t in coverage.index \
                else pd.Series(0, index=list(years))
            frac = float((cnt >= ncfg["min_headlines_per_year"]).mean())
            ok_news = frac >= ncfg.get("min_covered_year_fraction", 1.0)
        else:
            frac, ok_news = np.nan, True
        take = ok_price and ok_news and len(chosen) < u["n_tickers"]
        if take:
            chosen.append(t)
        rows.append({"ticker": t, "missing_days": missing, "ok_price": ok_price,
                     "covered_year_fraction": frac, "ok_news": ok_news, "selected": take})
    if len(chosen) < u["n_tickers"]:
        log.warning("only %d/%d candidates pass the rule. Loosen news.min_covered_year_fraction or extend "
                    "ranked_candidates, and record the change in the README.", len(chosen), u["n_tickers"])
    return chosen, pd.DataFrame(rows)


# ---------------------------------------------------------------- panel
def build_panel(cfg: dict, force: bool = False) -> pd.DataFrame:
    out = resolve(cfg["paths"]["processed"]) / "panel.parquet"
    if out.exists() and not force:
        return pd.read_parquet(out)
    with open(resolve(cfg["universe_config"])) as f:
        u = yaml.safe_load(f)
    interim = resolve(cfg["paths"]["interim"])
    cov_path = resolve(cfg["paths"]["reports"]) / "tables" / "news_coverage.csv"
    coverage = pd.read_csv(cov_path, index_col=0) if cov_path.exists() else None
    if coverage is not None:
        coverage.columns = coverage.columns.astype(int)
    # Two-tier design: the universe is chosen on price availability only. News coverage
    # does not drop tickers. It decides which ticker-years count in the sentiment comparison.
    use_cov = cfg["news"].get("universe_uses_coverage", False)
    tickers, decisions = select_universe(cfg, coverage if use_cov else None)
    decisions.to_csv(resolve(cfg["paths"]["reports"]) / "tables" / "universe_selection.csv", index=False)
    log.info("universe (%d): %s", len(tickers), tickers)

    gspc = load_prices(cfg, u["market_tickers"]["index"])
    calendar = gspc.loc[cfg["study"]["download_start"]:cfg["study"]["end"]].index
    vix = load_prices(cfg, u["market_tickers"]["vix"])
    prices = {t: load_prices(cfg, t) for t in tickers}
    tcfg = cfg["target"]
    panel = build_price_panel(prices, vix, calendar, tcfg["horizon"], tcfg["variance_floor"])

    scored_path = interim / "headlines_scored.parquet"
    if scored_path.exists():
        scored = pd.read_parquet(scored_path)
        covered = covered_ticker_years(scored, cfg["news"]["min_headlines_per_year"])
        sent = build_sentiment_features(scored, calendar, tickers, cfg["news"]["decay"], covered)
        sentiment_available = True
    else:
        log.warning("no scored headlines at %s; sentiment features are missing (NaN)", scored_path)
        sent = empty_sentiment(calendar, tickers)
        sentiment_available = False
    # Sentiment stays NaN in uncovered ticker-years. A dataset gap is not "no news".
    panel = panel.join(sent[SENT_COLS + [COVER_COL]], how="left")
    panel[COVER_COL] = panel[COVER_COL].fillna(0.0)
    cov_report = coverage_report(panel, cfg)
    cov_report.to_csv(resolve(cfg["paths"]["reports"]) / "tables" / "coverage_test_years.csv")
    log.info("news coverage of test ticker-days:\n%s", cov_report.to_string())
    out.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(out)
    with open(out.with_suffix(".json"), "w") as f:
        json.dump({"tickers": tickers, "sentiment_available": sentiment_available,
                   "n_rows": len(panel)}, f, indent=2)
    return panel


def coverage_report(panel: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """For each test year: the share of ticker-days with a realized target that are news-covered,
    and how many tickers are covered."""
    w = cfg["walk_forward"]
    p = panel[panel["y"].notna()]
    years = p.index.get_level_values("date").year
    rows = []
    for y in range(w["first_test_year"], w["last_test_year"] + 1):
        g = p[years == y]
        cov = g[COVER_COL] > 0
        tick = g[cov].index.get_level_values("ticker").unique()
        rows.append({"test_year": y, "ticker_days": len(g), "covered_ticker_days": int(cov.sum()),
                     "covered_fraction": round(float(cov.mean()), 4) if len(g) else np.nan,
                     "n_tickers_covered": len(tick), "n_tickers": g.index.get_level_values("ticker").nunique(),
                     "tickers_covered": " ".join(sorted(tick))})
    out = pd.DataFrame(rows).set_index("test_year")
    g = p[(years >= w["first_test_year"]) & (years <= w["last_test_year"])]
    out.loc["pooled"] = [len(g), int((g[COVER_COL] > 0).sum()), round(float((g[COVER_COL] > 0).mean()), 4),
                         np.nan, np.nan, ""]
    return out


def panel_meta(cfg: dict) -> dict:
    with open((resolve(cfg["paths"]["processed"]) / "panel.parquet").with_suffix(".json")) as f:
        return json.load(f)


# ---------------------------------------------------------------- walk-forward
@dataclass
class Fold:
    test_year: int
    train: tuple[pd.Timestamp, pd.Timestamp]
    val: tuple[pd.Timestamp, pd.Timestamp]
    test: tuple[pd.Timestamp, pd.Timestamp]


def walk_forward_folds(cfg: dict, years: list[int] | None = None) -> list[Fold]:
    w = cfg["walk_forward"]
    folds = []
    for y in range(w["first_test_year"], w["last_test_year"] + 1):
        if years and y not in years:
            continue
        folds.append(Fold(
            test_year=y,
            train=(pd.Timestamp(f"{w['train_start_year']}-01-01"), pd.Timestamp(f"{y - 2}-12-31")),
            val=(pd.Timestamp(f"{y - 1}-01-01"), pd.Timestamp(f"{y - 1}-12-31")),
            test=(pd.Timestamp(f"{y}-01-01"), pd.Timestamp(f"{y}-12-31")),
        ))
    return folds


def split_dates(calendar: pd.DatetimeIndex, start: pd.Timestamp, end: pd.Timestamp,
                purge: int = 0) -> pd.DatetimeIndex:
    """Trading days in [start, end], minus the last `purge` days.

    Those last days have targets that overlap the next window.
    """
    d = calendar[(calendar >= start) & (calendar <= end)]
    return d[:-purge] if purge else d


def fold_dates(cfg: dict, calendar: pd.DatetimeIndex, fold: Fold) -> dict[str, pd.DatetimeIndex]:
    h = cfg["target"]["horizon"] if cfg["walk_forward"].get("purge", True) else 0
    return {
        "train": split_dates(calendar, *fold.train, purge=h),
        "val": split_dates(calendar, *fold.val, purge=h),
        "train_val": split_dates(calendar, fold.train[0], fold.val[1], purge=h),
        "test": split_dates(calendar, *fold.test, purge=0),
    }


# ---------------------------------------------------------------- sequence windows
class WindowedPanel:
    """Dense [n_tickers, T, F] feature tensor plus index arrays for (ticker, t) samples.

    The window for sample (i, t) is X[i, t-W+1 : t+1]. It covers only rows dated
    on or before the forecast origin t.
    """

    def __init__(self, panel: pd.DataFrame, feature_cols: list[str], window: int,
                 required_cols: list[str] | None = None):
        """`required_cols` must have no NaN anywhere in the window (default: all features).

        Other features, i.e. sentiment in uncovered ticker-years, may be NaN. They are
        imputed after scaling, and `news_covered` tells the model when that happened.
        """
        self.feature_cols = list(feature_cols)
        self.window = window
        self.tickers = sorted(panel.index.get_level_values("ticker").unique())
        self.dates = pd.DatetimeIndex(sorted(panel.index.get_level_values("date").unique()))
        full = panel.reindex(pd.MultiIndex.from_product([self.dates, self.tickers], names=["date", "ticker"]))
        F, T, N = len(self.feature_cols), len(self.dates), len(self.tickers)
        X = full[self.feature_cols].to_numpy(np.float32).reshape(T, N, F).transpose(1, 0, 2)
        self.X = np.ascontiguousarray(X)                      # [N, T, F]
        self.y = full["y"].to_numpy(np.float32).reshape(T, N).T  # [N, T]
        req = [self.feature_cols.index(c) for c in (required_cols or self.feature_cols)]
        bad = np.isnan(self.X[:, :, req]).any(-1).astype(np.int32)  # [N, T]
        csum = np.concatenate([np.zeros((N, 1), np.int32), np.cumsum(bad, 1)], 1)
        # A window ending at t is valid if t >= W-1 and has no NaN in X[t-W+1..t].
        self.valid_window = np.zeros((N, T), bool)
        self.valid_window[:, window - 1:] = (csum[:, window:] - csum[:, :T - window + 1]) == 0
        self.date_pos = {d: i for i, d in enumerate(self.dates)}

    def sample_index(self, dates: pd.DatetimeIndex, require_target: bool = True) -> np.ndarray:
        """Return an [M, 2] array of (ticker_idx, t_idx) for valid samples on the given dates."""
        tpos = np.array([self.date_pos[d] for d in dates if d in self.date_pos], dtype=np.int64)
        ok = self.valid_window[:, tpos]
        if require_target:
            ok &= ~np.isnan(self.y[:, tpos])
        ti, tj = np.nonzero(ok)
        return np.stack([ti, tpos[tj]], 1)

    def fit_scaler(self, idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Mean/std from the feature rows at the training origins only."""
        rows = self.X[idx[:, 0], idx[:, 1]]
        mu = np.nanmean(rows, 0)
        sd = np.nanstd(rows, 0)
        sd[sd < 1e-8] = 1.0
        return mu.astype(np.float32), sd.astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--set", nargs="*", default=[])
    args = ap.parse_args()
    setup_logging()
    cfg = load_config(args.config, args.set)
    panel = build_panel(cfg, force=args.force)
    log.info("panel %s, %d tickers", panel.shape, panel.index.get_level_values("ticker").nunique())


if __name__ == "__main__":
    main()
