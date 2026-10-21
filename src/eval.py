"""Evaluation: per-year and pooled losses, Diebold-Mariano tests vs HAR-RV, regime splits (V5), MCS,
a vol-targeting check, plus figures and tables for the README.

    python -m src.eval --config configs/eval.yaml

All models are compared on the same (date, ticker) rows: the intersection of the rows every model has.
Deep-model metrics are reported as mean ± std over seeds. The DM test and the figures use the seed-averaged
forecast.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import load_config, resolve
from src.metrics import diebold_mariano, model_confidence_set, per_date_losses, qlike_vec
from src.utils import setup_logging

log = logging.getLogger("volsent.eval")


# ---------------------------------------------------------------- loading
def load_predictions(cfg: dict) -> dict[str, pd.DataFrame]:
    root = resolve(cfg["paths"]["artifacts"]) / "predictions"
    out = {}
    for label, rel in cfg["eval"]["models"].items():
        p = root / rel
        if not p.exists():
            log.warning("missing predictions for %s (%s), skipping", label, p)
            continue
        out[label] = pd.read_parquet(p)
    return out


def align(preds: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    keys = None
    for df in preds.values():
        k = pd.MultiIndex.from_frame(df[["date", "ticker"]].drop_duplicates())
        keys = k if keys is None else keys.intersection(k)
    out = {}
    for m, df in preds.items():
        idx = pd.MultiIndex.from_frame(df[["date", "ticker"]])
        out[m] = df[idx.isin(keys)].reset_index(drop=True)
    log.info("aligned on %d (date, ticker) rows", len(keys))
    return out


def seed_mean(df: pd.DataFrame) -> pd.DataFrame:
    """Seed-averaged forecast (log-vol averaged; variance averaged)."""
    g = df.groupby(["date", "ticker"], as_index=False)
    return g.agg(y=("y", "first"), rv2_true=("rv2_true", "first"), vix=("vix", "first"),
                 test_year=("test_year", "first"), y_pred=("y_pred", "mean"), var_pred=("var_pred", "mean"))


def covered_share(df: pd.DataFrame) -> pd.DataFrame:
    """Covered fraction of test ticker-days and the number of covered tickers, per test year."""
    d = df.drop_duplicates(["date", "ticker"])
    g = d.groupby("test_year")
    out = pd.DataFrame({
        "ticker_days": g.size(),
        "covered_ticker_days": g.news_covered.apply(lambda x: int((x > 0).sum())),
        "covered_fraction": g.news_covered.apply(lambda x: round(float((x > 0).mean()), 4)),
        "n_tickers_covered": g.apply(lambda x: x.loc[x.news_covered > 0, "ticker"].nunique()),
        "n_tickers": g.ticker.nunique(),
    })
    out.loc["pooled"] = [len(d), int((d.news_covered > 0).sum()), round(float((d.news_covered > 0).mean()), 4),
                         np.nan, d.ticker.nunique()]
    return out


# ---------------------------------------------------------------- metrics tables
def _losses(df: pd.DataFrame) -> pd.DataFrame:
    return df.assign(se=(df.y - df.y_pred) ** 2, ql=qlike_vec(df.rv2_true, df.var_pred))


def per_year_table(preds: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for m, df in preds.items():
        L = _losses(df)
        by = L.groupby(["seed", "test_year"])[["se", "ql"]].mean().reset_index()
        pooled = L.groupby("seed")[["se", "ql"]].mean().reset_index().assign(test_year="pooled")
        by = pd.concat([by, pooled])
        agg = by.groupby("test_year").agg(mse=("se", "mean"), mse_std=("se", "std"),
                                          qlike=("ql", "mean"), qlike_std=("ql", "std"),
                                          n_seeds=("seed", "nunique"))
        agg["model"] = m
        rows.append(agg.reset_index())
    t = pd.concat(rows)
    t[["mse_std", "qlike_std"]] = t[["mse_std", "qlike_std"]].fillna(0.0)
    return t


def dm_table(preds_mean: dict[str, pd.DataFrame], bench: str, h: int) -> pd.DataFrame:
    rows = []
    b = preds_mean[bench]
    for m, df in preds_mean.items():
        if m == bench:
            continue
        merged = df.merge(b[["date", "ticker", "y_pred", "var_pred"]], on=["date", "ticker"], suffixes=("", "_b"))
        la = per_date_losses(merged, "y_pred", "var_pred")
        lb = per_date_losses(merged, "y_pred_b", "var_pred_b")
        years = la.index.year
        for yr in list(np.unique(years)) + ["pooled"]:
            sel = slice(None) if yr == "pooled" else (years == yr)
            for loss in ("mse", "qlike"):
                r = diebold_mariano(la[loss].to_numpy()[sel], lb[loss].to_numpy()[sel], h=h)
                rows.append({"model": m, "benchmark": bench, "test_year": yr, "loss": loss, **r})
    return pd.DataFrame(rows)


def regime_masks(df: pd.DataFrame, regimes: dict) -> dict[str, np.ndarray]:
    out = {}
    for name, rule in regimes.items():
        m = np.ones(len(df), bool)
        if "vix_lt" in rule:
            m &= df.vix.to_numpy() < rule["vix_lt"]
        if "vix_ge" in rule:
            m &= df.vix.to_numpy() >= rule["vix_ge"]
        if "start" in rule:
            m &= (df.date >= pd.Timestamp(rule["start"])).to_numpy() & (df.date <= pd.Timestamp(rule["end"])).to_numpy()
        out[name] = m
    return out


def regime_table(preds_mean: dict[str, pd.DataFrame], regimes: dict, bench: str, h: int) -> pd.DataFrame:
    rows = []
    b = preds_mean[bench]
    for m, df in preds_mean.items():
        merged = df.merge(b[["date", "ticker", "y_pred", "var_pred"]], on=["date", "ticker"], suffixes=("", "_b"))
        L = _losses(merged)
        Lb = merged.assign(y_pred=merged.y_pred_b, var_pred=merged.var_pred_b).pipe(_losses)
        for rname, mask in regime_masks(merged, regimes).items():
            if mask.sum() == 0:
                continue
            row = {"model": m, "regime": rname, "n": int(mask.sum()),
                   "mse": L.se[mask].mean(), "qlike": L.ql[mask].mean(),
                   "mse_vs_har": L.se[mask].mean() / Lb.se[mask].mean(),
                   "qlike_vs_har": L.ql[mask].mean() / Lb.ql[mask].mean()}
            if m != bench:
                la = per_date_losses(merged[mask], "y_pred", "var_pred")
                lb = per_date_losses(merged[mask], "y_pred_b", "var_pred_b")
                row["dm_p_mse"] = diebold_mariano(la.mse.values, lb.mse.values, h)["p_value"]
                row["dm_p_qlike"] = diebold_mariano(la.qlike.values, lb.qlike.values, h)["p_value"]
            rows.append(row)
    return pd.DataFrame(rows)


def vol_target_table(preds_mean: dict[str, pd.DataFrame], target_ann: float = 0.10, max_lev: float = 3.0,
                     horizon: int = 5) -> pd.DataFrame:
    """Size each position as w = target / forecast vol and compare the realized vol of the position.

    Uses non-overlapping 5-day holding periods per ticker. Transaction costs are ignored.
    """
    tgt = target_ann * np.sqrt(horizon / 252)
    rows = []
    for m, df in preds_mean.items():
        df = df.sort_values(["ticker", "date"])
        df = df.groupby("ticker", group_keys=False).apply(lambda g: g.iloc[::horizon])
        w = np.minimum(tgt / np.sqrt(df.var_pred), max_lev)
        realized = w * np.sqrt(df.rv2_true)
        ratio = realized / tgt
        rows.append({"model": m, "mean_ratio": ratio.mean(), "median_ratio": ratio.median(),
                     "rmse_log_ratio": float(np.sqrt(np.mean(np.log(ratio) ** 2))),
                     "share_over_1.5x": float((ratio > 1.5).mean()), "n_periods": len(df)})
    return pd.DataFrame(rows)


def mcs_table(preds_mean: dict[str, pd.DataFrame], loss: str = "qlike", size: float = 0.1) -> pd.DataFrame:
    L = {m: per_date_losses(df)[loss] for m, df in preds_mean.items()}
    frame = pd.DataFrame(L).dropna()
    t = model_confidence_set(frame, size=size)
    t["loss"] = loss
    t = t.reset_index()
    return t.rename(columns={t.columns[0]: "model"})


def failure_cases(preds_mean: dict[str, pd.DataFrame], model: str, bench: str, k: int = 20,
                  headlines: pd.DataFrame | None = None) -> pd.DataFrame:
    """The k worst forecasts of `model` by QLIKE, with HAR's forecast and that day's most negative headline."""
    df = preds_mean[model].merge(preds_mean[bench][["date", "ticker", "y_pred", "var_pred"]],
                                 on=["date", "ticker"], suffixes=("", "_har"))
    df["qlike"] = qlike_vec(df.rv2_true, df.var_pred)
    df["qlike_har"] = qlike_vec(df.rv2_true, df.var_pred_har)
    ann = np.sqrt(252 / 5)
    out = df.nlargest(k, "qlike").assign(
        realized_ann=lambda d: np.exp(d.y) * ann, forecast_ann=lambda d: np.exp(d.y_pred) * ann,
        har_ann=lambda d: np.exp(d.y_pred_har) * ann, vix=lambda d: d.vix)
    out["direction"] = np.where(out.forecast_ann < out.realized_ann, "under", "over")
    cols = ["date", "ticker", "vix", "realized_ann", "forecast_ann", "har_ann", "direction", "qlike", "qlike_har"]
    out = out[cols]
    if headlines is not None and len(headlines):
        h = headlines.sort_values("p_neg", ascending=False).drop_duplicates(["trading_day", "ticker"])
        out = out.merge(h[["trading_day", "ticker", "headline", "p_neg"]].rename(columns={"trading_day": "date"}),
                        on=["date", "ticker"], how="left")
    return out


def under_over_summary(preds_mean: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Share of the QLIKE loss that comes from under-prediction, and the mean log forecast bias, per model."""
    rows = []
    for m, df in preds_mean.items():
        ql = qlike_vec(df.rv2_true, df.var_pred)
        under = df.var_pred < df.rv2_true
        rows.append({"model": m, "mean_log_bias": float((df.y_pred - df.y).mean()),
                     "share_under": float(under.mean()), "qlike_share_from_under": float(ql[under].sum() / ql.sum())})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- figures
def make_figures(preds_mean: dict[str, pd.DataFrame], reg: pd.DataFrame, bench: str, out: Path,
                 ticker: str = "AAPL") -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out.mkdir(parents=True, exist_ok=True)
    paths = []
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    # 1. Headline figure: rolling 63-day QLIKE ratio vs HAR, pooled over tickers.
    fig, ax = plt.subplots(figsize=(10, 4))
    b = per_date_losses(preds_mean[bench])["qlike"]
    for i, (m, df) in enumerate(preds_mean.items()):
        if m == bench:
            continue
        l = per_date_losses(df)["qlike"].reindex(b.index)
        ratio = l.rolling(63, min_periods=20).mean() / b.rolling(63, min_periods=20).mean()
        ax.plot(ratio.index, ratio.values, label=m, lw=1.2, color=colors[i % len(colors)])
    ax.axhline(1.0, color="k", lw=0.8, ls="--")
    ax.set_ylabel(f"rolling 63d QLIKE / {bench}")
    ax.set_title("Out-of-sample loss relative to HAR-RV (below 1 = better than HAR)")
    ax.legend(ncol=4, fontsize=8, frameon=False)
    fig.tight_layout()
    p = out / "rolling_qlike_vs_har.png"; fig.savefig(p, dpi=150); plt.close(fig); paths.append(p)

    # 2. Forecast vs realized vol for one ticker.
    fig, ax = plt.subplots(figsize=(10, 4))
    first = next(iter(preds_mean.values()))
    tk = ticker if ticker in set(first.ticker) else first.ticker.iloc[0]
    g = first[first.ticker == tk].sort_values("date")
    ax.plot(g.date, np.exp(g.y) * np.sqrt(252 / 5), color="0.6", lw=0.8, label="realized")
    for i, (m, df) in enumerate(preds_mean.items()):
        g = df[df.ticker == tk].sort_values("date")
        ax.plot(g.date, np.exp(g.y_pred) * np.sqrt(252 / 5), lw=1.0, label=m, color=colors[i % len(colors)])
    ax.set_ylabel("annualized 5-day vol")
    ax.set_title(f"{tk}: forecast vs realized 5-day volatility")
    ax.legend(ncol=5, fontsize=8, frameon=False)
    fig.tight_layout()
    p = out / f"forecast_vs_realized_{tk}.png"; fig.savefig(p, dpi=150); plt.close(fig); paths.append(p)

    # 3. Regime bars.
    if len(reg):
        piv = reg.pivot(index="regime", columns="model", values="qlike_vs_har")
        fig, ax = plt.subplots(figsize=(8, 4))
        piv.plot.bar(ax=ax, rot=0, width=0.8)
        ax.axhline(1.0, color="k", lw=0.8, ls="--")
        ax.set_ylabel("QLIKE / HAR")
        ax.set_title("Loss relative to HAR-RV by regime")
        ax.legend(fontsize=8, frameon=False, ncol=3)
        fig.tight_layout()
        p = out / "regimes_qlike_vs_har.png"; fig.savefig(p, dpi=150); plt.close(fig); paths.append(p)
    return paths


# ---------------------------------------------------------------- markdown
def fmt_pm(m: float, s: float, n: int) -> str:
    return f"{m:.4f} ± {s:.4f}" if n > 1 else f"{m:.4f}"


def results_markdown(yt: pd.DataFrame, dm: pd.DataFrame, bench: str) -> str:
    pooled = yt[yt.test_year == "pooled"].set_index("model")
    dmp = dm[dm.test_year == "pooled"].pivot(index="model", columns="loss", values="p_value")
    lines = ["| Model | MSE log-RV | QLIKE | DM p (MSE) | DM p (QLIKE) | seeds |", "|---|---|---|---|---|---|"]
    for m, r in pooled.sort_values("qlike").iterrows():
        pm = dmp.loc[m, "mse"] if m in dmp.index else np.nan
        pq = dmp.loc[m, "qlike"] if m in dmp.index else np.nan
        name = f"**{m}**" if m == bench else m
        lines.append(f"| {name} | {fmt_pm(r.mse, r.mse_std, r.n_seeds)} | {fmt_pm(r.qlike, r.qlike_std, r.n_seeds)} | "
                     f"{'—' if np.isnan(pm) else f'{pm:.3f}'} | {'—' if np.isnan(pq) else f'{pq:.3f}'} | {int(r.n_seeds)} |")
    per = yt[yt.test_year != "pooled"].pivot(index="model", columns="test_year", values="qlike")
    lines += ["", "QLIKE by test year:", "", "| Model | " + " | ".join(map(str, per.columns)) + " |",
              "|---|" + "---|" * len(per.columns)]
    for m, r in per.iterrows():
        lines.append(f"| {m} | " + " | ".join(f"{v:.4f}" for v in r.values) + " |")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/eval.yaml")
    ap.add_argument("--set", nargs="*", default=[])
    args = ap.parse_args()
    setup_logging()
    cfg = load_config(args.config, args.set)
    ecfg = cfg["eval"]
    bench, h = ecfg["benchmark"], cfg["target"]["horizon"]
    raw = load_predictions(cfg)
    tables = resolve(cfg["paths"]["reports"]) / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    if ecfg.get("mask_covered") and bench in raw and "news_covered" in raw[bench]:
        cov = covered_share(raw[bench])
        cov.to_csv(tables / "coverage_test_years.csv")
        print("News coverage of test ticker-days (the sentiment comparison uses covered rows only):")
        print(cov.to_string() + "\n")
    if ecfg.get("mask_covered"):
        # Sentiment comparison: covered ticker-days only, the same rows for every model (paired tests).
        for m, df in raw.items():
            if "news_covered" not in df:
                raise SystemExit(f"{m}: predictions lack news_covered; re-run training after rebuilding the panel")
            raw[m] = df[df.news_covered > 0]
    preds = align(raw)
    if bench not in preds:
        raise SystemExit(f"benchmark {bench} has no predictions; run V1 first")
    pm = {m: seed_mean(df) for m, df in preds.items()}

    yt = per_year_table(preds); yt.to_csv(tables / "results_by_year.csv", index=False)
    dm = dm_table(pm, bench, h); dm.to_csv(tables / "dm_tests.csv", index=False)
    extra = [b for b in ecfg.get("extra_benchmarks", []) if b in pm]
    if extra:
        dmx = pd.concat([dm_table(pm, b, h) for b in extra])
        dmx.to_csv(tables / "dm_tests_extra.csv", index=False)
        print("\nDM p-values vs " + ", ".join(extra) + " (pooled):")
        print(dmx[dmx.test_year == "pooled"].pivot_table(index="model", columns=["benchmark", "loss"],
                                                          values="p_value").round(4).to_string())
    reg = regime_table(pm, cfg["regimes"], bench, h); reg.to_csv(tables / "regimes.csv", index=False)
    vt = vol_target_table(pm); vt.to_csv(tables / "vol_target.csv", index=False)
    try:
        mcs = pd.concat([mcs_table(pm, "qlike"), mcs_table(pm, "mse")])
        mcs.to_csv(tables / "mcs.csv", index=False)
    except Exception as e:  # pragma: no cover
        log.warning("MCS failed: %s", e)
    uo = under_over_summary(pm); uo.to_csv(tables / "bias.csv", index=False)
    hl_path = resolve(cfg["paths"]["interim"]) / "headlines_scored.parquet"
    hl = pd.read_parquet(hl_path, columns=["trading_day", "ticker", "headline", "p_neg"]) if hl_path.exists() else None
    deep = [m for m in pm if m not in ("EWMA", "GARCH", "HAR", "HAR-X", "HAR+sent")]
    for m in deep[:4]:
        failure_cases(pm, m, bench, headlines=hl).to_csv(tables / f"failures_{m}.csv", index=False)
    md = results_markdown(yt, dm, bench)
    (resolve(cfg["paths"]["reports"]) / "results.md").write_text(md + "\n")
    make_figures(pm, reg, bench, resolve(cfg["paths"]["reports"]) / "figures")
    print(md)
    print("\nRegimes (QLIKE / HAR):")
    print(reg.pivot(index="model", columns="regime", values="qlike_vs_har").round(3).to_string())
    print("\nVol targeting:")
    print(vt.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
