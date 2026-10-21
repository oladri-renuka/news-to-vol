"""Before/after table for the VIX look-ahead fix. The metrics were fixed before the lagged run finished.

    python scripts/compare_vix_lag.py      # writes reports/vix_lag_comparison.md

"Before" = reports/archive_vix_same_day (VIX close of day t, A40/CUDA).
"After"  = reports/ (VIX close of day t-1, Apple M4/MPS).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BEFORE, AFTER = ROOT / "reports" / "archive_vix_same_day", ROOT / "reports"


def pooled(tier: str, root: Path) -> pd.DataFrame:
    sub = {"price": "tables", "sent": "sentiment/tables", "abl": "ablation/tables"}[tier]
    t = pd.read_csv(root / sub / "results_by_year.csv")
    return t[t.test_year == "pooled"].set_index("model")[["qlike", "qlike_std"]]


def dm(tier: str, root: Path, bench: str, extra: bool) -> pd.Series:
    sub = {"price": "tables", "sent": "sentiment/tables", "abl": "ablation/tables"}[tier]
    t = pd.read_csv(root / sub / ("dm_tests_extra.csv" if extra else "dm_tests.csv"))
    t = t[(t.test_year == "pooled") & (t.loss == "qlike") & (t.benchmark == bench)]
    return t.set_index("model")["p_value"]


def regimes(tier: str, root: Path) -> pd.DataFrame:
    sub = {"price": "tables", "sent": "sentiment/tables", "abl": "ablation/tables"}[tier]
    return pd.read_csv(root / sub / "regimes.csv").pivot(index="model", columns="regime", values="qlike_vs_har")


def fmt_p(p: float) -> str:
    return "—" if pd.isna(p) else ("<0.001" if p < 0.001 else f"{p:.3f}")


def main() -> None:
    out = ["# VIX look-ahead fix: before vs after", "",
           "Before: VIX close of day *t* as a feature (closes 16:15 ET, after the 16:00 origin), A40/CUDA. "
           "After: VIX close of day *t−1*, Apple M4/MPS. Backend differences alone move pooled QLIKE by about 0.0001.", ""]

    # Tier 1 headline
    out += ["## Tier 1: price-only, all ticker-days", "",
            "| Model | QLIKE before | QLIKE after | DM p vs HAR (before → after) | DM p vs HAR-X (before → after) | COVID / HAR (before → after) | VIX ≥ 30 / HAR (before → after) |",
            "|---|---|---|---|---|---|---|"]
    pb, pa = pooled("price", BEFORE), pooled("price", AFTER)
    db, da = dm("price", BEFORE, "HAR", False), dm("price", AFTER, "HAR", False)
    xb, xa = dm("price", BEFORE, "HAR-X", True), dm("price", AFTER, "HAR-X", True)
    rb, ra = regimes("price", BEFORE), regimes("price", AFTER)
    for m in ["LSTM-price", "Transformer-price", "HAR-X", "GARCH", "HAR", "EWMA"]:
        if m not in pa.index:
            continue
        out.append(f"| {m} | {pb.qlike.get(m, float('nan')):.4f} | {pa.qlike[m]:.4f} | "
                   f"{fmt_p(db.get(m))} → {fmt_p(da.get(m))} | {fmt_p(xb.get(m))} → {fmt_p(xa.get(m))} | "
                   f"{rb.covid_crash.get(m, float('nan')):.3f} → {ra.covid_crash[m]:.3f} | "
                   f"{rb.stressed.get(m, float('nan')):.3f} → {ra.stressed[m]:.3f} |")

    # Tier 2: what sentiment adds
    out += ["", "## Tier 2: does news help? (covered ticker-days)", "",
            "| Pair | ΔQLIKE from news, before | ΔQLIKE from news, after | DM p before | DM p after |", "|---|---|---|---|---|"]
    sb, sa = pooled("sent", BEFORE), pooled("sent", AFTER)
    for with_, without in [("LSTM-sent", "LSTM-price"), ("Transformer-sent", "Transformer-price"), ("HAR+sent", "HAR")]:
        if with_ not in sa.index:
            continue
        pb_ = dm("sent", BEFORE, without, True).get(with_) if without != "HAR" else dm("sent", BEFORE, "HAR", False).get(with_)
        pa_ = dm("sent", AFTER, without, True).get(with_) if without != "HAR" else dm("sent", AFTER, "HAR", False).get(with_)
        out.append(f"| {with_} vs {without} | {sb.qlike[with_] - sb.qlike[without]:+.4f} | "
                   f"{sa.qlike[with_] - sa.qlike[without]:+.4f} | {fmt_p(pb_)} | {fmt_p(pa_)} |")

    # V6: how much VIX matters
    out += ["", "## V6: effect of removing VIX (covered ticker-days)", "",
            "| Model | ΔQLIKE no-VIX, before | ΔQLIKE no-VIX, after | COVID / HAR full → no-VIX, before | after |",
            "|---|---|---|---|---|"]
    ab, aa = pooled("abl", BEFORE), pooled("abl", AFTER)
    gb, ga = regimes("abl", BEFORE), regimes("abl", AFTER)
    for fam in ["LSTM", "Transformer"]:
        full, nov = f"{fam}-sent", f"{fam}-noVIX"
        if nov not in aa.index:
            continue
        out.append(f"| {fam} | {ab.qlike[nov] - ab.qlike[full]:+.4f} | {aa.qlike[nov] - aa.qlike[full]:+.4f} | "
                   f"{gb.covid_crash[full]:.3f} → {gb.covid_crash[nov]:.3f} | {ga.covid_crash[full]:.3f} → {ga.covid_crash[nov]:.3f} |")

    text = "\n".join(out) + "\n"
    (ROOT / "reports" / "vix_lag_comparison.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
