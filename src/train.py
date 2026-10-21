"""Walk-forward training entry point. Every run is launched from a config file.

    python -m src.train --config configs/v1_baselines.yaml
    python -m src.train --config configs/v2_lstm_price.yaml --seeds 0 1 2
    python -m src.train --config configs/v2_lstm_price.yaml --smoke      # 1 fold, 1 seed, 2 epochs

Predictions are written to artifacts/predictions/<experiment>/<model>.parquet with columns
[date, ticker, test_year, seed, y, rv2_true, vix, y_pred, var_pred].
"""
from __future__ import annotations

import argparse
import copy
import json
import logging
import time

import numpy as np
import pandas as pd

from src.config import load_config, resolve
from src.data.dataset import WindowedPanel, build_panel, fold_dates, panel_meta, walk_forward_folds
from src.metrics import mse, qlike, smearing_factor
from src.models import baselines as B
from src.utils import Tracker, get_device, set_seed, setup_logging

log = logging.getLogger("volsent.train")


def feature_list(cfg: dict) -> list[str]:
    fcfg = cfg["features"]
    cols = []
    for group in cfg["experiment"].get("feature_groups", ["price"]):
        cols += fcfg[group]
    drop = set(cfg["experiment"].get("drop_features", []))
    return [c for c in cols if c not in drop]


def _attach_truth(pred: pd.DataFrame, panel: pd.DataFrame, test_year: int, seed: int) -> pd.DataFrame:
    truth = panel[["y", "rv2_true", "vix", "news_covered"]].reset_index()
    out = pred.merge(truth, on=["date", "ticker"], how="left")
    out["test_year"] = test_year
    out["seed"] = seed
    return out.dropna(subset=["y", "y_pred", "var_pred"])


# ---------------------------------------------------------------- baselines
def run_baselines(cfg: dict, panel: pd.DataFrame, folds, tracker: Tracker) -> dict[str, pd.DataFrame]:
    calendar = pd.DatetimeIndex(sorted(panel.index.get_level_values("date").unique()))
    h = cfg["target"]["horizon"]
    models = cfg["experiment"]["models"]
    preds: dict[str, list] = {m: [] for m in models}
    coefs = []
    for fold in folds:
        d = fold_dates(cfg, calendar, fold)
        fit_key = cfg["walk_forward"].get("baseline_fit_on", "train_val")
        fit_dates = d[fit_key]
        test = d["test"]
        t0 = time.time()
        for m in models:
            if m == "ewma":
                p = B.predict_ewma(panel, test, cfg["baselines"]["ewma_lambda"], h)
            elif m == "garch":
                p = B.predict_garch(panel, fit_dates.min(), fit_dates.max(), test, h, cfg["baselines"]["garch_dist"])
            elif m in ("har", "har_sent", "har_x"):
                cols = list(cfg["features"]["har"])
                if m == "har_x":
                    # HAR-X: the same linear model on every price feature. It separates
                    # "better inputs" from "deep model".
                    cols = [c for c in cfg["features"]["price"] if not c.startswith("dow_")]
                if m == "har_sent":
                    # Sentiment is NaN outside covered ticker-years, so fit_har/predict_har
                    # drop those rows: HAR+sent is fit and evaluated on covered ticker-days only.
                    cols += [c for c in cfg["features"]["sentiment"] if c != "news_covered"
                             and c not in set(cfg["experiment"].get("drop_features", []))]
                hcfg = cfg["baselines"]
                fitted = B.fit_har(panel, fit_dates, cols, hcfg.get("har_pooled", False), hcfg.get("har_ridge", 0.0),
                                   min_rows=hcfg.get("har_min_fit_rows", 250) if m == "har_sent" else 0)
                p = B.predict_har(panel, fitted, test, cols)
                fb = p.loc[p.pooled_fallback, "ticker"].unique()
                if len(fb):
                    log.info("[%d] %s: pooled fallback for %s (too few covered fit rows)", fold.test_year, m,
                             " ".join(sorted(fb)))
                ct = B.har_coef_table(fitted, cols)
                ct["model"], ct["test_year"] = m, fold.test_year
                coefs.append(ct)
            else:
                raise ValueError(m)
            p = _attach_truth(p, panel, fold.test_year, seed=0)
            preds[m].append(p)
            tracker.log_metrics({f"{m}.mse": mse(p.y, p.y_pred), f"{m}.qlike": qlike(p.rv2_true, p.var_pred)},
                                step=fold.test_year)
            log.info("[%d] %-9s n=%6d mse=%.4f qlike=%.4f", fold.test_year, m, len(p),
                     mse(p.y, p.y_pred), qlike(p.rv2_true, p.var_pred))
        log.info("fold %d done in %.1fs", fold.test_year, time.time() - t0)
    if coefs:
        out = resolve(cfg["paths"]["artifacts"]) / "predictions" / cfg["name"]
        out.mkdir(parents=True, exist_ok=True)
        pd.concat(coefs).to_csv(out / "har_coefficients.csv")
    return {m: pd.concat(v, ignore_index=True) for m, v in preds.items()}


# ---------------------------------------------------------------- deep models
def _batches(idx: np.ndarray, bs: int, rng: np.random.Generator | None):
    order = rng.permutation(len(idx)) if rng is not None else np.arange(len(idx))
    for i in range(0, len(idx), bs):
        yield idx[order[i:i + bs]]


def _predict(model, Xd, idx: np.ndarray, W: int, bs: int, device) -> np.ndarray:
    import torch

    model.eval()
    offs = torch.arange(-W + 1, 1, device=device)
    out = []
    with torch.inference_mode():
        for b in _batches(idx, bs * 4, None):
            bt = torch.as_tensor(b, device=device)
            xb = Xd[bt[:, :1], bt[:, 1:2] + offs]          # [B, W, F]
            out.append(model(xb, bt[:, 0]).float().cpu().numpy())
    return np.concatenate(out) if out else np.zeros(0)


def train_deep_fold(cfg: dict, wp: WindowedPanel, d: dict, seed: int, device, tracker: Tracker | None,
                    test_year: int):
    import torch

    from src.models.deep import build_model

    tcfg, mcfg = cfg["train"], cfg["model"]
    W = wp.window
    set_seed(seed)
    tr = wp.sample_index(d["train"])
    va = wp.sample_index(d["val"])
    te = wp.sample_index(d["test"])
    if tcfg.get("max_train_samples"):
        tr = tr[np.random.default_rng(seed).permutation(len(tr))[: tcfg["max_train_samples"]]]
    mu, sd = wp.fit_scaler(tr)
    Xs = np.nan_to_num((wp.X - mu) / sd, nan=0.0).astype(np.float32)
    ytr = wp.y[tr[:, 0], tr[:, 1]]
    y_mu, y_sd = float(ytr.mean()), float(ytr.std())
    Xd = torch.as_tensor(Xs, device=device)
    Yd = torch.as_tensor((wp.y - y_mu) / y_sd, device=device)

    model = build_model(mcfg, len(wp.feature_cols), len(wp.tickers)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=tcfg["lr"], weight_decay=tcfg.get("weight_decay", 0.0))
    offs = torch.arange(-W + 1, 1, device=device)
    rng = np.random.default_rng(seed)
    best, best_state, bad = np.inf, None, 0
    yva = wp.y[va[:, 0], va[:, 1]]
    for ep in range(tcfg["max_epochs"]):
        model.train()
        tot, n = 0.0, 0
        for b in _batches(tr, tcfg["batch_size"], rng):
            bt = torch.as_tensor(b, device=device)
            xb = Xd[bt[:, :1], bt[:, 1:2] + offs]
            yb = Yd[bt[:, 0], bt[:, 1]]
            loss = torch.nn.functional.mse_loss(model(xb, bt[:, 0]), yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if tcfg.get("grad_clip"):
                torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg["grad_clip"])
            opt.step()
            tot += loss.item() * len(b); n += len(b)
        pva = _predict(model, Xd, va, W, tcfg["batch_size"], device) * y_sd + y_mu
        vloss = mse(yva, pva)
        if tracker:
            tracker.log_metrics({f"{test_year}.train_loss": tot / n, f"{test_year}.val_mse": vloss}, step=ep)
        log.info("[%d seed %d] epoch %2d train=%.4f val_mse=%.4f", test_year, seed, ep, tot / n, vloss)
        if vloss < best - tcfg.get("min_delta", 0.0):
            best, bad = vloss, 0
            best_state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
        else:
            bad += 1
            if bad >= tcfg["patience"]:
                break
    model.load_state_dict(best_state)
    pva = _predict(model, Xd, va, W, tcfg["batch_size"], device) * y_sd + y_mu
    smear = smearing_factor(yva, pva)  # estimated on validation, never on test
    pte = _predict(model, Xd, te, W, tcfg["batch_size"], device) * y_sd + y_mu
    pred = pd.DataFrame({"date": wp.dates[te[:, 1]], "ticker": np.array(wp.tickers)[te[:, 0]],
                         "y_pred": pte, "var_pred": np.exp(2 * pte) * smear})
    state = {"model_state": best_state, "x_mu": mu, "x_sd": sd, "y_mu": y_mu, "y_sd": y_sd, "smear": smear,
             "feature_cols": wp.feature_cols, "tickers": wp.tickers, "window": W, "model_cfg": mcfg,
             "best_val_mse": best, "test_year": test_year}
    return pred, state


def run_deep(cfg: dict, panel: pd.DataFrame, folds, seeds: list[int], tracker: Tracker) -> dict[str, pd.DataFrame]:
    import torch

    device = get_device(cfg["train"].get("device", "auto"))
    log.info("device: %s", device)
    feats = feature_list(cfg)
    # Price features must be complete in every window. Sentiment may be NaN (uncovered
    # ticker-years), is imputed after scaling, and is flagged by news_covered.
    required = [f for f in feats if f not in set(cfg["features"]["sentiment"])]
    wp = WindowedPanel(panel, feats, cfg["model"]["window"], required_cols=required)
    calendar = wp.dates
    name = cfg["model"]["type"]
    out_models = resolve(cfg["paths"]["artifacts"]) / "models" / cfg["name"]
    out_models.mkdir(parents=True, exist_ok=True)
    preds = []
    for seed in seeds:
        for fold in folds:
            t0 = time.time()
            d = fold_dates(cfg, calendar, fold)
            p, state = train_deep_fold(cfg, wp, d, seed, device, tracker, fold.test_year)
            p = _attach_truth(p, panel, fold.test_year, seed)
            preds.append(p)
            m = {f"{name}.s{seed}.mse": mse(p.y, p.y_pred), f"{name}.s{seed}.qlike": qlike(p.rv2_true, p.var_pred)}
            tracker.log_metrics(m, step=fold.test_year)
            log.info("[%d seed %d] TEST mse=%.4f qlike=%.4f (%.0fs)", fold.test_year, seed,
                     *m.values(), time.time() - t0)
            if fold is folds[-1]:
                torch.save(state, out_models / f"{name}_seed{seed}_test{fold.test_year}.pt")
    return {name: pd.concat(preds, ignore_index=True)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seeds", type=int, nargs="*")
    ap.add_argument("--years", type=int, nargs="*", help="restrict to these test years")
    ap.add_argument("--smoke", action="store_true", help="1 fold, 1 seed, 2 epochs")
    ap.add_argument("--set", nargs="*", default=[], help="dotted overrides, e.g. train.max_epochs=3")
    args = ap.parse_args()
    setup_logging()
    cfg = load_config(args.config, args.set)
    seeds = args.seeds if args.seeds is not None else cfg["seeds"]
    years = args.years
    if args.smoke:
        seeds, years = seeds[:1], years or [cfg["walk_forward"]["first_test_year"]]
        cfg.setdefault("train", {})["max_epochs"] = 2
        cfg["name"] = cfg["name"] + "_smoke"

    panel = build_panel(cfg)
    meta = panel_meta(cfg)
    if "sentiment" in cfg["experiment"].get("feature_groups", []) or "har_sent" in cfg["experiment"].get("models", []):
        if not meta["sentiment_available"]:
            log.warning("sentiment features are all zero (no scored headlines). The results will NOT measure news.")
    folds = walk_forward_folds(cfg, years)
    tracker = Tracker(cfg, cfg["name"])
    with tracker.run({k: cfg[k] for k in ("experiment", "model", "train", "baselines") if k in cfg}):
        if cfg["experiment"]["kind"] == "baselines":
            preds = run_baselines(cfg, panel, folds, tracker)
        else:
            preds = run_deep(cfg, panel, folds, seeds, tracker)
        out = resolve(cfg["paths"]["artifacts"]) / "predictions" / cfg["name"]
        out.mkdir(parents=True, exist_ok=True)
        for m, df in preds.items():
            df.to_parquet(out / f"{m}.parquet")
            log.info("wrote %s (%d rows)", out / f"{m}.parquet", len(df))
        with open(out / "config.json", "w") as f:
            json.dump({k: v for k, v in cfg.items()}, f, indent=2, default=str)


if __name__ == "__main__":
    main()
