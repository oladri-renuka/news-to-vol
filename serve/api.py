"""FastAPI service.

    uvicorn serve.api:app --host 0.0.0.0 --port 8000

GET  /health
GET  /forecast/{ticker}    latest stored forecasts from the daily pipeline (SQLite)
POST /predict              run the deep model on a raw feature window (the latency benchmark uses this)
"""
from __future__ import annotations

import os
import sqlite3
import time
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
CKPT = Path(os.environ.get("VOLSENT_CHECKPOINT", ROOT / "artifacts/models/v2_lstm_price/lstm_seed0_test2023.pt"))
STORE = Path(os.environ.get("VOLSENT_STORE", ROOT / "artifacts/store/forecasts.sqlite"))

torch.set_num_threads(int(os.environ.get("VOLSENT_THREADS", "1")))
app = FastAPI(title="volsent", version="1.0")


@lru_cache(maxsize=1)
def model_bundle():
    import sys

    sys.path.insert(0, str(ROOT))
    from src.models.deep import build_model

    state = torch.load(CKPT, map_location="cpu", weights_only=False)
    m = build_model(state["model_cfg"], len(state["feature_cols"]), len(state["tickers"]))
    m.load_state_dict(state["model_state"])
    m.eval()
    return m, state


class PredictRequest(BaseModel):
    ticker: str
    window: list[list[float]] = Field(..., description="[W x F] raw (unscaled) features, oldest first")


class PredictResponse(BaseModel):
    ticker: str
    log_vol_5d: float
    vol_5d: float
    vol_annualized: float
    model: str
    inference_ms: float


@app.get("/health")
def health():
    ok = CKPT.exists()
    info = {"status": "ok" if ok else "no-model", "checkpoint": str(CKPT.name)}
    if ok:
        _, s = model_bundle()
        info.update(window=s["window"], features=s["feature_cols"], tickers=s["tickers"],
                    size_mb=round(CKPT.stat().st_size / 2**20, 3))
    return info


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    model, s = model_bundle()
    if req.ticker not in s["tickers"]:
        raise HTTPException(404, f"unknown ticker {req.ticker}")
    x = np.asarray(req.window, dtype=np.float32)
    if x.shape != (s["window"], len(s["feature_cols"])):
        raise HTTPException(422, f"window must be {s['window']} x {len(s['feature_cols'])}, got {list(x.shape)}")
    t0 = time.perf_counter()
    x = np.nan_to_num((x - s["x_mu"]) / s["x_sd"], nan=0.0)
    with torch.inference_mode():
        y = model(torch.from_numpy(x)[None], torch.tensor([s["tickers"].index(req.ticker)])).item()
    y = y * s["y_sd"] + s["y_mu"]
    ms = (time.perf_counter() - t0) * 1e3
    v = float(np.exp(y))
    return PredictResponse(ticker=req.ticker, log_vol_5d=y, vol_5d=v, vol_annualized=v * np.sqrt(252 / 5),
                           model=s["model_cfg"]["type"], inference_ms=ms)


@app.get("/forecast/{ticker}")
def forecast(ticker: str):
    if not STORE.exists():
        raise HTTPException(503, "forecast store missing; run the daily pipeline")
    con = sqlite3.connect(STORE)
    rows = con.execute(
        "SELECT asof, model, vol_5d, vol_ann FROM forecasts WHERE ticker=? AND asof=(SELECT MAX(asof) FROM forecasts WHERE ticker=?)",
        (ticker, ticker)).fetchall()
    con.close()
    if not rows:
        raise HTTPException(404, f"no forecasts for {ticker}")
    return {"ticker": ticker, "asof": rows[0][0],
            "forecasts": [{"model": m, "vol_5d": v, "vol_annualized": a} for _, m, v, a in rows]}
