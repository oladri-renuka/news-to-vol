"""CPU latency (p50/p95) and model size for the serving endpoint.

    python -m serve.benchmark                     # in-process via TestClient (includes HTTP + JSON overhead)
    python -m serve.benchmark --url http://localhost:8000    # against a running server / container
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default=None)
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--checkpoint", default=None)
    args = ap.parse_args()

    if args.checkpoint:
        import os
        os.environ["VOLSENT_CHECKPOINT"] = args.checkpoint
    from serve.api import CKPT, app, model_bundle

    model, s = model_bundle()
    payload = {"ticker": s["tickers"][0],
               "window": (np.random.default_rng(0).normal(size=(s["window"], len(s["feature_cols"]))) * s["x_sd"]
                          + s["x_mu"]).tolist()}
    if args.url:
        import httpx
        client = httpx.Client(base_url=args.url, timeout=10)
    else:
        from fastapi.testclient import TestClient
        client = TestClient(app)

    for _ in range(args.warmup):
        client.post("/predict", json=payload)
    e2e, inf = [], []
    for _ in range(args.n):
        t0 = time.perf_counter()
        r = client.post("/predict", json=payload)
        e2e.append((time.perf_counter() - t0) * 1e3)
        inf.append(r.json()["inference_ms"])
    n_params = sum(p.numel() for p in model.parameters())
    res = {
        "model": s["model_cfg"]["type"], "checkpoint": CKPT.name, "device": "cpu",
        "torch_threads": torch.get_num_threads(),
        "params": n_params, "size_mb_fp32_weights": round(n_params * 4 / 2**20, 3),
        "checkpoint_file_mb": round(CKPT.stat().st_size / 2**20, 3),
        "request_p50_ms": round(float(np.percentile(e2e, 50)), 3),
        "request_p95_ms": round(float(np.percentile(e2e, 95)), 3),
        "inference_p50_ms": round(float(np.percentile(inf, 50)), 3),
        "inference_p95_ms": round(float(np.percentile(inf, 95)), 3),
        "n_requests": args.n, "mode": "http" if args.url else "in-process TestClient",
    }
    out = ROOT / "reports" / "tables" / f"latency_{s['model_cfg']['type']}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
