"""Financial sentiment classifier: fine-tune on FinancialPhraseBank, compare with FinBERT, then
score every FNSPID headline once and cache the result.

    python -m src.models.sentiment --config configs/sentiment.yaml --stage cv      # 10-fold CV (GPU recommended)
    python -m src.models.sentiment --config configs/sentiment.yaml --stage train   # final model on all data
    python -m src.models.sentiment --config configs/sentiment.yaml --stage score   # batch inference, cached
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import load_config, resolve
from src.utils import Tracker, get_device, set_seed, setup_logging

log = logging.getLogger(__name__)

LABELS = ["negative", "neutral", "positive"]
LABEL2ID = {l: i for i, l in enumerate(LABELS)}
PHRASEBANK_FILES = {
    "sentences_allagree": "Sentences_AllAgree.txt",
    "sentences_75agree": "Sentences_75Agree.txt",
    "sentences_66agree": "Sentences_66Agree.txt",
    "sentences_50agree": "Sentences_50Agree.txt",
}


# ---------------------------------------------------------------- data
PHRASEBANK_URL = ("https://huggingface.co/datasets/takala/financial_phrasebank/resolve/main/"
                  "data/FinancialPhraseBank-v1.0.zip")


def ensure_phrasebank(zip_path: str | Path) -> Path:
    """Download the FinancialPhraseBank v1.0 zip (CC BY-NC-SA 3.0, ~0.7 MB) if it is missing."""
    path = resolve(zip_path)
    if not path.exists():
        import urllib.request

        path.parent.mkdir(parents=True, exist_ok=True)
        log.info("downloading FinancialPhraseBank to %s", path)
        tmp = path.with_suffix(".part")
        urllib.request.urlretrieve(PHRASEBANK_URL, tmp)
        tmp.rename(path)
    return path


def load_phrasebank(zip_path: str | Path, config_name: str = "sentences_allagree") -> pd.DataFrame:
    """Read FinancialPhraseBank v1.0 directly from the zip.

    The HF dataset uses a loading script, which `datasets>=4` no longer runs.
    Lines look like 'sentence text@label', encoded in latin-1.
    """
    fname = f"FinancialPhraseBank-v1.0/{PHRASEBANK_FILES[config_name]}"
    with zipfile.ZipFile(ensure_phrasebank(zip_path)) as z:
        raw = z.read(fname).decode("latin-1")
    rows = []
    for line in raw.splitlines():
        line = line.strip()
        if not line or "@" not in line:
            continue
        text, _, lab = line.rpartition("@")
        lab = lab.strip().lower()
        if lab in LABEL2ID:
            rows.append((text.strip(), LABEL2ID[lab]))
    df = pd.DataFrame(rows, columns=["text", "label"]).drop_duplicates("text").reset_index(drop=True)
    return df


# ---------------------------------------------------------------- training
def _batches(n: int, bs: int, rng: np.random.Generator | None = None):
    idx = rng.permutation(n) if rng is not None else np.arange(n)
    for i in range(0, n, bs):
        yield idx[i:i + bs]


def finetune(texts: list[str], labels: np.ndarray, scfg: dict, seed: int, device,
             val_texts: list[str] | None = None, val_labels: np.ndarray | None = None):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

    set_seed(seed)
    tok = AutoTokenizer.from_pretrained(scfg["model_name"])
    model = AutoModelForSequenceClassification.from_pretrained(
        scfg["model_name"], num_labels=3, id2label=dict(enumerate(LABELS)), label2id=LABEL2ID)
    model.to(device)
    enc = tok(texts, truncation=True, max_length=scfg["max_length"], padding="max_length", return_tensors="pt")
    y = torch.as_tensor(labels, dtype=torch.long)
    opt = torch.optim.AdamW(model.parameters(), lr=scfg["lr"], weight_decay=scfg["weight_decay"])
    steps = scfg["epochs"] * int(np.ceil(len(texts) / scfg["batch_size"]))
    sched = get_linear_schedule_with_warmup(opt, int(scfg["warmup_ratio"] * steps), steps)
    rng = np.random.default_rng(seed)
    for ep in range(scfg["epochs"]):
        model.train()
        tot = 0.0
        for b in _batches(len(texts), scfg["batch_size"], rng):
            out = model(input_ids=enc["input_ids"][b].to(device),
                        attention_mask=enc["attention_mask"][b].to(device), labels=y[b].to(device))
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); opt.zero_grad()
            tot += out.loss.item() * len(b)
        msg = f"epoch {ep + 1} train_loss={tot / len(texts):.4f}"
        if val_texts is not None:
            p = predict_proba(model, tok, val_texts, device, scfg["max_length"])
            msg += f" val_acc={(p.argmax(1) == val_labels).mean():.4f}"
        log.info(msg)
    return model, tok


def predict_proba(model, tok, texts: list[str], device, max_length: int = 64, batch_size: int = 256,
                  label_order: list[str] | None = None) -> np.ndarray:
    """Return probabilities ordered as LABELS, whatever order the model uses internally."""
    import torch

    model.eval()
    id2label = {int(k): v.lower() for k, v in model.config.id2label.items()}
    perm = [next(i for i, l in id2label.items() if l == name) for name in (label_order or LABELS)]
    out = []
    with torch.inference_mode():
        for i in range(0, len(texts), batch_size):
            enc = tok(texts[i:i + batch_size], truncation=True, max_length=max_length,
                      padding=True, return_tensors="pt").to(device)
            logits = model(**enc).logits.float()
            out.append(torch.softmax(logits, -1)[:, perm].cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, 3))


def macro_f1(y: np.ndarray, p: np.ndarray) -> float:
    from sklearn.metrics import f1_score

    return float(f1_score(y, p, average="macro"))


def run_cv(cfg: dict) -> pd.DataFrame:
    """Stratified K-fold CV of the fine-tuned model, with FinBERT scored on the same held-out folds."""
    import torch
    from sklearn.model_selection import StratifiedKFold
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    scfg = cfg["sentiment"]
    device = get_device(scfg.get("device", "auto"))
    rows = []
    for config_name in scfg["phrasebank_configs"]:
        df = load_phrasebank(scfg["phrasebank_zip"], config_name)
        log.info("%s: %d sentences, label counts %s", config_name, len(df), df.label.value_counts().to_dict())
        skf = StratifiedKFold(n_splits=scfg["cv_folds"], shuffle=True, random_state=cfg["seed"])
        fb_tok = AutoTokenizer.from_pretrained(scfg["finbert_name"])
        fb = AutoModelForSequenceClassification.from_pretrained(scfg["finbert_name"]).to(device)
        folds = list(skf.split(df.text, df.label))[: scfg.get("max_folds") or None]
        for k, (tr, te) in enumerate(folds):
            t0 = time.time()
            model, tok = finetune(df.text.iloc[tr].tolist(), df.label.values[tr], scfg, cfg["seed"] + k, device)
            p = predict_proba(model, tok, df.text.iloc[te].tolist(), device, scfg["max_length"])
            pf = predict_proba(fb, fb_tok, df.text.iloc[te].tolist(), device, scfg["max_length"])
            yte = df.label.values[te]
            rows.append({"config": config_name, "fold": k,
                         "model": scfg["model_name"], "acc": (p.argmax(1) == yte).mean(), "macro_f1": macro_f1(yte, p.argmax(1))})
            rows.append({"config": config_name, "fold": k,
                         "model": scfg["finbert_name"], "acc": (pf.argmax(1) == yte).mean(), "macro_f1": macro_f1(yte, pf.argmax(1))})
            log.info("%s fold %d: ft acc=%.4f f1=%.4f | finbert acc=%.4f f1=%.4f (%.0fs)", config_name, k,
                     rows[-2]["acc"], rows[-2]["macro_f1"], rows[-1]["acc"], rows[-1]["macro_f1"], time.time() - t0)
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
    res = pd.DataFrame(rows)
    out = resolve(cfg["paths"]["reports"]) / "tables"
    out.mkdir(parents=True, exist_ok=True)
    res.to_csv(out / "sentiment_cv_folds.csv", index=False)
    summ = res.groupby(["config", "model"])[["acc", "macro_f1"]].agg(["mean", "std"]).round(4)
    summ.to_csv(out / "sentiment_cv_summary.csv")
    print(summ.to_string())
    return res


def train_final(cfg: dict) -> Path:
    scfg = cfg["sentiment"]
    device = get_device(scfg.get("device", "auto"))
    frames = [load_phrasebank(scfg["phrasebank_zip"], c) for c in scfg["final_train_configs"]]
    df = pd.concat(frames).drop_duplicates("text")
    model, tok = finetune(df.text.tolist(), df.label.values, scfg, cfg["seed"], device)
    out = resolve(scfg["model_dir"])
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out)
    tok.save_pretrained(out)
    log.info("saved sentiment model to %s", out)
    return out


def score_headlines(cfg: dict, headlines: pd.DataFrame | None = None, out_path: Path | None = None) -> pd.DataFrame:
    """Score unique headline strings once. The cache key is the model directory plus the text hash."""
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    scfg = cfg["sentiment"]
    device = get_device(scfg.get("device", "auto"))
    interim = resolve(cfg["paths"]["interim"])
    if headlines is None:
        headlines = pd.read_parquet(interim / "headlines_aligned.parquet")
    out_path = out_path or interim / "headline_scores.parquet"
    src = scfg.get("score_with", "finetuned")
    model_dir = resolve(scfg["model_dir"]) if src == "finetuned" else scfg["finbert_name"]

    texts = pd.Series(headlines["headline"].unique())
    keys = texts.map(lambda s: hashlib.sha1(s.encode()).hexdigest())
    cache = pd.read_parquet(out_path) if out_path.exists() else pd.DataFrame(columns=["key", "p_neg", "p_neu", "p_pos", "scorer"])
    cache = cache[cache.scorer == str(model_dir)]
    todo = texts[~keys.isin(set(cache.key))]
    log.info("%d unique headlines, %d cached, %d to score on %s", len(texts), len(texts) - len(todo), len(todo), device)
    if len(todo):
        tok = AutoTokenizer.from_pretrained(model_dir)
        model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device)
        if device.type == "cuda":
            model.half()
        parts, t0 = [], time.time()
        chunk = scfg.get("score_chunk", 50_000)
        for i in range(0, len(todo), chunk):
            sub = todo.iloc[i:i + chunk].tolist()
            p = predict_proba(model, tok, sub, device, scfg["max_length"], scfg.get("score_batch_size", 512))
            parts.append(pd.DataFrame({"key": [hashlib.sha1(s.encode()).hexdigest() for s in sub],
                                       "p_neg": p[:, 0], "p_neu": p[:, 1], "p_pos": p[:, 2], "scorer": str(model_dir)}))
            log.info("scored %d/%d (%.0f/s)", min(i + chunk, len(todo)), len(todo), (i + len(sub)) / (time.time() - t0))
            cache = pd.concat([cache] + parts, ignore_index=True)
            cache.to_parquet(out_path)
            parts = []
    lookup = cache.set_index("key")[["p_neg", "p_neu", "p_pos"]]
    hk = headlines["headline"].map(lambda s: hashlib.sha1(s.encode()).hexdigest())
    scored = pd.concat([headlines.reset_index(drop=True), lookup.reindex(hk.values).reset_index(drop=True)], axis=1)
    return scored


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/sentiment.yaml")
    ap.add_argument("--stage", choices=["cv", "train", "score", "all"], required=True)
    ap.add_argument("--set", nargs="*", default=[])
    args = ap.parse_args()
    setup_logging()
    cfg = load_config(args.config, args.set)
    tracker = Tracker(cfg, f"sentiment-{args.stage}")
    with tracker.run(cfg.get("sentiment")):
        if args.stage in ("cv", "all"):
            res = run_cv(cfg)
            for (c, m), g in res.groupby(["config", "model"]):
                tag = f"{c}.{m.split('/')[-1]}"
                tracker.log_metrics({f"{tag}.acc": g.acc.mean(), f"{tag}.macro_f1": g.macro_f1.mean()})
        if args.stage in ("train", "all"):
            train_final(cfg)
        if args.stage in ("score", "all"):
            scored = score_headlines(cfg)
            path = resolve(cfg["paths"]["interim"]) / "headlines_scored.parquet"
            scored.to_parquet(path)
            log.info("wrote %s (%d rows)", path, len(scored))


if __name__ == "__main__":
    main()
