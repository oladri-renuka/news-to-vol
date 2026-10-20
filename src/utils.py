"""Seeding, device selection, logging and experiment tracking."""
from __future__ import annotations

import json
import logging
import os
import random
from contextlib import contextmanager
from pathlib import Path

import numpy as np

log = logging.getLogger("volsent")


def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def get_device(preference: str = "auto"):
    import torch

    if preference != "auto":
        return torch.device(preference)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class Tracker:
    """Thin wrapper over MLflow / W&B so the training code stays backend-agnostic.

    backend: "mlflow" (default, local file store), "wandb", or "none".
    """

    def __init__(self, cfg: dict, run_name: str):
        tcfg = cfg.get("tracking", {})
        self.backend = tcfg.get("backend", "mlflow")
        self.run_name = run_name
        self._run = None
        if self.backend == "mlflow":
            try:
                import mlflow
            except ImportError:
                log.warning("mlflow not installed; tracking disabled")
                self.backend = "none"
                return
            from src.config import resolve

            mlflow.set_tracking_uri(tcfg.get("uri") or "sqlite:///" + str(resolve("mlflow.db")))
            mlflow.set_experiment(tcfg.get("experiment", cfg.get("name", "volsent")))
            self._mlflow = mlflow
        elif self.backend == "wandb":
            import wandb

            self._wandb = wandb

    @contextmanager
    def run(self, params: dict | None = None):
        if self.backend == "mlflow":
            with self._mlflow.start_run(run_name=self.run_name):
                if params:
                    self._mlflow.log_params(_flatten(params))
                yield self
        elif self.backend == "wandb":
            self._run = self._wandb.init(name=self.run_name, config=params, reinit=True)
            try:
                yield self
            finally:
                self._run.finish()
        else:
            yield self

    def log_metrics(self, metrics: dict, step: int | None = None) -> None:
        metrics = {k: float(v) for k, v in metrics.items() if v is not None and np.isfinite(v)}
        if self.backend == "mlflow":
            self._mlflow.log_metrics(metrics, step=step)
        elif self.backend == "wandb" and self._run is not None:
            self._run.log(metrics, step=step)

    def log_artifact(self, path: str | Path) -> None:
        if self.backend == "mlflow":
            self._mlflow.log_artifact(str(path))


def _flatten(d: dict, prefix: str = "") -> dict:
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_flatten(v, key + "."))
        else:
            s = json.dumps(v) if isinstance(v, (list, tuple)) else str(v)
            out[key] = s[:250]
    return out
