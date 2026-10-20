#!/usr/bin/env bash
# Run the full study end to end on the pod. Each stage caches its outputs, so a rerun is safe.
#   bash scripts/run_all.sh 2>&1 | tee logs/run_all.log
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs
python -m pytest -q                                   # leakage and unit tests first
python -m src.data.prices                             # ~1 min
bash scripts/fetch_fnspid.sh                          # ~29 GB download, then filter and align
python -m src.models.sentiment --stage all            # CV vs FinBERT, final model, score headlines (GPU)
python -m src.data.dataset --force                    # universe rule, features, sentiment features
python -m src.train --config configs/v1_baselines.yaml
python -m src.train --config configs/v4_har_sent.yaml
for c in v2_lstm_price v2_transformer_price v3_lstm_sent v3_transformer_sent; do
  python -m src.train --config configs/$c.yaml
done
for c in configs/v6_*.yaml; do python -m src.train --config "$c"; done
python -m src.eval --config configs/eval.yaml
python -m src.eval --config configs/eval_ablation.yaml
python -m serve.benchmark
python -m src.pipeline.daily --offline --asof 2023-12-29
# Pack what the Mac, the dashboard and CI need (small): predictions, checkpoints, tables, the panel.
tar czf results.tar.gz artifacts/predictions artifacts/models artifacts/store reports data/processed/panel.json \
  data/interim/headlines_scored.parquet mlflow.db
echo "done -> results.tar.gz (copy it back with: runpodctl send results.tar.gz)"
