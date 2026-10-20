PY ?= python
.PHONY: setup data news sentiment panel baselines deep sweep ablation eval test serve bench dashboard daily all smoke

setup:            ## install dependencies
	$(PY) -m pip install -r requirements.txt

data:             ## prices (yfinance, cached Parquet)
	$(PY) -m src.data.prices

news:             ## FNSPID download + filter + 4pm-ET alignment + coverage table (~29 GB download; run on the pod)
	bash scripts/fetch_fnspid.sh

sentiment:        ## 10-fold CV vs FinBERT, final fine-tune, cached scoring of all headlines (GPU)
	$(PY) -m src.models.sentiment --stage all

panel:            ## universe selection + features + sentiment features
	$(PY) -m src.data.dataset --force

baselines:        ## V1 (EWMA, GARCH, HAR, HAR-X) + V4 (HAR + sentiment)
	$(PY) -m src.train --config configs/v1_baselines.yaml
	$(PY) -m src.train --config configs/v4_har_sent.yaml

deep:             ## V2 + V3, 3 seeds each
	for c in v2_lstm_price v2_transformer_price v3_lstm_sent v3_transformer_sent; do \
	  $(PY) -m src.train --config configs/$$c.yaml || exit 1; done

ablation:         ## V6
	for c in configs/v6_*.yaml; do $(PY) -m src.train --config $$c || exit 1; done

eval:             ## tables, DM tests, regimes (V5), MCS, figures
	$(PY) -m src.eval --config configs/eval.yaml
	$(PY) -m src.eval --config configs/eval_ablation.yaml

test:
	$(PY) -m pytest -q

daily:            ## one run of the daily batch pipeline
	$(PY) -m src.pipeline.daily

serve:
	uvicorn serve.api:app --host 0.0.0.0 --port 8000

bench:            ## CPU latency p50/p95 + model size
	CUDA_VISIBLE_DEVICES= $(PY) -m serve.benchmark

dashboard:
	streamlit run dashboard/app.py

smoke:            ## a fast end-to-end check (1 fold, 1 seed, 2 epochs)
	$(PY) -m src.train --config configs/v2_lstm_price.yaml --smoke --set tracking.backend=none

all: data news sentiment panel baselines deep ablation eval
