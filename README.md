# Volatility forecasting with news sentiment

**Result (2017–2023 walk-forward, 30 large caps, 3 seeds):** deep models beat HAR-RV on next-week volatility, and **news sentiment does not help**.
- The price-only LSTM's pooled QLIKE is **0.495 ± 0.004 vs 0.576** for HAR (Diebold–Mariano p = 0.002). A *linear* HAR on the same price inputs (HAR-X) recovers ~80% of that gain.
- On the 82% of ticker-days with FNSPID news coverage, adding headline sentiment makes the LSTM significantly *worse* (0.510 vs 0.499, p = 0.029), leaves the Transformer unchanged (p = 0.45), and makes HAR significantly worse (p < 0.001).
- The biggest misses are earnings and legal events whose headlines read as *neutral*. Tone is not the signal; the event is.

![Rolling QLIKE relative to HAR](reports/figures/rolling_qlike_vs_har.png)

## How to run (3 commands)

```bash
pip install -r requirements.txt
bash scripts/run_all.sh          # tests -> data -> sentiment -> V1..V6 -> eval -> benchmark -> pipeline
streamlit run dashboard/app.py
```

Individual stages are `make data | news | sentiment | panel | baselines | deep | ablation | eval | test | bench | daily | dashboard`. Every run starts from a YAML file in `configs/`. Override any key with `--set key=value`, and use `--smoke` for a 1-fold, 1-seed, 2-epoch check. On a RunPod PyTorch image, run `bash scripts/runpod_setup.sh` first.

## Question

Does daily news sentiment, added to an LSTM or Transformer, improve **5-day realized-volatility** forecasts over strong econometric baselines? And in which regimes (calm vs. stressed)? The project forecasts volatility, not returns.

## Data

| Dataset | Use | License |
|---|---|---|
| yfinance | Adjusted daily OHLCV for the universe, ^GSPC and ^VIX, 2009–2023 (2009 is feature warm-up only) | Apache-2.0 library; Yahoo data for personal/research use |
| [takala/financial_phrasebank](https://huggingface.co/datasets/takala/financial_phrasebank) | `sentences_allagree` and `sentences_75agree`, used to train and evaluate the sentiment model (downloaded automatically) | CC BY-NC-SA 3.0 |
| [FNSPID](https://huggingface.co/datasets/Zihan1004/FNSPID) | Headlines with date and ticker, from both `nasdaq_exteral_data.csv` and `All_external.csv`. 255,601 headlines for the universe | CC BY 4.0 |

No raw data is committed (`/data/` is in `.gitignore`). `make data` caches prices as Parquet. `make news` downloads FNSPID (~27 GB) and keeps only the universe headlines.

**Universe rule** (`configs/universe.yaml`): walk down the S&P 500 members on 2010-01-01, ranked by end-2009 market cap. Keep a ticker if it has continuous adjusted prices under the same symbol through 2023. Stop at 30. BRK-B is excluded because it joined the index in Feb 2010. Decisions are logged in `reports/tables/universe_selection.csv`.

**News coverage: a two-tier design.** This rule was set *after* the FNSPID coverage audit and *before* any sentiment modelling. The audit (`reports/tables/news_coverage.csv`) showed that requiring ≥20 headlines in every year would keep only 13 tickers. FNSPID has whole ticker-years with no headlines at all, e.g. MSFT before 2022, most AAPL/AMZN years, and JPM/BAC/JNJ from 2021. These are dataset gaps, not quiet periods. So:

* **Tier 1, price-only (V1, V2):** all 30 tickers, every ticker-day. News coverage does not affect the universe.
* **Tier 2, sentiment (V3, V4, V6):** a ticker-year is *covered* if it has ≥20 headlines. Sentiment comparisons are evaluated on **covered ticker-days only**, and the same rows are used for every model in the comparison (baselines and price-only models included), so the Diebold–Mariano tests are paired.
* **A gap is not "no news".** In uncovered ticker-years every sentiment feature, `has_news` included, is *missing* (NaN), never 0. `has_news = 0` means "no headline that day" only within a covered year. Deep models impute the missing values after scaling and get a `news_covered` input. HAR+sentiment is fit on covered rows only. Market-wide sentiment aggregates use covered ticker-days only.

Coverage of the test period (`reports/sentiment/tables/coverage_test_years.csv`):

| Test year | 2017 | 2018 | 2019 | 2020 | 2021 | 2022 | 2023 | Pooled |
|---|---|---|---|---|---|---|---|---|
| Covered ticker-days | 80.0% | 93.3% | 93.3% | 96.7% | 63.3% | 73.3% | 73.3% | **81.9%** (43,145 / 52,650) |
| Tickers covered (of 30) | 24 | 28 | 28 | 29 | 19 | 22 | 22 | |

**News alignment** (`src/data/news.py`, tested in `tests/test_news_alignment.py`): FNSPID timestamps are parsed as UTC and converted to New York time. A headline at or after **16:00 ET**, or on a weekend/holiday, counts for the **next** trading day. **99.8%** of universe headlines are stamped `00:00:00`, i.e. they carry a date but no time. These are treated as published *after the close*, which can delay a signal by a day but can never leak one. The timezone audit (`python -m src.data.news --audit`) found only 6 session-specific timed headlines, too few to check the "UTC" label, and with nearly every row date-only, the label barely matters.

## Target and features

* **Target:** `RV(t,t+5) = sqrt(Σ_{i=1..5} r²_{t+i})` from daily log returns. Models predict `log RV`.
* **Price features** (data through day *t* only; `tests/test_features.py` perturbs the future and checks that nothing at *t* moves): HAR daily/weekly/monthly RV, Parkinson and Garman–Klass vol (daily and 5-day), |r|, r, 60-day volume z-score, log VIX and its change, day-of-week.
* **Sentiment features:** for each ticker and day, the mean of `p_pos − p_neg`, the log headline count, the max `p_neg`, and a `has_news` flag. Also market-wide mean sentiment and count, and `news_covered`. All use exponential decay `S_t = x_t + 0.5·S_{t−1}`, so yesterday's news counts a little today. All are missing in uncovered ticker-years (see above).
* **From log forecasts to variance forecasts.** QLIKE needs a variance forecast. For log-space models it is `exp(2ŷ)·smear`, where `smear` is the Duan smearing factor estimated on the fit window (validation year for deep models), never on test data. GARCH and EWMA produce variances natively.

## Models

| Model | Notes |
|---|---|
| EWMA λ=0.94 | RiskMetrics, flat 5-day term structure |
| GARCH(1,1)-t | `arch`, one fit per ticker per walk-forward window. Test forecasts filter through each origin with fixed parameters |
| **HAR-RV** | Per-ticker OLS of log RV on log daily/weekly/monthly RV. **The bar to beat** |
| HAR-X | Same OLS on *all* price features. Not in the original spec; added to separate "better inputs" from "deep model" |
| HAR-RV + sentiment | HAR + the six sentiment features (V4), fit on covered rows. A ticker with < 250 covered fit rows (e.g. MSFT, covered only from 2022) uses a pooled regression across tickers |
| LSTM | 2 layers, hidden 64, 60-day window, 8-dim ticker embedding, dropout 0.2 |
| Transformer | 3 pre-norm encoder layers, 4 heads, d_model 64, sinusoidal positions, ticker embedding added to each token, read-out at the last token |
| Sentiment | `distilroberta-base` fine-tuned on FinancialPhraseBank (3 classes), compared with `ProsusAI/finbert` |

**Training.** Walk-forward: train 2010–2015, validate 2016, test 2017, then roll forward one year at a time to test 2023 (7 folds). The last 5 days of each train and validation window are **purged**, because their targets overlap the next window. Scalers are fit on training rows only and refit at every step. Deep models use MSE on standardized log RV, Adam 1e-3, batch 256, early stopping on the validation year (patience 5), **3 seeds**, and report mean ± std. Baselines have no hyperparameters, so they are fit on train + val. That gives them an extra, more recent year: the honest direction.

## Results

All numbers come from one run on a RunPod A40 (`reports/`). DM is a Diebold–Mariano test on the cross-sectional mean loss per date, with Newey–West variance (4 lags, since 5-day targets overlap) and the HLN correction. Deep-model DM tests use the seed-averaged forecast.

### Sentiment model (FinancialPhraseBank, 10-fold stratified CV)

| Config | Model | Accuracy | Macro-F1 |
|---|---|---|---|
| allagree (2,259 sentences) | **distilroberta, fine-tuned** | **0.977 ± 0.008** | **0.969 ± 0.012** |
| | FinBERT (off the shelf) | 0.972 ± 0.010 | 0.963 ± 0.014 |
| 75agree (3,448 sentences) | distilroberta, fine-tuned | 0.943 ± 0.012 | 0.928 ± 0.017 |
| | FinBERT (off the shelf) | 0.946 ± 0.012 | 0.936 ± 0.016 |

The two are a tie within one standard deviation. FinBERT's numbers are optimistic, because it was trained on FinancialPhraseBank, so these folds are not truly held out for it. The final model was trained on all of `75agree` and scored the 196,826 unique FNSPID headlines in about 30 seconds on the A40.

### Tier 1: price-only models (V1, V2), all 52,650 ticker-days

<!-- RESULTS:START -->
| Model | MSE log-RV | QLIKE | DM p vs HAR (MSE / QLIKE) | DM p vs HAR-X (QLIKE) | In 90% MCS (QLIKE) | seeds |
|---|---|---|---|---|---|---|
| LSTM-price | 0.2284 ± 0.0029 | **0.4945 ± 0.0038** | <0.001 / 0.002 | 0.003 | ✓ | 3 |
| Transformer-price | 0.2304 ± 0.0014 | 0.4987 ± 0.0039 | <0.001 / <0.001 | 0.084 | ✓ | 3 |
| HAR-X | 0.2354 | 0.5114 | <0.001 / 0.008 | — | | 1 |
| GARCH(1,1)-t | 0.2966 | 0.5503 | <0.001 / 0.151 | <0.001 | | 1 |
| **HAR-RV** | 0.2653 | 0.5756 | — | 0.008 | | 1 |
| EWMA | 0.2972 | 0.6007 | <0.001 / 0.257 | <0.001 | | 1 |

QLIKE by test year (lowest in bold):

| Model | 2017 | 2018 | 2019 | 2020 | 2021 | 2022 | 2023 |
|---|---|---|---|---|---|---|---|
| EWMA | 0.698 | 0.637 | 0.629 | 0.791 | 0.496 | 0.434 | 0.518 |
| GARCH | 0.601 | 0.601 | 0.550 | 0.717 | 0.454 | 0.445 | 0.483 |
| HAR | 0.611 | 0.594 | 0.548 | 0.919 | 0.449 | 0.421 | 0.483 |
| HAR-X | 0.619 | 0.562 | 0.527 | 0.599 | **0.444** | 0.370 | 0.458 |
| LSTM-price | 0.574 | 0.491 | 0.533 | **0.557** | 0.476 | 0.368 | 0.462 |
| Transformer-price | **0.552** | **0.462** | **0.518** | 0.680 | 0.490 | **0.355** | **0.431** |
<!-- RESULTS:END -->

### Tier 2: does news help? (V3, V4), covered ticker-days only

These are the same 43,145 rows for every model, with paired tests.

| Model | MSE log-RV | QLIKE | In 90% MCS (QLIKE) |
|---|---|---|---|
| LSTM-price | 0.2303 ± 0.0030 | **0.4988 ± 0.0039** | ✓ |
| Transformer + sentiment | 0.2322 ± 0.0011 | 0.5033 ± 0.0019 | ✓ |
| Transformer-price | 0.2328 ± 0.0020 | 0.5042 ± 0.0051 | ✓ |
| LSTM + sentiment | 0.2355 ± 0.0005 | 0.5100 ± 0.0045 | ✓ |
| HAR-X | 0.2384 | 0.5216 | |
| **HAR-RV** | 0.2703 | 0.5930 | |
| HAR + sentiment | 0.2894 | 0.6617 | |

**What sentiment adds**, with each model compared to itself without news:

| Pair | QLIKE without → with news | DM p (QLIKE) | DM p (MSE) | Verdict |
|---|---|---|---|---|
| LSTM | 0.4988 → 0.5100 | **0.029** | <0.001 | significantly **worse** |
| Transformer | 0.5042 → 0.5033 | 0.45 | 0.069 | no difference |
| HAR | 0.5930 → 0.6617 | **<0.001** | <0.001 | significantly **worse** |

**Is the gain from sentiment or from the deep model? (V4)** It is from the deep model and the price inputs. Every deep model beats HAR by a wide margin with or without news, HAR + sentiment loses to plain HAR, and sentiment never improves a model significantly.

### Regimes (V5), QLIKE relative to HAR (< 1 is better)

VIX is taken at the forecast origin. Tier 1 uses all ticker-days; the sentiment rows use covered days.

| Model | VIX < 20 | 20 ≤ VIX < 30 | VIX ≥ 30 | COVID crash | 2022 |
|---|---|---|---|---|---|
| GARCH | 1.001 | 1.016 | 0.671 | 0.607 | 1.056 |
| HAR-X | 0.983 | 0.894 | 0.494 | 0.413 | 0.878 |
| LSTM-price | 0.936 | 0.878 | **0.449** | **0.325** | 0.866 |
| Transformer-price | **0.899** | **0.869** | 0.659 | 0.589 | **0.831** |
| *covered days:* LSTM + sentiment | 0.927 | 0.890 | 0.512 | 0.404 | 0.855 |
| *covered days:* Transformer + sentiment | 0.888 | 0.847 | 0.628 | 0.555 | 0.807 |
| *covered days:* HAR + sentiment | 1.054 | 1.147 | 1.292 | 1.284 | 0.953 |

* **Calm markets are close to a tie:** HAR is hard to beat when nothing happens.
* **The gap opens in stress.** HAR's fixed log-linear coefficients mean-revert too fast, and any model with range-based vol and VIX reacts faster.
* **The two deep models break differently.** The LSTM is best in the COVID crash (0.33× HAR). The Transformer is best in calm and moderate markets but lags in the crash (0.59×), which makes 2020 its only losing year.
* **Sentiment hurts most in crises.** On covered days the LSTM goes from 0.33× to 0.40× HAR's loss in the COVID crash, and HAR + sentiment is 1.29× worse than HAR in stressed markets. Headline counts and negativity spike far beyond anything in training, and the models extrapolate badly.

### Ablations (V6), on the V3 sentiment models, covered days

| Change | LSTM QLIKE (p vs full) | Transformer QLIKE (p vs full) | COVID crash, ratio to HAR (LSTM / Transformer) |
|---|---|---|---|
| Full model (V3) | 0.5100 | 0.5033 | 0.40 / 0.56 |
| No VIX | 0.5327 (**0.037**) | 0.5399 (**0.009**) | **0.60 / 0.82** |
| No range-based vol | 0.5278 (**<0.001**) | 0.5226 (**<0.001**) | 0.40 / 0.60 |
| 20-day window instead of 60 | 0.5097 (0.25) | 0.5165 (**<0.001**) | 0.39 / 0.56 |

* **VIX matters most, and almost entirely in crises.** Removing it costs both models significantly, with the largest damage in the COVID crash. It is what lets a model react to a market-wide shock before realized vol catches up.
* **Range-based vol** (Parkinson, Garman–Klass) helps significantly in calm and moderate years.
* **Long memory barely matters.** A 20-day window is as good as 60 for the LSTM. The Transformer benefits slightly from the longer window.

Vol-targeting check (`reports/*/tables/vol_target.csv`, transaction costs ignored): sizing positions by deep-model forecasts cuts the share of 5-day periods where realized vol exceeds 1.5× target by about a third (7.7% vs 11.4% for HAR on covered days). They do it by forecasting slightly higher vol, so the mean realized/target ratio is 0.87–0.89 vs 0.96 for HAR.

## Failure analysis

`reports/sentiment/tables/failures_<model>.csv` lists each deep model's 20 worst forecasts by QLIKE, with HAR's forecast and that day's most negative headline.

* **The worst misses are event jumps, and their headlines read as neutral.** Both the price-only and sentiment LSTMs have the same top failures: *under*-predictions before single-stock events in calm markets (VIX 12–15).
  * **HPQ, 21–27 Feb 2019:** forecast ≈15% annualized, realized ≈135%, an earnings-driven drop. The most negative headlines that week were "HP (HPQ) to Post Q1 Earnings: What's in Store for the Stock?" and "Notable earnings after Wednesday's close", with **p_neg = 0.00**.
  * **QCOM, 11–15 Apr 2019:** forecast ≈19%, realized ≈175%. That was the Apple–Qualcomm settlement, a +23% day. Headlines beforehand were neutral, at most p_neg = 0.04.
  * **The pattern:** news *polarity* is the wrong signal for volatility. An earnings date or a pending legal ruling is a vol event whatever its tone. A scheduled-event flag (earnings calendar) or an event-type classifier is the natural next feature; tone alone is not.
* **Why HAR + sentiment fails: overfitting, not the fallback.** On its own per-ticker fits (38,376 rows), HAR + sentiment's QLIKE is 0.670 vs 0.595 for HAR on the same rows. Six extra regressors per ticker overfit, and the worst year is 2020 (1.217 vs 0.934). The pooled-fallback forecasts (4,769 rows) are nearly as good as HAR (0.594 vs 0.578).
* **Asymmetric bias.** Under-predictions make up about 42% of the deep models' QLIKE loss, against 55% for HAR. The deep models' mean log bias is close to zero (−0.01 to −0.03). GARCH and EWMA sit high (+0.18–0.20 on the full sample) because they forecast the *mean* variance, whose log exceeds the mean log-variance (Jensen). That is why they look poor on log-MSE but reasonable on QLIKE.
* **2021 is where the deep models lose.** Both are worse than HAR in 2021 (LSTM 0.476, Transformer 0.490, HAR 0.449). Early-stopped on the 2020 crash year, they over-forecast the calmer 2021 market for months, visible as the late-2021 hump in the figure. The validation year was not representative of the test year.

## Deployment

* **Daily pipeline** (`src/pipeline/daily.py`): pulls the latest prices from yfinance, refits HAR on a trailing 3-year window, runs EWMA and the price-only LSTM, and appends to SQLite (`forecasts`, `realized` and `headlines` tables). A bar for today is dropped until 16:30 ET, so an intraday partial bar is never used.
* **Scheduled run** (`.github/workflows/daily.yml`, weekdays at 22:30 UTC):
  * Downloads the trained LSTM from the [`model-latest`](https://github.com/oladri-renuka/news-to-vol/releases/tag/model-latest) release and installs only `requirements-pipeline.txt` plus CPU torch.
  * Runs the pipeline, then publishes the updated `forecasts.sqlite` to the [`forecasts`](https://github.com/oladri-renuka/news-to-vol/releases/tag/forecasts) release, so the history accumulates across runs.
  * To use it locally: `gh release download forecasts -p forecasts.sqlite -D artifacts/store`.
* **FNSPID ends in 2023.** Live headlines need a separate news source. If `data/live/headlines.csv` exists (`published_at_utc, ticker, headline`), the pipeline aligns and scores it with the same 16:00 ET rule and stores it for the dashboard. The deployed model uses price features only, which is also the better model per the results above.
* **API** (`serve/api.py`, `serve/Dockerfile`): `GET /forecast/{ticker}` and `POST /predict` (raw 60×17 feature window → 5-day vol). `.github/workflows/docker.yml` builds the image on every change to the serving code, starts the container, checks `/health`, checks that a malformed request gets a 422, and benchmarks latency inside the container.
* **Latency** (`python -m serve.benchmark`, CPU, 1 thread, 500 requests):

| Model | Params | Size | Machine | Inference p50 / p95 | Request p50 / p95 (TestClient unless noted) |
|---|---|---|---|---|---|
| LSTM | 56,881 | 0.22 MB | Docker container on a GitHub Actions runner, real HTTP via uvicorn | 0.69 / 0.70 ms | 2.9 / 3.0 ms |
| LSTM | 56,881 | 0.22 MB | RunPod x86 CPU | 0.97 / 1.89 ms | 8.5 / 12.0 ms |
| LSTM | 56,881 | 0.22 MB | Apple M4 | 4.1 / 4.9 ms | 10.8 / 15.3 ms |

  The serving image is 1.0 GB, almost all of it CPU PyTorch; the model itself is 0.22 MB.
* **Dashboard** (`dashboard/app.py`): forecast vs realized per ticker, rolling QLIKE vs HAR, per-year error table, the latest live forecasts, and the day's most negative headlines.

## Limitations

* **Survivorship bias.** Requiring a firm to trade through 2023 drops failures and acquisitions, and the end-2009 ranking is approximate (compiled from public tables, not CRSP). The universe is large, liquid US stocks only.
* **FNSPID timestamps.** 99.8% of headlines have no time of day, so the conservative next-day rule can delay real news by one day. This biases the test *against* sentiment and is the strongest caveat on the null result.
* **The sentiment results cover only part of the sample.** They hold on covered ticker-days (82%), which are not a random subset: coverage is better for some firms and years. They answer "does news help *when we have news*?".
* **Headline polarity is a coarse signal.** The model scores tone sentence by sentence and never reads article bodies or event types. The failure analysis suggests event *type* matters more than tone.
* **The VIX closes at 16:15 ET**, 15 minutes after stocks. Using the day-*t* VIX is a small look-ahead.
* **The volatility proxy** is close-to-close returns (no intraday RV), which makes QLIKE noisy for single stock-weeks.
* **CUDA vs MPS.** Deep-model numbers differ slightly between GPU backends, e.g. LSTM pooled QLIKE 0.4945 on CUDA vs 0.4946 on Apple MPS. All reported numbers come from the A40 run.
* **The vol-targeting check ignores transaction costs** and uses non-overlapping 5-day holds.
* **The deep models are tuned only by early stopping.** There is no hyperparameter search, by design (spec settings), so the comparison is fair but not maximal.

## Repo layout

```
configs/      base.yaml (shared), universe.yaml, sentiment.yaml, v1..v6 experiment configs, eval*.yaml, pipeline.yaml
src/data/     prices.py, news.py (alignment), features.py, sentiment_features.py, dataset.py (universe, splits, windows)
src/models/   baselines.py, deep.py, sentiment.py
src/train.py  walk-forward runner (baselines + deep)      src/eval.py  losses, DM, MCS, regimes, figures
src/pipeline/ daily.py                                    serve/       FastAPI, benchmark, Dockerfile
dashboard/    Streamlit app                               notebooks/   analysis only, never training
reports/      tier 1 (figures/, tables/), sentiment/ (tier 2), ablation/ (V6)
tests/        leakage and unit tests (pytest)             scripts/     runpod_setup.sh, fetch_fnspid.sh, run_all.sh
```

Runs are tracked in MLflow (`mlflow ui --backend-store-uri sqlite:///mlflow.db`). Set `tracking.backend: wandb` for Weights & Biases.

## Compute

One RunPod A40, about 2 hours end to end:

| Stage | Time |
|---|---|
| FNSPID download (27 GB) + filtering | 4 min |
| Sentiment CV (2 configs × 10 folds) + final fine-tune + scoring | 7 min |
| V1 + V4 baselines | < 1 min |
| V2 + V3 deep models (4 models × 7 folds × 3 seeds) | 42 min |
| V6 ablations (6 configs × 7 folds × 3 seeds) | 64 min |

The price-only experiments also run on a laptop (Apple M4: the LSTM ~25 s per fold, the Transformer ~1 min per fold).
