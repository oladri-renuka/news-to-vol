# VIX look-ahead fix: before vs after

Before: VIX close of day *t* as a feature (closes 16:15 ET, after the 16:00 origin), A40/CUDA. After: VIX close of day *t−1*, Apple M4/MPS. Backend differences alone move pooled QLIKE by about 0.0001.

## Tier 1: price-only, all ticker-days

| Model | QLIKE before | QLIKE after | DM p vs HAR (before → after) | DM p vs HAR-X (before → after) | COVID / HAR (before → after) | VIX ≥ 30 / HAR (before → after) |
|---|---|---|---|---|---|---|
| LSTM-price | 0.4945 | 0.4977 | 0.002 → 0.002 | 0.003 → <0.001 | 0.325 → 0.373 | 0.449 → 0.464 |
| Transformer-price | 0.4987 | 0.5053 | <0.001 → <0.001 | 0.084 → 0.163 | 0.589 → 0.639 | 0.659 → 0.692 |
| HAR-X | 0.5114 | 0.5142 | 0.008 → 0.011 | — → — | 0.413 → 0.428 | 0.494 → 0.496 |
| GARCH | 0.5503 | 0.5503 | 0.151 → 0.151 | <0.001 → <0.001 | 0.607 → 0.607 | 0.671 → 0.671 |
| HAR | 0.5756 | 0.5756 | — → — | 0.008 → 0.011 | 1.000 → 1.000 | 1.000 → 1.000 |
| EWMA | 0.6007 | 0.6007 | 0.257 → 0.257 | <0.001 → <0.001 | 0.678 → 0.678 | 0.717 → 0.717 |

## Tier 2: does news help? (covered ticker-days)

| Pair | ΔQLIKE from news, before | ΔQLIKE from news, after | DM p before | DM p after |
|---|---|---|---|---|
| LSTM-sent vs LSTM-price | +0.0112 | +0.0091 | 0.029 | 0.062 |
| Transformer-sent vs Transformer-price | -0.0009 | -0.0025 | 0.452 | 0.809 |
| HAR+sent vs HAR | +0.0687 | +0.0687 | <0.001 | <0.001 |

## V6: effect of removing VIX (covered ticker-days)

| Model | ΔQLIKE no-VIX, before | ΔQLIKE no-VIX, after | COVID / HAR full → no-VIX, before | after |
|---|---|---|---|---|
| LSTM | +0.0228 | +0.0228 | 0.404 → 0.598 | 0.443 → 0.600 |
| Transformer | +0.0366 | +0.0373 | 0.555 → 0.820 | 0.577 → 0.884 |
