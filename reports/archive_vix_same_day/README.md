# Archived run: same-day VIX (superseded)

These are the results of the RunPod A40 run (2026-10-21) in which the VIX features at forecast origin t used
**VIX close of day t**. VIX closes at 16:15 ET, 15 minutes after the 16:00 stock close that defines the
origin, so this was a look-ahead. The ablation showed VIX matters most in crises, so the leak sat exactly
under the headline crisis results.

The current results in `reports/` use VIX from t-1 (`features.vix_lag: 1`). This archive is kept unchanged
so the before/after comparison is on record. The README reports both.
