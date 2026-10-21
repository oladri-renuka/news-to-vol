import numpy as np
import pandas as pd

from src.data.features import garman_klass_var, parkinson_var, price_features, realized_vol_target

FEATURES = ["ret", "abs_ret", "log_rv_d", "log_rv_w", "log_rv_m", "log_parkinson", "log_gk",
            "log_parkinson_w", "log_gk_w", "volume_z", "log_vix", "dlog_vix"]


def test_target_matches_definition(ohlcv):
    r = np.log(ohlcv.Close).diff()
    rv = realized_vol_target(r, 5)
    t = 100
    expected = np.sqrt((r.iloc[t + 1:t + 6] ** 2).sum())
    assert np.isclose(rv.iloc[t], expected)
    assert rv.iloc[-5:].isna().all()  # no future data at the end of the sample


def test_features_do_not_look_ahead(ohlcv, vix):
    """Rewriting everything after day t must not change any feature at or before t."""
    t = 250
    base = price_features(ohlcv, vix)
    shocked = ohlcv.copy()
    shocked.iloc[t + 1:] *= 3.0
    shocked.iloc[t + 1:, shocked.columns.get_loc("Volume")] *= 10
    vix2 = vix.copy(); vix2.iloc[t + 1:] += 30
    alt = price_features(shocked, vix2)
    pd.testing.assert_frame_equal(base[FEATURES].iloc[: t + 1], alt[FEATURES].iloc[: t + 1])
    # ...while the target at t-5 (whose window ends at t) is also unchanged, and at t it changes.
    assert np.isclose(base.y.iloc[t - 5], alt.y.iloc[t - 5])


def test_target_uses_future(ohlcv, vix):
    t = 250
    base = price_features(ohlcv, vix)
    shocked = ohlcv.copy()
    shocked.iloc[t + 3:, :4] *= 1.5   # jump inside (t, t+5]
    alt = price_features(shocked, vix)
    assert not np.isclose(base.y.iloc[t], alt.y.iloc[t])


def test_range_estimators_nonnegative_and_sensible(ohlcv):
    pk = parkinson_var(ohlcv.High, ohlcv.Low)
    gk = garman_klass_var(ohlcv.Open, ohlcv.High, ohlcv.Low, ohlcv.Close)
    assert (pk >= 0).all() and (gk >= 0).all()
    r2 = np.log(ohlcv.Close).diff() ** 2
    # Range estimators should be the same order of magnitude as squared returns.
    assert 0.1 < pk.mean() / r2.mean() < 10


def test_parkinson_constant():
    # ln(H/L) = 0.02  ->  0.02^2 / (4 ln 2)
    v = parkinson_var(pd.Series([np.exp(0.02)]), pd.Series([1.0]))
    assert np.isclose(v.iloc[0], 0.0004 / (4 * np.log(2)))


def test_same_day_vix_is_not_a_feature(ohlcv, vix):
    """VIX closes at 16:15 ET, after the 16:00 forecast origin. Changing day t's VIX must not change
    any model input at t. It may only change inputs from t+1 on."""
    from src.config import load_config

    feats = [c for c in load_config("configs/base.yaml")["features"]["price"]]
    t = 250
    base = price_features(ohlcv, vix, vix_lag=1)
    v2 = vix.copy(); v2.iloc[t] *= 2.0
    alt = price_features(ohlcv, v2, vix_lag=1)
    pd.testing.assert_frame_equal(base[feats].iloc[: t + 1], alt[feats].iloc[: t + 1])
    assert not np.isclose(base.log_vix.iloc[t + 1], alt.log_vix.iloc[t + 1])


def test_same_day_vix_leak_is_detectable(ohlcv, vix):
    """Sanity check for the test above: with vix_lag=0 (the archived run), the leak shows up."""
    t = 250
    base = price_features(ohlcv, vix, vix_lag=0)
    v2 = vix.copy(); v2.iloc[t] *= 2.0
    alt = price_features(ohlcv, v2, vix_lag=0)
    assert not np.isclose(base.log_vix.iloc[t], alt.log_vix.iloc[t])
