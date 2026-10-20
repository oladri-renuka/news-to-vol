import numpy as np
import pandas as pd

from src.config import load_config
from src.data.dataset import WindowedPanel, fold_dates, walk_forward_folds


def test_walk_forward_schedule():
    cfg = load_config("configs/base.yaml")
    folds = walk_forward_folds(cfg)
    assert [f.test_year for f in folds] == list(range(2017, 2024))
    f0 = folds[0]
    assert f0.train == (pd.Timestamp("2010-01-01"), pd.Timestamp("2015-12-31"))
    assert f0.val[0].year == 2016 and f0.test[0].year == 2017
    assert folds[-1].train[1].year == 2021 and folds[-1].val[0].year == 2022


def test_purge_and_disjoint():
    cfg = load_config("configs/base.yaml")
    cal = pd.bdate_range("2010-01-01", "2018-12-31")
    d = fold_dates(cfg, cal, walk_forward_folds(cfg, [2017])[0])
    h = cfg["target"]["horizon"]
    # The last train origin's 5-day target window must end before the first val day.
    last_train = cal.get_loc(d["train"].max())
    first_val = cal.get_loc(d["val"].min())
    assert last_train + h < first_val
    last_val = cal.get_loc(d["val"].max())
    assert last_val + h < cal.get_loc(d["test"].min())
    assert len(d["train"].intersection(d["val"])) == 0
    assert len(d["val"].intersection(d["test"])) == 0


def test_windows_only_use_past(small_panel):
    wp = WindowedPanel(small_panel, ["log_rv_d", "log_rv_w"], window=20)
    idx = wp.sample_index(wp.dates[100:200])
    i, t = idx[0]
    win = wp.X[i, t - 19:t + 1]
    expected = small_panel.xs(wp.tickers[i], level="ticker")[["log_rv_d", "log_rv_w"]].iloc[t - 19:t + 1].to_numpy()
    np.testing.assert_allclose(win, expected, rtol=1e-6)
    assert not np.isnan(win).any()


def test_scaler_fit_on_train_rows_only(small_panel):
    wp = WindowedPanel(small_panel, ["log_rv_d"], window=20)
    tr = wp.sample_index(wp.dates[100:300])
    wp.X[:, 400:, 0] += 1000.0  # contaminate the "future"
    mu, _ = wp.fit_scaler(tr)
    assert abs(mu[0]) < 100
