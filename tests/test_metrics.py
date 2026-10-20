import numpy as np

from src.metrics import diebold_mariano, newey_west_var, qlike_vec, smearing_factor


def test_qlike_zero_at_truth_and_asymmetric():
    assert np.allclose(qlike_vec([1.0, 2.0], [1.0, 2.0]), 0.0)
    under = qlike_vec([1.0], [0.5])[0]
    over = qlike_vec([1.0], [2.0])[0]
    assert under > over > 0  # under-prediction is penalized more


def test_qlike_minimized_by_conditional_mean():
    rng = np.random.default_rng(0)
    rv2 = rng.chisquare(5, 20000) / 5
    grid = np.linspace(0.5, 1.5, 101)
    losses = [qlike_vec(rv2, np.full_like(rv2, h)).mean() for h in grid]
    assert abs(grid[int(np.argmin(losses))] - rv2.mean()) < 0.03


def test_dm_detects_better_model():
    rng = np.random.default_rng(0)
    base = rng.normal(1, 0.3, 800) ** 2
    better = base * 0.8
    r = diebold_mariano(better, base, h=5)
    assert r["dm_stat"] < 0 and r["p_value"] < 0.01


def test_dm_null_not_rejected_often():
    rng = np.random.default_rng(1)
    rej = 0
    for _ in range(200):
        a = rng.normal(0, 1, 300) ** 2
        b = rng.normal(0, 1, 300) ** 2
        rej += diebold_mariano(a, b, h=1)["p_value"] < 0.05
    assert rej / 200 < 0.12


def test_newey_west_iid_close_to_var():
    x = np.random.default_rng(2).normal(size=20000)
    assert abs(newey_west_var(x, 4) - 1.0) < 0.05


def test_smearing_unbiased_variance():
    rng = np.random.default_rng(3)
    e = rng.normal(0, 0.4, 100000)
    s = smearing_factor(e, np.zeros_like(e))
    assert abs(s - np.exp(2 * 0.4 ** 2)) < 0.02
