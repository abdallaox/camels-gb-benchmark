"""Metrics sanity and trainer resume behaviour."""
import numpy as np
import pytest

from gbhydro.dataset import PROCESSED
from gbhydro.metrics import fhv, flv, kge, nse


def test_metrics_perfect_and_biased():
    obs = np.abs(np.random.default_rng(0).normal(2, 1, 400))
    assert nse(obs, obs) == pytest.approx(1.0)
    assert kge(obs, obs) == pytest.approx(1.0)
    assert fhv(obs, obs) == pytest.approx(0.0, abs=1e-6)
    assert flv(obs, obs) == pytest.approx(0.0, abs=1e-6)
    assert nse(obs * 1.5, obs) < 1 and fhv(obs * 1.5, obs) > 0        # overestimating peaks -> positive FHV
    assert nse(np.full_like(obs, obs.mean()), obs) == pytest.approx(0.0)
    assert np.isnan(nse(obs[:1], obs[:1]))


def test_metrics_ignore_missing_obs():
    obs = np.abs(np.random.default_rng(1).normal(2, 1, 400))
    sim = obs.copy()
    obs_nan = obs.copy(); obs_nan[::7] = np.nan
    assert nse(sim, obs_nan) == pytest.approx(1.0)


@pytest.mark.skipif(not PROCESSED.exists(), reason="run scripts/data/prepare_data.py first")
def test_trainer_resumes_from_last_checkpoint(tmp_path):
    from gbhydro.dataset import Study
    from gbhydro.models import build
    from gbhydro.study import STUDY
    from gbhydro.train import Trainer
    study = Study()
    cfg = dict(STUDY["training"], epochs=2, validate_n_basins=3)
    model = build("lstm", study)
    tr = Trainer(study, model, tmp_path / "r", cfg, seed=0, max_batches=2, log=lambda *_: None)
    tr.fit()
    assert tr.epoch == 2 and (tmp_path / "r" / "last.pt").exists()
    cfg3 = dict(cfg, epochs=3)
    tr2 = Trainer(study, build("lstm", study), tmp_path / "r", cfg3, seed=0, max_batches=2, log=lambda *_: None)
    assert tr2.epoch == 2                                   # resumed, not restarted
    tr2.fit()
    assert tr2.epoch == 3
    log = (tmp_path / "r" / "log.csv").read_text().strip().splitlines()
    assert len(log) == 4                                    # header + 3 epochs, no duplicates


def test_drift_penalty_charges_sinks_and_is_zero_at_steady_state():
    """The storage-drift penalty is zero when a chunk ends with the water it started with, grows with the square of
    the net gain relative to precipitation, and is the same for [B] (HBV) and [B, n] (gated) state layouts."""
    import torch
    from gbhydro.train import Trainer
    p = torch.full((2, 365), 3.0)                                       # 1095 mm of rain per basin
    s_in = {"a": torch.tensor([100.0, 100.0]), "b": torch.tensor([[10.0, 10.0], [10.0, 10.0]])}
    steady = {"a": torch.tensor([90.0, 110.0]), "b": torch.tensor([[20.0, 10.0], [5.0, 5.0]])}   # totals unchanged
    sink = {"a": torch.tensor([100.0, 100.0]), "b": torch.tensor([[10.0, 10.0], [120.0, 120.0]])}  # basin 2 gains 220 mm
    pen = Trainer._drift_penalty
    zero = pen(None, s_in, steady, p)
    assert torch.allclose(zero, torch.zeros(()), atol=1e-8)
    expected = 0.5 * (220.0 / (1095.0 + 1.0)) ** 2                      # mean over the two basins
    assert pen(None, s_in, sink, p).item() == pytest.approx(expected, rel=1e-5)
    double = {"a": torch.tensor([100.0, 100.0]), "b": torch.tensor([[10.0, 10.0], [230.0, 230.0]])}
    assert pen(None, s_in, double, p).item() == pytest.approx(4 * expected, rel=1e-5)
