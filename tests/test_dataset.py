"""Window alignment, normalisation provenance and loss sanity for gbhydro.dataset."""
import numpy as np
import pandas as pd
import pytest
import torch

from gbhydro.data import DATA_DIR, load_timeseries
from gbhydro.dataset import PROCESSED, Study
from gbhydro.loss import nse_star

pytestmark = pytest.mark.skipif(not PROCESSED.exists(), reason="run scripts/data/prepare_data.py first")


@pytest.fixture(scope="module")
def study():
    return Study()


@pytest.mark.skipif(not (DATA_DIR / "timeseries").exists(), reason="needs the raw CAMELS-GB time series in data/camels_gb")
def test_window_matches_raw_csv(study):
    """A model window equals the raw CAMELS-GB record: Thames at Kingston, 1995-01-15."""
    row = int(np.where(study.basins == 39001)[0][0])
    day = study.dates.get_loc(pd.Timestamp("1995-01-15"))
    w = study.window(np.array([row]), np.array([day]))
    raw = load_timeseries(39001).loc["1994-01-16":"1995-01-15"]
    assert w["x_d"].shape == (1, 365, 3)
    assert torch.allclose(w["p"][0], torch.tensor(raw["precipitation"].to_numpy(np.float32)))
    assert torch.allclose(w["t"][0], torch.tensor(raw["temperature"].to_numpy(np.float32)))
    assert abs(float(w["y"][0]) - raw["discharge_spec"].iloc[-1]) < 1e-5
    back = w["x_d"][0] * study.dyn_std + study.dyn_mean          # standardised inputs invert to raw
    assert torch.allclose(back[:, 0], w["p"][0], atol=1e-3)


def test_normalisation_uses_train_basins_and_period_only(study):
    tr, (t0, t1) = study.sets["train"], study.periods["train"]
    p = study._raw["precipitation"][tr, t0:t1 + 1]
    assert abs(float(p.mean()) - float(study.dyn_mean[0])) < 1e-3
    assert torch.isfinite(study.x_s).all()
    assert abs(float(study.x_s[tr].mean())) < 1e-4


def test_samples_have_observed_targets(study):
    b, t = study.samples("train", "train")
    assert len(b) > 1_000_000
    assert torch.isfinite(study.target[b, t]).all()
    assert t.min() >= study.seq_length - 1
    held, _ = study.samples("spatial_test", "test")
    assert not set(held) & set(study.sets["train"])


def test_nse_star_is_zero_for_perfect_prediction():
    y = torch.rand(8)
    std = torch.rand(8) + 0.5
    assert float(nse_star(y, y, std)) == 0.0
    assert float(nse_star(y + 1, y, std)) > 0
