"""Per-catchment HBV (LocalHBV) and the batched DDS calibrator."""
import math

import torch

from gbhydro.calibrate import dds, dds_propose, nse_per_row, simulate_q
from gbhydro.models.hbv import PARAMS, LocalHBV, RegionalHBV, hbv_step
from tests.test_hbv import synthetic


def test_local_matches_regional_with_same_parameters():
    torch.manual_seed(0)
    reg = RegionalHBV(33)
    with torch.no_grad():
        reg.pnet.head.weight.normal_(0, 2)
    b = synthetic(B=5, L=80)
    b["rows"] = torch.tensor([3, 0, 4, 1, 2])
    k = reg.pnet(b["x_s"])
    loc = LocalHBV(5)
    with torch.no_grad():                       # write the regional parameters into the table, in unit space
        z = torch.stack([k[p] for p in PARAMS], 1)
        loc.u[b["rows"]] = (z - loc.lo) / (loc.hi - loc.lo)
    out_r, out_l = reg(b), loc(b)
    assert torch.allclose(out_r["q_routed"], out_l["q_routed"], atol=1e-4)
    for p in PARAMS:
        assert torch.allclose(out_r["params"][p], out_l["params"][p], atol=1e-4), p
    assert torch.allclose(simulate_q(hbv_step, out_l["params"], b["p"], b["t"], b["pet"]), out_l["q_routed"], atol=1e-4)
    assert all(not q.requires_grad for q in loc.parameters())


def test_dds_proposal_stays_in_bounds_and_perturbs_something():
    gen = torch.Generator().manual_seed(0)
    u = torch.rand(50, 14, generator=gen)
    for i in (1, 500, 999, 1000):
        v = dds_propose(u, i, 1000, gen)
        assert v.shape == u.shape and (v >= 0).all() and (v <= 1).all()
        assert ((v != u).sum(1) >= 1).all()
    late = (dds_propose(u, 999, 1000, gen) != u).float().sum(1).mean()
    early = (dds_propose(u, 2, 1000, gen) != u).float().sum(1).mean()
    assert late < early                           # fewer dimensions perturbed as the search matures


def test_dds_maximises_and_never_worsens():
    target = torch.tensor([[0.2, 0.8, 0.5]]).repeat(20, 1)

    def objective(u):
        f = -((u - target) ** 2).sum(1)
        f[0] = float("nan")                       # chain 0 always fails: its start must survive untouched
        return f

    gen = torch.Generator().manual_seed(1)
    u0 = torch.rand(20, 3, generator=gen)
    seen = []
    u, f, trace = dds(objective, u0, 300, gen, log=lambda i, fb, better: seen.append(fb[1:].clone()))
    assert torch.equal(u[0], u0[0]) and math.isinf(f[0])
    assert (f[1:] > -0.01).all(), f
    for a, b in zip(seen[:-1], seen[1:]):
        assert (b >= a).all()
    assert len(trace) == 300


def test_nse_per_row_masks_and_flags():
    obs = torch.tensor([[1.0, 2.0, 3.0, 4.0], [1.0, 1.0, 1.0, 1.0], [5.0, float("nan"), 7.0, 9.0]])
    sim = torch.tensor([[1.0, 2.0, 3.0, 4.0], [1.0, 1.0, 1.0, 1.0], [5.0, 100.0, 7.0, 9.0]])
    mask = torch.isfinite(obs)
    nse = nse_per_row(sim, obs, mask)
    assert nse[0] == 1.0 and torch.isnan(nse[1]) and nse[2] == 1.0
