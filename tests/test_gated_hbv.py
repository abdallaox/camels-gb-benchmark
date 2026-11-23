"""Gated HBV: mass balance P - ET - Q - dS = 0, gate fractions in (0,1), non-negative stores, shapes, gradients,
ordered initial unit biases."""
import pytest
import torch

from gbhydro.models.gated_hbv import GATES, INITIAL, STORES, GatedHBV


def synthetic(B=5, L=90, n_dyn=3, n_static=33, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"p": torch.relu(torch.randn(B, L, generator=g) * 4 + 2), "t": torch.randn(B, L, generator=g) * 6 + 4,
            "pet": torch.relu(torch.randn(B, L, generator=g) * 0.8 + 1.5), "x_d": torch.randn(B, L, n_dyn, generator=g),
            "x_s": torch.randn(B, n_static, generator=g)}


@pytest.mark.parametrize("constrained", [True, False])
def test_mass_balance_and_nonnegative_stores(constrained):
    torch.manual_seed(0)
    m = GatedHBV(3, 33, n=6, constrained=constrained)
    with torch.no_grad():                                   # random, large weights: gates far from their start
        for p in m.parameters():
            p.add_(torch.randn_like(p) * 1.5)
    b = synthetic()
    out = m(b)
    st, fl = out["stores"], out["fluxes"]
    ds = sum(st[k][:, -1] for k in STORES) - sum(INITIAL.values())
    residual = b["p"].sum(1) - fl["et"].sum(1) - fl["q"].sum(1) - ds
    assert torch.allclose(residual, torch.zeros_like(residual), atol=1e-3), residual
    for k in STORES:
        assert (out["units"][k] >= -1e-6).all(), k
    assert (fl["q"] >= 0).all() and (fl["et"] >= 0).all()
    assert torch.allclose(fl["snow"] + fl["rain"], b["p"])


def test_shapes_and_gate_ranges():
    m = GatedHBV(3, 33, n=4)
    b = synthetic(B=3, L=20)
    out = m(b)
    assert out["y_hat"].shape == (3,) and out["stores"]["SM"].shape == (3, 20) and out["units"]["SLZ"].shape == (3, 20, 4)
    ctx = torch.cat([b["x_d"][:, 0], b["x_s"]], 1)
    for name, store in GATES.items():
        x = ctx if store is None else torch.cat([torch.rand(3, 1), ctx], 1)
        g = m.gates[name](x)
        assert g.shape == (3, 1 if store is None else 4) and (g > 0).all() and (g < 1).all(), name


def test_initial_unit_biases_are_ordered_fast_to_slow():
    m = GatedHBV(3, 33, n=8)
    for name, store in GATES.items():
        if store is None:
            continue
        bias = m.gates[name].l2.bias
        assert (bias[:-1] > bias[1:]).all(), name            # strictly decreasing: unit 1 fastest


def test_gradients_reach_every_gate():
    m = GatedHBV(3, 33, n=4)
    m(synthetic(B=2, L=30))["y_hat"].sum().backward()
    for n, p in m.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), n


def test_et_bounded_by_pet_and_snow_by_temperature():
    """Constrained taps (15 Nov 2026, ceiling moved to 2 degC on 20 Nov 2026): total ET never exceeds PET or the soil
    water available; no snow above ~4 degC, almost all precipitation can be snow well below 0 degC; the unconstrained
    variant keeps the old free fractions."""
    torch.manual_seed(2)
    m = GatedHBV(3, 33, n=4)
    with torch.no_grad():
        for p in m.parameters():
            p.add_(torch.randn_like(p) * 1.5)
    b = synthetic(B=4, L=60)
    b["t"][0] = 10.0; b["t"][1] = -5.0
    out = m(b)
    fl = out["fluxes"]
    assert (fl["et"] <= b["pet"] + 1e-5).all()
    assert (fl["snow"][0] <= 1e-6).all()                                          # warm basin (10 degC): no snow
    assert (fl["snow"][1] <= b["p"][1] + 1e-6).all() and fl["snow"][1].sum() > 0  # cold basin: snow possible
    old = GatedHBV(3, 33, n=4, snow_tt=0.0)                                       # gatedc2 ceiling: shut at +2 degC
    with torch.no_grad():
        old.gates["snow"].l2.bias.fill_(8.0); m.gates["snow"].l2.bias.fill_(8.0)  # snow gate ~1 in both
    b3 = synthetic(B=1, L=20); b3["t"][:] = 2.0; b3["p"][:] = 10.0
    assert (old(b3)["fluxes"]["snow"] < 0.03 * b3["p"]).all()                     # old: 2% of P at +2 degC
    sn = m(b3)["fluxes"]["snow"]
    assert (sn <= 0.5 * b3["p"] + 1e-3).all() and (sn > 0.45 * b3["p"]).all()        # new ceiling: 50% at TT_max = 2
    free = GatedHBV(3, 33, n=4, constrained=False)
    with torch.no_grad():
        free.gates["et"].l2.bias.fill_(6.0)                                        # ET gate ~1: ET = whole soil store
    b2 = synthetic(B=2, L=30)
    assert (free(b2)["fluxes"]["et"] > b2["pet"]).any()                           # old variant can exceed PET
