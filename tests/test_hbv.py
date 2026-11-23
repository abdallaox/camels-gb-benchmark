"""Regional HBV: mass balance closes, stores stay non-negative, parameters stay inside their ranges,
routing kernel sums to one and matches HBV_Lab's integer weights closely."""
import torch

from gbhydro.models.hbv import INITIAL, PARAMS, RANGES, STORES, RegionalHBV, routing_weights


def synthetic(B=6, L=120, seed=0):
    g = torch.Generator().manual_seed(seed)
    p = torch.relu(torch.randn(B, L, generator=g) * 4 + 2)
    t = torch.randn(B, L, generator=g) * 6 + 4
    pet = torch.relu(torch.randn(B, L, generator=g) * 0.8 + 1.5)
    x_s = torch.randn(B, 33, generator=g)
    return {"p": p, "t": t, "pet": pet, "x_s": x_s}


def test_mass_balance_closes_per_window():
    torch.manual_seed(1)
    m = RegionalHBV(33)
    with torch.no_grad():
        m.pnet.head.weight.normal_(0, 2)               # push parameters away from mid-range
    out = m(synthetic())
    st, fl = out["stores"], out["fluxes"]
    s_end = sum(st[n][:, -1] for n in STORES)
    s_start = sum(INITIAL.values())
    inflow = fl["p_eff"].sum(1)                        # rain + SFCF-corrected snowfall
    outflow = fl["et"].sum(1) + fl["q"].sum(1)
    residual = inflow - outflow - (s_end - s_start)
    assert torch.allclose(residual, torch.zeros_like(residual), atol=1e-3), residual
    for n in STORES:
        assert (st[n] >= -1e-6).all(), n
    assert (fl["q"] >= 0).all() and (fl["et"] >= 0).all()


def test_shapes_and_parameter_ranges():
    m = RegionalHBV(33)
    b = synthetic(B=4, L=50)
    out = m(b)
    assert out["y_hat"].shape == (4,) and out["q_routed"].shape == (4, 50)
    for p in PARAMS:
        lo, hi = RANGES[p]
        v = out["params"][p]
        assert v.shape == (4,) and (v > lo).all() and (v < hi).all(), p
    fresh = RegionalHBV(33)                            # zero head weights and bias -> mid-range everywhere
    with torch.no_grad():
        fresh.pnet.head.weight.zero_()
    k = fresh.pnet(b["x_s"])
    for p in PARAMS:
        mid = (RANGES[p][0] + RANGES[p][1]) / 2
        assert torch.allclose(k[p], torch.full((4,), mid), atol=1e-5), p


def test_routing_weights_sum_to_one_and_match_hbv_lab_shape():
    w = routing_weights(torch.tensor([1.0, 2.0, 3.0, 4.0, 7.3, 10.0]))
    assert torch.allclose(w.sum(1), torch.ones(6), atol=1e-6)
    assert torch.allclose(w[0, :2], torch.tensor([1.0, 0.0]), atol=1e-6)             # MAXBAS 1: no routing
    assert torch.allclose(w[1, :2], torch.tensor([0.5, 0.5]), atol=1e-6)             # MAXBAS 2: as HBV_Lab
    assert torch.allclose(w[2, :3], torch.tensor([2 / 9, 5 / 9, 2 / 9]), atol=1e-6)  # MAXBAS 3 (HBV_Lab: .25 .5 .25)
    assert (w[2, 3:] == 0).all()


def test_gradients_reach_every_parameter():
    m = RegionalHBV(33)
    m(synthetic(B=3, L=40))["y_hat"].sum().backward()
    for n, p in m.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), n


def test_continuous_mode_state_chaining_matches_one_run():
    """Running two chunks with the state carried over equals one run over both chunks (HBV and gated HBV)."""
    from gbhydro.models.gated_hbv import GatedHBV
    for m in (RegionalHBV(33), GatedHBV(3, 33, n=4)):
        g = torch.Generator().manual_seed(3)
        B, L = 3, 80
        b = {"p": torch.relu(torch.randn(B, L, generator=g) * 4 + 2), "t": torch.randn(B, L, generator=g) * 6 + 4,
             "pet": torch.relu(torch.randn(B, L, generator=g) + 1.5), "x_d": torch.randn(B, L, 3, generator=g),
             "x_s": torch.randn(B, 33, generator=g)}
        with torch.no_grad():
            full = m(b)
            first = m({k: (v[:, :40] if v.dim() > 1 and v.shape[1] == L else v) for k, v in b.items()})
            second = m({k: (v[:, 40:] if v.dim() > 1 and v.shape[1] == L else v) for k, v in b.items()}, state=first["state"])
        q_full = full["fluxes"]["q"]
        q_chain = torch.cat([first["fluxes"]["q"], second["fluxes"]["q"]], 1)
        assert torch.allclose(q_full, q_chain, atol=1e-5), type(m).__name__
        for k in full["stores"]:
            assert torch.allclose(full["stores"][k][:, -1], second["stores"][k][:, -1], atol=1e-4), (type(m).__name__, k)
