"""Run trained weights once, continuously, over the whole record (evaluation and all plots use this).

    sim = simulate_continuous(study, model, rows)       # dict of [n_basins, T] arrays: q, stores, fluxes
    err = cold_start_error(study, model, rows, days)    # continuous-run store minus windowed store, same day

For the HBV models the recurrence is the model's own day step, started once from the standard initial stores
on the first day of the record (1970-10-01) and never reset. For the LSTM the sequence is simply the full
record, which is what NeuralHydrology's sequence-to-sequence evaluation would do.
Windowed (training-style) simulation is Trainer.simulate; the difference between the two on the same day is
the cold-start store error reported per catchment.
"""
import numpy as np
import torch

from gbhydro.models.gated_hbv import GatedHBV
from gbhydro.models.hbv import RegionalHBV


@torch.no_grad()
def simulate_continuous(study, model, rows: np.ndarray, batch_size: int = 128, units_days: np.ndarray | None = None) -> dict:
    """Continuous simulation for basin rows over the full record. Returns numpy arrays [len(rows), T]
    under keys 'q' (simulated discharge, routed for the RPL-HBV), 'obs', plus every store and flux series
    the model exposes ('stores/<name>', 'fluxes/<name>'). The first 365 days are the spin-up and are kept
    (callers decide what to exclude). units_days: for the gated model, also return the per-unit stores on these
    day indices as 'units/<store>' [len(rows), len(units_days), n] (the full unit series would be too large)."""
    model.eval()
    T = study.target.shape[1]
    out = {}
    for i in range(0, len(rows), batch_size):
        r = torch.as_tensor(rows[i:i + batch_size], device=study.device)
        batch = {"x_d": study.x_d[r], "x_s": study.x_s[r], "p": study._raw["precipitation"][r],
                 "t": study._raw["temperature"][r], "pet": study._raw["pet"][r], "rows": r}
        res = _run(model, batch, units_days)
        for k, v in res.items():
            out.setdefault(k, []).append(v.float().cpu().numpy())
    out = {k: np.concatenate(v) for k, v in out.items()}
    out["obs"] = study.target[torch.as_tensor(rows, device=study.device)].cpu().numpy()
    assert out["q"].shape == (len(rows), T)
    return out


def _run(model, batch, units_days=None) -> dict:
    if isinstance(model, (RegionalHBV, GatedHBV)):
        r = model(batch)
        res = {"q": r["q_routed"] if "q_routed" in r else r["fluxes"]["q"]}
        res.update({f"stores/{k}": v for k, v in r["stores"].items()})
        res.update({f"fluxes/{k}": v for k, v in r["fluxes"].items()})
        if units_days is not None and "units" in r:
            d = torch.as_tensor(np.asarray(units_days), device=batch["p"].device)
            res.update({f"units/{k}": v[:, d] for k, v in r["units"].items()})
        return res
    # LSTM: full-sequence pass, prediction for every day
    x = torch.cat([batch["x_d"], batch["x_s"][:, None, :].expand(-1, batch["x_d"].shape[1], -1)], dim=-1)
    h, _ = model.lstm(x)
    return {"q": model.head(model.dropout(h)).squeeze(-1)}


@torch.no_grad()
def cold_start_error(study, model, rows: np.ndarray, days: np.ndarray, continuous: dict | None = None,
                     batch_size: int = 1024) -> dict:
    """For HBV models: store level from the continuous run minus the store level at the end of a training-style
    365-day window ending the same day, for every (row, day) pair. Returns {store: array [n_pairs]} in mm,
    plus 'basin_row' and 'day'. Positive = the windowed run under-fills the store (cold start)."""
    cont = continuous or simulate_continuous(study, model, rows)
    pos = {r: i for i, r in enumerate(rows)}
    b = np.repeat(rows, len(days)); d = np.tile(days, len(rows))
    err = {}
    for i in range(0, len(b), batch_size):
        w = study.window(b[i:i + batch_size], d[i:i + batch_size])
        st = model(w)["stores"]
        for k, v in st.items():
            windowed = v[:, -1].float().cpu().numpy()
            ci = np.array([pos[x] for x in b[i:i + batch_size]])
            continuous_level = cont[f"stores/{k}"][ci, d[i:i + batch_size]]
            err.setdefault(k, []).append(continuous_level - windowed)
    err = {k: np.concatenate(v) for k, v in err.items()}
    err["basin_row"], err["day"] = b, d
    return err
