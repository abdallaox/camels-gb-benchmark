"""Per-catchment calibration of the HBV core by dynamically dimensioned search (DDS, Tolson and Shoemaker 2007),
batched over catchments and restarts: every chain (basin x restart) is an independent DDS search, and one
evaluation runs all chains through the calibration period in a single batched pass of the day step.

    u, f, trace = dds(objective, u0, iters, gen)      # u in [0, 1]^(chains x dims), f = objective (maximised)

Objective here: NSE of the continuously simulated discharge over the training period (first target day onwards),
per chain, which is the per-basin equivalent of the NSE* loss the learned models minimise (per basin the two differ
by a constant). The simulation starts from the fixed initial stores on the first day of the record, as the
RPL-HBV does, so the calibration period includes the same one-year fill from the cold start.
"""
import math

import torch

from gbhydro.models.hbv import INITIAL, route


def dds_propose(u_best: torch.Tensor, i: int, m: int, gen: torch.Generator, r: float = 0.2) -> torch.Tensor:
    """One DDS proposal for every chain. u_best [C, D] in [0, 1]. Iteration i of m (1-based): each dimension is
    perturbed with probability 1 - ln(i)/ln(m) (at least one), by N(0, r) in unit space, reflected at the bounds."""
    C, D = u_best.shape
    p = 1.0 - math.log(i) / math.log(m) if m > 1 else 1.0
    sel = torch.rand(C, D, generator=gen, device=u_best.device) < p
    none = ~sel.any(1)
    if none.any():                                             # DDS: perturb one random dimension at least
        j = torch.randint(0, D, (int(none.sum()),), generator=gen, device=u_best.device)
        sel[none.nonzero(as_tuple=True)[0], j] = True
    u = u_best + sel * r * torch.randn(C, D, generator=gen, device=u_best.device)
    u = torch.where(u < 0, -u, u)
    u = torch.where(u > 1, 2 - u, u)
    return u.clamp(0.0, 1.0)                                    # still outside after reflection: set to the bound


def dds(objective, u0: torch.Tensor, iters: int, gen: torch.Generator, start_iter: int = 0, f0=None, log=None,
        checkpoint=None):
    """Greedy DDS maximisation from u0 [C, D]. objective(u) -> [C] (NaN = failed, never accepted).
    Returns (u_best, f_best, trace) with trace = list of (iteration, median f_best). Resume with start_iter/f0."""
    u_best = u0.clone()
    f_best = objective(u_best) if f0 is None else f0.clone()
    f_best = torch.nan_to_num(f_best, nan=-float("inf"))
    trace = []
    for i in range(start_iter + 1, iters + 1):
        u = dds_propose(u_best, i, iters, gen)
        f = torch.nan_to_num(objective(u), nan=-float("inf"))
        better = f > f_best
        u_best = torch.where(better[:, None], u, u_best)
        f_best = torch.where(better, f, f_best)
        trace.append((i, float(f_best.median())))
        if log:
            log(i, f_best, better)
        if checkpoint and (i % 50 == 0 or i == iters):
            checkpoint(i, u_best, f_best)
    return u_best, f_best, trace


@torch.no_grad()
def simulate_q(step, k: dict, p: torch.Tensor, t: torch.Tensor, pet: torch.Tensor) -> torch.Tensor:
    """Routed discharge [C, L] of the HBV core from the fixed initial stores, parameters k (dict of [C])."""
    C, L = p.shape
    s = {n: torch.full((C,), v, device=p.device, dtype=p.dtype) for n, v in INITIAL.items()}
    q = []
    for i in range(L):
        s, f = step(p[:, i], t[:, i], pet[:, i], s, k)
        q.append(f["q"])
    return route(torch.stack(q, 1), k["MAXBAS"])


def nse_per_row(sim: torch.Tensor, obs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """NSE per row over the masked days; NaN where fewer than two observations or zero variance."""
    n = mask.sum(1)
    o = torch.where(mask, obs, torch.zeros_like(obs))
    mean = o.sum(1) / n.clamp(min=1)
    ss_res = torch.where(mask, (sim - obs) ** 2, torch.zeros_like(obs)).sum(1)
    ss_tot = torch.where(mask, (obs - mean[:, None]) ** 2, torch.zeros_like(obs)).sum(1)
    nse = 1 - ss_res / ss_tot
    return torch.where((n >= 2) & (ss_tot > 0), nse, torch.full_like(nse, float("nan")))
