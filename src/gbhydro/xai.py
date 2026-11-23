"""Explainable-AI layer shared by the three models: integrated gradients of the last day's flow with respect to the
raw forcing of the preceding L days (P, T, PET in mm/d, degC, mm/d) and to the standardised static attributes.

    ig = integrated_gradients(model, study, rows, days)     # dict of arrays, see below

Every model is driven through the same function of the raw forcing: the standardised x_d is recomputed from the raw
window inside the graph so that gradients reach the raw values, whichever representation the model uses. The baseline
is the same for all models: no rain, temperature and PET at their training-period means (constant), and the mean
training catchment (attributes = 0 standardised). The HBV-structured models start the window from the stores of the
continuous run on the day before the window (passed as `states`), held fixed along the path; the LSTM starts from
zeros as in training. Attributions sum to f(x) - f(baseline) (completeness), checked and returned.

Outputs (numpy):
    'forcing'   [n, L, 3]   attribution to P, T, PET at each day of the window (index L-1 = the target day)
    'static'    [n, n_attr] attribution to each attribute
    'f', 'f0'   [n]         model output at the input and at the baseline (mm/d)
    'lag_mass'  [n, L]      |forcing attribution| summed over the 3 inputs, by lag (0 = target day)
"""
import numpy as np
import torch

FORCING = ("precipitation", "temperature", "pet")


def _forward(model, study, raw: torch.Tensor, x_s: torch.Tensor, state):
    """raw [B, L, 3] (P, T, PET in physical units) -> last-day flow [B]."""
    j = [study.dyn_names.index(v) for v in FORCING]
    x_d = (raw - study.dyn_mean[j]) / study.dyn_std[j]
    batch = {"x_d": x_d, "x_s": x_s, "p": raw[..., 0], "t": raw[..., 1], "pet": raw[..., 2]}
    out = model(batch, state) if state is not None else model(batch)
    return out["y_hat"]


def integrated_gradients(model, study, rows: np.ndarray, days: np.ndarray, states: dict | None = None,
                         L: int | None = None, steps: int = 32, alpha_batch: int = 8) -> dict:
    """rows, days: equal-length arrays of (basin row, target day index). states: model state dict for the day before
    each window ([n] or [n, units] tensors), or None (LSTM / cold start)."""
    model.eval()
    L = L or study.seq_length
    n = len(rows); dev = study.device
    j = [study.dyn_names.index(v) for v in FORCING]
    r = torch.as_tensor(rows, device=dev); t = torch.as_tensor(days, device=dev)
    idx = t[:, None] - torch.arange(L - 1, -1, -1, device=dev)[None, :]
    bb = r[:, None].expand_as(idx)
    raw = torch.stack([study._raw[v][bb, idx] for v in FORCING], -1)                 # [n, L, 3]
    x_s = study.x_s[r]
    base = torch.zeros_like(raw); base[..., 1] = study.dyn_mean[j[1]]; base[..., 2] = study.dyn_mean[j[2]]
    base_s = torch.zeros_like(x_s)
    alphas = (torch.arange(steps, device=dev, dtype=raw.dtype) + 1) / steps           # right Riemann sum, as Captum
    grad_raw, grad_s = torch.zeros_like(raw), torch.zeros_like(x_s)
    for a0 in range(0, steps, alpha_batch):
        al = alphas[a0:a0 + alpha_batch]; k = len(al)
        xr = (base[None] + al[:, None, None, None] * (raw - base)[None]).reshape(k * n, L, 3).requires_grad_(True)
        xs = (base_s[None] + al[:, None, None] * (x_s - base_s)[None]).reshape(k * n, -1).requires_grad_(True)
        st = None if states is None else {kk: v.repeat(k, *([1] * (v.dim() - 1))) for kk, v in states.items()}
        y = _forward(model, study, xr, xs, st)
        gr, gs = torch.autograd.grad(y.sum(), [xr, xs])
        grad_raw += gr.reshape(k, n, L, 3).sum(0); grad_s += gs.reshape(k, n, -1).sum(0)
    ig_raw = (raw - base) * grad_raw / steps; ig_s = (x_s - base_s) * grad_s / steps
    with torch.no_grad():
        f = _forward(model, study, raw, x_s, states); f0 = _forward(model, study, base, base_s, states)
    out = {"forcing": ig_raw.cpu().numpy(), "static": ig_s.cpu().numpy(), "f": f.cpu().numpy(), "f0": f0.cpu().numpy()}
    out["lag_mass"] = np.abs(out["forcing"]).sum(-1)[:, ::-1]                        # lag 0 = target day
    out["completeness_err"] = out["forcing"].sum((1, 2)) + out["static"].sum(1) - (out["f"] - out["f0"])
    return out


def memory_days(lag_mass: np.ndarray, share: float = 0.9) -> np.ndarray:
    """Lag (days) by which `share` of the in-window attribution mass is reached, per row."""
    c = np.cumsum(lag_mass, 1) / np.maximum(lag_mass.sum(1, keepdims=True), 1e-12)
    return (c < share).sum(1)
