"""Model registry: name -> constructor(study, **kwargs). Every model maps a batch dict to {"y_hat": [B], ...}.
`load_run` rebuilds a trained run with the options it was trained with (read from its config.yml), so that variants
that share a parameter set but not a structure (gated constrained / unconstrained) are never silently confused."""
import torch
import yaml

from gbhydro.data import ROOT
from gbhydro.models.gated_hbv import GatedHBV
from gbhydro.models.hbv import LocalHBV, RegionalHBV
from gbhydro.models.lstm import LSTM


def build(name: str, study, **kw):
    n_dyn, n_static = len(study.dyn_names), len(study.attr_names)
    if name == "lstm":
        m = LSTM(n_dyn, n_static, **kw)
    elif name == "hbv":                    # compiled step on GPU: 3.5x faster than the eager loop (13 Nov smoke test)
        m = RegionalHBV(n_static, **{"compile_step": study.device.type == "cuda", **kw})
    elif name == "hbv_local":              # per-catchment LC-HBV (scripts/training/calibrate_hbv.py), gauged baseline
        m = LocalHBV(len(study.basins), **{"compile_step": study.device.type == "cuda", **kw})
    elif name == "gated":
        m = GatedHBV(n_dyn, n_static, **{"compile_step": study.device.type == "cuda", **kw})
    else:
        raise ValueError(f"unknown model {name!r}")
    m.kw = {k: v for k, v in kw.items() if k != "compile_step"}      # recorded in the run's config.yml by the trainer
    return m


def run_config(run: str) -> dict:
    for d in (ROOT / "runs" / run, ROOT / "results" / run):
        if (d / "config.yml").exists():
            return yaml.safe_load((d / "config.yml").read_text())
    raise FileNotFoundError(f"no config.yml for run {run!r} under runs/ or results/")


def load_run(run: str, study, checkpoint: str = "best.pt", **extra):
    """Build the model of a run with its recorded options (plus `extra`, e.g. compile_step=False) and load its weights."""
    cfg = run_config(run)
    kw = dict(cfg.get("model_kw") or {}); kw.update(extra)
    m = build(cfg["model"], study, **kw)
    m.load_state_dict(torch.load(ROOT / "runs" / run / checkpoint, map_location=study.device)["model"])
    return m
