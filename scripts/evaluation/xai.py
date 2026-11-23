"""Integrated gradients for one trained run over all catchments.
For each catchment, attribution of the flow on `n_days` target days spread over the test period (the 15th of every
k-th month) to the preceding 365 days of P, T, PET and to the static attributes. The HBV-structured models start each
window from the continuous run's stores. Writes results/<run>/ig.npz (per-catchment mean profiles) and ig_summary.csv.
Usage: uv run python scripts/evaluation/xai.py --run gatedc3_n16_s0 [--n-days 24] [--steps 32] [--basins 0] [--batch 64]"""
import argparse
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from gbhydro.continuous import simulate_continuous
from gbhydro.data import ROOT
from gbhydro.dataset import Study
from gbhydro.models import build, load_run
from gbhydro.xai import FORCING, integrated_gradients, memory_days

ap = argparse.ArgumentParser()
ap.add_argument("--run", required=True); ap.add_argument("--n-days", type=int, default=24); ap.add_argument("--steps", type=int, default=32)
ap.add_argument("--basins", type=int, default=0, help="0 = all"); ap.add_argument("--batch", type=int, default=64); ap.add_argument("--alpha-batch", type=int, default=8)
a = ap.parse_args()
device = os.environ.get("GBHYDRO_DEVICE", "cpu"); run_dir = Path(os.environ.get("GBHYDRO_RUN_DIR", ROOT / "runs")) / a.run
study = Study(inputs="matched", device=device)
kind = "gated" if "gated" in a.run else ("hbv" if "hbv" in a.run else "lstm")
model = load_run(a.run, study, **({"compile_step": False} if kind != "lstm" else {}))
rows_all = np.arange(len(study.basins)) if a.basins == 0 else np.arange(a.basins)
t0, t1 = study.periods["test"]
cand = [i for i in range(t0, t1 + 1) if study.dates[i].day == 15]
days = np.array(cand[:: max(1, len(cand) // a.n_days)][: a.n_days])
print(f"{a.run}: {len(rows_all)} basins x {len(days)} days ({study.dates[days[0]].date()} .. {study.dates[days[-1]].date()}), {a.steps} steps, device {device}", flush=True)

L = study.seq_length
store_keys, unit_keys, full = [], [], None
if kind != "lstm":
    sim_model = build(kind, study, **model.kw); sim_model.load_state_dict(model.state_dict())
    t_start = time.time()
    full = simulate_continuous(study, sim_model, rows_all, batch_size=256, units_days=days - L)   # stores for the window start states
    store_keys = [k for k in full if k.startswith("stores/")]
    unit_keys = [k for k in full if k.startswith("units/")]
    print(f"  continuous run for start states in {time.time() - t_start:.0f}s", flush=True)

n_attr = study.x_s.shape[1]
acc = {"forcing": np.zeros((len(rows_all), L, 3)), "static": np.zeros((len(rows_all), n_attr)), "lag_mass": np.zeros((len(rows_all), L)),
       "f": np.zeros(len(rows_all)), "f0": np.zeros(len(rows_all)), "abs_forcing": np.zeros((len(rows_all), 3)), "abs_static": np.zeros((len(rows_all), n_attr)),
       "mem50": np.zeros(len(rows_all)), "mem90": np.zeros(len(rows_all)), "comp_err": np.zeros(len(rows_all))}
t_start = time.time()
for d in days:
    for b0 in range(0, len(rows_all), a.batch):
        rr = rows_all[b0:b0 + a.batch]; dd = np.full(len(rr), d)
        states = None
        if kind != "lstm":
            day0 = d - L                                                           # stores at the end of the day before the window
            if kind == "gated":                                                    # per-unit stores from the continuous run
                di = int(np.where(days == d)[0][0])
                states = {k.split("/")[1]: torch.as_tensor(full[k][rr, di], device=device, dtype=torch.float32) for k in unit_keys}
            else:
                states = {k.split("/")[1]: torch.as_tensor(full[k][rr, day0], device=device, dtype=torch.float32) for k in store_keys}
        ig = integrated_gradients(model, study, rr, dd, states=states, steps=a.steps, alpha_batch=a.alpha_batch)
        sl = slice(b0, b0 + len(rr))
        acc["forcing"][sl] += ig["forcing"]; acc["static"][sl] += ig["static"]; acc["lag_mass"][sl] += ig["lag_mass"]
        acc["f"][sl] += ig["f"]; acc["f0"][sl] += ig["f0"]; acc["comp_err"][sl] += np.abs(ig["completeness_err"])
        acc["abs_forcing"][sl] += np.abs(ig["forcing"]).sum(1); acc["abs_static"][sl] += np.abs(ig["static"])
        acc["mem50"][sl] += memory_days(ig["lag_mass"], 0.5); acc["mem90"][sl] += memory_days(ig["lag_mass"], 0.9)
    print(f"  day {study.dates[d].date()} done, {time.time() - t_start:.0f}s", flush=True)
nd = len(days)
out = {k: v / nd for k, v in acc.items()}
out["basins"] = study.basins[rows_all]; out["days"] = days; out["attr_names"] = np.array(study.attr_names); out["forcing_names"] = np.array(FORCING)
res_dir = ROOT / "results" / a.run; res_dir.mkdir(parents=True, exist_ok=True)
np.savez_compressed(res_dir / "ig.npz", **out)
held = set(int(b) for b in open(ROOT / "config/basins_spatial_test.txt").read().split())
s = pd.DataFrame(index=pd.Index([int(b) for b in out["basins"]], name="basin"))
s["set"] = ["held-out" if b in held else "trained" for b in s.index]
tot = out["abs_forcing"].sum(1) + out["abs_static"].sum(1)
for i, v in enumerate(FORCING):
    s[f"share_{v}"] = out["abs_forcing"][:, i] / tot
s["share_static"] = out["abs_static"].sum(1) / tot
s["mem50_days"], s["mem90_days"] = out["mem50"], out["mem90"]
s["f_mean"], s["f0_mean"], s["comp_err_mean"] = out["f"], out["f0"], out["comp_err"]
top = np.argsort(-out["abs_static"], 1)[:, :3]
s["top_attrs"] = [",".join(study.attr_names[j] for j in row) for row in top]
s.to_csv(res_dir / "ig_summary.csv")
print(s.drop(columns="top_attrs").describe().round(3).to_string())
print("mean |static| share by attribute (top 10):")
print(pd.Series(out["abs_static"].mean(0) / tot.mean(), index=study.attr_names).sort_values(ascending=False).head(10).round(3).to_string())
