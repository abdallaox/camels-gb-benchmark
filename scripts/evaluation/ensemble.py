"""Five-seed ensemble mean (continuous evaluation) for an HBV-structured model family: average the simulated flow of
the seeds day by day, score the test period. Writes results/<prefix>ensemble/metrics_test.csv and the per-catchment
ensemble flow for the test period as results/<prefix>ensemble/sim_test.npz (for hydrograph figures).
Usage: uv run python scripts/evaluation/ensemble.py hbvc_b128_s   (-> results/hbvc_b128_ensemble)   |  gatedc3_n16_s  |  lstm_s"""
import sys

import numpy as np
import pandas as pd
import torch

from gbhydro.continuous import simulate_continuous
from gbhydro.data import ROOT
from gbhydro.dataset import Study
from gbhydro.metrics import ALL
from gbhydro.models import build, load_run

prefix = sys.argv[1]; kind = "gated" if "gated" in prefix else "hbv"
torch.set_num_threads(4)
study = Study(inputs="matched", device="cpu"); rows = np.arange(len(study.basins)); a, z = study.periods["test"]
q = None
for s in range(5):
    m = load_run(f"{prefix}{s}", study)
    sim = simulate_continuous(study, m, rows, batch_size=256)
    q = sim["q"] if q is None else q + sim["q"]
    print(f"  seed {s} done", flush=True)
q /= 5; obs = sim["obs"]
held = set(int(b) for b in open(ROOT / "config/basins_spatial_test.txt").read().split())
d = pd.DataFrame(index=pd.Index([int(b) for b in study.basins], name="basin"))
for name, fn in ALL.items():
    d[name] = [fn(q[i, a:z + 1], obs[i, a:z + 1]) for i in rows]
d["set"] = ["held-out" if b in held else "trained" for b in d.index]
out = ROOT / "results" / f"{prefix.rstrip('_s')}_ensemble"; out.mkdir(exist_ok=True)
d.to_csv(out / "metrics_test.csv")
np.savez_compressed(out / "sim_test.npz", q=q[:, a:z + 1].astype(np.float32), obs=obs[:, a:z + 1].astype(np.float32), basins=study.basins, dates=study.dates[a:z + 1].values)
print(f"{prefix} ensemble: NSE median trained {d[d.set == 'trained'].NSE.median():.3f} held-out {d[d.set == 'held-out'].NSE.median():.3f} | KGE {d[d.set == 'trained'].KGE.median():.3f} / {d[d.set == 'held-out'].KGE.median():.3f} | mean NSE clip {d.NSE.clip(-1).mean():.3f}")
