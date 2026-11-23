"""Continuous evaluation of any run: one pass over the record from the fixed initial stores, metrics on the test
period for every catchment (the fair-comparison rule of 14 Nov: windowed and continuous arms compared on continuous
evaluation). Writes results/<run>/metrics_test_continuous.csv and prints medians next to the run's own metrics_test.csv.
Usage: uv run python scripts/evaluation/eval_continuous.py <run> [<run> ...]"""
import sys

import numpy as np
import pandas as pd
import torch

from gbhydro.continuous import simulate_continuous
from gbhydro.data import ROOT
from gbhydro.dataset import Study
from gbhydro.metrics import ALL
from gbhydro.models import build, load_run

torch.set_num_threads(4)
study = Study(inputs="matched", device="cpu")
rows = np.arange(len(study.basins)); a, z = study.periods["test"]
held = set(int(b) for b in open(ROOT / "config/basins_spatial_test.txt").read().split())
for run in sys.argv[1:]:
    kind = "gated" if "gated" in run else ("hbv" if "hbv" in run else "lstm")
    m = load_run(run, study)
    sim = simulate_continuous(study, m, rows, batch_size=256)
    d = pd.DataFrame(index=pd.Index([int(b) for b in study.basins], name="basin"))
    for name, fn in ALL.items():
        d[name] = [fn(sim["q"][i, a:z + 1], sim["obs"][i, a:z + 1]) for i in rows]
    d["set"] = ["held-out" if b in held else "trained" for b in d.index]
    d.to_csv(ROOT / f"results/{run}/metrics_test_continuous.csv")
    w = pd.read_csv(ROOT / f"results/{run}/metrics_test.csv", index_col="basin")
    print(f"{run}: median NSE windowed-eval trained {w[w.set == 'trained'].NSE.median():.3f} held-out {w[w.set == 'held-out'].NSE.median():.3f}"
          f" | continuous-eval trained {d[d.set == 'trained'].NSE.median():.3f} held-out {d[d.set == 'held-out'].NSE.median():.3f}"
          f" | mean NSE (clip -1) {w.NSE.clip(-1).mean():.3f} -> {d.NSE.clip(-1).mean():.3f} | FLV median {w.FLV.median():+.0f}% -> {d.FLV.median():+.0f}%", flush=True)
