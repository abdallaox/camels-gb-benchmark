"""Hydrographs: observed flow and the three models' five-seed ensemble means (continuous evaluation) for contrasting
catchments over two water years in the test period. LSTM ensemble is simulated here from runs/lstm_s0..4.
Usage: uv run python scripts/figures/plot_hydrographs.py [basins=89003,42004,43009,12002] [start=2002-10-01] [years=2]
   -> figures/hydrographs.png"""
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from gbhydro.continuous import simulate_continuous
from gbhydro.data import ROOT, load_attributes
from gbhydro.dataset import Study
from gbhydro.metrics import nse
from gbhydro.models import build, load_run

basins = [int(b) for b in (sys.argv[1] if len(sys.argv) > 1 else "89003,42004,43009,12002").split(",")]
start = pd.Timestamp(sys.argv[2] if len(sys.argv) > 2 else "2002-10-01"); years = int(sys.argv[3]) if len(sys.argv) > 3 else 2
end = start + pd.DateOffset(years=years) - pd.Timedelta(days=1)
LABEL = {89003: "Orchy at Glen Orchy (wet upland)", 42004: "Test at Broadlands (chalk)", 43009: "Stour at Hammoon (clay)", 12002: "Dee at Park (snowy)"}
INK, MUTED, GRID = "#222222", "#666666", "#dddddd"
COL = {"LSTM": "#2a78d6", "RPL-HBV": "#eb6834", "RFL-HBV": "#2e9e5b"}

study = Study(inputs="matched", device="cpu"); torch.set_num_threads(4)
rows = np.array([int(np.where(study.basins == b)[0][0]) for b in basins])
a, z = study.periods["test"]; dates = study.dates
lstm = None
for s in range(5):
    m = load_run(f"lstm_s{s}", study)
    q = simulate_continuous(study, m, rows)["q"]; lstm = q if lstm is None else lstm + q
sims = {"LSTM": lstm / 5}
for k, pre in [("RPL-HBV", "hbvc_b128"), ("RFL-HBV", "gatedc3_n16")]:
    e = np.load(ROOT / "results" / f"{pre}_ensemble" / "sim_test.npz")
    idx = [int(np.where(e["basins"] == b)[0][0]) for b in basins]
    full = np.full((len(basins), len(dates)), np.nan, dtype=np.float32); full[:, a:z + 1] = e["q"][idx]; sims[k] = full
obs = study.target[torch.as_tensor(rows)].cpu().numpy()
w = (dates >= start) & (dates <= end)
fig, axes = plt.subplots(len(basins), 1, figsize=(14, 2.6 * len(basins)), sharex=True)
for ax, b, i in zip(axes, basins, range(len(basins))):
    ax.plot(dates[w], obs[i, w], color=INK, lw=1.6, label="observed")
    scores = []
    for k, c in COL.items():
        ax.plot(dates[w], sims[k][i, w], color=c, lw=1.0, label=k)
        scores.append(f"{k} NSE {nse(sims[k][i, a:z + 1], obs[i, a:z + 1]):.2f}")
    ax.set_title(f"{b}  {LABEL[b] if b in LABEL else load_attributes().gauge_name.loc[b]}    test-period " + "   ".join(scores), loc="left", fontsize=9, color=INK)
    ax.set_ylabel("flow (mm/d)", fontsize=8, color=MUTED); ax.tick_params(labelsize=8, colors=MUTED)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(GRID)
    ax.grid(color=GRID, lw=0.5, axis="y"); ax.set_axisbelow(True)
axes[0].legend(frameon=False, fontsize=8, ncol=4, loc="upper right")

fig.tight_layout(); fig.savefig(ROOT / "figures/hydrographs.png", dpi=140); print("wrote figures/hydrographs.png")
