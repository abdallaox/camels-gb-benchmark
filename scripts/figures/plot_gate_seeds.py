"""Seed stability of the learned rules: the five RFL-HBV seeds overlaid, daily flux vs own store from the continuous
runs, for contrasting catchments. Usage: uv run python scripts/figures/plot_gate_seeds.py [prefix=gatedc3_n16_s] [basins]
   -> figures/gate_seeds_<prefix>.png"""
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from gbhydro.continuous import simulate_continuous
from gbhydro.data import ROOT, load_attributes
from gbhydro.dataset import Study
from gbhydro.models import build, load_run

prefix = sys.argv[1] if len(sys.argv) > 1 else "gatedc3_n16_s"
basins = [int(b) for b in (sys.argv[2] if len(sys.argv) > 2 else "89003,42004,43009,12002").split(",")]
LABEL = {89003: "Orchy (wet upland)", 42004: "Test (chalk)", 43009: "Stour at Hammoon (clay)", 12002: "Dee at Park (snowy)"}
INK, MUTED, GRID = "#222222", "#666666", "#dddddd"
SEEDS = ["#2a78d6", "#eb6834", "#2e9e5b", "#9b59b6", "#c9a227"]

study = Study(inputs="matched", device="cpu")
rows = np.array([int(np.where(study.basins == b)[0][0]) for b in basins])
sims = []
for s in range(5):
    m = load_run(f"{prefix}{s}", study)
    sims.append(simulate_continuous(study, m, rows))
att = load_attributes()
pet = study._raw["pet"].cpu().numpy()[rows]; temp = study._raw["temperature"].cpu().numpy()[rows]
w = study.dates >= pd.Timestamp("1975-10-01")

def panels(g, i):
    return [("ET efficiency (days with PET > 0.5 mm)", g["stores/SM"][i], g["fluxes/et"][i] / np.maximum(pet[i], 0.1), "soil store (mm)", "ET / PET", w & (pet[i] > 0.5)),
            ("recharge", g["stores/SM"][i], g["fluxes/rech"][i], "soil store (mm)", "recharge (mm/d)", w),
            ("percolation", g["stores/SUZ"][i], g["fluxes/perc"][i], "upper store (mm)", "percolation (mm/d)", w),
            ("fast outflow", g["stores/SUZ"][i], g["fluxes/fast"][i], "upper store (mm)", "fast flow (mm/d)", w),
            ("baseflow", g["stores/SLZ"][i], g["fluxes/base"][i], "lower store (mm)", "baseflow (mm/d)", w),
            ("melt (days with snow)", temp[i], g["fluxes/melt"][i], "temperature (degC)", "melt (mm/d)", w & (g["stores/SP"][i] > 1))]

fig, axes = plt.subplots(6, len(basins), figsize=(3.4 * len(basins), 13))
for j, (b, i) in enumerate(zip(basins, range(len(basins)))):
    for s, g in enumerate(sims):
        for k, (name, x, y, xl, yl, mask) in enumerate(panels(g, i)):
            ax = axes[k, j]
            ax.scatter(x[mask], y[mask], s=1.5, color=SEEDS[s], alpha=0.2, lw=0, rasterized=True)
            if s == 0:
                ax.set_xlabel(xl, fontsize=7.5, color=MUTED); ax.set_ylabel(yl, fontsize=7.5, color=MUTED); ax.tick_params(labelsize=7, colors=MUTED)
                for sp in ("top", "right"):
                    ax.spines[sp].set_visible(False)
                for sp in ("left", "bottom"):
                    ax.spines[sp].set_color(GRID)
                ax.grid(color=GRID, lw=0.4); ax.set_axisbelow(True)
                if k == 0:
                    ax.set_ylim(0, 1.05); ax.set_title(f"{b}  {LABEL.get(b, att.gauge_name.loc[b])}", loc="left", fontsize=8.5, color=INK)
            if j == 0 and k == 0 and s == 4:
                for t in range(5):
                    ax.scatter([], [], color=SEEDS[t], s=12, label=f"seed {t}")
                ax.legend(frameon=False, fontsize=7, loc="lower right", ncol=5, columnspacing=0.6, handletextpad=0.2)
fig.suptitle(f"Seed stability of the learned rules: five RFL-HBV seeds ({prefix}0-4), daily values 1975-2015", x=0.01, ha="left", fontsize=10.5, color=INK)
fig.tight_layout(rect=(0, 0, 1, 0.98))
out = ROOT / "figures" / f"gate_seeds_{prefix.rstrip('_s')}.png"; fig.savefig(out, dpi=140); print("wrote", out)
