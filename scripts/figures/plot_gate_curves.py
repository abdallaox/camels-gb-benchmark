"""Learned process rules next to HBV's: daily flux against its own store from the continuous runs, for contrasting
catchments. Rows: ET efficiency vs soil store, recharge vs soil store, percolation vs upper store, fast outflow vs
upper store, baseflow vs lower store, melt vs temperature. Columns: catchments. Gated in green, RPL-HBV in orange.
Usage: uv run python scripts/figures/plot_gate_curves.py [hbv_run=hbvc_b128_s0] [gated_run=gatedc3_n16_s0] [basins=89003,42004,43009,12002]
   -> figures/gate_curves_<gated_run>.png"""
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from gbhydro.continuous import simulate_continuous
from gbhydro.data import ROOT, load_attributes
from gbhydro.dataset import Study
from gbhydro.models import build, load_run

hbv_run = sys.argv[1] if len(sys.argv) > 1 else "hbvc_b128_s0"
gated_run = sys.argv[2] if len(sys.argv) > 2 else "gatedc3_n16_s0"
basins = [int(b) for b in (sys.argv[3] if len(sys.argv) > 3 else "89003,42004,43009,12002").split(",")]
LABEL = {89003: "Orchy (wet upland)", 42004: "Test (chalk)", 43009: "Stour at Hammoon (clay)", 12002: "Dee at Park (snowy)"}
INK, MUTED, GRID, HBV, GATED = "#222222", "#666666", "#dddddd", "#eb6834", "#2e9e5b"

study = Study(inputs="matched", device="cpu")
rows = np.array([int(np.where(study.basins == b)[0][0]) for b in basins])
sims = {}
for kind, run in [("hbv", hbv_run), ("gated", gated_run)]:
    m = load_run(run, study)
    sims[kind] = simulate_continuous(study, m, rows)
att = load_attributes()
pet = study._raw["pet"].cpu().numpy()[rows]; temp = study._raw["temperature"].cpu().numpy()[rows]
w = study.dates >= pd.Timestamp("1975-10-01")
h, g = sims["hbv"], sims["gated"]
params = pd.read_csv(ROOT / f"results/{hbv_run}/parameters_test.csv", index_col="basin").loc[basins]

# (x, y) per model per row: name, HBV (x, y), gated (x, y), xlabel, ylabel
def panels(i):
    return [
        ("ET efficiency", (h["stores/soil_moisture"][i], h["fluxes/et"][i] / np.maximum(pet[i], 0.1)), (g["stores/SM"][i], g["fluxes/et"][i] / np.maximum(pet[i], 0.1)), "soil store (mm)", "ET / PET"),
        ("recharge", (h["stores/soil_moisture"][i], h["fluxes/recharge"][i]), (g["stores/SM"][i], g["fluxes/rech"][i]), "soil store (mm)", "recharge (mm/d)"),
        ("percolation", (h["stores/upper_storage"][i], h["fluxes/perc"][i]), (g["stores/SUZ"][i], g["fluxes/perc"][i]), "upper store (mm)", "percolation (mm/d)"),
        ("fast outflow", (h["stores/upper_storage"][i], h["fluxes/q0"][i] + h["fluxes/q1"][i]), (g["stores/SUZ"][i], g["fluxes/fast"][i]), "upper store (mm)", "fast flow (mm/d)"),
        ("baseflow", (h["stores/lower_storage"][i], h["fluxes/q2"][i]), (g["stores/SLZ"][i], g["fluxes/base"][i]), "lower store (mm)", "baseflow (mm/d)"),
        ("melt", (temp[i], h["fluxes/melt"][i]), (temp[i], g["fluxes/melt"][i]), "temperature (degC)", "melt (mm/d)"),
    ]

fig, axes = plt.subplots(6, len(basins), figsize=(3.4 * len(basins), 13))
for j, (b, i) in enumerate(zip(basins, range(len(basins)))):
    for k, (name, (hx, hy), (gx, gy), xl, yl) in enumerate(panels(i)):
        ax = axes[k, j]
        if name == "melt":      # only days with snow on the ground
            mh, mg = w & (h["stores/snowpack"][i] > 1), w & (g["stores/SP"][i] > 1)
        else:
            mh = mg = w
        ax.scatter(hx[mh], hy[mh], s=2, color=HBV, alpha=0.25, lw=0, rasterized=True)
        ax.scatter(gx[mg], gy[mg], s=2, color=GATED, alpha=0.25, lw=0, rasterized=True)
        ax.set_xlabel(xl, fontsize=7.5, color=MUTED); ax.set_ylabel(yl, fontsize=7.5, color=MUTED)
        ax.tick_params(labelsize=7, colors=MUTED)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            ax.spines[sp].set_color(GRID)
        ax.grid(color=GRID, lw=0.4); ax.set_axisbelow(True)
        if name == "ET efficiency":
            ax.set_ylim(0, 1.05)
        if name == "baseflow" or name == "percolation" or name == "fast outflow":
            ax.set_ylim(bottom=0)
        if k == 0:
            p = params.loc[b]
            ax.set_title(f"{b}  {LABEL.get(b, att.gauge_name.loc[b])}\nHBV: FC {p['FC']:.0f}  LP {p['LP']:.2f}  BETA {p['BETA']:.1f}  PERC {p['PERC']:.1f}  K2 {p['K2']:.4f}",
                         loc="left", fontsize=8, color=INK)
        if k == 0 and j == 0:
            ax.scatter([], [], color=HBV, s=12, label=f"RPL-HBV ({hbv_run})"); ax.scatter([], [], color=GATED, s=12, label=f"RFL-HBV ({gated_run})")
            ax.legend(frameon=False, fontsize=7.5, loc="lower right")
fig.suptitle("Learned flux rules against HBV's: daily values 1975-2015 from the continuous runs", x=0.01, ha="left", fontsize=10.5, color=INK)
fig.tight_layout(rect=(0, 0, 1, 0.98))
out = ROOT / "figures" / f"gate_curves_{gated_run}.png"; fig.savefig(out, dpi=140); print("wrote", out)
