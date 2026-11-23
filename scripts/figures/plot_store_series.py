"""Example store series for the paper: for a few contrasting catchments, the model soil store (RPL-HBV and RFL-HBV,
seed 0) next to ERA5-Land 0-100 cm soil water, all as standardised anomalies (day-of-year climatology removed, divided
by the standard deviation over 1975-2015), plus one snowy catchment with the snowpack in mm against ERA5-Land snow
water equivalent. Continuous simulation from runs/<run>/best.pt.
Usage: uv run python scripts/figures/plot_store_series.py [start=2005-10-01] [years=4] -> figures/store_series.png"""
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import xarray as xr

from gbhydro.continuous import simulate_continuous
from gbhydro.data import ROOT, load_attributes
from gbhydro.dataset import Study
from gbhydro.models import load_run

sys.path.insert(0, str(ROOT / "scripts" / "evaluation"))
from store_realism import LAYERS, SPINUP_END, anomalies  # noqa: E402

start = pd.Timestamp(sys.argv[1] if len(sys.argv) > 1 else "2005-10-01")
years = int(sys.argv[2]) if len(sys.argv) > 2 else 4
end = start + pd.DateOffset(years=years) - pd.Timedelta(days=1)
INK, ERA5, HBV, GATED = "#222222", "#2a78d6", "#eb6834", "#2e9e5b"
RUNS = {"RPL-HBV": "hbvc_b128_s0", "RFL-HBV": "gatedc3_n16_s0"}
# (basin, label): wet upland, lowland clay/mixed, chalk, and a snowy catchment for the snowpack panel
SOIL = [(76007, "Eden at Sheepmount, wet upland"), (39001, "Thames at Kingston, large lowland"), (39019, "Lambourn at Shaw, chalk")]
SNOW = [(12007, "Dee at Mar Lodge, snowpack")]

torch.set_num_threads(4)
study = Study(inputs="matched", device="cpu")
era = xr.open_dataset(ROOT / "data/processed/era5land_catchments.nc")
basins = [b for b, _ in SOIL + SNOW]
rows = np.array([int(np.where(study.basins == b)[0][0]) for b in basins])
dates = study.dates
full = dates >= SPINUP_END; w = (dates >= start) & (dates <= end)
wts = np.array(list(LAYERS.values()), dtype=float); wts /= wts.sum()
era_soil = sum(wi * era[v].values[rows] for v, wi in zip(LAYERS, wts))
era_swe = era["snow_depth_water_equivalent_mean"].values[rows]
sims = {}
for name, run in RUNS.items():
    sim = simulate_continuous(study, load_run(run, study), rows)
    gated = "gated" in run
    sims[name] = {"soil": sim["stores/SM"] if gated else sim["stores/soil_moisture"],
                  "snow": sim["stores/SP"] if gated else sim["stores/snowpack"] + sim["stores/liquid_water"]}


def z(x):
    a = anomalies(x, dates)
    return a / np.nanstd(a[:, full], axis=1, keepdims=True)


def corr(a, b):
    m = full & np.isfinite(a) & np.isfinite(b)
    return np.corrcoef(a[m], b[m])[0, 1]


fig, axes = plt.subplots(len(basins), 1, figsize=(10, 2.1 * len(basins)), sharex=True)
d = dates[w]
for k, (ax, (b, label)) in enumerate(zip(axes, SOIL + SNOW)):
    snowy = k >= len(SOIL)
    obs = era_swe[k] if snowy else z(era_soil)[k]
    ax.plot(d, obs[w], color=ERA5, lw=1.4, label="ERA5-Land")
    txt = []
    for (name, s), c in zip(sims.items(), (HBV, GATED)):
        mod = s["snow"][k] if snowy else z(s["soil"])[k]
        r = corr(mod, obs)
        ax.plot(d, mod[w], color=c, lw=1.1, label=name)
        txt.append(f"{name} r = {r:.2f}")
    ax.text(0.005, 0.95, f"{label}.  " + ", ".join(txt), transform=ax.transAxes, fontsize=9, color=INK, va="top")
    ax.set_ylabel("snow water, mm" if snowy else "soil water anomaly, s.d.", fontsize=9, color=INK)
    if not snowy:
        ax.axhline(0, color="#bbbbbb", lw=0.6, zorder=0); ax.set_ylim(-3.2, 4.4)      # headroom for the label row
    else:
        ax.set_ylim(0, None)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(labelsize=9, colors=INK)
axes[0].legend(frameon=False, loc="upper right", fontsize=9, ncol=3, bbox_to_anchor=(1, 1.22))
fig.tight_layout(h_pad=0.4, rect=(0, 0, 1, 0.98))
fig.savefig(ROOT / "figures/store_series.png", dpi=150); print("wrote figures/store_series.png")
