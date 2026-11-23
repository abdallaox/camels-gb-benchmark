"""ERA5-Land snow water equivalent (Caravan catchment means) for the snowiest catchments, with the RPL-HBV snowpack
from a continuous run of a checkpoint for reference.

Usage: uv run python scripts/figures/plot_snow_series.py [runs=hbvc_b128_s0,gatedc3_n16_s0] [start=2005-10-01] [years=5] [n_basins=6]
   -> figures/snow_series_<runs>.png   (HBV runs in orange, gated runs in green)
Needs data/processed/era5land_catchments.nc (scripts/data/prepare_era5land.py) and runs/<run>/best.pt.
"""
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

runs = (sys.argv[1] if len(sys.argv) > 1 else "hbvc_b128_s0,gatedc3_n16_s0").split(",")
start = pd.Timestamp(sys.argv[2] if len(sys.argv) > 2 else "2005-10-01")
years = int(sys.argv[3]) if len(sys.argv) > 3 else 5
n_basins = int(sys.argv[4]) if len(sys.argv) > 4 else 6
end = start + pd.DateOffset(years=years) - pd.Timedelta(days=1)

INK, MUTED, GRID = "#222222", "#666666", "#dddddd"
ERA5, HBV, GATED, RAIN = "#2a78d6", "#eb6834", "#2e9e5b", "#bbbbbb"

era = xr.open_dataset(ROOT / "data/processed/era5land_catchments.nc")
swe = era["snow_depth_water_equivalent_mean"].to_pandas()           # basin x time
mean_swe = swe.mean(axis=1).sort_values(ascending=False)
basins = [int(b) for b in mean_swe.index[:n_basins]]
att = load_attributes()
names, elev = att["gauge_name"], att["elev_mean"]

study = Study(inputs="matched", device="cpu")
rows = np.array([int(np.where(study.basins == b)[0][0]) for b in basins])
sims = {}
for run in runs:
    kind = "gated" if "gated" in run else "hbv"
    model = load_run(run, study)
    sim = simulate_continuous(study, model, rows)
    sims[run] = sim["stores/SP"] if kind == "gated" else sim["stores/snowpack"] + sim["stores/liquid_water"]
dates = study.dates
precip = study._raw["precipitation"].cpu().numpy()

fig, axes = plt.subplots(len(basins), 1, figsize=(14, 2.3 * len(basins)), sharex=True)
for ax, b, r, i in zip(axes, basins, rows, range(len(basins))):
    w = (dates >= start) & (dates <= end)
    d = dates[w]
    ax.bar(d, precip[r, w], width=1.0, color=RAIN, lw=0, label="precipitation (CAMELS-GB)")
    ax.plot(d, swe.loc[b].to_numpy()[w], color=ERA5, lw=1.4, label="ERA5-Land SWE (Caravan)")
    for run, sp in sims.items():
        ax.plot(d, sp[i, w], color=GATED if "gated" in run else HBV, lw=1.1, label=f"{'RFL-HBV' if 'gated' in run else 'RPL-HBV'} snowpack ({run})")
    ax.text(0.005, 0.93, f"{b}  {names.loc[b]}   mean elevation {elev.loc[b]:.0f} m   "
            f"mean SWE 1970-2015: {mean_swe.loc[b]:.1f} mm", transform=ax.transAxes, fontsize=9, color=INK, va="top")
    ax.set_ylabel("mm", fontsize=8, color=MUTED)
    ax.tick_params(labelsize=8, colors=MUTED)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(GRID)
    ax.grid(color=GRID, lw=0.5, axis="y"); ax.set_axisbelow(True)
    ax.set_ylim(0, max(np.nanmax(swe.loc[b].to_numpy()[w]), max(np.nanmax(sp[i, w]) for sp in sims.values()), 20) * 1.15)
axes[0].legend(frameon=False, loc="upper right", fontsize=8, ncol=4)
axes[0].set_title(f"Snow water equivalent {start.date()} to {end.date()}: the {len(basins)} snowiest CAMELS-GB catchments by ERA5-Land mean SWE",
                  loc="left", fontsize=10.5, color=INK)
fig.tight_layout(h_pad=0.5)
out = ROOT / "figures" / f"snow_series_{'_'.join(runs)}.png"
fig.savefig(out, dpi=140); print("wrote", out)
print(pd.DataFrame({"name": names.loc[basins], "elev_m": elev.loc[basins].round(0), "mean_swe_mm": mean_swe.loc[basins].round(1),
                    "max_swe_mm": swe.loc[basins].max(axis=1).round(0),
                    **{f"{run}_mean_mm": [sp[i].mean().round(1) for i in range(len(basins))] for run, sp in sims.items()}}).to_string())
