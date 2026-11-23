"""Store realism figure for the paper: cumulative distributions over catchments of (left) the anomaly correlation between
the model soil store and ERA5-Land 0-100 cm soil water, all 669 catchments, and (right) the correlation between the
model snowpack and ERA5-Land snow water equivalent, catchments with median annual peak SWE above 20 mm. Seed median per
catchment. Reads results/<run>/store_realism.csv (scripts/evaluation/store_realism.py); the LC-HBV joins when present.
Usage: uv run python scripts/figures/plot_store_cdf.py -> figures/store_cdf.png"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from gbhydro.data import ROOT

INK = "#222222"
MODELS = {"RPL-HBV": ([f"hbvc_b128_s{s}" for s in range(5)], "#eb6834", "-"),
          "RFL-HBV (n = 16)": ([f"gatedc3_n16_s{s}" for s in range(5)], "#2e9e5b", "-"),
          "LC-HBV (per catchment)": ([f"hbv_local_r{s}" for s in range(5)], "#8e44ad", "--")}
SNOWY_MM = 20.0
TAB = pd.read_csv(ROOT / "results/overview/table7_store_realism.csv", index_col=0)   # medians as in Table 3
TABCOL = {"soil_corr_anom": "soil_anom_med", "snow_corr": "snow_corr_med"}


def seed_median(runs):
    fs = [ROOT / f"results/{r}/store_realism.csv" for r in runs]
    ds = [pd.read_csv(f, index_col="basin") for f in fs if f.exists()]
    if not ds:
        return None
    m = pd.concat(ds).groupby("basin").median(numeric_only=True)
    m["snowy"] = ds[0].era_peak_swe_median_mm > SNOWY_MM
    return m


PANELS = [("soil_corr_anom", None, "soil realism (compared with ERA5-Land)", "all 669 catchments"),
          ("snow_corr", "snowy", "snow realism (compared with ERA5-Land)", None)]
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
for ax, (col, sub, xlabel, note) in zip(axes, PANELS):
    for name, (runs, c, ls) in MODELS.items():
        d = seed_median(runs)
        if d is None:
            continue
        v = d[col] if sub is None else d.loc[d[sub], col]
        v = np.sort(v.dropna().clip(lower=-0.2).to_numpy())
        if note is None:
            note = f"{len(v)} catchments with peak SWE above {SNOWY_MM:.0f} mm"
        ax.plot(v, np.arange(1, len(v) + 1) / len(v), color=c, ls=ls, lw=2, label=f"{name}, median {TAB.loc[name.replace(' (per catchment)', ''), TABCOL[col]]:.2f}")
    ax.set_xlim(-0.2, 1); ax.set_ylim(0, 1)
    ax.set_xticks([-0.2, 0, 0.2, 0.4, 0.6, 0.8, 1]); ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
    ax.set_xlabel(xlabel, fontsize=10, color=INK); ax.set_ylabel("fraction of catchments", fontsize=10, color=INK)
    ax.set_title(note, loc="left", fontsize=10, color=INK)
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(labelsize=9, colors=INK)
fig.tight_layout()
fig.savefig(ROOT / "figures/store_cdf.png", dpi=150); print("wrote figures/store_cdf.png")
