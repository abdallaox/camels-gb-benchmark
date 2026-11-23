"""Spatial pattern figure: every gauge coloured by its test-period NSE (seed median per catchment; restart median
for the LC-HBV). Layout: the four models in one row, LSTM first. Darker is better; scores below 0.6 share the lightest colour. Light grey outline: union of the catchment
polygons. Reads results/<run>/metrics_test_full.csv.
Usage: uv run python scripts/figures/plot_spatial_maps.py -> figures/spatial_maps.png"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import colors

from gbhydro.data import ROOT, load_attributes, load_outline_rings

INK = "#222222"
MODELS = {"LSTM": [f"lstm_s{s}" for s in range(5)], "RPL-HBV": [f"hbvc_b128_s{s}" for s in range(5)],
          "RFL-HBV (n = 16)": [f"gatedc3_n16_s{s}" for s in range(5)], "LC-HBV (per catchment)": [f"hbv_local_r{s}" for s in range(5)]}
VMIN, VMAX = 0.6, 0.95


def seed_median(runs, metric):
    ds = [pd.read_csv(ROOT / f"results/{r}/metrics_test_full.csv", index_col="basin") for r in runs]
    return pd.concat([d[metric] for d in ds], axis=1).median(axis=1), ds[0].set


att = load_attributes()
# subtle land outline: union of the catchment polygons, gaps closed with a 5 km buffer in and out, exterior rings only
rings = load_outline_rings()
cmap = plt.get_cmap("viridis_r")     # dark = better, light = worse; everything below VMIN takes the lightest colour
norm = colors.Normalize(VMIN, VMAX)


def draw(ax, name, runs):
    v, st = seed_median(runs, "NSE")
    a = att.loc[v.index]
    for rx, ry in rings:
        ax.plot(rx, ry, color="#cccccc", lw=0.6, zorder=1)
    sc = ax.scatter(a.gauge_easting / 1e3, a.gauge_northing / 1e3, c=v, cmap=cmap, norm=norm, s=11, lw=0, zorder=2)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_title(f"{name}\nNSE median {v.median():.2f}", loc="left", fontsize=10, color=INK)
    return sc


fig = plt.figure(figsize=(15, 6.2))
gs = fig.add_gridspec(1, 4, left=0.01, right=0.93, top=0.92, bottom=0.01, wspace=0.03)
axes = [fig.add_subplot(gs[0, j]) for j in range(4)]
for ax, (name, runs) in zip(axes, MODELS.items()):
    sc = draw(ax, name, runs)
cb = fig.colorbar(sc, ax=axes, shrink=0.75, pad=0.01, extend="min")
cb.set_label("NSE, test period 1998 to 2008", fontsize=9, color=INK)
cb.ax.tick_params(labelsize=9, colors=INK)
fig.savefig(ROOT / "figures/spatial_maps.png", dpi=170, bbox_inches="tight"); print("wrote figures/spatial_maps.png")
