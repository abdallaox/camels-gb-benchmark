"""Where performance is given up for structure: maps of the test-period NSE difference between the LSTM and each
HBV-based model (seed median per catchment; restart median for the LC-HBV, training catchments only).
Red: LSTM better. Same style as figures/spatial_maps.png. Usage: uv run python scripts/figures/plot_gap_maps.py
-> figures/gap_maps.png"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import colors

from gbhydro.data import ROOT, load_attributes, load_outline_rings

INK = "#222222"
LIM = 0.3
RUNS = {"LSTM": [f"lstm_s{s}" for s in range(5)], "RPL-HBV": [f"hbvc_b128_s{s}" for s in range(5)],
        "RFL-HBV": [f"gatedc3_n16_s{s}" for s in range(5)], "LC-HBV": [f"hbv_local_r{s}" for s in range(5)]}


def seed_median(runs):
    ds = [pd.read_csv(ROOT / f"results/{r}/metrics_test_full.csv", index_col="basin") for r in runs if (ROOT / f"results/{r}/metrics_test_full.csv").exists()]
    return pd.concat([d.NSE for d in ds], axis=1).median(axis=1), ds[0].set


att = load_attributes()
nse = {k: seed_median(v) for k, v in RUNS.items()}
rings = load_outline_rings()
norm = colors.Normalize(-LIM, LIM)
panels = [k for k in ["RPL-HBV", "RFL-HBV", "LC-HBV"] if k in nse]
fig, axes = plt.subplots(1, len(panels), figsize=(4 * len(panels), 5.4))
fig.subplots_adjust(left=0.01, right=0.9, top=0.9, bottom=0.01, wspace=0.02)
for ax, k in zip(axes, panels):
    v, st = nse[k]
    diff = (nse["LSTM"][0] - v)
    if k == "LC-HBV":
        diff = diff[st == "trained"]
    diff = diff.dropna()
    a = att.loc[diff.index]
    for rx, ry in rings:
        ax.plot(rx, ry, color="#cccccc", lw=0.6, zorder=1)
    sc = ax.scatter(a.gauge_easting / 1e3, a.gauge_northing / 1e3, c=diff.clip(-LIM, LIM), cmap="RdBu_r", norm=norm, s=8, lw=0, zorder=2)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_title(f"LSTM minus {k}\nmedian {diff.median():+.3f}, LSTM better in {(diff > 0).mean() * 100:.0f}%\nby more than 0.1 in {(diff > 0.1).mean() * 100:.0f}%",
                 loc="left", fontsize=9.5, color=INK)
cb = fig.colorbar(sc, ax=axes, shrink=0.7, pad=0.01, extend="both")
cb.set_label("difference in test NSE (red: LSTM better)", fontsize=9, color=INK)
cb.ax.tick_params(labelsize=9, colors=INK)
fig.savefig(ROOT / "figures/gap_maps.png", dpi=150, bbox_inches="tight"); print("wrote figures/gap_maps.png")
