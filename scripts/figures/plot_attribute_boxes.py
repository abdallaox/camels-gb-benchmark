"""Performance by catchment type as box charts (the figure form of Table 3): test-period NSE of the three regional
models and the LC-HBV (calibrated on every catchment) over all 669 catchments, grouped by baseflow index category, aridity category, highly productive aquifer share,
urban share, mean elevation and reported abstraction as a share of mean flow (England only), the same classes as scripts/evaluation/analysis_overview.py. Box = interquartile range, line =
median, whiskers = 10th to 90th percentile. Usage: uv run python scripts/figures/plot_attribute_boxes.py
-> figures/attribute_boxes.png"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from gbhydro.data import ROOT, load_attributes

INK, MUTED = "#222222", "#555555"
COL = {"LSTM": "#2a78d6", "RPL-HBV": "#eb6834", "RFL-HBV": "#2e9e5b", "LC-HBV": "#8e44ad"}
pb = pd.read_csv(ROOT / "results/overview/per_basin_seed_median_nse.csv", index_col=0)
lc = pd.concat([pd.read_csv(ROOT / f"results/hbv_local_r{r}/metrics_test_full.csv", index_col="basin").NSE for r in range(5)], axis=1).median(axis=1)
NSE = {"LSTM": pb["LSTM"], "RPL-HBV": pb["HBV regional (continuous)"], "RFL-HBV": pb["RFL-HBV n=16 (constrained)"], "LC-HBV": lc.reindex(pb.index)}
a = load_attributes().reindex(pb.index)
GROUPS = {
    "baseflow index (categories)": pd.qcut(a.baseflow_index, 3, labels=["low", "mid", "high"]),
    "aridity (categories)": pd.qcut(a.aridity, 3, labels=["wet", "mid", "dry"]),
    "highly productive aquifer (% of area)": pd.cut(a.frac_high_perc, [-1, 10, 50, 101], labels=["< 10", "10 to 50", "> 50"]),
    "urban share (% of area)": pd.cut(a.urban_perc, [-1, 2, 10, 101], labels=["< 2", "2 to 10", "> 10"]),
    "mean elevation (m)": pd.cut(a.elev_mean, [0, 150, 350, 2000], labels=["< 150", "150 to 350", "> 350"]),
    "abstraction (% of mean flow, England)": None,
}
# licensed abstraction (groundwater + surface water) as a share of mean flow; given for 441 catchments in England
hum = pd.read_csv(ROOT / "data/camels_gb/data/CAMELS_GB_humaninfluence_attributes.csv", index_col="gauge_id").reindex(pb.index)
hyd = pd.read_csv(ROOT / "data/camels_gb/data/CAMELS_GB_hydrologic_attributes.csv", index_col="gauge_id").reindex(pb.index)
has = hum.groundwater_abs.notna() | hum.surfacewater_abs.notna()
abs_share = 100 * (hum.groundwater_abs.fillna(0) + hum.surfacewater_abs.fillna(0)) / hyd.q_mean
abs_share[~has] = np.nan
GROUPS["abstraction (% of mean flow, England)"] = pd.cut(abs_share, [-1, 5, 20, 50, 1e9], labels=["< 5", "5 to 20", "20 to 50", "> 50"])

fig, axes = plt.subplots(2, 3, figsize=(11.5, 7.4), sharey=True)
axes = axes.flat
w, off = 0.13, {"LSTM": -0.225, "RPL-HBV": -0.075, "RFL-HBV": 0.075, "LC-HBV": 0.225}   # group spans 0.58, gap 0.42
for ax, (title, cls) in zip(axes, GROUPS.items()):
    levels = list(cls.cat.categories)
    for name, v in NSE.items():
        data = [v[cls == lv].dropna().clip(lower=-0.5).to_numpy() for lv in levels]
        ax.boxplot(data, positions=np.arange(len(levels)) + off[name], widths=w, whis=(10, 90), showfliers=False,
                   patch_artist=True, medianprops={"color": "white", "lw": 1.6},
                   whiskerprops={"color": COL[name], "lw": 1.1}, capprops={"color": COL[name], "lw": 1.1},
                   boxprops={"facecolor": COL[name], "edgecolor": COL[name], "lw": 0.8})
    for k in range(1, len(levels)):   # faint divider between classes
        ax.axvline(k - 0.5, color="#dddddd", lw=0.8, zorder=0)
    ax.set_xlim(-0.5, len(levels) - 0.5)
    ax.set_xticks(np.arange(len(levels)))
    ax.set_xticklabels([f"{lv}\n({int((cls == lv).sum())})" for lv in levels], fontsize=9.5)
    ax.set_title(title, loc="left", fontsize=10.5, color=INK, pad=8)
    ax.set_xlim(-0.6, len(levels) - 0.4)
    ax.set_ylim(0.2, 1); ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(labelsize=9.5, colors=INK)
    ax.tick_params(axis="y", labelleft=True)
for ax in list(axes)[:5:3]:
    ax.set_ylabel("test NSE", fontsize=10, color=INK)
handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in COL.values()]
fig.legend(handles, list(COL), frameon=False, fontsize=10.5, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.01))
fig.tight_layout(h_pad=2.5, w_pad=1.5, rect=(0, 0.04, 1, 1))
fig.savefig(ROOT / "figures/attribute_boxes.png", dpi=150, bbox_inches="tight"); print("wrote figures/attribute_boxes.png")
