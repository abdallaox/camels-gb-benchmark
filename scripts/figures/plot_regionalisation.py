"""What regionalisation costs: the RPL-HBV (parameters from attributes, one model for all catchments) against the
per-catchment LC-HBV (same equations, 14 free parameters per catchment), training catchments, test period.
figures/regionalisation.png: left, test NSE against soil store realism (anomaly correlation with ERA5-Land) as medians
with interquartile bars, as in the trade-off figure; middle and right, maps of RPL-HBV minus LC-HBV NSE and
realism. figures/regionalisation_scatter.png (supplement): per-catchment NSE scatter with the 1:1 line, coloured by
aquifer share. Usage: uv run python scripts/figures/plot_regionalisation.py"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import colors

from gbhydro.data import ROOT, load_attributes, load_outline_rings

INK = "#222222"
COL = {"RPL-HBV": "#eb6834", "LC-HBV": "#8e44ad"}


def seed_median(runs):
    return pd.concat([pd.read_csv(ROOT / f"results/{r}/metrics_test_full.csv", index_col="basin").NSE for r in runs], axis=1).median(axis=1)


reg = seed_median([f"hbvc_b128_s{s}" for s in range(5)])
cal = seed_median([f"hbv_local_r{s}" for s in range(5)])
sets = pd.read_csv(ROOT / "results/hbvc_b128_s0/metrics_test_full.csv", index_col="basin").set
soil = pd.read_csv(ROOT / "results/overview/per_basin_soil_anom_corr.csv", index_col="basin")
m = (sets == "trained") & reg.notna() & cal.notna()
NSE = {"RPL-HBV": reg[m], "LC-HBV": cal[m]}
SOIL = {"RPL-HBV": soil["RPL-HBV"].reindex(reg[m].index), "LC-HBV": soil["LC-HBV"].reindex(reg[m].index)}
att = load_attributes().loc[reg[m].index]
d_nse = NSE["RPL-HBV"] - NSE["LC-HBV"]
d_soil = SOIL["RPL-HBV"] - SOIL["LC-HBV"]
rings = load_outline_rings()


def style(ax):
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(labelsize=9, colors=INK)


def draw_map(ax, v, lim, title, label):
    for rx, ry in rings:
        ax.plot(rx, ry, color="#cccccc", lw=0.6, zorder=1)
    sc = ax.scatter(att.gauge_easting / 1e3, att.gauge_northing / 1e3, c=v.clip(-lim, lim), cmap="RdBu", norm=colors.Normalize(-lim, lim), s=9, lw=0, zorder=2)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_title(title, loc="left", fontsize=9.5, color=INK)
    cb = ax.figure.colorbar(sc, ax=ax, shrink=0.55, pad=0.02, extend="both")
    cb.set_label(label, fontsize=8.5, color=INK); cb.ax.tick_params(labelsize=8.5, colors=INK)


fig, axes = plt.subplots(1, 3, figsize=(13, 4.8), gridspec_kw={"width_ratios": [1.5, 0.95, 0.95]})
ax = axes[0]
for k, c in COL.items():
    x, y = NSE[k].clip(0), SOIL[k]
    xq, yq = x.quantile([0.25, 0.5, 0.75]).to_numpy(), y.quantile([0.25, 0.5, 0.75]).to_numpy()
    ax.errorbar(xq[1], yq[1], xerr=[[xq[1] - xq[0]], [xq[2] - xq[1]]], yerr=[[yq[1] - yq[0]], [yq[2] - yq[1]]],
                fmt="o", ms=10, color=c, elinewidth=1.3, capsize=0, zorder=5, label=f"{k}  (NSE {xq[1]:.2f}, realism {yq[1]:.2f})")
ax.set_xlim(0.6, 0.9); ax.set_ylim(0.6, 0.9)
ax.set_xlabel("test NSE (median and interquartile range over catchments)", fontsize=9, color=INK)
ax.set_ylabel("soil realism (compared with ERA5-Land)", fontsize=9, color=INK)
ax.set_title(f"training catchments ({int(m.sum())}), same equations,\nparameters from attributes or fitted per catchment", loc="left", fontsize=9.5, color=INK)
ax.legend(frameon=False, fontsize=9, loc="lower left")
style(ax)
draw_map(axes[1], d_nse, 0.2, f"RPL-HBV minus LC-HBV: test NSE\nmedian {d_nse.median():+.3f}, RPL-HBV lower in {(d_nse < 0).mean() * 100:.0f}%",
         "difference in test NSE (blue: RPL-HBV better)")
draw_map(axes[2], d_soil, 0.1, f"RPL-HBV minus LC-HBV: soil realism\nmedian {d_soil.median():+.3f}, RPL-HBV lower in {(d_soil < 0).mean() * 100:.0f}%",
         "difference in soil realism (blue: RPL-HBV better)")
fig.tight_layout(w_pad=1.5)
fig.savefig(ROOT / "figures/regionalisation.png", dpi=150, bbox_inches="tight"); print("wrote figures/regionalisation.png")

# supplement: per-catchment NSE scatter
fig, ax = plt.subplots(figsize=(6.6, 5.6))
ax.plot([0, 1], [0, 1], color="#bbbbbb", lw=1, zorder=1)
sc = ax.scatter(cal[m].clip(0), reg[m].clip(0), c=att.frac_high_perc, cmap="viridis_r", norm=colors.Normalize(0, 100), s=12, lw=0, alpha=0.85, zorder=2)
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal")
ax.set_xlabel("LC-HBV: test NSE (restart median)", fontsize=9, color=INK)
ax.set_ylabel("RPL-HBV: test NSE (seed median)", fontsize=9, color=INK)
ax.set_title(f"training catchments ({int(m.sum())}): RPL-HBV below the line in {(d_nse < 0).mean() * 100:.0f}%\nworse by more than 0.1 in {(d_nse < -0.1).mean() * 100:.0f}%, "
             f"better by more than 0.1 in {(d_nse > 0.1).mean() * 100:.0f}%", loc="left", fontsize=9.5, color=INK)
cb = fig.colorbar(sc, ax=ax, shrink=0.75, pad=0.02)
cb.set_label("highly productive aquifer (% of area)", fontsize=9, color=INK); cb.ax.tick_params(labelsize=9, colors=INK)
style(ax)
fig.tight_layout(); fig.savefig(ROOT / "figures/regionalisation_scatter.png", dpi=150, bbox_inches="tight"); print("wrote figures/regionalisation_scatter.png")
for name, v in (("NSE", d_nse), ("soil realism", d_soil)):
    print(name, "median", round(v.median(), 3), "regional lower in", f"{(v < 0).mean() * 100:.0f}%", "| Spearman with aquifer", round(v.corr(att.frac_high_perc, method="spearman"), 2),
          "aridity", round(v.corr(att.aridity, method="spearman"), 2), "BFI", round(v.corr(att.baseflow_index, method="spearman"), 2))
