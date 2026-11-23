"""Maps of where the RFL-HBV's learned rules have the LC-HBV's shape: rule similarity (correlation of the probed
curves, scripts/evaluation/probe_rules.py) per catchment (results/overview/table11_rule_agreement.csv), one map per rule. Same style
as figures/spatial_maps.png. Usage: uv run python scripts/figures/plot_rule_agreement_maps.py -> figures/rule_agreement_maps.png"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import colors

from gbhydro.data import ROOT, load_attributes, load_outline_rings

INK = "#222222"
RULES = [("et", "ET rule"), ("recharge", "recharge rule"), ("fast", "fast flow rule"),
         ("baseflow", "baseflow rule"), ("melt", "melt rule")]
REF = "LC-HBV"

att = load_attributes()
d = pd.read_csv(ROOT / "results/overview/table11_rule_agreement.csv", index_col="basin")
rings = load_outline_rings()
cmap = plt.get_cmap("viridis_r"); norm = colors.Normalize(0, 1)  # correlations below 0 drawn as 0
fig, axes = plt.subplots(2, 3, figsize=(11, 10))
fig.subplots_adjust(left=0.01, right=0.9, top=0.95, bottom=0.01, wspace=0.02, hspace=0.1)
axes.flat[-1].set_axis_off()
for ax, (rule, title) in zip(axes.flat, RULES):
    v = d[f"RFL-HBV vs {REF}: {rule}"].dropna()
    a = att.loc[v.index]
    for rx, ry in rings:
        ax.plot(rx, ry, color="#cccccc", lw=0.6, zorder=1)
    rest = att.loc[d.index.difference(v.index)]          # catchments where the rule is not evaluated (no snow for melt)
    ax.scatter(rest.gauge_easting / 1e3, rest.gauge_northing / 1e3, color="#e4e4e4", s=3, lw=0, zorder=1.5)
    sc = ax.scatter(a.gauge_easting / 1e3, a.gauge_northing / 1e3, c=v, cmap=cmap, norm=norm, s=11, lw=0, zorder=2)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    # median in the wettest and driest third of the catchments (aridity, all 669)
    third = pd.qcut(att.aridity.loc[d.index], 3, labels=["wet", "mid", "dry"]).reindex(v.index)
    med = {k: v[third == k].median() for k in ["wet", "dry"]}
    print(f"{rule}: median {v.median():.2f}, wettest third {med['wet']:.2f}, driest third {med['dry']:.2f}, n {len(v)}")
    sub = f"median {v.median():.2f}" + ("" if rule == "melt" else f"\nwettest third {med['wet']:.2f}, driest third {med['dry']:.2f}")
    ax.set_title(f"{title}\n{sub}", loc="left", fontsize=9.5, color=INK)
cb = fig.colorbar(sc, ax=axes, shrink=0.5, pad=0.01)
cb.set_label(f"rule similarity (compared with {REF})", fontsize=9, color=INK)
cb.ax.tick_params(labelsize=9, colors=INK)
fig.savefig(ROOT / "figures/rule_agreement_maps.png", dpi=150, bbox_inches="tight"); print("wrote figures/rule_agreement_maps.png")
