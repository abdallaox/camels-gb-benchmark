"""Performance across the range of catchment attributes: catchments split into five equal-count groups per attribute
(a separate 'none' group where many catchments have zero), median test NSE of each model per group (seed medians per
catchment; restart median for the LC-HBV, training catchments only). Flow signatures appear as descriptors only.
Usage: uv run python scripts/figures/plot_attribute_bins.py -> figures/attribute_bins.png, results/overview/table8c_attribute_bins.csv"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from gbhydro.data import ROOT, load_attributes

INK = "#222222"
MODELS = {"LSTM": ("#2a78d6", "-"), "RPL-HBV": ("#eb6834", "-"), "RFL-HBV": ("#2e9e5b", "-"), "LC-HBV": ("#8e44ad", "--")}
ATTRS = [("aridity", "aridity (PET / P)", 2), ("p_mean", "mean precipitation (mm/d)", 1), ("dpsbar", "mean slope (m/km)", 0),
         ("frac_high_perc", "highly productive aquifer (% of area)", 0), ("baseflow_index", "baseflow index", 2),
         ("groundwater_abs", "groundwater abstraction (mm/d, England only)", 3)]

pb = pd.read_csv(ROOT / "results/overview/per_basin_seed_median_nse.csv", index_col="basin")
att = load_attributes().loc[pb.index]
scores = {"LSTM": pb["LSTM"], "RPL-HBV": pb["HBV regional (continuous)"], "RFL-HBV": pb["RFL-HBV n=16 (constrained)"]}
cal = [ROOT / f"results/hbv_local_r{s}/metrics_test_full.csv" for s in range(5)]
if all(f.exists() for f in cal):
    C = pd.concat([pd.read_csv(f, index_col="basin").NSE for f in cal], axis=1).median(axis=1).reindex(pb.index)
    C[pb.set == "held-out"] = np.nan
    scores["LC-HBV"] = C


def groups(x, nd):
    """Five equal-count groups; if more than a fifth of the values are zero, zero is its own group and the rest are quartiles."""
    x = x.dropna()
    if (x == 0).mean() > 0.2:
        pos = x[x > 0]
        q = pd.qcut(pos, 4, labels=False, duplicates="drop")
        g = pd.Series(np.nan, index=x.index); g[x == 0] = 0; g[pos.index] = q + 1
        edges = [0] + list(pos.quantile([0, 0.25, 0.5, 0.75, 1]))
        labels = ["none"] + [f"{edges[i]:.{nd}f}-{edges[i + 1]:.{nd}f}" for i in range(1, 5)]
    else:
        g = pd.qcut(x, 5, labels=False)
        edges = list(x.quantile([0, 0.2, 0.4, 0.6, 0.8, 1]))
        labels = [f"{edges[i]:.{nd}f}-{edges[i + 1]:.{nd}f}" for i in range(5)]
    return g.astype(int), labels


rows = []
fig, axes = plt.subplots(2, 3, figsize=(12, 7), sharey=True)
for ax, (a, title, nd) in zip(axes.flat, ATTRS):
    g, labels = groups(att[a], nd)
    for name, (col, ls) in MODELS.items():
        if name not in scores:
            continue
        s = scores[name].reindex(g.index)
        med = s.groupby(g).median(); n = s.groupby(g).count()
        ax.plot(med.index, med.values, color=col, ls=ls, lw=2, marker="o", ms=4.5, label=name)
        for k in med.index:
            rows.append({"attribute": a, "group": labels[k], "n": int(n[k]), "model": name, "median_NSE": med[k]})
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, fontsize=8.5, color=INK)
    ax.set_title(f"{title}\n({int(g.notna().sum())} catchments, {len(labels)} groups of equal count)", loc="left", fontsize=10, color=INK)
    ax.set_ylim(0.3, 1.0); ax.set_yticks([0.4, 0.6, 0.8, 1.0])
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(labelsize=9, colors=INK)
for ax in axes[:, 0]:
    ax.set_ylabel("median test NSE", fontsize=9.5, color=INK)
axes[0, 0].legend(frameon=False, fontsize=9, loc="lower left")
fig.text(0.5, 0.005, "catchments grouped from low to high attribute value", ha="center", fontsize=9.5, color=INK)
fig.tight_layout(rect=(0, 0.02, 1, 1), w_pad=1.5, h_pad=2.0)
fig.savefig(ROOT / "figures/attribute_bins.png", dpi=150); print("wrote figures/attribute_bins.png")
pd.DataFrame(rows).to_csv(ROOT / "results/overview/table8c_attribute_bins.csv", index=False)
