"""Rule similarity (correlation of the probed curves) of the learned rules with the LC-HBV over all catchments: one panel per rule, cumulative
distribution of the rule agreement between a model's probed curve and the per-catchment LC-HBV's (the reference,
since its parameters were fitted to each catchment): RFL-HBV in green, RPL-HBV in orange (same equations as the
reference, so this shows how high the agreement can be). Falls back to the RPL-HBV as reference when the
calibrated runs are absent. Reads results/overview/table11_rule_agreement.csv (scripts/evaluation/probe_rules.py agreement).
Usage: uv run python scripts/figures/plot_rule_agreement.py -> figures/rule_agreement.png"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from gbhydro.data import ROOT

INK = "#222222"
RULES = [("et", "ET rule"), ("recharge", "recharge rule"), ("fast", "fast flow rule"),
         ("baseflow", "baseflow rule"), ("melt", "melt rule")]
d = pd.read_csv(ROOT / "results/overview/table11_rule_agreement.csv", index_col="basin")
if any(c.startswith("RFL-HBV vs LC-HBV") for c in d.columns):
    REF = "LC-HBV"
    PAIRS = [("RFL-HBV vs LC-HBV", "#2e9e5b", "-", "RFL-HBV"), ("RPL-HBV vs LC-HBV", "#eb6834", "-", "RPL-HBV")]
else:
    REF = "RPL-HBV"
    PAIRS = [("RFL-HBV vs RPL-HBV", "#2e9e5b", "-", "RFL-HBV")]

fig, axes = plt.subplots(2, 3, figsize=(11, 6.6))
for ax, (rule, title) in zip(axes.flat, RULES):
    n = 0
    for pair, col, ls, label in PAIRS:
        c = f"{pair}: {rule}"
        v = np.sort(d[c].dropna().to_numpy()); n = max(n, len(v))
        ax.plot(v, np.arange(1, len(v) + 1) / len(v), color=col, ls=ls, lw=2, label=f"{label}, median {np.median(v):.2f}")
    ax.set_xlim(-1, 1); ax.set_ylim(0, 1); ax.set_xticks([-1, -0.5, 0, 0.5, 1]); ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
    ax.set_title(f"{title}  ({n} catchments)", loc="left", fontsize=10, color=INK)
    ax.set_xlabel(f"rule similarity (compared with {REF})", fontsize=9.5, color=INK)
    ax.set_ylabel("fraction of catchments", fontsize=9.5, color=INK)
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(labelsize=9, colors=INK)
axes.flat[-1].set_axis_off()
fig.tight_layout(w_pad=1.5, h_pad=2.0)
fig.savefig(ROOT / "figures/rule_agreement.png", dpi=150); print("wrote figures/rule_agreement.png, reference:", REF)
