"""Performance figures for the paper: cumulative distributions over catchments of the test-period metrics (seed
median per catchment) for every model. figures/performance_cdf.png = training catchments, six panels (NSE, KGE,
volume bias, FHV, FLV, FMS; the LC-HBV included when its results exist); figures/ungauged_cdf.png = held-out
catchments, NSE and KGE, regional models only. Reads results/<run>/metrics_test_full.csv.
Usage: uv run python scripts/figures/plot_performance_cdf.py"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from gbhydro.data import ROOT

INK, MUTED = "#222222", "#555555"
MODELS = {"LSTM": ([f"lstm_s{s}" for s in range(5)], "#2a78d6", "-"),
          "RPL-HBV": ([f"hbvc_b128_s{s}" for s in range(5)], "#eb6834", "-"),
          "RFL-HBV (n = 16)": ([f"gatedc3_n16_s{s}" for s in range(5)], "#2e9e5b", "-"),
          "LC-HBV (per catchment)": ([f"hbv_local_r{s}" for s in range(5)], "#8e44ad", "--")}
# metric: (axis label, x limits, ticks, zero line)
PANELS = {"NSE": ("NSE", (-0.5, 1), [-0.5, 0, 0.5, 1], None),
          "KGE": ("KGE", (-0.5, 1), [-0.5, 0, 0.5, 1], None),
          "PBIAS": ("volume bias (%)", (-50, 50), [-50, -25, 0, 25, 50], 0),
          "FHV": ("high-flow bias FHV (%)", (-60, 40), [-60, -30, 0, 30], 0),
          "FLV": ("low-flow bias FLV (%)", (-200, 200), [-200, -100, 0, 100, 200], 0),
          "FMS": ("mid-slope bias FMS (%)", (-60, 40), [-60, -30, 0, 30], 0)}


def table_median(runs, metric, st):
    """The median as the paper's tables give it: median over catchments for each seed, then median over seeds."""
    ds = [pd.read_csv(ROOT / f"results/{r}/metrics_test_full.csv", index_col="basin") for r in runs if (ROOT / f"results/{r}/metrics_test_full.csv").exists()]
    return float(np.median([d.loc[d.set == st, metric].median() for d in ds]))


def seed_median(runs, metric):
    ds = [pd.read_csv(ROOT / f"results/{r}/metrics_test_full.csv", index_col="basin") for r in runs if (ROOT / f"results/{r}/metrics_test_full.csv").exists()]
    if not ds:
        return None
    return pd.DataFrame({"v": pd.concat([d[metric] for d in ds], axis=1).median(axis=1), "set": ds[0].set})


def panel(ax, metric, st, legend):
    label, lim, ticks, zero = PANELS[metric]
    n, lines = 0, []
    for name, (runs, col, ls) in MODELS.items():
        d = seed_median(runs, metric)
        if d is None:
            continue
        lab = name.replace(" (per catchment)", "*") if st == "held-out" else name   # * = calibrated on these catchments
        n = max(n, int(d[d.set == st].v.notna().sum()))
        v = np.sort(d[d.set == st].v.dropna().to_numpy())
        ax.plot(v.clip(*lim), np.arange(1, len(v) + 1) / len(v), color=col, ls=ls, lw=2, label=lab)
        med = table_median(runs, metric, st)
        lines.append((f"{med:+.0f}%" if zero is not None else f"{med:.2f}", col))
    if zero is not None:
        ax.axvline(zero, color="#bbbbbb", lw=0.8, ls=":", zorder=0)
    ax.set_xlim(*lim); ax.set_ylim(0, 1)
    ax.set_xticks(ticks); ax.set_yticks([0, 0.5, 1])
    ax.set_xlabel(label, fontsize=10, color=INK)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(labelsize=9, colors=INK)
    # medians as in the paper's tables, one line per model in its colour, top left
    x0, y0 = 0.03, 0.96
    ax.text(x0, y0, "median", transform=ax.transAxes, fontsize=8, color=MUTED, va="top")
    for i, (t, col) in enumerate(lines):
        ax.text(x0, y0 - 0.085 * (i + 1), t, transform=ax.transAxes, fontsize=8.5, color=col, va="top")
    if legend:
        ax.legend(frameon=False, fontsize=9, loc="upper left", bbox_to_anchor=(0.0, 0.58))
    return n


fig = plt.figure(figsize=(10, 10.5))
gs = fig.add_gridspec(3, 2, height_ratios=[1.5, 1, 1], wspace=0.22, hspace=0.4, left=0.08, right=0.98, top=0.94, bottom=0.06)
for c, metric in enumerate(["NSE", "KGE"]):
    ax = fig.add_subplot(gs[0, c])
    n = panel(ax, metric, "trained", legend=(metric == "NSE"))
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
    if c == 0:
        ax.set_ylabel("fraction of catchments", fontsize=10, color=INK)
for (r, c), metric in zip([(1, 0), (1, 1), (2, 0), (2, 1)], ["PBIAS", "FHV", "FLV", "FMS"]):
    ax = fig.add_subplot(gs[r, c])
    panel(ax, metric, "trained", legend=False)
    if c == 0:
        ax.set_ylabel("fraction of catchments", fontsize=10, color=INK)
fig.suptitle(f"training catchments (n = {n}), test period 1998 to 2008, seed median per catchment", x=0.01, ha="left", fontsize=10, color=MUTED)
fig.savefig(ROOT / "figures/performance_cdf.png", dpi=150); print(f"wrote figures/performance_cdf.png  (trained, n = {n})")

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
for ax, metric in zip(axes, ["NSE", "KGE"]):
    n = panel(ax, metric, "held-out", legend=(metric == "NSE"))
    ax.set_xlabel(f"{metric}, test period 1998 to 2008", fontsize=10, color=INK)
axes[0].set_ylabel("fraction of catchments", fontsize=10, color=INK)
fig.tight_layout()
fig.savefig(ROOT / "figures/ungauged_cdf.png", dpi=150); print(f"wrote figures/ungauged_cdf.png  (held-out, n = {n})")
