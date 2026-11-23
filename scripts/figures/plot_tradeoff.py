"""Performance against explainability: test NSE against soil-store realism (anomaly correlation with ERA5-Land), both
seed medians per catchment, trained catchments. figures/tradeoff.png (main): the three HBV-structured models as medians
with bars spanning the 25th to 75th percentile of catchments; the LSTM has an NSE but no store, drawn as a vertical
dashed line. figures/tradeoff_sweep.png (supplement): the same with the unconstrained RFL-HBV and the RFL-HBV as n
grows from 2 to 48 units per store (seed 0; n = 8 the five-seed median; results/overview/table10_size_sweep.csv);
figures/size_sweep.png (main): the sweep panel alone.
Usage: uv run python scripts/figures/plot_tradeoff.py"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from gbhydro.data import ROOT

INK = "#222222"
COL = {"RPL-HBV": "#eb6834", "RFL-HBV": "#2e9e5b", "LC-HBV": "#8e44ad", "RFL-HBV, unconstrained": "#9e9e9e"}
nse = pd.read_csv(ROOT / "results/overview/per_basin_seed_median_nse.csv", index_col="basin")
soil = pd.read_csv(ROOT / "results/overview/per_basin_soil_anom_corr.csv", index_col="basin").reindex(nse.index)
NSE = {"RPL-HBV": nse["HBV regional (continuous)"], "RFL-HBV": nse["RFL-HBV n=16 (constrained)"], "RFL-HBV, unconstrained": nse["RFL-HBV n=8 (unconstrained)"]}
SOIL = {"RPL-HBV": soil["RPL-HBV"], "RFL-HBV": soil["RFL-HBV (n = 16)"], "RFL-HBV, unconstrained": soil["RFL-HBV, unconstrained"]}
if "LC-HBV" in soil:
    NSE["LC-HBV"] = pd.concat([pd.read_csv(ROOT / f"results/hbv_local_r{s}/metrics_test_full.csv", index_col="basin").NSE for s in range(5)], axis=1).median(axis=1).reindex(nse.index)
    SOIL["LC-HBV"] = soil["LC-HBV"]
else:
    COL.pop("LC-HBV")
# legend NSE = median of the per-seed medians from Table 1 (the paper's convention); the dot is the median over
# catchments of the seed-median NSE, which can differ in the third decimal
_t1 = pd.read_csv(ROOT / "results/overview/table1b_metrics_full.csv")
_t1 = _t1[_t1.set == "trained"].set_index("model").NSE
T1 = {"RPL-HBV": _t1.get("RPL-HBV"), "RFL-HBV": _t1.get("RFL-HBV ($n = 16$)")}
T1 = {k: v for k, v in T1.items() if v is not None}
SW = pd.read_csv(ROOT / "results/overview/table10_size_sweep.csv", index_col="n"); SW.index = SW.index.astype(int)


def style(ax):
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(labelsize=9, colors=INK)


m = nse.set == "trained"


def medians_panel(ax, models, lstm_label):
    for k in models:
        c = COL[k]
        x, y = NSE[k][m].clip(0), SOIL[k][m]
        xq, yq = x.quantile([0.25, 0.5, 0.75]).to_numpy(), y.quantile([0.25, 0.5, 0.75]).to_numpy()
        open_ = "unconstrained" in k
        ax.errorbar(xq[1], yq[1], xerr=[[xq[1] - xq[0]], [xq[2] - xq[1]]], yerr=[[yq[1] - yq[0]], [yq[2] - yq[1]]],
                    fmt="o", ms=10, color=c, mfc="white" if open_ else c, mew=1.8, elinewidth=1.3, capsize=0, zorder=5,
                    label=f"{k}  (NSE {T1.get(k, xq[1]):.2f}, realism {yq[1]:.2f})")
    lx = nse["LSTM"][m].median()
    ax.axvline(lx, color="#2a78d6", lw=1.8, ls="--", zorder=2)
    ax.text(lx - 0.006, 0.89, lstm_label.format(lx), color="#2a78d6", fontsize=9, ha="right", va="top")
    ax.set_xlim(0.6, 0.9); ax.set_ylim(0.6, 0.9)
    ax.set_xlabel("test NSE 1998 to 2008 (median and interquartile range over catchments)", fontsize=9.5, color=INK)
    ax.set_ylabel("soil realism (compared with ERA5-Land)", fontsize=9.5, color=INK)
    style(ax)


# main figure: one panel, three HBV-structured models, trained catchments
fig, ax = plt.subplots(figsize=(6.4, 5.2))
medians_panel(ax, ["RPL-HBV", "LC-HBV", "RFL-HBV"], "LSTM median NSE {:.2f}" + chr(10) + "(no store to validate)")
ax.legend(frameon=False, fontsize=9, loc="lower left")
ax.set_title(f"trained catchments ({int(m.sum())})", loc="left", fontsize=10.5, color=INK)
fig.tight_layout(); fig.savefig(ROOT / "figures/tradeoff.png", dpi=150); print("wrote figures/tradeoff.png")

# supplement figure: the same with the unconstrained model and the size sweep
fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2))
medians_panel(axes[0], list(COL), "LSTM median NSE {:.2f}" + chr(10) + "(no store to validate)")
axes[0].plot(SW["NSE_trained"], SW["soil_anom_trained"], color=INK, lw=0.9, zorder=3)
axes[0].scatter(SW["NSE_trained"], SW["soil_anom_trained"], s=14, color=INK, zorder=4, label="RFL-HBV, n = 2 to 32 (right)")
axes[0].legend(frameon=False, fontsize=8.5, loc="lower left")
axes[0].set_title(f"trained catchments ({int(m.sum())})", loc="left", fontsize=10.5, color=INK)
def sweep_panel(ax):
    """The RFL-HBV at every n on the NSE-realism plane, with the three models' medians and the LSTM line."""
    ax.plot(SW["NSE_trained"], SW["soil_anom_trained"], color=INK, lw=1, zorder=3)
    ax.scatter(SW["NSE_trained"], SW["soil_anom_trained"], s=28, color=INK, zorder=4)
    for n_, r in SW.iterrows():
        ax.annotate(f"n = {n_}", (r["NSE_trained"], r["soil_anom_trained"]), xytext=(5, 5), textcoords="offset points", fontsize=8.5, color=INK)
    for k, off, ha in (("RPL-HBV", (-8, -3), "right"), ("LC-HBV", (0, -14), "center"), ("RFL-HBV", (-8, -12), "right")):
        if k not in COL:
            continue
        x, y = NSE[k][m].median(), SOIL[k][m].median()
        ax.scatter(x, y, s=80, color=COL[k], zorder=5)
        ax.annotate(k + (" (n = 16, 5 seeds)" if k == "RFL-HBV" else ""), (x, y), xytext=off, textcoords="offset points", fontsize=8.5, color=COL[k], ha=ha, va="center")
    lx = nse["LSTM"][m].median(); ax.axvline(lx, color="#2a78d6", lw=1.8, ls="--", zorder=2)
    ax.text(lx - 0.004, 0.705, f"LSTM {lx:.2f}", color="#2a78d6", fontsize=8.5, ha="right")
    ax.set_xlim(0.73, 0.89); ax.set_ylim(0.7, 0.85)
    ax.set_xlabel("median test NSE", fontsize=9.5, color=INK); ax.set_ylabel("median soil realism", fontsize=9.5, color=INK)
    ax.set_title("RFL-HBV as n grows, trained catchments", loc="left", fontsize=10.5, color=INK)
    style(ax)


sweep_panel(axes[1])
fig.tight_layout(w_pad=2); fig.savefig(ROOT / "figures/tradeoff_sweep.png", dpi=150); print("wrote figures/tradeoff_sweep.png")

# main-text figure of the size sweep alone
fig, ax = plt.subplots(figsize=(6.4, 5.2))
sweep_panel(ax)
fig.tight_layout(); fig.savefig(ROOT / "figures/size_sweep.png", dpi=150); print("wrote figures/size_sweep.png")
