"""Explainability layer 2 (integrated gradients, scripts/evaluation/xai.py) compared across the three models, seed 0.
Reads results/<run>/ig.npz and ig_summary.csv. Writes results/overview/table9_xai.csv (input shares, memory, top
attributes, agreement between models), table9b_xai_by_bfi.csv (memory by BFI tercile) and figures/xai.png: input
importance for the three models, left the share of attribution by input group, right the share of the attribute
attribution for every one of the 33 attributes. figures/xai_memory.png (supplement): cumulative attribution against
lag and effective memory by BFI group.
Usage: uv run python scripts/figures/plot_xai.py"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from gbhydro.data import ROOT, load_attributes

INK, MUTED, GRID = "#222222", "#666666", "#dddddd"
RUNS = {"LSTM": ("lstm_s0", "#2a78d6"), "RPL-HBV": ("hbvc_b128_s0", "#eb6834"), "RFL-HBV": ("gatedc3_n16_s0", "#2e9e5b")}
att = load_attributes()
k2 = pd.read_csv(ROOT / "results/hbvc_b128_s0/parameters_test.csv", index_col="basin")["K2"]

ig, summ = {}, {}
for label, (run, _) in RUNS.items():
    ig[label] = np.load(ROOT / f"results/{run}/ig.npz", allow_pickle=True)
    summ[label] = pd.read_csv(ROOT / f"results/{run}/ig_summary.csv", index_col="basin")
basins = summ["LSTM"].index
a = att.loc[basins]
bfi = a.baseflow_index; tau = 1.0 / k2.loc[basins]

# table 9: medians per model, correlations of memory with catchment descriptors, agreement between models
rows = {}
for label, s in summ.items():
    r = {"share_P": s.share_precipitation.median(), "share_T": s.share_temperature.median(), "share_PET": s.share_pet.median(),
         "share_static": s.share_static.median(), "mem50_med_d": s.mem50_days.median(), "mem50_10th_d": s.mem50_days.quantile(.1),
         "mem50_90th_d": s.mem50_days.quantile(.9), "mem90_med_d": s.mem90_days.median(),
         "mem50_held_out_d": s[s.set == "held-out"].mem50_days.median(), "comp_err_med": s.comp_err_mean.median(),
         # means over catchments, the statistic of Fig. 12 and the text (Table S9 of the supplement, 22 Nov 2026)
         "share_P_mean": s.share_precipitation.mean(), "share_T_mean": s.share_temperature.mean(),
         "share_PET_mean": s.share_pet.mean(), "share_static_mean": s.share_static.mean()}
    for name, x in [("rho_mem50_bfi", bfi), ("rho_mem50_hbv_tau", tau), ("rho_mem50_aridity", a.aridity), ("rho_mem50_area", a.area)]:
        r[name] = spearmanr(x, s.mem50_days).statistic
    sh = pd.Series(ig[label]["abs_static"].mean(0), index=ig[label]["attr_names"]); sh = sh / sh.sum()
    r["top5_attrs"] = ", ".join(f"{k} {v:.2f}" for k, v in sh.sort_values(ascending=False).head(5).items())
    rows[label] = r
T9 = pd.DataFrame(rows).T
lab = list(summ)
for i in range(3):
    for j in range(i + 1, 3):
        T9.loc[lab[i], f"rho_mem50_vs_{lab[j]}"] = spearmanr(summ[lab[i]].mem50_days, summ[lab[j]].mem50_days).statistic
T9.to_csv(ROOT / "results/overview/table9_xai.csv")
print(T9.drop(columns="top5_attrs").astype(float).round(3).to_string()); print(T9.top5_attrs.to_string())

terc = pd.qcut(bfi, 3, labels=["low BFI", "mid BFI", "high BFI"])
T9b = pd.DataFrame({label: s.mem50_days.groupby(terc, observed=True).median() for label, s in summ.items()})
T9b.loc["1/K2 of RPL-HBV (d)"] = tau.groupby(terc, observed=True).median().to_numpy()
T9b.to_csv(ROOT / "results/overview/table9b_xai_by_bfi.csv"); print(T9b.round(1).to_string())


def style(ax):
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(labelsize=9, colors=INK)


NAMES = {"area": "area", "elev_mean": "mean elevation", "dpsbar": "mean slope", "p_mean": "mean precipitation", "pet_mean": "mean PET",
         "aridity": "aridity", "p_seasonality": "precipitation seasonality", "frac_snow": "snow fraction", "high_prec_freq": "heavy-rain day frequency",
         "high_prec_dur": "heavy-rain spell length", "low_prec_freq": "dry day frequency", "low_prec_dur": "dry spell length",
         "dwood_perc": "deciduous woodland", "ewood_perc": "evergreen woodland", "grass_perc": "grassland", "crop_perc": "cropland", "urban_perc": "urban",
         "sand_perc": "sand", "silt_perc": "silt", "clay_perc": "clay", "organic_perc": "organic soil", "porosity_hypres": "soil porosity",
         "conductivity_hypres": "soil conductivity", "tawc": "available water capacity", "root_depth": "root depth", "soil_depth_pelletier": "soil depth",
         "frac_high_perc": "highly productive aquifer (fractured)", "frac_mod_perc": "moderately productive aquifer (fractured)",
         "inter_high_perc": "highly productive aquifer (intergranular)", "inter_mod_perc": "moderately productive aquifer (intergranular)",
         "no_gw_perc": "no groundwater", "num_reservoir": "number of reservoirs", "reservoir_cap": "reservoir capacity"}

# SHAP-style importance bar chart: mean over catchments of each input's share of the catchment's total absolute
# attribution, the three forcings and the 33 attributes on one scale, sorted by the mean over the three models
imp = {}
for label in RUNS:
    d = ig[label]
    tot = d["abs_forcing"].sum(1) + d["abs_static"].sum(1)
    cols = [str(n) for n in d["forcing_names"]] + [str(n) for n in d["attr_names"]]
    imp[label] = pd.Series(100 * np.concatenate([d["abs_forcing"], d["abs_static"]], 1).__truediv__(tot[:, None]).mean(0), index=cols)
imp = pd.DataFrame(imp)
FORCING = {"precipitation": "rain (365 days)", "temperature": "temperature (365 days)", "pet": "PET (365 days)"}
order = imp.mean(axis=1).sort_values(ascending=True).index
imp.loc[order[::-1]].round(2).to_csv(ROOT / "results/overview/table9c_input_importance.csv")
def bars(order, out, figsize):
    """Horizontal bars of the attribution share per input, one bar per model, inputs from `order` bottom to top."""
    fig, ax = plt.subplots(figsize=figsize)
    y = np.arange(len(order)); w = 0.27
    for i, (label, (run, col)) in enumerate(RUNS.items()):
        ax.barh(y + (i - 1) * w, imp.loc[order, label], w, color=col, lw=0, label=label)
    ax.set_yticks(y); ax.set_yticklabels([FORCING.get(a, NAMES.get(a, a)) for a in order], fontsize=9, color=INK); ax.set_ylim(-0.6, len(order) - 0.4)
    ax.set_xscale("log"); ax.set_xlim(0.3, 60); ax.set_xticks([0.3, 1, 3, 10, 30]); ax.set_xticklabels(["0.3", "1", "3", "10", "30"])
    ax.set_xlabel("share of the total attribution (%, mean over catchments, log scale)", fontsize=9.5, color=INK)
    ax.legend(frameon=False, fontsize=9.5, loc="lower right")
    ax.tick_params(axis="y", length=0)
    style(ax)
    fig.tight_layout(); fig.savefig(ROOT / out, dpi=150); print("wrote", out)


bars(order, "figures/xai_full.png", (8.5, 10))                      # all 36 inputs, supplement
bars(order[-9:], "figures/xai.png", (8.5, 3.6))                     # three forcings and the six largest attributes, main text

# supplement: memory
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.4))
lags = np.arange(1, 366)
for label, (run, col) in RUNS.items():
    lm = ig[label]["lag_mass"]; prof = lm / lm.sum(1, keepdims=True)
    med = np.cumsum(np.median(prof, 0)); med = med / med[-1]
    ax1.plot(lags, med, color=col, lw=2, label=f"{label}  (half within {summ[label].mem50_days.median():.0f} days)")
ax1.set_xscale("log"); ax1.set_xlim(1, 365); ax1.set_ylim(0, 1); ax1.axhline(0.5, color="#bbbbbb", lw=0.8, ls=":")
ax1.set_xticks([1, 10, 100, 365]); ax1.set_xticklabels(["1", "10", "100", "365"])
ax1.set_xlabel("days before the target day (log scale)", fontsize=9.5, color=INK)
ax1.set_ylabel("cumulative share of attribution to P, T and PET", fontsize=9.5, color=INK)
ax1.set_title("how far back the flow depends on the forcing", loc="left", fontsize=10, color=INK)
ax1.legend(frameon=False, fontsize=9, loc="upper left")
terc5 = pd.qcut(bfi, 5, labels=False)
edges = bfi.quantile([0, .2, .4, .6, .8, 1]).to_numpy()
labels = [f"{edges[i]:.2f}-{edges[i + 1]:.2f}" for i in range(5)]
for label, (run, col) in RUNS.items():
    s = summ[label]; rho = spearmanr(bfi, s.mem50_days).statistic
    g = s.mem50_days.groupby(terc5)
    ax2.plot(range(5), g.median(), color=col, lw=2, marker="o", ms=5, label=f"{label}  (Spearman with BFI {rho:.2f})")
    ax2.fill_between(range(5), g.quantile(.25), g.quantile(.75), color=col, alpha=0.12, lw=0)
ax2.set_yscale("log"); ax2.set_ylim(5, 300); ax2.set_yticks([5, 10, 20, 50, 100, 200]); ax2.set_yticklabels(["5", "10", "20", "50", "100", "200"])
ax2.set_xticks(range(5)); ax2.set_xticklabels(labels, fontsize=8.5)
ax2.set_xlabel("observed baseflow index, five groups of equal count", fontsize=9.5, color=INK)
ax2.set_ylabel("effective memory: days holding half the attribution", fontsize=9.5, color=INK)
ax2.set_title("effective memory by baseflow index", loc="left", fontsize=10, color=INK)
ax2.legend(frameon=False, fontsize=9, loc="upper left")
style(ax1); style(ax2)
fig.tight_layout(w_pad=1.5); fig.savefig(ROOT / "figures/xai_memory.png", dpi=150); print("wrote figures/xai_memory.png")
