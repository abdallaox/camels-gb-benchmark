"""Pearson correlation of catchment attributes with each model's test NSE (floored at 0 so a few failed catchments do not
dominate) and with the gaps between models: a dot plot with one row per attribute: left panel the four models' NSE
(one coloured dot each), right panel the two gaps between models. Attributes with no correlation anywhere are dropped.
Flow signatures (Q95, Q5, runoff ratio, BFI ...) appear as descriptors only; they are never model inputs.
Writes results/overview/table8b_attribute_correlations_figure.csv and figures/attribute_correlations.png.
Usage: uv run python scripts/figures/plot_attribute_correlations.py"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

from gbhydro.data import ROOT, load_attributes

INK = "#222222"
COLOURS = {"LSTM": "#2a78d6", "RPL-HBV": "#eb6834", "RFL-HBV": "#2e9e5b", "LC-HBV": "#8e44ad"}
ATTRS = [("urban_perc", "urban"), ("crop_perc", "crop"), ("grass_perc", "grass"), ("dwood_perc", "deciduous wood"), ("frac_snow", "snow fraction"),
         ("aridity", "aridity"), ("pet_mean", "PET mean"), ("p_mean", "P mean"), ("p_seasonality", "P seasonality"),
         ("Q95", "Q95"), ("Q5", "Q5"), ("q_mean", "Q mean"), ("runoff_ratio", "runoff ratio"), ("baseflow_index", "baseflow index"), ("slope_fdc", "FDC slope"),
         ("frac_high_perc", "highly productive aquifer"), ("no_gw_perc", "no groundwater"), ("clay_perc", "clay"), ("sand_perc", "sand"), ("tawc", "available water capacity"),
         ("elev_mean", "elevation mean"), ("dpsbar", "slope"), ("area", "area"), ("groundwater_abs", "groundwater abstraction"), ("num_reservoir", "reservoirs")]
pb = pd.read_csv(ROOT / "results/overview/per_basin_seed_median_nse.csv", index_col="basin")
att = load_attributes().loc[pb.index]
L, H, G = pb["LSTM"].clip(lower=0), pb["HBV regional (continuous)"].clip(lower=0), pb["RFL-HBV n=16 (constrained)"].clip(lower=0)
scores = {"LSTM\nNSE": L, "RPL-HBV\nNSE": H, "RFL-HBV\nNSE": G}
cal = [ROOT / f"results/hbv_local_r{s}/metrics_test_full.csv" for s in range(5)]
if all(f.exists() for f in cal):
    C = pd.concat([pd.read_csv(f, index_col="basin").NSE for f in cal], axis=1).median(axis=1).reindex(pb.index).clip(lower=0)
    C[pb.set == "held-out"] = np.nan            # gauged baseline: training catchments only
    scores["LC-HBV\nNSE"] = C
scores.update({"LSTM minus\nRPL-HBV": L - H, "RFL minus\nRPL-HBV": G - H})

rho = pd.DataFrame(index=[n for _, n in ATTRS], columns=list(scores), dtype=float); pv = rho.copy()
for a, name in ATTRS:
    x = att[a]
    for k, s in scores.items():
        m = x.notna() & s.notna()
        r = pearsonr(x[m], s[m]); rho.loc[name, k], pv.loc[name, k] = r.statistic, r.pvalue
out = pd.concat({"r": rho, "p": pv}, axis=1); out.to_csv(ROOT / "results/overview/table8b_attribute_correlations_figure.csv")
print(rho.round(2).to_string())

KEY = {"aridity", "P mean", "area", "highly productive aquifer", "baseflow index", "groundwater abstraction"}
DROP = {"snow fraction", "Q5", "no groundwater", "clay", "sand", "available water capacity"}
keep = [n for _, n in ATTRS if n not in DROP]
rho = rho.loc[keep]
models = [k for k in scores if "minus" not in k]; gaps = [k for k in scores if "minus" in k]
fig, axes = plt.subplots(1, 2, figsize=(10, 7.2), sharey=True, gridspec_kw={"width_ratios": [1.15, 1]})
y = np.arange(len(keep))[::-1]
for ax, cols, title in ((axes[0], models, "test NSE of each model"), (axes[1], gaps, "difference between models")):
    for yi, n in zip(y, keep):
        if n in KEY:
            ax.axhspan(yi - 0.5, yi + 0.5, color="#eeeeee", lw=0, zorder=0)
        ax.plot([-0.5, 0.5], [yi, yi], color="#dddddd", lw=0.8, zorder=1)
    ax.axvline(0, color=INK, lw=0.8, zorder=2)
    for k in cols:
        name = k.split(chr(10))[0]
        col = COLOURS.get(name, "#888888")
        marker, mfc = ("o", col) if name in COLOURS else (("D", col) if "LSTM" in k else ("s", "white"))
        ax.plot(rho.loc[keep, k].to_numpy(), y, ls="none", marker=marker, ms=7 if marker == "o" else 6, mfc=mfc, mec=col, mew=1.6,
                label=k.replace(chr(10), " "), zorder=3)
    ax.set_xlim(-0.5, 0.5); ax.set_xticks([-0.4, -0.2, 0, 0.2, 0.4])
    ax.set_title(title, loc="left", fontsize=10, color=INK)
    ax.legend(frameon=False, fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.06), ncol=2, handletextpad=0.3, columnspacing=1.2)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(axis="y", length=0); ax.tick_params(labelsize=9, colors=INK)
axes[0].set_yticks(y); axes[0].set_yticklabels(keep, fontsize=9.5, color=INK)
for lab, n in zip(axes[0].get_yticklabels(), keep):
    if n in KEY:
        lab.set_fontweight("bold")
fig.text(0.5, 0.005, "Pearson correlation with the score (positive: higher attribute, higher score)", ha="center", fontsize=9.5, color=INK)
fig.tight_layout(rect=(0, 0.02, 1, 1), w_pad=1.0)
fig.savefig(ROOT / "figures/attribute_correlations.png", dpi=150); print("wrote figures/attribute_correlations.png")
