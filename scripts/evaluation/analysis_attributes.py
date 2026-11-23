"""Research question 4: which catchment characteristics explain performance, the gaps between models, and store
realism. Spearman rank correlations of per-catchment scores (seed medians) with CAMELS-GB attributes, over all 669
catchments. Flow-derived attributes (BFI, runoff ratio) are used here as descriptors only, never as inputs.
Writes results/overview/table8_attribute_correlations.csv. Usage: uv run python scripts/evaluation/analysis_attributes.py"""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from gbhydro.data import ROOT, load_attributes

ATTRS = ["baseflow_index", "runoff_ratio", "aridity", "p_mean", "pet_mean", "p_seasonality", "frac_snow", "elev_mean", "dpsbar", "area",
         "frac_high_perc", "inter_high_perc", "no_gw_perc", "clay_perc", "sand_perc", "tawc", "soil_depth_pelletier",
         "urban_perc", "crop_perc", "dwood_perc", "num_reservoir", "reservoir_cap", "groundwater_abs", "surfacewater_abs"]
pb = pd.read_csv(ROOT / "results/overview/per_basin_seed_median_nse.csv", index_col="basin")
soil = pd.read_csv(ROOT / "results/overview/per_basin_soil_anom_corr.csv", index_col="basin")
att = load_attributes().loc[pb.index]
scores = {"NSE LSTM": pb["LSTM"], "NSE HBV": pb["HBV regional (continuous)"], "NSE gated": pb["RFL-HBV n=16 (constrained)"],
          "LSTM minus HBV": pb["LSTM"] - pb["HBV regional (continuous)"], "gated minus HBV": pb["RFL-HBV n=16 (constrained)"] - pb["HBV regional (continuous)"],
          "LSTM minus gated": pb["LSTM"] - pb["RFL-HBV n=16 (constrained)"],
          "soil realism HBV": soil["RPL-HBV"], "soil realism gated": soil["RFL-HBV (n = 16)"]}
rows = []
for a in ATTRS:
    x = att[a]; ok = x.notna()
    r = {"attribute": a, "n": int(ok.sum())}
    for k, s in scores.items():
        m = ok & s.notna()
        rho, p = spearmanr(x[m], s[m])
        r[k] = rho; r[k + " p"] = p
    rows.append(r)
T8 = pd.DataFrame(rows).set_index("attribute"); T8.to_csv(ROOT / "results/overview/table8_attribute_correlations.csv")
show = T8[[k for k in scores]].round(2)
print("Spearman rho of per-catchment scores with attributes (|rho| >= 0.2 marked *)")
print(show.map(lambda v: f"{v:+.2f}{'*' if abs(v) >= 0.2 else ' '}").to_string())
