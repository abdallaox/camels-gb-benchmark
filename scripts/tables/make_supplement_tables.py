"""Supplement tables. Table S1: the 33 static attributes every regional model receives (config/study.yml, `extended`),
with description, unit and the range over the 669 study catchments, read from the CAMELS-GB attribute files so the
table never drifts from what the models used. Usage: uv run python scripts/tables/make_supplement_tables.py"""
import glob

import pandas as pd
import yaml

from gbhydro.data import ROOT

OUT = ROOT / "paper/tables"; OUT.mkdir(parents=True, exist_ok=True)
GROUP = {"topographic": "Topography", "climatic": "Climate", "landcover": "Land cover", "soil": "Soil",
         "hydrogeology": "Hydrogeology", "humaninfluence": "Human influence"}
DESC = {  # descriptions and units follow Coxon et al. (2020), Tables 3 to 8
    "area": ("catchment area", "km$^2$"), "elev_mean": ("mean elevation", "m"), "dpsbar": ("mean drainage path slope", "m km$^{-1}$"),
    "p_mean": ("mean daily precipitation", "mm d$^{-1}$"), "pet_mean": ("mean daily PET", "mm d$^{-1}$"),
    "aridity": ("aridity, PET / P", "--"), "p_seasonality": ("precipitation seasonality", "--"),
    "frac_snow": ("fraction of precipitation falling as snow", "--"),
    "high_prec_freq": ("frequency of high-precipitation days ($\\geq 5 \\times$ mean)", "d yr$^{-1}$"),
    "high_prec_dur": ("mean duration of high-precipitation events", "d"),
    "low_prec_freq": ("frequency of dry days ($< 1$ mm d$^{-1}$)", "d yr$^{-1}$"), "low_prec_dur": ("mean duration of dry periods", "d"),
    "dwood_perc": ("deciduous woodland", "\\%"), "ewood_perc": ("evergreen woodland", "\\%"), "grass_perc": ("grass and pasture", "\\%"),
    "crop_perc": ("crops", "\\%"), "urban_perc": ("urban", "\\%"),
    "sand_perc": ("sand content", "\\%"), "silt_perc": ("silt content", "\\%"), "clay_perc": ("clay content", "\\%"),
    "organic_perc": ("organic content", "\\%"), "porosity_hypres": ("porosity (HYPRES)", "\\%"),
    "conductivity_hypres": ("saturated hydraulic conductivity (HYPRES)", "cm h$^{-1}$"), "tawc": ("total available water content", "mm"),
    "root_depth": ("root depth", "m"), "soil_depth_pelletier": ("depth to bedrock", "m"),
    "frac_high_perc": ("fractured aquifer, highly productive", "\\%"), "frac_mod_perc": ("fractured aquifer, moderately productive", "\\%"),
    "inter_high_perc": ("intergranular aquifer, highly productive", "\\%"), "inter_mod_perc": ("intergranular aquifer, moderately productive", "\\%"),
    "no_gw_perc": ("rocks with essentially no groundwater", "\\%"),
    "num_reservoir": ("number of reservoirs", "--"), "reservoir_cap": ("total reservoir capacity", "ML"),
}

attrs = yaml.safe_load((ROOT / "config/study.yml").read_text())["static_attributes"]["extended"]
basins = [int(b) for b in (ROOT / "config/basins_all.txt").read_text().split()]
rows = {}
for f in sorted(glob.glob(str(ROOT / "data/camels_gb/data/CAMELS_GB_*_attributes.csv"))):
    d = pd.read_csv(f, index_col="gauge_id").loc[basins]
    g = f.split("CAMELS_GB_")[1].split("_attributes")[0]
    for c in d.columns:
        if c in attrs:
            rows[c] = (GROUP[g], d[c].min(), d[c].median(), d[c].max())
assert set(rows) == set(attrs), set(attrs) - set(rows)


def fmt(v):
    return f"{v:.0f}" if abs(v) >= 100 else (f"{v:.1f}" if abs(v) >= 10 else f"{v:.2f}")


body = "group & attribute & description & unit & min & median & max \\\\\n\\midrule\n"
last = None
for a in attrs:
    g, lo, med, hi = rows[a]
    body += f"{g if g != last else ''} & \\texttt{{{a.replace('_', '\\_')}}} & {DESC[a][0]} & {DESC[a][1]} & {fmt(lo)} & {fmt(med)} & {fmt(hi)} \\\\\n"
    last = g
tex = ("\\begin{table}[htbp]\n\\centering\n\\footnotesize\n\\setlength{\\tabcolsep}{4pt}\n"
       "\\caption{The 33 static catchment attributes given to the regional models, with their range over the 669 study "
       "catchments. Names, descriptions and units follow \\citet{coxon2020}. No attribute derived from observed flow is "
       "included.}\n\\label{tab:attributes}\n"
       "\\resizebox{\\ifdim\\width>\\textwidth\\textwidth\\else\\width\\fi}{!}{%\n"
       f"\\begin{{tabular}}{{lllrrrr}}\n\\toprule\n{body}\\bottomrule\n\\end{{tabular}}}}\n\\end{{table}}\n")
(OUT / "tableS1_attributes.tex").write_text(tex, encoding="utf-8"); print("wrote", OUT / "tableS1_attributes.tex")
