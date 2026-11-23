"""LaTeX tables for the paper from the CSVs in results/overview (never retype a number).
Writes paper/tables/table1.tex (performance), table3.tex (by catchment type), table7.tex (store realism),
table9.tex (attribution). Usage: uv run python scripts/tables/make_tables.py"""
import numpy as np
import pandas as pd

from gbhydro.data import ROOT

OUT = ROOT / "paper/tables"; OUT.mkdir(parents=True, exist_ok=True)
NAMES = {"LSTM": "LSTM", "HBV regional (continuous)": "RPL-HBV", "RFL-HBV n=16 (constrained)": "RFL-HBV ($n = 16$)",
         "RFL-HBV n=8 (unconstrained)": "RFL-HBV, unconstrained", "HBV regional (windowed, wide ranges)": "RPL-HBV, windowed"}


def write(name, body, caption, label, cols):
    tex = ("\\begin{table}[htbp]\n\\centering\n\\footnotesize\n\\setlength{\\tabcolsep}{4pt}\n" f"\\caption{{{caption}}}\n\\label{{{label}}}\n"
           "\\resizebox{\\ifdim\\width>\\textwidth\\textwidth\\else\\width\\fi}{!}{%\n"      # shrink only if wider than the text
           f"\\begin{{tabular}}{{{cols}}}\n\\toprule\n{body}\\bottomrule\n\\end{{tabular}}}}\n\\end{{table}}\n")
    (OUT / f"{name}.tex").write_text(tex, encoding="utf-8"); print("wrote", OUT / f"{name}.tex")


def esc(x):
    return str(x).replace("%", "\\%").replace("<", "$<$").replace(">", "$>$")


def rng(v, r, nd=2):
    return f"{v:.{nd}f}" + (f" ({r.replace('..', ' to ')})" if isinstance(r, str) and r else "")


# tables 1 (training catchments, test period) and 5 (held-out catchments = ungauged), from the per-run full metrics
GROUPS = {"LSTM": ([f"lstm_s{s}" for s in range(5)], "lstm_ensemble"),
          "RPL-HBV": ([f"hbvc_b128_s{s}" for s in range(5)], "hbvc_b128_ensemble"),
          "RFL-HBV ($n = 16$)": ([f"gatedc3_n16_s{s}" for s in range(5)], "gatedc3_n16_ensemble"),
          "RFL-HBV, unconstrained": (["gatedc_s0"], None),
          "LC-HBV (per catchment)": ([f"hbv_local_r{s}" for s in range(5)], None)}


def seed_stats(runs, st):
    """Per-seed medians over the catchments of set `st`, then the median and range over seeds; share above 0.8 / below 0.5."""
    ds = [pd.read_csv(ROOT / f"results/{r}/metrics_test_full.csv", index_col="basin") for r in runs if (ROOT / f"results/{r}/metrics_test_full.csv").exists()]
    if not ds:
        return None
    ds = [d[d.set == st] for d in ds]
    per = pd.DataFrame([d[["NSE", "KGE", "PBIAS", "FHV", "FLV", "FMS"]].median() for d in ds])
    out = per.median().to_dict()
    out["n"] = int(ds[0].NSE.notna().sum()); out["seeds"] = len(ds)
    out["NSE_range"] = f"{per.NSE.min():.3f}..{per.NSE.max():.3f}" if len(ds) > 1 else ""
    out["KGE_range"] = f"{per.KGE.min():.3f}..{per.KGE.max():.3f}" if len(ds) > 1 else ""
    out["gt08"] = np.median([(d.NSE > 0.8).mean() for d in ds]); out["lt05"] = np.median([(d.NSE < 0.5).mean() for d in ds])
    return out


def catchment_nse(runs, st):
    """Per-catchment NSE, median over seeds (or restarts), on the catchments of set `st`."""
    ds = [pd.read_csv(ROOT / f"results/{r}/metrics_test_full.csv", index_col="basin") for r in runs]
    m = pd.concat([d.NSE for d in ds], axis=1).median(axis=1)
    return m[ds[0].set.reindex(m.index) == st]


def ens_median(name, st):
    f = ROOT / f"results/{name}/metrics_test.csv"
    if name is None or not f.exists():
        return None
    d = pd.read_csv(f, index_col="basin"); return float(d[d.set == st].NSE.median())


body = ("model & NSE & KGE & NSE $>$ 0.8 & NSE $<$ 0.5 & LSTM ahead & gap to LSTM & ensemble NSE & bias (\\%) & FHV (\\%) & "
        "FLV (\\%) & FMS (\\%) \\\\\n\\midrule\n")
tr = {}
lstm_c = catchment_nse(GROUPS["LSTM"][0], "trained")
main = ("model & NSE & KGE & NSE $<$ 0.5 & LSTM ahead & gap to LSTM & FHV (\\%) & FLV (\\%) \\\\\n\\midrule\n")
for label, (runs, ens) in GROUPS.items():
    s = seed_stats(runs, "trained")
    if s is None:
        continue
    tr[label] = s; e = ens_median(ens, "trained")
    if label == "LSTM":
        win = "--"
    else:
        d = (lstm_c - catchment_nse(runs, "trained")).dropna()
        win = f"{100 * (d > 0).mean():.0f}\\% & {d.median():.3f}"
    win = win if win != "--" else "-- & --"
    body += (f"{label} & {rng(s['NSE'], s['NSE_range'], 3)} & {rng(s['KGE'], s['KGE_range'], 3)} & {100 * s['gt08']:.0f}\\% & {100 * s['lt05']:.0f}\\% & {win} & "
             f"{'--' if e is None else f'{e:.2f}'} & {s['PBIAS']:+.1f} & {s['FHV']:+.0f} & {s['FLV']:+.0f} & {s['FMS']:+.0f} \\\\\n")
    if "unconstrained" not in label:
        main += (f"{label.replace(' (per catchment)', '').replace(' ($n = 16$)', '')} & {s['NSE']:.3f} & {s['KGE']:.3f} & "
                 f"{100 * s['lt05']:.0f}\\% & {win} & {s['FHV']:+.0f} & {s['FLV']:+.0f} \\\\\n")
n_tr = next(iter(tr.values()))["n"]
write("table1_main", main, f"Test-period performance on the {n_tr} training catchments, medians. LSTM ahead is the share of "
      "catchments in which the LSTM has the higher NSE, and the gap is the median NSE difference to it. Table~S5 adds the "
      "ranges over seeds.", "tab:performance_main", "lrrrrrrr")
write("table1", body, f"Test-period (1998 to 2008) performance on the {n_tr} training catchments: medians over catchments of the "
      "per-seed values, with the range over five seeds in brackets (the unconstrained RFL-HBV has one seed), the share of "
      "catchments with NSE above 0.8 and below 0.5, the share of catchments in which the LSTM has the higher NSE and the median "
      "NSE difference to the LSTM (seed or restart median per catchment), the median NSE of the five-member ensemble mean, and the median percent bias "
      "and the high-flow, low-flow and mid-slope biases of the flow duration curve (FHV, FLV, FMS) of \\citet{yilmaz2008}.",
      "tab:performance", "lllllrrrrrrr")

body = "model & NSE & KGE & NSE $<$ 0.5 & ensemble NSE & drop from training \\\\\n\\midrule\n"
for label, (runs, ens) in GROUPS.items():
    s = seed_stats(runs, "held-out")
    if s is None or "unconstrained" in label:
        continue
    e = ens_median(ens, "held-out")
    body += (f"{label.replace(' (per catchment)', '*')} & {rng(s['NSE'], s['NSE_range'])} & {rng(s['KGE'], s['KGE_range'])} & {100 * s['lt05']:.0f}\\% & "
             f"{'--' if e is None else f'{e:.2f}'} & "
             + f"{tr[label]['NSE'] - s['NSE']:+.3f}".replace("-", "$-$").replace("+", "$+$") + " \\\\\n")
    n_ho = s["n"]
write("table5", body, f"Prediction in the {n_ho} held-out catchments, test period, medians with the seed range in brackets. "
      "The drop is the fall in median NSE from the training catchments. *The LC-HBV is calibrated on each of these "
      "catchments and is the gauged reference.",
      "tab:ungauged", "lllrrr")

# table 3: medians by catchment type
t = pd.read_csv(ROOT / "results/overview/table3_by_catchment_type.csv")
keep = ["BFI tercile", "aridity tercile", "chalk/high-productivity aquifer (frac_high_perc)", "urban_perc", "elev_mean (m)", "abstraction (% of mean flow)"]
labels = {"BFI tercile": "BFI", "aridity tercile": "aridity", "chalk/high-productivity aquifer (frac_high_perc)": "productive aquifer share",
          "urban_perc": "urban share", "elev_mean (m)": "mean elevation (m)", "abstraction (% of mean flow)": "abstraction (\\% of mean flow, England)"}
body = "grouping & class & $n$ & LSTM & RPL-HBV & RFL-HBV & LC-HBV & LSTM $-$ RPL & RFL $-$ RPL & LC $-$ RPL \\\\\n\\midrule\n"
for g in keep:
    for i, (_, r) in enumerate(t[t.group == g].iterrows()):
        body += (f"{labels[g] if i == 0 else ''} & {esc(r.level)} & {r.n} & {r.LSTM:.2f} & {r['HBV regional (continuous)']:.2f} & "
                 f"{r['RFL-HBV n=16 (constrained)']:.2f} & {r['LC-HBV']:.2f} & {r['LSTM-HBV']:+.3f} & {r['gated-HBV']:+.3f} & "
                 f"{r['LC-RPL']:+.3f} \\\\\n")
write("table3", body, "Median test NSE (seed medians, restart medians for the LC-HBV, the 667 catchments with a test record) "
      "by catchment class, and the median paired difference between models. The LC-HBV is calibrated on every catchment.",
      "tab:by_type", "llrrrrrrrr")

# table 3b: by catchment area (results/overview/table6_by_area.csv, all catchments)
t = pd.read_csv(ROOT / "results/overview/table6_by_area.csv")
t = t[t.set == "all"]
body = "area (km$^2$) & $n$ & LSTM & RPL-HBV & RFL-HBV & LSTM $-$ RPL & RFL $-$ RPL & LSTM $<$ 0.5 & RPL $<$ 0.5 & RFL $<$ 0.5 \\\\\n\\midrule\n"
for _, r in t.iterrows():
    body += (f"{esc(r['area km2'])} & {r.n} & {r.LSTM:.2f} & {r.HBV:.2f} & {r.gated:.2f} & {r['LSTM-HBV']:+.3f} & {r['gated-HBV']:+.3f} & "
             f"{r['LSTM<0.5 %']:.0f}\\% & {r['HBV<0.5 %']:.0f}\\% & {r['gated<0.5 %']:.0f}\\% \\\\\n")
write("table3b", body, "Median test NSE (seed medians, the 667 catchments with a test record) by catchment area, the median paired difference between models, "
      "and the share of catchments with NSE below 0.5.", "tab:by_area", "lrrrrrrrrr")

# table 7: store realism
t = pd.read_csv(ROOT / "results/overview/table7_store_realism.csv", index_col=0)
body = "model & soil anomaly corr. & 10th pct & held-out & snow corr. & snow peak bias (\\%) & snow mean ratio \\\\\n\\midrule\n"
for name, r in t[~t.index.str.contains("unconstrained")].iterrows():
    body += (f"{name} & {rng(r.soil_anom_med, r.soil_anom_med_range)} & {r.soil_anom_10th:.2f} & {r.soil_anom_held:.2f} & "
             f"{rng(r.snow_corr_med, r.snow_corr_med_range)} & {r.snow_peak_bias_med:.0f} & {r.snow_mean_ratio_med:.2f} \\\\\n")
write("table7", body, f"Store realism, medians with the seed range in brackets. Soil: all 669 catchments, with the 10th "
      f"percentile and the held-out catchments. Snow: the {int(t.n_snowy.iloc[0])} snowy catchments, with the bias of the "
      "annual peak and the ratio of the mean snowpack to ERA5-Land.", "tab:realism", "lllllrr")

# table 10: size sweep of the RFL-HBV (results/overview/table10_size_sweep.csv from scripts/evaluation/size_sweep_table.py)
t = pd.read_csv(ROOT / "results/overview/table10_size_sweep.csv", index_col="n")
body = "$n$ & weights & seeds & NSE trained & NSE held-out & KGE trained & soil realism & soil 10th pct & snow corr. & best epoch \\\\\n\\midrule\n"
for n, r in t.iterrows():
    body += (f"{int(n)} & {int(r.n_params):,} & {int(r.seeds)} & {r.NSE_trained:.3f} & {r.NSE_heldout:.3f} & {r.KGE_trained:.3f} & "
             f"{r.soil_anom:.3f} & {r.soil_anom_10th:.2f} & {r.snow_corr:.2f} & {int(r.best_epoch)} \\\\\n")
write("table10", body, "The RFL-HBV at every size $n$ (units per store and hidden neurons per gate): number of weights, median "
      "test-period NSE on the training and held-out catchments and KGE on the training catchments, median and 10th percentile "
      "of the soil store realism over all 669 catchments, median snowpack correlation in the 94 snowy catchments, and the best "
      "epoch. Seed 0 at every size except $n = 8$, which is the median over five seeds.", "tab:sweep", "rrrrrrrrrr")

# table 9: attribution
t = pd.read_csv(ROOT / "results/overview/table9_xai.csv", index_col=0)
body = "model & rainfall & temperature & PET & attributes \\\\\n\\midrule\n"
for name, r in t.iterrows():
    body += (f"{name} & {100 * r.share_P_mean:.1f} & {100 * r.share_T_mean:.1f} & {100 * r.share_PET_mean:.1f} & "
             f"{100 * r.share_static_mean:.1f} \\\\\n")
write("table9", body, "Integrated-gradient attribution of the test-period flow (24 target days per catchment, seed 0): share (\\%) of "
      "the absolute attribution carried by each group of inputs, mean over catchments as in Fig.~12 of the main text, the "
      "forcings summed over the 365-day window.",
      "tab:xai", "lrrrr")


# table 3c: flow bias by abstraction class (results/overview/table3c_bias_by_abstraction.csv from scripts/evaluation/bias_by_abstraction.py)
t = pd.read_csv(ROOT / "results/overview/table3c_bias_by_abstraction.csv", index_col=0)
n = t.loc["n"]; t = t.drop(index="n")
body = "model & " + " & ".join(f"{c.replace('<', '$<$').replace('>', '$>$')}\\% ($n = {int(n[c])}$)" for c in t.columns) + " \\\\\n\\midrule\n"
for name, r in t.iterrows():
    body += name + " & " + " & ".join(f"{v:+.0f}" for v in r.values) + " \\\\\n"
write("table3c", body, "Median test-period bias in total flow (\\%), seed or restart median per catchment, by reported abstraction (surface and groundwater) "
      "as a share of the mean flow, for the 441 catchments in England with abstraction data. Positive: the model "
      "overestimates the flow.", "tab:bias_abstraction", "lrrrr")
