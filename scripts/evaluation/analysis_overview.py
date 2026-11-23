"""First-pass overview of the committed results: model medians per seed, seed-median per basin, paired differences,
and performance by catchment type. Reads results/<run>/metrics_test_full.csv (continuous evaluation). Prints tables; writes
results/overview/*.csv so the paper tables can be rebuilt. Usage: uv run python scripts/evaluation/analysis_overview.py"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from gbhydro.data import ROOT, load_attributes

OUT = ROOT / "results/overview"; OUT.mkdir(exist_ok=True)
MODELS = {  # label: (run prefix, seeds, evaluation)
    "LSTM": ("lstm_s", range(5), "continuous"),
    "HBV regional (continuous)": ("hbvc_b128_s", range(5), "continuous"),
    "RFL-HBV n=16 (constrained)": ("gatedc3_n16_s", range(5), "continuous"),
    "RFL-HBV n=8 (unconstrained)": ("gatedc_s", [0], "continuous"),
}
METRICS = ["NSE", "KGE", "FHV", "FLV", "FMS"]
att = load_attributes()


def load(prefix, seeds):
    frames = []
    for s in seeds:
        # one continuous pass over the record for every model (methods)
        d = pd.read_csv(ROOT / f"results/{prefix}{s}/metrics_test_full.csv", index_col="basin")
        d["seed"] = s
        frames.append(d)
    return pd.concat(frames)


def summarise(d):
    """Per seed: median NSE/KGE per set; mean NSE clipped at -1; share of basins above thresholds; signature medians."""
    rows = []
    for (s, st), g in d.groupby(["seed", "set"]):
        rows.append({"seed": s, "set": st, "n": len(g), "NSE_med": g.NSE.median(), "NSE_mean_clip": g.NSE.clip(-1).mean(),
                     "NSE>0.8": (g.NSE > 0.8).mean(), "NSE<0.5": (g.NSE < 0.5).mean(), "KGE_med": g.KGE.median(),
                     "FHV_med": g.FHV.median(), "FLV_med": g.FLV.median(), "FMS_med": g.FMS.median()})
    return pd.DataFrame(rows)


all_runs, seed_tables, per_basin = {}, {}, {}
for label, (prefix, seeds, ev) in MODELS.items():
    d = load(prefix, seeds); all_runs[label] = d
    t = summarise(d); seed_tables[label] = t
    # seed-median per basin (robust single number per basin per model)
    pb = d.groupby("basin")[METRICS].median(); pb["set"] = d.groupby("basin")["set"].first()
    per_basin[label] = pb

print("=" * 100)
print("TABLE 1  test period 1998-2008, per model: median over seeds of each per-seed statistic (min-max over seeds)")
rows = []
for label, t in seed_tables.items():
    for st, g in t.groupby("set"):
        r = {"model": label, "set": st, "n_basins": int(g.n.iloc[0]), "eval": MODELS[label][2], "seeds": len(g)}
        for c in ["NSE_med", "KGE_med", "NSE_mean_clip", "NSE>0.8", "NSE<0.5", "FHV_med", "FLV_med", "FMS_med"]:
            r[c] = g[c].median(); r[c + "_range"] = f"{g[c].min():.3f}..{g[c].max():.3f}" if len(g) > 1 else ""
        rows.append(r)
T1 = pd.DataFrame(rows); T1.to_csv(OUT / "table1_model_medians.csv", index=False)
show = T1[["model", "set", "NSE_med", "NSE_med_range", "KGE_med", "KGE_med_range", "NSE_mean_clip", "NSE>0.8", "NSE<0.5", "FHV_med", "FLV_med", "FMS_med"]]
print(show.round(3).to_string(index=False))

print("\n" + "=" * 100)
print("TABLE 2  paired per-basin differences (seed-median NSE per basin), test period")
pairs = [("LSTM", "HBV regional (continuous)"), ("LSTM", "RFL-HBV n=16 (constrained)"), ("RFL-HBV n=16 (constrained)", "HBV regional (continuous)"),
         ("RFL-HBV n=16 (constrained)", "RFL-HBV n=8 (unconstrained)")]
rows = []
for a, b in pairs:
    for st in ["trained", "held-out", "all"]:
        pa, pb_ = per_basin[a], per_basin[b]
        m = pa.index if st == "all" else pa.index[pa.set == st]
        diff = (pa.loc[m, "NSE"] - pb_.loc[m, "NSE"]).dropna()
        rows.append({"A": a, "B": b, "set": st, "n": len(diff), "median A-B": diff.median(), "10th": diff.quantile(.1), "90th": diff.quantile(.9),
                     "A better %": 100 * (diff > 0).mean(), "A better by >0.05 %": 100 * (diff > 0.05).mean(), "B better by >0.05 %": 100 * (diff < -0.05).mean()})
T2 = pd.DataFrame(rows); T2.to_csv(OUT / "table2_paired_differences.csv", index=False)
print(T2.round(3).to_string(index=False))

print("\n" + "=" * 100)
print("TABLE 3  seed-median NSE by catchment type (all 669 basins; groups from CAMELS-GB attributes, flow-derived ones used only here)")
pb = pd.DataFrame({k: v.NSE for k, v in per_basin.items()}); pb["set"] = per_basin["LSTM"].set
a = att.loc[pb.index]
groups = {
    "BFI tercile": pd.qcut(a.baseflow_index, 3, labels=["low", "mid", "high"]),
    "aridity tercile": pd.qcut(a.aridity, 3, labels=["wet", "mid", "dry"]),
    "chalk/high-productivity aquifer (frac_high_perc)": pd.cut(a.frac_high_perc, [-1, 10, 50, 101], labels=["<10%", "10-50%", ">50%"]),
    "urban_perc": pd.cut(a.urban_perc, [-1, 2, 10, 101], labels=["<2%", "2-10%", ">10%"]),
    "elev_mean (m)": pd.cut(a.elev_mean, [0, 150, 350, 2000], labels=["<150", "150-350", ">350"]),
    # licensed abstraction (groundwater + surface water) as a share of mean flow, England only (NaN elsewhere)
    "abstraction (% of mean flow)": pd.cut((100 * (a.groundwater_abs.fillna(0) + a.surfacewater_abs.fillna(0)) / a.q_mean)
                                           .where(a.groundwater_abs.notna() | a.surfacewater_abs.notna()),
                                           [-1, 5, 20, 50, 1e9], labels=["<5%", "5-20%", "20-50%", ">50%"]),
    "reservoirs": pd.cut(a.num_reservoir.fillna(0), [-1, 0, 2, 1000], labels=["none", "1-2", ">2"]),
    "snow fraction": pd.cut(a.frac_snow, [-1, 0.02, 0.05, 1], labels=["<2%", "2-5%", ">5%"]),
    "flow completeness (train)": None,
}
# LC-HBV restart median per catchment (gauged baseline, calibrated on every catchment), for the class table only
lc = pd.concat([pd.read_csv(ROOT / f"results/hbv_local_r{r}/metrics_test_full.csv", index_col="basin").NSE for r in range(5)],
               axis=1).median(axis=1).reindex(pb.index)
rows = []
for gname, g in groups.items():
    if g is None:
        continue
    for lev, idx in pb.groupby(g).groups.items():
        r = {"group": gname, "level": str(lev), "n": len(idx)}
        for k in pb.columns.drop("set"):
            r[k] = pb.loc[idx, k].median()
        r["LSTM-HBV"] = (pb.loc[idx, "LSTM"] - pb.loc[idx, "HBV regional (continuous)"]).median()
        r["gated-HBV"] = (pb.loc[idx, "RFL-HBV n=16 (constrained)"] - pb.loc[idx, "HBV regional (continuous)"]).median()
        r["LC-HBV"] = lc.loc[idx].median()
        r["LC-RPL"] = (lc.loc[idx] - pb.loc[idx, "HBV regional (continuous)"]).median()
        rows.append(r)
T3 = pd.DataFrame(rows); T3.to_csv(OUT / "table3_by_catchment_type.csv", index=False)
print(T3.round(3).to_string(index=False))

print("\n" + "=" * 100)
print("TABLE 4  seed spread: per-basin NSE standard deviation over the 5 seeds (median over basins) and best epochs")
for label, (prefix, seeds, ev) in MODELS.items():
    if len(list(seeds)) < 2:
        continue
    d = all_runs[label]; sd = d.groupby("basin").NSE.std()
    be = [json.load(open(ROOT / f"results/{prefix}{s}/summary_test.json"))["best_epoch"] for s in seeds]
    print(f"{label:40s} median per-basin seed SD of NSE {sd.median():.3f} (90th {sd.quantile(.9):.3f}); best epochs {be}")

print("\n" + "=" * 100)
print("TABLE 5  worst catchments (seed-median NSE < 0.3 for all three main models)")
bad = pb[(pb[["LSTM", "HBV regional (continuous)", "RFL-HBV n=16 (constrained)"]] < 0.3).all(axis=1)]
cols = ["gauge_name", "area", "baseflow_index", "aridity", "frac_high_perc", "urban_perc", "num_reservoir", "groundwater_abs", "benchmark_catch"]
print(bad.round(2).join(att.loc[bad.index, cols]).to_string())
pb.join(att[cols]).to_csv(OUT / "per_basin_seed_median_nse.csv")
print("\nwrote", sorted(p.name for p in OUT.iterdir()))
