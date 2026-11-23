"""Size sweep of the RFL-HBV: one row per n (units per store = hidden neurons per gate) from results/gatedc2_n<n>_s0
(seed 0) and, at n = 8, the five seeds gatedc2_s0..4 (seed medians). Writes results/overview/table10_size_sweep.csv,
read by scripts/figures/plot_tradeoff.py. Usage: uv run python scripts/evaluation/size_sweep_table.py"""
import json

import pandas as pd

from gbhydro.data import ROOT

SNOWY_PEAK_MM = 20
rows = []
for n in [2, 4, 8, 16, 32, 48, 64]:
    runs = [f"gatedc2_s{s}" for s in range(5)] if n == 8 else [f"gatedc2_n{n}_s0"]
    runs = [r for r in runs if (ROOT / f"results/{r}/metrics_test.csv").exists()]
    if not runs:
        continue
    met = pd.concat([pd.read_csv(ROOT / f"results/{r}/metrics_test.csv", index_col="basin") for r in runs])
    st = met.groupby("basin").set.first(); met = met.groupby("basin")[["NSE", "KGE"]].median()
    r = {"n": n, "NSE_trained": met.NSE[st == "trained"].median(), "NSE_heldout": met.NSE[st == "held-out"].median(),
         "KGE_trained": met.KGE[st == "trained"].median()}
    sr = [ROOT / f"results/{x}/store_realism.csv" for x in runs]
    if all(f.exists() for f in sr):
        d = pd.concat([pd.read_csv(f, index_col="basin") for f in sr]).groupby("basin").median(numeric_only=True)
        st = st.reindex(d.index)   # store realism covers all 669 basins, the test metrics the 667 with a test record
        snowy = d[d.era_peak_swe_median_mm > SNOWY_PEAK_MM]
        r.update({"soil_anom": d.soil_corr_anom.median(), "soil_anom_trained": d.soil_corr_anom[st == "trained"].median(),
                  "soil_anom_heldout": d.soil_corr_anom[st == "held-out"].median(), "soil_anom_10th": d.soil_corr_anom.quantile(0.1),
                  "snow_corr": snowy.snow_corr.median(), "snow_peak_bias": snowy.snow_peak_bias_pct.median(),
                  "snow_mean_ratio": snowy.snow_mean_ratio.median()})
    r["n_params"] = 6 * n * n + 272 * n + 1          # snow gate 36 -> n -> 1, six gates 37 -> n -> n
    r["best_epoch"] = int(pd.Series([json.load(open(ROOT / f"results/{x}/summary_test.json"))["best_epoch"] for x in runs]).median())
    r["seeds"] = len(runs)
    rows.append(r)
T = pd.DataFrame(rows).set_index("n")
T.to_csv(ROOT / "results/overview/table10_size_sweep.csv")
print(T.round(3).to_string())
