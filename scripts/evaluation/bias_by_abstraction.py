"""Median test-period percent bias (seed or restart median per catchment) by licensed abstraction as a share of
mean flow, 441 English catchments, the classes of scripts/figures/plot_attribute_boxes.py.
Usage: uv run python scripts/evaluation/bias_by_abstraction.py -> results/overview/table3c_bias_by_abstraction.csv"""
import pandas as pd, numpy as np
from gbhydro.data import ROOT
R = {"LSTM": [f"lstm_s{s}" for s in range(5)], "RPL": [f"hbvc_b128_s{s}" for s in range(5)],
     "RFL": [f"gatedc3_n16_s{s}" for s in range(5)], "LC": [f"hbv_local_r{s}" for s in range(5)]}
hum = pd.read_csv(ROOT/"data/camels_gb/data/CAMELS_GB_humaninfluence_attributes.csv", index_col="gauge_id")
hyd = pd.read_csv(ROOT/"data/camels_gb/data/CAMELS_GB_hydrologic_attributes.csv", index_col="gauge_id")
has = hum.groundwater_abs.notna() | hum.surfacewater_abs.notna()
a = 100*(hum.groundwater_abs.fillna(0)+hum.surfacewater_abs.fillna(0))/hyd.q_mean; a[~has] = np.nan
cls = pd.cut(a, [-1,5,20,50,1e9], labels=["<5","5-20","20-50",">50"])
rows, scored = {}, None
for m, runs in R.items():
    b = pd.concat([pd.read_csv(ROOT/f"results/{r}/metrics_test_full.csv", index_col="basin")["PBIAS"] for r in runs], axis=1).median(axis=1)
    scored = b.dropna().index if scored is None else scored.intersection(b.dropna().index)
    rows[m] = {str(k): round(float(b[cls.reindex(b.index) == k].median()), 1) for k in cls.cat.categories}
n = cls.reindex(scored).value_counts().reindex(cls.cat.categories)   # study catchments with a test score only
out = pd.DataFrame(rows).T.rename(index={"RPL": "RPL-HBV", "RFL": "RFL-HBV", "LC": "LC-HBV"}); out.loc["n"] = n.values
out.to_csv(ROOT / "results/overview/table3c_bias_by_abstraction.csv"); print(out)
