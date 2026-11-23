"""Water-balance diagnostics of an RFL-HBV continuous run over the record (sink check).

Per catchment, from 1975-10-01 to 2015-09-30 (after a 5-year spin-up): mean of each store 2005-2014 (mm), drift of
each store (slope of a straight-line fit, mm per decade), ET / PET, Q / P, and the sink = change in total storage
over the period as a share of precipitation (%). A catchment is a sink if it stores more than 5% of its rain.

Usage: uv run python scripts/evaluation/diag_continuous.py <run> [<run> ...]   -> results/<run>/diag_continuous.csv
"""
import sys

import numpy as np
import pandas as pd

from gbhydro.continuous import simulate_continuous
from gbhydro.data import ROOT
from gbhydro.dataset import Study
from gbhydro.models import load_run

START, END = pd.Timestamp("1975-10-01"), pd.Timestamp("2015-09-30")
STORES = ["SP", "SM", "SUZ", "SLZ"]


def main(run: str, study: Study):
    model = load_run(run, study)
    rows = np.arange(len(study.basins))
    sim = simulate_continuous(study, model, rows, batch_size=256)
    dates = study.dates
    w = (dates >= START) & (dates <= END)
    late = (dates >= pd.Timestamp("2005-01-01")) & (dates <= pd.Timestamp("2014-12-31"))
    p = study._raw["precipitation"].cpu().numpy()[rows]
    pet = study._raw["pet"].cpu().numpy()[rows]
    yrs = (dates[w] - START).days.values / 3652.5                        # decades since START
    d = pd.DataFrame(index=pd.Index([int(b) for b in study.basins], name="basin"))
    total = 0
    for k in STORES:
        s = sim[f"stores/{k}"]
        total = total + s
        d[f"{k}_mean_2005_14"] = s[:, late].mean(1)
        x = yrs - yrs.mean()
        d[f"{k}_drift_mm_per_decade"] = ((s[:, w] - s[:, w].mean(1, keepdims=True)) * x).sum(1) / (x * x).sum()
    i0, i1 = np.where(w)[0][[0, -1]]
    P = np.nansum(p[:, w], 1)
    d["ET_over_PET"] = sim["fluxes/et"][:, w].sum(1) / np.nansum(pet[:, w], 1)
    d["Q_over_P"] = sim["fluxes/q"][:, w].sum(1) / P
    d["sink_pct_of_P"] = 100 * (total[:, i1] - total[:, i0 - 1]) / P
    out = ROOT / f"results/{run}/diag_continuous.csv"
    d.to_csv(out)
    print(f"{run}: sinks (>5% of P) {int((d.sink_pct_of_P > 5).sum())} of {len(d)}, median sink {d.sink_pct_of_P.median():.1f}%, "
          f"median SP 2005-14 {d.SP_mean_2005_14.median():.1f} mm, SP drift > 10 mm/decade in {int((d.SP_drift_mm_per_decade > 10).sum())}, "
          f"median SLZ 2005-14 {d.SLZ_mean_2005_14.median():.0f} mm, ET/PET {d.ET_over_PET.median():.2f}", flush=True)


if __name__ == "__main__":
    study = Study(inputs="matched", device="cpu")
    for run in sys.argv[1:]:
        main(run, study)
