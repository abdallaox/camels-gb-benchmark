"""Full metric set per catchment for the main runs, from one continuous pass over the record with the local best.pt:
NSE, KGE and its components (r, alpha, beta), percent bias, FHV, FLV, FMS on the test period.
Writes results/<run>/metrics_test_full.csv per run and results/overview/table1b_metrics_full.csv (medians over catchments
of the per-seed values, trained / held-out, with the seed range), plus paper/tables/table2.tex.
The LSTM is simulated continuously here as well; its windowed and continuous medians agree (results/lstm_s0/continuous_vs_windowed.csv).
Usage: uv run python scripts/evaluation/eval_full_metrics.py [run ...]   (default: lstm_s0..4, hbvc_b128_s0..4, gatedc3_n16_s0..4, gatedc_s0)"""
import sys
import time

import numpy as np
import pandas as pd
import torch

from gbhydro.continuous import simulate_continuous
from gbhydro.data import ROOT
from gbhydro.dataset import Study
from gbhydro.metrics import ALL, EXTRA
from gbhydro.models import build, load_run

torch.set_num_threads(4)
GROUPS = {"LSTM": [f"lstm_s{s}" for s in range(5)], "RPL-HBV": [f"hbvc_b128_s{s}" for s in range(5)],
          "RFL-HBV ($n = 16$)": [f"gatedc3_n16_s{s}" for s in range(5)], "RFL-HBV, unconstrained": ["gatedc_s0"]}
runs = sys.argv[1:] or [r for rs in GROUPS.values() for r in rs]
study = Study(inputs="matched", device="cpu")
rows = np.arange(len(study.basins)); a, z = study.periods["test"]
held = set(int(b) for b in open(ROOT / "config/basins_spatial_test.txt").read().split())
FUNS = {**ALL, **EXTRA}
for run in runs:
    out = ROOT / f"results/{run}/metrics_test_full.csv"
    if out.exists():
        continue
    kind = "gated" if "gated" in run else ("hbv" if "hbv" in run else "lstm")
    m = load_run(run, study)
    t0 = time.time(); sim = simulate_continuous(study, m, rows, batch_size=256)
    d = pd.DataFrame(index=pd.Index([int(b) for b in study.basins], name="basin"))
    for name, fn in FUNS.items():
        d[name] = [fn(sim["q"][i, a:z + 1], sim["obs"][i, a:z + 1]) for i in rows]
    d["set"] = ["held-out" if b in held else "trained" for b in d.index]
    d.to_csv(out)
    print(f"{run}: {time.time() - t0:.0f}s, median NSE {d.NSE.median():.3f} KGE {d.KGE.median():.3f} r {d.r.median():.3f} "
          f"alpha {d.alpha.median():.3f} beta {d.beta.median():.3f} PBIAS {d.PBIAS.median():+.1f}%", flush=True)

COLS = ["NSE", "KGE", "r", "alpha", "beta", "PBIAS", "FHV", "FLV", "FMS"]
rows_out = []
for label, rs in GROUPS.items():
    ds = [pd.read_csv(ROOT / f"results/{r}/metrics_test_full.csv", index_col="basin") for r in rs if (ROOT / f"results/{r}/metrics_test_full.csv").exists()]
    if not ds:
        continue
    for st in ["trained", "held-out"]:
        med = pd.DataFrame([d[d.set == st][COLS].median() for d in ds])
        r = {"model": label, "set": st, "n_basins": int((ds[0].set == st).sum()), "seeds": len(ds)}
        for c in COLS:
            r[c] = med[c].median(); r[c + "_range"] = f"{med[c].min():.3f}..{med[c].max():.3f}" if len(ds) > 1 else ""
        rows_out.append(r)
T = pd.DataFrame(rows_out); T.to_csv(ROOT / "results/overview/table1b_metrics_full.csv", index=False)
print(T[["model", "set"] + COLS].round(3).to_string())

body = "model & catchments & NSE & KGE & $r$ & $\\alpha$ & $\\beta$ & bias (\\%) & FHV (\\%) & FLV (\\%) & FMS (\\%) \\\\\n\\midrule\n"
for _, r in T.iterrows():
    body += (f"{r.model} & {r.set} ({r.n_basins}) & {r.NSE:.3f} & {r.KGE:.3f} & {r.r:.3f} & {r.alpha:.2f} & {r.beta:.2f} & "
             f"{r.PBIAS:+.1f} & {r.FHV:+.0f} & {r.FLV:+.0f} & {r.FMS:+.0f} \\\\\n")
tex = ("\\begin{table}[htbp]\n\\centering\n\\footnotesize\n\\setlength{\\tabcolsep}{4pt}\n\\caption{Test-period (1998 to 2008) metrics from one continuous run over the record: "
       "medians over catchments of the per-seed values (five seeds, one for the unconstrained RFL-HBV). $r$, $\\alpha$ and $\\beta$ are the "
       "correlation, variability ratio and mean ratio that make up KGE \\citep{gupta2009}. FHV, FLV and FMS are the high-flow, low-flow and "
       "mid-slope biases of the flow duration curve \\citep{yilmaz2008}.}\n\\label{tab:metrics_full}\n"
       "\\resizebox{\\ifdim\\width>\\textwidth\\textwidth\\else\\width\\fi}{!}{%\n"
       f"\\begin{{tabular}}{{llrrrrrrrrr}}\n\\toprule\n{body}\\bottomrule\n\\end{{tabular}}}}\n\\end{{table}}\n")
(ROOT / "paper/tables").mkdir(parents=True, exist_ok=True); (ROOT / "paper/tables/table2.tex").write_text(tex, encoding="utf-8")
print("wrote paper/tables/table2.tex")
