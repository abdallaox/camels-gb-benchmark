"""Learned rules figure for the paper, from the probes of scripts/evaluation/probe_rules.py: every model is given the same store
level (mm) and the same weather, and the figure shows the flux it releases, for three contrasting catchments.
Columns: ET efficiency vs soil store, recharge vs soil store, fast flow vs upper store, baseflow vs lower store, melt vs
temperature (snowpack of SP_MELT mm). Line = mean over the five seeds (restarts for the LC-HBV), band = seed range.
Solid where the model's own store lies between its 1st and 99th percentile in the continuous run, dashed beyond (the
model is asked about levels it never reaches). The HBV curves of the soil store stop at FC.
Usage: uv run python scripts/figures/plot_gate_rules.py [basins=46003,79002,43005] -> figures/gate_rules.png"""
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from gbhydro.data import ROOT, load_attributes

sys.path.insert(0, str(ROOT / "scripts" / "evaluation"))
from probe_rules import MODELS, RULES, model_curves  # noqa: E402

basins = [int(b) for b in (sys.argv[1] if len(sys.argv) > 1 else "46003,79002,43005").split(",")]
LABEL = {60010: "Tywi at Capel Dewi, wet upland", 15007: "Tay at Pitnacree, snowy upland", 43005: "Avon at Amesbury, chalk",
         89003: "Orchy at Glen Orchy, wet upland", 42004: "Test at Broadlands, chalk", 43009: "Stour at Hammoon, clay",
         12002: "Dee at Park, snowy", 46003: "Dart at Austins Bridge, wet upland", 79002: "Nith at Friars Carse, snowy upland"}
INK = "#222222"
STYLE = {"RPL-HBV": ("#eb6834", 1.6), "RFL-HBV": ("#2e9e5b", 2.8), "LC-HBV": ("#8e44ad", 1.6)}
PANELS = [("soil store, mm", "ET / PET"), ("soil store, mm", "recharge, mm/d"), ("upper store, mm", "fast flow, mm/d"),
          ("lower store, mm", "baseflow, mm/d"), ("temperature, \N{DEGREE SIGN}C", "melt, mm/d")]

cur = {name: model_curves(runs) for name, runs in MODELS.items()}
cur = {k: v for k, v in cur.items() if v is not None}
seeds = {name: [np.load(ROOT / f"results/rules/probe_{r}.npz")["curves"] for r in runs if (ROOT / f"results/rules/probe_{r}.npz").exists()]
         for name, runs in MODELS.items() if name in cur}
first = next(iter(cur.values()))
rows = {int(b): i for i, b in enumerate(first["basins"])}
att = load_attributes()
snowy = pd.read_csv(ROOT / "results/hbvc_b128_s0/store_realism.csv", index_col="basin").era_peak_swe_median_mm.loc[basins] > 20
nse = {name: pd.concat([pd.read_csv(ROOT / f"results/{r}/metrics_test_full.csv", index_col="basin").NSE for r in runs], axis=1).median(axis=1)
       for name, runs in MODELS.items() if name in cur and all((ROOT / f"results/{r}/metrics_test_full.csv").exists() for r in runs)}
sp_melt = float(np.load(ROOT / f"results/rules/probe_{MODELS['RPL-HBV'][0]}.npz")["sp_melt"])

fig, axes = plt.subplots(len(basins), 5, figsize=(13, 3.4 * len(basins)))
for i, b in enumerate(basins):
    r = rows[b]
    for k, (xl, yl) in enumerate(PANELS):
        ax = axes[i, k]; top = 0
        if k == 4 and not snowy.loc[b]:
            ax.set_axis_off(); ax.text(0.5, 0.5, "snow negligible", transform=ax.transAxes, ha="center", fontsize=9, color=INK)
            continue
        for name, c in cur.items():
            col, lw = STYLE[name]
            x, y = c["levels"][r, k], c["curves"][r, k]
            ok = np.isfinite(y)
            if ok.sum() < 2:
                continue
            st = np.stack([s[r, k] for s in seeds[name]])
            with np.errstate(all="ignore"):
                lo, hi = np.nanmin(st, 0), np.nanmax(st, 0)
            ax.fill_between(x[ok], lo[ok], hi[ok], color=col, alpha=0.12, lw=0)
            inside = np.ones_like(ok) if k == 4 else (x >= c["p1"][r, k]) & (x <= c["p99"][r, k])
            # dashed everywhere, solid on top where the model reaches the level (continued to the neighbouring point)
            ax.plot(x[ok], y[ok], color=col, ls=(0, (2.5, 2.5)), lw=lw * 0.7, zorder=2)
            sol = inside & ok
            sol[1:] |= inside[:-1] & ok[1:]; sol[:-1] |= inside[1:] & ok[:-1]
            ys = np.where(sol, y, np.nan)
            ax.plot(x, ys, color=col, ls="-", lw=lw, label=name, zorder=3 if name == "RFL-HBV" else 2.5)
            if sol.any():
                top = max(top, float(np.nanmax(ys)))
        ax.set_xlabel(xl, fontsize=9, color=INK); ax.set_ylabel(yl, fontsize=9, color=INK)
        ax.set_ylim(0, 1.05 if k == 0 else max(top, 1e-3) * 1.3)       # axis from the reached parts; extrapolations may leave the panel
        if k < 4:
            ax.set_xlim(left=0)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.tick_params(labelsize=8, colors=INK)
handles, labels = axes[0, 0].get_legend_handles_labels()
empty = [ax for ax in axes[:, 4] if not ax.axison]           # a "snow negligible" panel, if any, holds the legend
(empty[0] if empty else axes[0, 0]).legend(handles, labels, frameon=False, fontsize=9, loc="lower center" if empty else "lower right")
fig.tight_layout(h_pad=4.0, w_pad=0.8, rect=(0, 0, 1, 0.97))
for i, b in enumerate(basins):                       # row titles above the first panel, placed after the layout
    scores = ", ".join(f"{n} {v.loc[b]:.2f}" for n, v in nse.items())
    y = axes[i, 0].get_position().y1
    fig.text(axes[i, 0].get_position().x0, y + 0.012, f"{LABEL.get(b, att.gauge_name.loc[b])}.  Test NSE: {scores}", fontsize=10, color=INK, va="bottom")
out = ROOT / "figures/gate_rules.png"
fig.savefig(out, dpi=150); print("wrote", out.relative_to(ROOT), list(cur), f"melt at a snowpack of {sp_melt:.0f} mm")
