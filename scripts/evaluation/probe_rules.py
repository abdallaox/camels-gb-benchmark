"""Probe the trained rules: same store level, same weather, every model.

Each trained model is treated as a function from (store level, the day's weather) to the day's flux. For every
catchment the probed store is set to a grid of levels in mm, all other stores are empty, and one day of the model is
run with the real rain, temperature and PET of every day from 1975-10-01 to 2015-09-30; the flux is averaged over
the days. The three HBV-based models answer the same question on the same axes, so their rules can be compared
directly (a partial-dependence plot). Rules:
    ET efficiency (ET / PET) against the soil store, days with PET > 0.5 mm
    recharge (mm/d) against the soil store
    fast flow (mm/d) against the upper store
    baseflow (mm/d) against the lower store
    melt (mm/d) against temperature, for a snowpack of SP_MELT mm, days November to April with the temperature
    replaced by the grid value (the RFL-HBV's standardised temperature input is replaced too)
Grid: NGRID levels from 0 to the largest 99th percentile of that store among the three models' continuous runs
(seed median), per catchment, so every run of every model is probed on the same levels. The HBV soil store cannot
exceed FC, so the HBV curves are undefined above it. The RFL-HBV's n units are filled in the proportions the model
itself shows at that level in its continuous run (binned by the store total; --equal fills them 1/n each).

Step 1 (ranges): simulate each run continuously 1970-2015 and save the 1st and 99th percentiles of its stores from
1975-10-01, the RFL-HBV's unit shares by level and the HBV FC -> results/rules/range_<run>.npz
Step 2 (probe): -> results/rules/probe_<run>.npz: curves [basin, rule, grid], levels [basin, rule, grid] (mm, or degC
for melt), p1 and p99 of the run's own store per rule [basin, rule], FC [basin] for the HBV runs
Step 3 (agreement): seed-mean curves per model (grid points where every seed is defined); per catchment
and rule, the rule similarity = Pearson correlation of the two curves over the grid points where both are defined
(>= 4 points) and the level is at most the higher 99th percentile of the two models (both curves interpolated onto
NGRID levels of that range). It scores the shape only, like the soil and snow realism; the rate is reported
separately as the ratio of the mean fluxes (table11b_rule_rate.csv, 2 = twice as fast as the reference). Melt in the 94
snowy catchments -> results/overview/table11_rule_agreement.csv (per basin) and *_summary.csv.

Usage: uv run python scripts/evaluation/probe_rules.py ranges <run> ...
       uv run python scripts/evaluation/probe_rules.py probe <run> ... [--equal]
       uv run python scripts/evaluation/probe_rules.py agreement
"""
import sys
import time

import numpy as np
import pandas as pd
import torch

from gbhydro.continuous import simulate_continuous
from gbhydro.data import ROOT
from gbhydro.dataset import Study
from gbhydro.models import load_run
from gbhydro.models.gated_hbv import STORES as GSTORES, GatedHBV, gated_step
from gbhydro.models.hbv import STORES as HSTORES, hbv_step

RULES = ["et", "recharge", "fast", "baseflow", "melt"]
RULE_STORE = {"et": "SM", "recharge": "SM", "fast": "SUZ", "baseflow": "SLZ", "melt": "SP"}
HBV_NAME = {"SP": "snowpack", "SM": "soil_moisture", "SUZ": "upper_storage", "SLZ": "lower_storage"}
NGRID = 41
TGRID = np.linspace(-5.0, 10.0, NGRID)       # degC, melt rule
SP_MELT = 20.0                               # mm of snowpack for the melt probe = the ERA5-Land peak that defines a snowy catchment
NSHARE_BINS = 20                             # bins of the store total for the RFL-HBV unit shares
START = pd.Timestamp("1975-10-01")
COLD_MONTHS = [11, 12, 1, 2, 3, 4]
CHUNK_DAYS = 40                              # days per batch: basins x grid x days rows
OUT = ROOT / "results/rules"
MODELS = {"RPL-HBV": [f"hbvc_b128_s{s}" for s in range(5)], "RFL-HBV": [f"gatedc3_n16_s{s}" for s in range(5)],
          "LC-HBV": [f"hbv_local_r{s}" for s in range(5)]}


def hbv_parameters(model, study):
    with torch.no_grad():
        return model.parameters_for({"x_s": study.x_s, "rows": torch.arange(len(study.basins), device=study.device)})


# ------------------------------------------------------------------------------------------------------ step 1
def ranges_for_run(run, study):
    model = load_run(run, study)
    gated = isinstance(model, GatedHBV)
    B = len(study.basins)
    w = np.nonzero(study.dates >= START)[0]
    t0 = time.time()
    sim = simulate_continuous(study, model, np.arange(B), batch_size=256, units_days=w[::5] if gated else None)
    print(f"{run}: simulated in {time.time() - t0:.0f} s", flush=True)
    out = {"basins": np.array([int(b) for b in study.basins])}
    for st in GSTORES:
        v = sim[f"stores/{st}" if gated else f"stores/{HBV_NAME[st]}"][:, w]
        out[f"p1_{st}"], out[f"p99_{st}"] = np.percentile(v, 1, axis=1), np.percentile(v, 99, axis=1)
        if st == "SP":                                   # typical snowpack on snow days, to choose SP_MELT
            out["sp_snowday_median"] = np.array([np.median(r[r > 1]) if (r > 1).sum() >= 30 else np.nan for r in v])
        if gated:
            u = sim[f"units/{st}"]                       # [B, D, n]
            tot = u.sum(2)
            n = u.shape[2]
            centres = np.full((B, NSHARE_BINS), np.nan); shares = np.full((B, NSHARE_BINS, n), np.nan)
            for i in range(B):
                m = tot[i] > 1e-6
                if m.sum() < 50:
                    continue
                t, ui = tot[i][m], u[i][m]
                edges = np.unique(np.quantile(t, np.linspace(0, 1, NSHARE_BINS + 1)))
                idx = np.clip(np.searchsorted(edges, t, side="right") - 1, 0, len(edges) - 2)
                for k in range(len(edges) - 1):
                    s = idx == k
                    if s.sum() >= 5:
                        centres[i, k] = np.median(t[s]); shares[i, k] = (ui[s] / t[s][:, None]).mean(0)
            out[f"centres_{st}"], out[f"shares_{st}"] = centres, shares
    if not gated:
        out["FC"] = hbv_parameters(model, study)["FC"].cpu().numpy()
    OUT.mkdir(exist_ok=True)
    np.savez_compressed(OUT / f"range_{run}.npz", **out)
    print(f"{run}: ranges saved", flush=True)


# ------------------------------------------------------------------------------------------------------ step 2
def common_grid(B):
    """Per store, NGRID levels from 0 to the largest seed-median 99th percentile among the models -> {store: [B, NGRID]}."""
    top = {st: np.zeros(B) for st in GSTORES}
    for runs in MODELS.values():
        zs = [np.load(OUT / f"range_{r}.npz") for r in runs if (OUT / f"range_{r}.npz").exists()]
        if not zs:
            continue
        for st in GSTORES:
            top[st] = np.maximum(top[st], np.median(np.stack([z[f"p99_{st}"] for z in zs]), 0))
    return {st: np.linspace(0, 1, NGRID)[None, :] * np.maximum(top[st], 1.0)[:, None] for st in GSTORES}


def unit_levels(lev, rng, st, n, equal):
    """Split the store levels lev [B, G] over the n units: shares interpolated from the run's own continuous run
    (constant beyond the visited range), or 1/n each -> tensor [B, G, n]."""
    B, G = lev.shape
    sh = np.full((B, G, n), 1.0 / n)
    if not equal and f"centres_{st}" in rng:
        c, s = rng[f"centres_{st}"], rng[f"shares_{st}"]
        for i in range(B):
            ok = np.isfinite(c[i])
            if ok.sum() < 2:
                continue
            order = np.argsort(c[i][ok]); cc, ss = c[i][ok][order], s[i][ok][order]
            for j in range(n):
                sh[i, :, j] = np.interp(lev[i], cc, ss[:, j])
        sh /= sh.sum(2, keepdims=True)
    return torch.as_tensor(lev[:, :, None] * sh, dtype=torch.float32)


@torch.no_grad()
def probe_run(run, study, equal=False):
    model = load_run(run, study); model.eval()
    gated = isinstance(model, GatedHBV)
    B, G, A = len(study.basins), NGRID, study.x_s.shape[1]
    grid = common_grid(B)
    rng = np.load(OUT / f"range_{run}.npz")
    days = np.nonzero(study.dates >= START)[0]
    cold = days[np.isin(study.dates.month.values[days], COLD_MONTHS)]
    raw = study._raw
    if gated:
        gates, n = dict(model.gates.items()), model.n
        it = study.dyn_names.index("temperature"); tm, ts = study.dyn_mean[it], study.dyn_std[it]
    else:
        k_all = hbv_parameters(model, study)
    tgrid = torch.as_tensor(TGRID, dtype=torch.float32)
    # one probe setting per store: (store, levels [B, G], days, rules read from it, temperature replaced by the grid)
    settings = [("SM", grid["SM"], days, ["et", "recharge"], False), ("SUZ", grid["SUZ"], days, ["fast"], False),
                ("SLZ", grid["SLZ"], days, ["baseflow"], False), ("SP", np.full((B, G), SP_MELT), cold, ["melt"], True)]
    C = np.full((B, len(RULES), G), np.nan, dtype=np.float32)
    L = np.full((B, len(RULES), G), np.nan, dtype=np.float32)
    t0 = time.time()
    for st, lev, dset, rules, rep_t in settings:
        lev_t = torch.as_tensor(lev, dtype=torch.float32)
        unit = unit_levels(lev, rng, st, n, equal) if gated else None
        sums = {r: torch.zeros(B, G) for r in rules}; cnt = {r: torch.zeros(B, G) for r in rules}
        for d0 in range(0, len(dset), CHUNK_DAYS):
            dd = torch.as_tensor(dset[d0:d0 + CHUNK_DAYS]); D = len(dd)
            ex = lambda v: v[:, None, :].expand(B, G, D).reshape(-1)                   # [B, D] -> [B*G*D]
            P, T, PET = ex(raw["precipitation"][:, dd]), ex(raw["temperature"][:, dd]), ex(raw["pet"][:, dd])
            if rep_t:
                T = tgrid[None, :, None].expand(B, G, D).reshape(-1)
            if gated:
                xd = study.x_d[:, dd][:, None].expand(B, G, D, -1).reshape(B * G * D, -1).clone()
                if rep_t:
                    xd[:, it] = (T - tm) / ts
                ctx = torch.cat([xd, study.x_s[:, None, None, :].expand(B, G, D, A).reshape(-1, A)], 1)
                s = {k: torch.zeros(B * G * D, n) for k in GSTORES}
                s[st] = unit[:, :, None, :].expand(B, G, D, n).reshape(-1, n)
                _, f = gated_step(P, T, PET, s, ctx, gates, model.constrained, model.snow_tt)
                fl = {"et": f["et"], "recharge": f["rech"], "fast": f["fast"], "baseflow": f["base"], "melt": f["melt"]}
            else:
                k = {kk: v[:, None, None].expand(B, G, D).reshape(-1) for kk, v in k_all.items()}
                s = {kk: torch.zeros(B * G * D) for kk in HSTORES}
                s[HBV_NAME[st]] = lev_t[:, :, None].expand(B, G, D).reshape(-1)
                _, f = hbv_step(P, T, PET, s, k)
                fl = {"et": f["et"], "recharge": f["recharge"], "fast": f["q0"] + f["q1"], "baseflow": f["q2"], "melt": f["melt"]}
            for r in rules:
                v = fl[r].reshape(B, G, D)
                if r == "et":                                        # efficiency on days with PET above 0.5 mm
                    pet = PET.reshape(B, G, D); m = pet > 0.5
                    sums[r] += torch.where(m, v / pet.clamp(min=0.5), torch.zeros(())).sum(2); cnt[r] += m.sum(2)
                else:
                    sums[r] += v.sum(2); cnt[r] += D
        for r in rules:
            curve = (sums[r] / cnt[r].clamp(min=1)).numpy()
            if not gated and st == "SM":
                curve[lev > rng["FC"][:, None]] = np.nan              # the HBV soil store cannot exceed FC
            i = RULES.index(r)
            C[:, i, :] = curve
            L[:, i, :] = TGRID[None, :] if r == "melt" else lev
        print(f"{run}: {st} probed, {time.time() - t0:.0f} s", flush=True)
    p1 = np.stack([rng[f"p1_{RULE_STORE[r]}"] for r in RULES], 1)
    p99 = np.stack([rng[f"p99_{RULE_STORE[r]}"] for r in RULES], 1)
    extra = {"FC": rng["FC"]} if "FC" in rng else {}
    np.savez_compressed(OUT / f"probe_{run}{'_equal' if equal else ''}.npz", curves=C, levels=L, p1=p1, p99=p99,
                        basins=rng["basins"], rules=np.array(RULES), sp_melt=SP_MELT, **extra)
    print(f"{run}: probe saved", flush=True)


# ------------------------------------------------------------------------------------------------------ step 3
def model_curves(runs, suffix=""):
    """Seed-mean curves [basin, rule, grid] (grid points where every seed is defined), the common levels,
    and the seed-median p1 and p99 of the own store [basin, rule]; None if no run is probed."""
    fs = [OUT / f"probe_{r}{suffix}.npz" for r in runs]
    zs = [np.load(f) for f in fs if f.exists()]
    if not zs:
        return None
    stack = np.stack([z["curves"] for z in zs])
    ok = np.isfinite(stack).all(0)
    with np.errstate(all="ignore"):
        mean = np.where(ok, np.nanmean(stack, 0), np.nan)
    return {"curves": mean, "levels": zs[0]["levels"], "basins": zs[0]["basins"],
            "p1": np.median(np.stack([z["p1"] for z in zs]), 0), "p99": np.median(np.stack([z["p99"] for z in zs]), 0),
            "FC": np.median(np.stack([z["FC"] for z in zs]), 0) if "FC" in zs[0] else None, "n_runs": len(zs)}


def resample(x, y, xf):
    """Curve y(x) interpolated onto xf, NaN outside the range where y is defined."""
    ok = np.isfinite(y)
    if ok.sum() < 2:
        return np.full(len(xf), np.nan)
    out = np.interp(xf, x[ok], y[ok])
    out[(xf < x[ok].min()) | (xf > x[ok].max())] = np.nan
    return out


def agreement(a, b, x=None, top=None, rate=False):
    """Rule similarity: Pearson correlation of two curves over the grid points where both are defined (at least 4);
    NaN if undefined or a curve is constant. rate=True returns mean a / mean b instead (how many times as fast).
    With x and top the curves are first interpolated onto NGRID levels from 0 to top (the levels one of the two
    models reaches)."""
    if x is not None:
        xf = np.linspace(0, max(top, 1e-6), NGRID)
        a, b = resample(x, a, xf), resample(x, b, xf)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 4:
        return np.nan
    a, b = a[m], b[m]
    if rate:
        return float(np.mean(a) / np.mean(b)) if np.mean(b) > 1e-9 else np.nan
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def main_agreement():
    cur = {name: model_curves(runs) for name, runs in MODELS.items()}
    cur = {k: v for k, v in cur.items() if v is not None}
    basins = next(iter(cur.values()))["basins"]
    held = set(int(b) for b in open(ROOT / "config/basins_spatial_test.txt").read().split())
    d = pd.DataFrame(index=pd.Index(basins, name="basin"))
    d["set"] = ["held-out" if b in held else "trained" for b in basins]
    r = d.copy()                                                      # rate ratios, same layout
    for a, b in [("RFL-HBV", "RPL-HBV"), ("RFL-HBV", "LC-HBV"), ("RPL-HBV", "LC-HBV")]:
        if a in cur and b in cur:
            for k, rule in enumerate(RULES):
                lev, top = cur[a]["levels"][:, k], np.maximum(cur[a]["p99"][:, k], cur[b]["p99"][:, k])
                for tab, rt in ((d, False), (r, True)):
                    tab[f"{a} vs {b}: {rule}"] = [agreement(cur[a]["curves"][i, k], cur[b]["curves"][i, k],
                                                            None if rule == "melt" else lev[i], top[i], rate=rt)
                                                  for i in range(len(basins))]
    # melt only in the 94 snowy catchments of the snow realism (ERA5-Land median annual peak above 20 mm)
    era = pd.read_csv(ROOT / "results/hbvc_b128_s0/store_realism.csv", index_col="basin").era_peak_swe_median_mm
    snowy = era.reindex(d.index) > 20
    for c in [c for c in d.columns if c.endswith(": melt")]:
        d.loc[~snowy, c] = np.nan
        r.loc[~snowy, c] = np.nan
    d.to_csv(ROOT / "results/overview/table11_rule_agreement.csv")
    r.to_csv(ROOT / "results/overview/table11b_rule_rate.csv")
    print("rate ratio medians:", r.drop(columns="set").median().round(2).to_string())
    v = d.drop(columns="set")
    summ = pd.DataFrame({"median": v.median(), "share above 0.9": (v > 0.9).sum() / v.notna().sum(), "n": v.notna().sum()})
    print(summ.to_string())
    summ.to_csv(ROOT / "results/overview/table11_rule_agreement_summary.csv")


if __name__ == "__main__":
    torch.set_num_threads(8)
    cmd, args = sys.argv[1], [a for a in sys.argv[2:] if not a.startswith("--")]
    if cmd == "agreement":
        main_agreement()
    else:
        study = Study(inputs="matched", device="cpu")
        for run in args:
            if cmd == "ranges":
                if (OUT / f"range_{run}.npz").exists():
                    print(f"{run}: ranges exist, skipping"); continue
                ranges_for_run(run, study)
            elif cmd == "probe":
                probe_run(run, study, equal="--equal" in sys.argv)
            else:
                raise SystemExit(__doc__)
