"""Per-catchment HBV calibration (the conventional, gauged-only baseline) by DDS, batched on one GPU.

  uv run python scripts/training/calibrate_hbv.py --iters 3000 --restarts 5            # runs hbv_local_r0 .. hbv_local_r4
  uv run python scripts/training/calibrate_hbv.py --smoke                              # 8 basins, 5 iterations, CPU-sized

Every catchment (all 669, held-out ones included: this model is calibrated on each catchment's own record) gets its
own 14 HBV parameters, found by dynamically dimensioned search (Tolson and Shoemaker 2007) on the NSE of a continuous
simulation over the training period, with the same core, ranges, initial stores and target days as the regional
HBV. `--restarts` independent searches from random starts play the role of the five seeds of the learned models.
All basins x restarts are evaluated in one batched pass per iteration, so the cost is that of one 27-year run per
iteration. Resumes from <RUN_DIR>/<name>_dds/dds_state.pt. Each restart is written as a run folder that
gbhydro.models.load_run can rebuild (model hbv_local): best.pt, config.yml, parameters_test.csv, calibration.csv
(train-period NSE), metrics_test.csv and metrics_test_full.csv (continuous run over the record, test period).
Env: GBHYDRO_RUN_DIR, GBHYDRO_DEVICE.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from gbhydro.calibrate import dds, nse_per_row, simulate_q
from gbhydro.continuous import simulate_continuous
from gbhydro.dataset import Study
from gbhydro.metrics import ALL, EXTRA
from gbhydro.models.hbv import PARAMS, LocalHBV, hbv_step
from gbhydro.study import DEVICE, RUN_DIR, STUDY


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=3000)
    ap.add_argument("--restarts", type=int, default=5)
    ap.add_argument("--name", default="hbv_local")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=DEVICE)
    ap.add_argument("--compile", default="auto", choices=["auto", "none", "default", "reduce-overhead"],
                    help="torch.compile mode for the day step (auto = default on CUDA, none on CPU)")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    if a.smoke:
        a.iters, a.restarts, a.name = 5, 2, "smoke_" + a.name

    study = Study(inputs="matched", device=a.device)
    dev = study.device
    n_all = len(study.basins)
    rows = np.arange(n_all if not a.smoke else 8)
    t0, t1 = study.periods["train"]
    first_target = max(t0, study.seq_length - 1)                      # same first target day as the trainer
    n, R = len(rows), a.restarts
    bidx = torch.as_tensor(np.tile(rows, R), device=dev)              # chain c = restart c // n, basin rows[c % n]
    p, t, pet = (study._raw[k][bidx, t0:t1 + 1] for k in ("precipitation", "temperature", "pet"))
    obs = study.target[bidx, t0:t1 + 1]
    mask = torch.isfinite(obs)
    mask[:, :first_target - t0] = False
    mode = a.compile if a.compile != "auto" else ("default" if dev.type == "cuda" else "none")
    step = hbv_step if mode == "none" else torch.compile(hbv_step, mode=mode)
    model = LocalHBV(n_all).to(dev)
    print(f"{a.name}: {n} basins x {R} restarts = {n * R} chains, {p.shape[1]} days, {a.iters} iterations, "
          f"device {dev}, compile {mode}", flush=True)

    def objective(u):
        q = simulate_q(step, model.params_from_unit(u), p, t, pet)
        return nse_per_row(q, obs, mask)

    state_dir = Path(RUN_DIR) / f"{a.name}_dds"; state_dir.mkdir(parents=True, exist_ok=True)
    ck_file = state_dir / "dds_state.pt"
    gen = torch.Generator(device=dev).manual_seed(a.seed)
    if ck_file.exists():
        ck = torch.load(ck_file, map_location=dev)
        u0, f0, start, trace0 = ck["u_best"], ck["f_best"], ck["iter"], ck["trace"]
        gen.set_state(ck["gen_state"].cpu() if dev.type == "cpu" else ck["gen_state"])
        print(f"resumed at iteration {start}, median train NSE {float(f0.median()):.3f}", flush=True)
    else:
        u0, f0, start, trace0 = torch.rand(n * R, len(PARAMS), generator=gen, device=dev), None, 0, []

    clock = {"t": time.time()}

    def log(i, f_best, better):
        if i % 10 == 0 or i <= 3:
            dt = (time.time() - clock["t"]) / (10 if i % 10 == 0 else 1); clock["t"] = time.time()
            fb = f_best[torch.isfinite(f_best)]
            if len(fb) == 0:
                print(f"iter {i:5d}  no chain has a finite objective yet", flush=True); return
            print(f"iter {i:5d}  median NSE {fb.median():.4f}  10th pct {fb.quantile(0.1):.3f}  "
                  f"improved {int(better.sum())}/{len(better)}  {dt:.1f} s/iter", flush=True)

    def checkpoint(i, u_best, f_best):
        torch.save({"u_best": u_best, "f_best": f_best, "iter": i, "gen_state": gen.get_state(),
                    "trace": trace0 + trace}, ck_file)

    trace = []
    u_best, f_best, trace_new = dds(objective, u0, a.iters, gen, start_iter=start, f0=f0, log=log,
                                   checkpoint=lambda i, u, f: checkpoint(i, u, f))
    trace.extend(trace_new)
    pd.DataFrame(trace0 + trace, columns=["iteration", "median_train_NSE"]).to_csv(state_dir / "dds_trace.csv", index=False)

    # ---- write one run folder per restart and evaluate it continuously over the record
    rows_all = np.arange(n_all)
    a_test, z_test = study.periods["test"]
    held = set(int(b) for b in study.basins[study.sets["spatial_test"]])
    basins = [int(b) for b in study.basins[rows]]
    for k in range(R):
        run = Path(RUN_DIR) / f"{a.name}_r{k}"; run.mkdir(parents=True, exist_ok=True)
        u_k, f_k = u_best[k * n:(k + 1) * n], f_best[k * n:(k + 1) * n]
        with torch.no_grad():
            model.u[torch.as_tensor(rows, device=dev)] = u_k
        torch.save({"model": model.state_dict(), "iters": a.iters, "restart": k}, run / "best.pt")
        (run / "config.yml").write_text(yaml.safe_dump({
            "model": "hbv_local", "model_kw": {}, "seed": k, "n_params": int(n * len(PARAMS)),
            "calibration": {"method": "DDS (Tolson and Shoemaker 2007), r = 0.2, greedy", "iterations": a.iters,
                            "objective": "NSE, continuous simulation over the training period from the fixed initial stores",
                            "period": [str(study.dates[t0].date()), str(study.dates[t1].date())],
                            "first_target_day": str(study.dates[first_target].date()), "restarts": R, "restart": k,
                            "ranges": model.ranges}}))
        pars = pd.DataFrame(model.basin_parameters(rows), index=pd.Index(basins, name="basin"))
        pars.to_csv(run / "parameters_test.csv")
        (run / "ranges.json").write_text(json.dumps(model.ranges, indent=1))
        cal = pd.DataFrame({"train_NSE": f_k.cpu().numpy()}, index=pd.Index(basins, name="basin"))
        cal["train_NSE"] = cal["train_NSE"].replace(-np.inf, np.nan)
        at_bound = ((u_k < 0.02) | (u_k > 0.98)).float().mean(0).cpu().numpy()
        cal.to_csv(run / "calibration.csv")
        sim = simulate_continuous(study, model, rows, batch_size=256)
        d = pd.DataFrame(index=pd.Index(basins, name="basin"))
        for name, fn in {**ALL, **EXTRA}.items():
            d[name] = [fn(sim["q"][i, a_test:z_test + 1], sim["obs"][i, a_test:z_test + 1]) for i in range(n)]
        d["set"] = ["held-out" if b in held else "trained" for b in d.index]
        d.to_csv(run / "metrics_test_full.csv")
        d[list(ALL) + ["set"]].to_csv(run / "metrics_test.csv")
        summary = d.groupby("set")[["NSE", "KGE"]].median().round(3)
        (run / "summary_test.json").write_text(json.dumps(
            {"iterations": a.iters, "train_NSE_median": float(np.nanmedian(cal["train_NSE"])),
             **{f"{s}_{m}": float(v) for s, r in summary.iterrows() for m, v in r.items()},
             "share_at_bound": {q: float(v) for q, v in zip(PARAMS, at_bound)}}, indent=1))
        print(f"{run.name}: train NSE median {np.nanmedian(cal['train_NSE']):.3f}; test\n{summary.to_string()}\n"
              f"at a bound (>2% of range): " + ", ".join(f"{q} {100 * v:.0f}%" for q, v in zip(PARAMS, at_bound) if v > 0.1),
              flush=True)


if __name__ == "__main__":
    main()
