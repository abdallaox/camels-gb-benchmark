"""One training loop for every model.

    trainer = Trainer(study, model, run_dir, cfg)   # cfg = STUDY["training"] (+ overrides)
    trainer.fit()                                   # resumes from run_dir/last.pt if present
    trainer.evaluate("all", "test")                 # per-basin metrics + simulated series at best epoch

Early stopping: median validation NSE over `validate_n_basins` fixed random training basins,
patience `early_stopping_patience`; best weights kept in best.pt. TensorBoard logs in run_dir/tb.

Two training modes (cfg["mode"]):
  windowed    365-day windows, loss on the last day, every observed day a sample (Kratzert 2019 / Lees 2021).
  continuous  each basin is run through the training period in order, in chunks of `chunk_days`, carrying the
              model state between chunks (truncated backpropagation); loss on every observed target day of a
              chunk (the same target days as windowed mode). Validation and test are one continuous run over the
              whole record. For models that carry hydrological stores, this removes the cold start that windowed
              training imposes on slow stores (decision of 14 Nov 2026).
"""
import csv
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.tensorboard import SummaryWriter

from gbhydro.continuous import simulate_continuous
from gbhydro.loss import EPS, nse_star
from gbhydro.metrics import all_metrics


class Trainer:
    def __init__(self, study, model, run_dir: Path, cfg: dict, seed: int = 0,
                 max_batches: int | None = None, log=print):
        self.study, self.model, self.run_dir, self.cfg, self.seed = study, model, Path(run_dir), cfg, seed
        self.max_batches = max_batches            # smoke tests only
        self.log = log
        self.run_dir.mkdir(parents=True, exist_ok=True)
        torch.manual_seed(seed); np.random.seed(seed)
        self.model.to(study.device)
        self.opt = torch.optim.Adam(self.model.parameters(), lr=self._lr(0))
        self.epoch, self.best, self.best_epoch, self.since_best = 0, -np.inf, None, 0
        self.tb = SummaryWriter(self.run_dir / "tb")
        self.step = 0
        n_val = int(cfg.get("validate_n_basins", 0) or 0)
        tr = study.sets["train"]
        self.val_rows = np.sort(np.random.default_rng(0).choice(tr, n_val, replace=False)) if n_val else tr
        (self.run_dir / "config.yml").write_text(yaml.safe_dump({"training": cfg, "seed": seed, "model": model.name, "model_kw": getattr(model, "kw", {}),
                                                                 "n_params": sum(p.numel() for p in model.parameters())}))
        if (self.run_dir / "last.pt").exists():
            self._load("last.pt")

    def _lr(self, epoch):
        sched = {int(k): float(v) for k, v in self.cfg["learning_rate"].items()}
        return sched[max(k for k in sched if k <= epoch)]

    # ------------------------------------------------------------------ checkpoints
    def _save(self, name):
        torch.save({"model": self.model.state_dict(), "opt": self.opt.state_dict(), "epoch": self.epoch,
                    "best": self.best, "best_epoch": self.best_epoch, "since_best": self.since_best,
                    "step": self.step}, self.run_dir / name)

    def _load(self, name):
        ck = torch.load(self.run_dir / name, map_location=self.study.device)
        self.model.load_state_dict(ck["model"]); self.opt.load_state_dict(ck["opt"])
        self.epoch, self.best, self.best_epoch = ck["epoch"], ck["best"], ck["best_epoch"]
        self.since_best, self.step = ck["since_best"], ck["step"]
        self.log(f"resumed {name} at epoch {self.epoch} (best {self.best:.3f} @ {self.best_epoch})")

    # ------------------------------------------------------------------ training
    @property
    def continuous(self) -> bool:
        return self.cfg.get("mode", "windowed") == "continuous"

    def _epoch_windowed(self):
        cfg = self.cfg
        for i, b in enumerate(self.study.batches("train", "train", cfg["batch_size"], shuffle=True,
                                                  seed=self.seed * 1000 + self.epoch)):
            if self.max_batches and i >= self.max_batches:
                break
            yield nse_star(self.model(b)["y_hat"], b["y"], b["std"])

    def _epoch_continuous(self):
        """Chunks in time order; within a chunk the training basins in shuffled mini-batches; model state carried
        per basin from chunk to chunk (detached). Loss on observed days at or after the first target day."""
        cfg, st = self.cfg, self.study
        rows = st.sets["train"]
        t0, t1 = st.periods["train"]
        first_target = max(t0, st.seq_length - 1)                      # same first target day as windowed mode
        chunk, nb = int(cfg.get("chunk_days", 365)), int(cfg.get("batch_basins", 32))
        lam = float(cfg.get("drift_penalty", 0.0))
        rng = np.random.default_rng(self.seed * 1000 + self.epoch)
        table, n_steps = {}, 0
        for c0 in range(t0, t1 + 1, chunk):
            c1 = min(c0 + chunk, t1 + 1)
            order = rng.permutation(len(rows))
            for j in range(0, len(order), nb):
                idx = order[j:j + nb]
                r = torch.as_tensor(rows[idx], device=st.device)
                batch = {"x_d": st.x_d[r, c0:c1], "x_s": st.x_s[r], "p": st._raw["precipitation"][r, c0:c1],
                         "t": st._raw["temperature"][r, c0:c1], "pet": st._raw["pet"][r, c0:c1], "rows": r}
                state = {k: v[idx] for k, v in table.items()} if table else None
                out = self.model(batch, state)
                for k, v in out["state"].items():
                    if k not in table:
                        table[k] = torch.zeros((len(rows),) + v.shape[1:], device=st.device, dtype=v.dtype)
                    table[k][idx] = v.detach()
                q = out["q_routed"] if "q_routed" in out else out["fluxes"]["q"]          # [B, chunk]
                y = st.target[r, c0:c1]
                mask = torch.isfinite(y) & torch.isfinite(st.q_std[r])[:, None]
                mask[:, :max(0, first_target - c0)] = False
                if not mask.any():
                    continue
                y = torch.where(mask, y, torch.zeros_like(y))              # NaN targets would poison the backward pass
                std = torch.nan_to_num(st.q_std[r], nan=1.0)[:, None].expand_as(y)
                loss = torch.mean(((q - y) ** 2 / (std + EPS) ** 2)[mask])
                if lam > 0 and state is not None:                          # first chunk (filling from the cold start) exempt
                    loss = loss + lam * self._drift_penalty(state, out["state"], batch["p"])
                yield loss
                n_steps += 1
                if self.max_batches and n_steps >= self.max_batches:
                    return

    @staticmethod
    def _storage(state: dict | None) -> torch.Tensor | None:
        """Total water held in all stores, per basin [B], from a model state dict ([B] or [B, n] tensors)."""
        return None if state is None else sum(v.flatten(1).sum(1) if v.dim() > 1 else v for v in state.values())

    def _drift_penalty(self, state_in, state_out, p) -> torch.Tensor:
        """Storage-drift penalty for one chunk: mean over basins of ((S_end - S_start) / sum P)^2. Charges a model for
        water that enters the stores and never leaves (a learned sink) or for draining stores it did not fill; a
        seasonal cycle cancels over a water-year chunk. The caller skips the first chunk, where the stores fill from
        the cold start; if called without a start state the fixed initial storage is used. cfg key `drift_penalty`
        = weight (0 = off; the gated runs of 15 Nov 2026 use 30)."""
        s_end = Trainer._storage(state_out)
        s_start = Trainer._storage(state_in)
        if s_start is None:                                             # first chunk: the model's fixed initial stores
            s_start = torch.full_like(s_end, float(sum(self.model.INITIAL.values())))
        return torch.mean(((s_end - s_start) / (p.sum(1) + 1.0)) ** 2)

    def fit(self):
        cfg = self.cfg
        while self.epoch < cfg["epochs"]:
            self.epoch += 1
            for g in self.opt.param_groups:
                g["lr"] = self._lr(self.epoch - 1)
            self.model.train()
            t0, losses, nan_run = time.time(), [], 0
            for loss in (self._epoch_continuous() if self.continuous else self._epoch_windowed()):
                if not torch.isfinite(loss):
                    nan_run += 1
                    if nan_run > cfg.get("allow_nan_batches", 5):
                        raise RuntimeError("loss NaN repeatedly")
                    continue
                nan_run = 0
                self.opt.zero_grad(set_to_none=True)
                loss.backward()
                gn = torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg["clip_gradient_norm"])
                self.opt.step()
                li = loss.item(); losses.append(li)
                self.step += 1
                if self.step % 50 == 0:
                    self.tb.add_scalar("train/loss", li, self.step)
                    self.tb.add_scalar("train/grad_norm", float(gn), self.step)
            val = self.validate()
            self.tb.add_scalar("val/NSE_median", val["NSE"], self.epoch)
            self.tb.add_scalar("val/KGE_median", val["KGE"], self.epoch)
            self.tb.add_scalar("train/lr", self._lr(self.epoch - 1), self.epoch)
            improved = val["NSE"] > self.best
            if improved:
                self.best, self.best_epoch, self.since_best = val["NSE"], self.epoch, 0
                self._save("best.pt")
            else:
                self.since_best += 1
            self._save("last.pt")
            row = {"epoch": self.epoch, "train_loss": float(np.mean(losses)) if losses else np.nan,
                   "val_NSE": val["NSE"], "val_KGE": val["KGE"], "val_basins": val["n"],
                   "seconds": round(time.time() - t0), "best": improved}
            self._log_row(row)
            self.log(f"epoch {self.epoch:2d}  loss {row['train_loss']:.4f}  val NSE {val['NSE']:.3f}  "
                     f"KGE {val['KGE']:.3f}  ({row['seconds']}s){'  *' if improved else ''}")
            if self.since_best >= cfg.get("early_stopping_patience", 5):
                self.log(f"early stop: no improvement for {self.since_best} epochs (best {self.best:.3f} @ {self.best_epoch})")
                break
        self.tb.flush()
        return self.best, self.best_epoch

    def _log_row(self, row):
        f = self.run_dir / "log.csv"
        new = not f.exists()
        with open(f, "a", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(row));
            if new: w.writeheader()
            w.writerow(row)

    # ------------------------------------------------------------------ evaluation
    @torch.no_grad()
    def simulate(self, rows: np.ndarray, period: str, batch_size: int = 2048) -> pd.DataFrame:
        """Simulation for every observed day of `period` in basins `rows` (windowed: sequence-to-one per day;
        continuous: one run over the whole record, sliced). Returns a long DataFrame: basin, day index, sim, obs."""
        self.model.eval()
        if self.continuous:
            if self.max_batches:
                rows = rows[:8]
            c = simulate_continuous(self.study, self.model, rows)
            t0, t1 = self.study.periods[period]
            bi, ti = np.nonzero(np.isfinite(c["obs"][:, t0:t1 + 1]))
            return pd.DataFrame({"basin": self.study.basins[rows[bi]], "day": ti + t0, "sim": c["q"][bi, ti + t0],
                                 "obs": c["obs"][bi, ti + t0]})
        b, t = self._samples(rows, period)
        sims = []
        for i in range(0, len(b), batch_size):
            w = self.study.window(b[i:i + batch_size], t[i:i + batch_size])
            if self.max_batches and i // batch_size >= self.max_batches:
                break
            sims.append(self.model(w)["y_hat"].float().cpu().numpy())
        sim = np.concatenate(sims) if sims else np.array([])
        n = len(sim)
        return pd.DataFrame({"basin": self.study.basins[b[:n]], "day": t[:n], "sim": sim,
                             "obs": self.study.target[b[:n], t[:n]].cpu().numpy()})

    def _samples(self, rows, period):
        """(basin rows, end days) of every observed day in `period` for the given basin rows."""
        t0, t1 = self.study.periods[period]
        t0 = max(t0, self.study.seq_length - 1)
        obs = torch.isfinite(self.study.target[rows][:, t0:t1 + 1]).cpu().numpy()
        bi, ti = np.nonzero(obs)
        return rows[bi], ti + t0

    def validate(self) -> dict:
        df = self.simulate(self.val_rows, "validation")
        m = df.groupby("basin").apply(lambda g: pd.Series(all_metrics(g["sim"].to_numpy(), g["obs"].to_numpy())),
                                      include_groups=False)
        return {"NSE": float(m["NSE"].median()), "KGE": float(m["KGE"].median()), "n": int(m["NSE"].notna().sum())}

    def evaluate(self, basin_set: str = "all", period: str = "test", tag: str | None = None):
        """Per-basin metrics and simulated series at the best epoch, written to run_dir."""
        if (self.run_dir / "best.pt").exists():
            ck = torch.load(self.run_dir / "best.pt", map_location=self.study.device)
            self.model.load_state_dict(ck["model"])
        df = self.simulate(self.study.sets[basin_set], period)
        m = df.groupby("basin").apply(lambda g: pd.Series(all_metrics(g["sim"].to_numpy(), g["obs"].to_numpy())),
                                      include_groups=False)
        train = set(self.study.basins[self.study.sets["train"]])
        m["set"] = ["trained" if b in train else "held-out" for b in m.index]
        tag = tag or f"{period}"
        m.to_csv(self.run_dir / f"metrics_{tag}.csv")
        df["date"] = self.study.dates[df["day"].to_numpy()]
        df.drop(columns="day").to_parquet(self.run_dir / f"sim_{tag}.parquet", index=False)
        if hasattr(self.model, "basin_parameters"):                 # HBV models: per-basin parameters at best epoch
            rows = self.study.sets[basin_set]
            pars = pd.DataFrame(self.model.basin_parameters(self.study.x_s[rows]), index=self.study.basins[rows])
            pars.index.name = "basin"
            pars.to_csv(self.run_dir / f"parameters_{tag}.csv")
            if hasattr(self.model, "ranges"):
                (self.run_dir / "ranges.json").write_text(json.dumps(self.model.ranges, indent=1))
        summary = m.groupby("set")[["NSE", "KGE"]].median().round(3)
        self.log(f"{tag}: best epoch {self.best_epoch}\n{summary.to_string()}")
        (self.run_dir / f"summary_{tag}.json").write_text(json.dumps(
            {"best_epoch": self.best_epoch, **{f"{s}_{k}": float(v) for s, r in summary.iterrows() for k, v in r.items()}},
            indent=1))
        return m

