"""All basins as one tensor; windows are index slices. No data loader, no workers.

    study = Study(device="cuda:0")
    for batch in study.batches("train", "train", batch_size=256, shuffle=True, seed=0):
        batch["x_d"]   [B, 365, n_dyn]  standardised dynamic inputs (train-basin, train-period stats)
        batch["x_s"]   [B, n_attr]      standardised static attributes (train-basin stats)
        batch["p"], ["t"], ["pet"]  [B, 365]  raw P (mm/d), T (degC), PET (mm/d) for the HBV cores
        batch["y"]     [B]              observed discharge on the last day, mm/d
        batch["std"]   [B]              per-basin train-period discharge std (for the NSE* loss)
        batch["basin"] [B]              row index into study.basins

A window = 365 days of input ending on day t; only day t is a target (Kratzert 2019 / Lees 2021).
Windows exist for every day t in the period whose target is observed. Validation and test
use the same windows, so evaluation equals NeuralHydrology's sequence-to-one evaluation.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from gbhydro.data import ROOT
from gbhydro.study import STUDY, read_basins

PROCESSED = Path(ROOT / "data" / "processed" / "camels_gb.npz")


class Study:
    def __init__(self, inputs: str = "matched", statics: str = "extended", seq_length: int | None = None,
                 device: str = "cpu", path: Path = PROCESSED):
        z = np.load(path, allow_pickle=False)
        self.device = torch.device(device)
        self.basins = z["basins"]                                  # int gauge ids
        self.dates = pd.DatetimeIndex(z["dates"])
        self.seq_length = seq_length or STUDY["training"]["seq_length"]
        fnames, anames = list(z["forcing_names"]), list(z["attr_names"])
        self.dyn_names = list(STUDY["dynamic_inputs"][inputs])
        self.attr_names = list(STUDY["static_attributes"][statics])
        dyn_idx = [fnames.index(v) for v in self.dyn_names]
        attr_idx = [anames.index(a) for a in self.attr_names]

        f = torch.from_numpy(z["forcing"])
        self._raw = {k: f[:, :, fnames.index(k)].to(self.device) for k in ("precipitation", "temperature", "pet")}
        self.target = torch.from_numpy(z["target"]).to(self.device)          # [B, T], NaN = unobserved
        x_d = f[:, :, dyn_idx]
        x_s = torch.from_numpy(z["attrs"][:, attr_idx])

        # periods and basin sets
        self.periods = {k: (self.dates.get_loc(pd.Timestamp(a)), self.dates.get_loc(pd.Timestamp(b)))
                        for k, (a, b) in STUDY["periods"].items()}
        row = {int(b): i for i, b in enumerate(self.basins)}
        self.sets = {k: np.array([row[int(b)] for b in read_basins(k)]) for k in ("all", "train", "spatial_test")}

        # normalisation: train basins, train period only
        tr, (t0, t1) = self.sets["train"], self.periods["train"]
        xd_tr = x_d[tr, t0:t1 + 1].reshape(-1, len(dyn_idx))
        self.dyn_mean, self.dyn_std = xd_tr.mean(0).to(self.device), xd_tr.std(0).to(self.device)
        self.attr_mean, self.attr_std = x_s[tr].mean(0), x_s[tr].std(0)
        y_tr = self.target[:, t0:t1 + 1].cpu().numpy()
        self.q_std = torch.tensor([np.nanstd(r) if np.isfinite(r).sum() > 1 else np.nan for r in y_tr],
                                  dtype=torch.float32).to(self.device)
        self.x_d = ((x_d.to(self.device) - self.dyn_mean) / self.dyn_std)
        self.x_s = ((x_s - self.attr_mean) / self.attr_std).to(self.device)

    # ------------------------------------------------------------------ windows
    def samples(self, basin_set: str, period: str) -> tuple[np.ndarray, np.ndarray]:
        """(basin rows, end days) of every valid window: target observed, window inside the record,
        and (for training) a usable per-basin std."""
        rows, (t0, t1) = self.sets[basin_set], self.periods[period]
        t0 = max(t0, self.seq_length - 1)
        obs = torch.isfinite(self.target[rows][:, t0:t1 + 1]).cpu().numpy()
        if period == "train":
            obs &= torch.isfinite(self.q_std[rows]).cpu().numpy()[:, None]
        b, t = np.nonzero(obs)
        return rows[b], t + t0

    def batches(self, basin_set: str, period: str, batch_size: int, shuffle: bool = False, seed: int = 0):
        b, t = self.samples(basin_set, period)
        order = np.random.default_rng(seed).permutation(len(b)) if shuffle else np.arange(len(b))
        for i in range(0, len(order), batch_size):
            yield self.window(b[order[i:i + batch_size]], t[order[i:i + batch_size]])

    def window(self, b: np.ndarray, t: np.ndarray) -> dict:
        b = torch.as_tensor(b, device=self.device)
        t = torch.as_tensor(t, device=self.device)
        idx = t[:, None] - torch.arange(self.seq_length - 1, -1, -1, device=self.device)[None, :]   # [B, L]
        bb = b[:, None].expand_as(idx)
        return {"x_d": self.x_d[bb, idx], "x_s": self.x_s[b],
                "p": self._raw["precipitation"][bb, idx], "t": self._raw["temperature"][bb, idx],
                "pet": self._raw["pet"][bb, idx],
                "y": self.target[b, t], "std": self.q_std[b], "basin": b, "day": t}

    def n_samples(self, basin_set: str, period: str) -> int:
        return len(self.samples(basin_set, period)[0])
