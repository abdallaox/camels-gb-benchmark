"""Train and evaluate one run with the shared pipeline.

  uv run python scripts/training/train.py --model lstm --seed 0                 # run name lstm_s0
  uv run python scripts/training/train.py --model lstm --seed 0 --smoke         # 20 batches, CPU-sized check

Resumes automatically if runs/<name>/last.pt exists. Env: GBHYDRO_RUN_DIR, GBHYDRO_DEVICE.
"""
import argparse
import json
from pathlib import Path

import torch

from gbhydro.dataset import Study
from gbhydro.models import build
from gbhydro.study import DEVICE, RUN_DIR, STUDY
from gbhydro.train import Trainer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--inputs", default="matched", choices=["matched", "full"])
    ap.add_argument("--name", default=None)
    ap.add_argument("--cfg", default=None, help='JSON overrides on the training config, e.g. {"batch_basins": 128}')
    ap.add_argument("--kw", default="{}", help='model kwargs as JSON, e.g. \'{"hidden_size": 128}\'')
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--mode", default="windowed", choices=["windowed", "continuous"],
                    help="continuous = training_continuous overrides in study.yml (HBV-structured models)")
    ap.add_argument("--device", default=DEVICE)
    a = ap.parse_args()

    kw = json.loads(a.kw)
    name = a.name or f"{a.model}{'_full' if a.inputs == 'full' else ''}{'_' + '_'.join(f'{k}{v}' for k, v in kw.items()) if kw else ''}_s{a.seed}"
    cfg = dict(STUDY["training"])
    if a.mode == "continuous":
        cfg.update(STUDY["training_continuous"])
        if not a.name:
            name = name.replace(a.model, a.model + "c", 1)               # hbvc_s0, gatedc_s0
    if a.cfg:
        cfg.update(json.loads(a.cfg))
    if a.epochs:
        cfg["epochs"] = a.epochs
    if a.smoke:
        cfg.update(epochs=2, validate_n_basins=5)
        name = f"smoke_{name}"
    print(f"run {name}  device {a.device}  model {a.model} {kw}")
    study = Study(inputs=a.inputs, device=a.device)
    model = build(a.model, study, **kw)
    print(f"{sum(p.numel() for p in model.parameters()):,} parameters; "
          f"{study.n_samples('train', 'train'):,} training windows")
    tr = Trainer(study, model, Path(RUN_DIR) / name, cfg, seed=a.seed, max_batches=20 if a.smoke else None)
    tr.fit()
    tr.evaluate("all", "test")


if __name__ == "__main__":
    main()
