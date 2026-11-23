"""Shared study settings: config/study.yml, basin lists, environment overrides.

Environment overrides (e.g. on a cluster):
  GBHYDRO_DATA_DIR  NeuralHydrology data folder (default data/nh in the repo)
  GBHYDRO_RUN_DIR   where runs are written (default runs/ in the repo)
  GBHYDRO_DEVICE    cpu or cuda:0 (default cpu)
"""
import os
from pathlib import Path

import yaml

from gbhydro.data import NH_DATA_DIR, ROOT

STUDY = yaml.safe_load((ROOT / "config" / "study.yml").read_text())
DATA_DIR = Path(os.environ.get("GBHYDRO_DATA_DIR", NH_DATA_DIR))
RUN_DIR = Path(os.environ.get("GBHYDRO_RUN_DIR", ROOT / "runs"))
DEVICE = os.environ.get("GBHYDRO_DEVICE", "cpu")


def read_basins(name: str) -> list[str]:
    return (ROOT / STUDY["basins"][name]).read_text().split()
