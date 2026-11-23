"""Store realism: soil and snow stores of a continuous run against ERA5-Land (Caravan catchment means), per catchment.

Soil: the model's soil store (HBV soil_moisture; gated SM) against ERA5-Land root-zone soil water, layers 1-3
(0-100 cm) weighted by thickness, in m3/m3. Scores: Pearson correlation of the daily series (raw) and of the
anomalies after removing each series' own smoothed day-of-year climatology (anomaly correlation, the number
the study uses: the seasonal cycle is easy, the anomalies are the test). Both over 1975-2015 (after a 5-year
spin-up) and over the test years 1998-2008.
Snow: the model's snowpack (HBV snowpack + liquid water; gated SP) against ERA5-Land SWE: daily correlation,
annual-peak bias in % (median over years of (model peak - ERA5 peak) / ERA5 peak), and mean SWE ratio. Reported
for every catchment; `snowy` flags catchments whose ERA5-Land median annual peak SWE exceeds 10 mm.

Usage: uv run python scripts/evaluation/store_realism.py <run> [<run> ...]   -> results/<run>/store_realism.csv
Needs runs/<run>/best.pt and data/processed/era5land_catchments.nc (scripts/data/prepare_era5land.py).
"""
import sys
import time

import numpy as np
import pandas as pd
import torch
import xarray as xr

from gbhydro.continuous import simulate_continuous
from gbhydro.data import ROOT
from gbhydro.dataset import Study
from gbhydro.models import build, load_run

LAYERS = {"volumetric_soil_water_layer_1_mean": 7, "volumetric_soil_water_layer_2_mean": 21, "volumetric_soil_water_layer_3_mean": 72}
SPINUP_END, TEST = pd.Timestamp("1975-10-01"), (pd.Timestamp("1998-01-01"), pd.Timestamp("2008-12-31"))
SNOWY_PEAK_MM = 10.0


def anomalies(x: np.ndarray, dates: pd.DatetimeIndex, half_window: int = 15) -> np.ndarray:
    """x [B, T] minus its day-of-year climatology smoothed over +-half_window days (circular)."""
    doy = dates.dayofyear.values - 1
    clim = np.zeros((x.shape[0], 366)); cnt = np.zeros(366)
    np.add.at(clim.T, doy, np.nan_to_num(x).T); np.add.at(cnt, doy, 1)
    clim = clim / np.maximum(cnt, 1)
    k = np.ones(2 * half_window + 1) / (2 * half_window + 1)
    pad = np.concatenate([clim[:, -half_window:], clim, clim[:, :half_window]], 1)
    smooth = np.stack([np.convolve(r, k, mode="valid") for r in pad])
    return x - smooth[:, doy]


def corr_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a - np.nanmean(a, 1, keepdims=True); b = b - np.nanmean(b, 1, keepdims=True)
    num = np.nansum(a * b, 1); den = np.sqrt(np.nansum(a * a, 1) * np.nansum(b * b, 1))
    return np.where(den > 0, num / np.where(den > 0, den, 1), np.nan)


def main(run: str, study: Study, era: xr.Dataset):
    model = load_run(run, study)
    kind = "gated" if type(model).__name__ == "GatedHBV" else "hbv"
    rows = np.arange(len(study.basins)); dates = study.dates
    t0 = time.time(); sim = simulate_continuous(study, model, rows, batch_size=256); print(f"{run}: simulated in {time.time() - t0:.0f}s", flush=True)
    soil = sim["stores/SM"] if kind == "gated" else sim["stores/soil_moisture"]
    snow = sim["stores/SP"] if kind == "gated" else sim["stores/snowpack"] + sim["stores/liquid_water"]
    w = np.array(list(LAYERS.values()), dtype=float); w /= w.sum()
    era_soil = sum(wi * era[v].values for v, wi in zip(LAYERS, w))            # [B, T] m3/m3
    era_swe = era["snow_depth_water_equivalent_mean"].values
    full = dates >= SPINUP_END; test = (dates >= TEST[0]) & (dates <= TEST[1])
    d = pd.DataFrame(index=pd.Index([int(b) for b in study.basins], name="basin"))
    d["soil_corr_raw"] = corr_rows(soil[:, full], era_soil[:, full])
    d["soil_corr_anom"] = corr_rows(anomalies(soil, dates)[:, full], anomalies(era_soil, dates)[:, full])
    d["soil_corr_anom_test"] = corr_rows(anomalies(soil, dates)[:, test], anomalies(era_soil, dates)[:, test])
    d["soil_mean_mm"] = soil[:, full].mean(1); d["soil_range_mm"] = np.percentile(soil[:, full], 95, 1) - np.percentile(soil[:, full], 5, 1)
    d["era_soil_mean"] = era_soil[:, full].mean(1)
    d["snow_corr"] = corr_rows(snow[:, full], era_swe[:, full])
    yrs = pd.Series(dates.year, index=dates); wy = np.where(dates.month >= 10, dates.year + 1, dates.year)  # water years
    pk_m = pd.DataFrame(snow.T, index=dates).groupby(wy).max(); pk_e = pd.DataFrame(era_swe.T, index=dates).groupby(wy).max()
    pk_m, pk_e = pk_m.loc[1976:2015], pk_e.loc[1976:2015]
    d["era_peak_swe_median_mm"] = pk_e.median().values
    d["snow_peak_bias_pct"] = (100 * (pk_m - pk_e) / pk_e.where(pk_e > 1)).median().values
    d["snow_mean_ratio"] = snow[:, full].mean(1) / np.where(era_swe[:, full].mean(1) > 0, era_swe[:, full].mean(1), np.nan)
    d["snowy"] = d.era_peak_swe_median_mm > SNOWY_PEAK_MM
    held = set(int(b) for b in open(ROOT / "config/basins_spatial_test.txt").read().split())
    d["set"] = ["held-out" if b in held else "trained" for b in d.index]
    out = ROOT / "results" / run / "store_realism.csv"; out.parent.mkdir(exist_ok=True); d.to_csv(out)
    s = d[d.snowy]
    print(f"  soil anomaly corr median {d.soil_corr_anom.median():.3f} (10th {d.soil_corr_anom.quantile(.1):.3f}), raw {d.soil_corr_raw.median():.3f}; "
          f"snowy n={len(s)}: snow corr median {s.snow_corr.median():.3f}, peak bias median {s.snow_peak_bias_pct.median():+.0f}%, mean ratio {s.snow_mean_ratio.median():.2f}")
    return d


if __name__ == "__main__":
    torch.set_num_threads(4)
    study = Study(inputs="matched", device="cpu")
    era = xr.open_dataset(ROOT / "data/processed/era5land_catchments.nc")
    assert list(int(b) for b in era.basin.values) == [int(b) for b in study.basins]
    for run in sys.argv[1:]:
        main(run, study, era)
