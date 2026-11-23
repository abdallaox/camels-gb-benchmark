# Benchmarking differentiable hydrological models across Great Britain

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24477646.svg)](https://doi.org/10.5281/zenodo.24477646)

Code, results and figures for the paper *Benchmarking differentiable hydrological models across Great Britain:
performance and explainability* (in preparation).

The study compares four rainfall-runoff models on 669 catchments of CAMELS-GB under one protocol:

| model | what is learned | run names |
|---|---|---|
| **LSTM** | everything (black box), regional | `lstm_s0..4` |
| **RPL-HBV** | regional parameter learning: the 14 HBV parameters from the catchment attributes | `hbvc_b128_s0..4` |
| **RFL-HBV** | regional flux learning: every HBV flux rule learned by a small network, HBV's stores kept | `gatedc3_n16_s0..4` |
| **LC-HBV** | nothing regional: HBV calibrated on each catchment's own record (DDS) | `hbv_local_r0..4` |

Run names are the folder names under `results/` (and `runs/` once trained). In the code the three HBV-based models
are called `hbv`, `gated` and `hbv_local`.

## Repository layout

```
config/          study.yml (periods, inputs, attributes, training settings) and the catchment lists
src/gbhydro/     the package: data loading, models (models/lstm.py, hbv.py, gated_hbv.py), training, evaluation
scripts/
  data/          1. build the input arrays from CAMELS-GB and ERA5-Land
  training/      2. train the learned models and calibrate the LC-HBV
  evaluation/    3. metrics, ensembles, store realism, probed flux rules, attribution, summary tables (CSV)
  figures/       4. every figure of the paper and supplement
  tables/        5. every LaTeX table of the paper and supplement
results/         per-run metrics and diagnostics (CSV/NPZ) and results/overview/ summary tables
figures/         the figures as they appear in the paper
paper/tables/    the LaTeX tables as they appear in the paper; paper/figures_tikz/ the two model schematics
cluster/         Slurm job scripts used for training on a GPU cluster
tests/           unit tests (mass balance, flux fractions, window alignment, loss, calibration)
```

## Setup

Python 3.12 and [uv](https://docs.astral.sh/uv/):

```bash
uv sync                 # creates .venv from uv.lock
uv run pytest           # unit tests
```

On Linux, PyTorch comes from the CUDA 12.6 index; elsewhere the PyPI build is used.

## Data

The data are not redistributed here. Download them into `data/` (ignored by git):

| data | source | place in |
|---|---|---|
| CAMELS-GB v1 (Coxon et al. 2020) | EIDC, https://doi.org/10.5285/8344e4f3-d2ea-44f5-8afa-86d2987543a9 | `data/camels_gb/` (so that `data/camels_gb/data/CAMELS_GB_*_attributes.csv` exists) |
| ERA5-Land catchment means from Caravan v1.6 (Kratzert et al. 2023) | Zenodo, https://doi.org/10.5281/zenodo.15529786 | `data/caravan/archive/Caravan-nc.tar.gz` |

ERA5-Land soil water and snow are used only to check the model stores, never as inputs.

## Reproducing the paper

Every command runs from the repository root. Steps 2 and 3 need a GPU for reasonable run times; steps 4 and 5 run
on a laptop from the committed `results/`.

**1. Data** (minutes)

```bash
uv run python scripts/data/make_basin_lists.py        # optional: config/ already holds the lists used
uv run python scripts/data/prepare_data.py            # -> data/processed/camels_gb.npz
uv run python scripts/data/prepare_era5land.py extract
uv run python scripts/data/prepare_era5land.py build  # -> data/processed/era5land_catchments.nc
```

**2. Training** (one GPU; about half an hour per LSTM run, 2 hours per RPL-HBV run, 5 hours per RFL-HBV run and
6 hours for all five LC-HBV restarts).
Each run writes `runs/<name>/` (weights `best.pt`) and `results/<name>/`. On a Slurm cluster wrap any of these in
`sbatch cluster/train.sbatch ...` (arguments of `train.py`) or `sbatch cluster/run.sbatch <script> ...`.

```bash
for s in 0 1 2 3 4; do
  uv run python scripts/training/train.py --model lstm --seed $s                                   # lstm_s$s
  uv run python scripts/training/train.py --model hbv --mode continuous --seed $s --name hbvc_b128_s$s
  uv run python scripts/training/train.py --model gated --mode continuous --seed $s --name gatedc3_n16_s$s \
      --kw '{"n": 16}' --cfg '{"drift_penalty": 30}'
done
uv run python scripts/training/calibrate_hbv.py --iters 3000 --restarts 5                         # hbv_local_r0..4

# ablations (supplement): unconstrained RFL-HBV, and the size sweep with the snow switch at 0 degC
uv run python scripts/training/train.py --model gated --mode continuous --seed 0 --name gatedc_s0 \
    --kw '{"constrained": false, "snow_tt": 0.0}'
for s in 0 1 2 3 4; do
  uv run python scripts/training/train.py --model gated --mode continuous --seed $s --name gatedc2_s$s \
      --kw '{"snow_tt": 0.0}' --cfg '{"drift_penalty": 30}'                                        # n = 8
done
for n in 2 4 16 32 48; do
  uv run python scripts/training/train.py --model gated --mode continuous --seed 0 --name gatedc2_n${n}_s0 \
      --kw "{\"n\": $n, \"snow_tt\": 0.0}" --cfg '{"drift_penalty": 30}'
done
```

The trained weights of every run are archived with the Zenodo release of this repository (see below), so steps 3
to 5 can be rerun without training: unpack them into `runs/`.

**3. Evaluation** (CPU or GPU, hours)

```bash
uv run python scripts/evaluation/eval_full_metrics.py          # continuous evaluation of the main runs -> metrics_test_full.csv
uv run python scripts/evaluation/eval_continuous.py lstm_s0    # LSTM: windowed against continuous evaluation
for p in lstm_s hbvc_b128_s gatedc3_n16_s; do uv run python scripts/evaluation/ensemble.py $p; done
for r in hbvc_b128_s{0..4} gatedc3_n16_s{0..4} hbv_local_r{0..4} gatedc_s0 gatedc2_s{0..4} gatedc2_n{2,4,16,32,48}_s0; do
  uv run python scripts/evaluation/store_realism.py $r         # soil and snow against ERA5-Land
done
uv run python scripts/evaluation/diag_continuous.py gatedc3_n16_s{0..4} gatedc_s0   # water balance, sinks
RULES="hbvc_b128_s0 hbvc_b128_s1 hbvc_b128_s2 hbvc_b128_s3 hbvc_b128_s4 gatedc3_n16_s0 gatedc3_n16_s1 gatedc3_n16_s2 \
  gatedc3_n16_s3 gatedc3_n16_s4 hbv_local_r0 hbv_local_r1 hbv_local_r2 hbv_local_r3 hbv_local_r4"
uv run python scripts/evaluation/probe_rules.py ranges $RULES   # probed flux rules (partial dependence)
uv run python scripts/evaluation/probe_rules.py probe $RULES
uv run python scripts/evaluation/probe_rules.py agreement
for r in lstm_s0 hbvc_b128_s0 gatedc3_n16_s0; do uv run python scripts/evaluation/xai.py --run $r; done
uv run python scripts/evaluation/analysis_overview.py
uv run python scripts/evaluation/analysis_attributes.py
uv run python scripts/evaluation/bias_by_abstraction.py
uv run python scripts/evaluation/size_sweep_table.py
```

**4. Figures** and **5. Tables**: each script writes one figure (or a few) to `figures/`, or the LaTeX tables to
`paper/tables/`. The maps below give the script for every figure and table.

### Figures

| paper | file | script (`scripts/figures/`) |
|---|---|---|
| Fig. 1, 2 | `schematic_regional_hbv.pdf`, `schematic_gated_hbv.pdf` | TikZ sources in `paper/figures_tikz/` |
| Fig. 3, 7 | `performance_cdf.png`, `ungauged_cdf.png` | `plot_performance_cdf.py` |
| Fig. 4 | `spatial_maps.png` | `plot_spatial_maps.py` |
| Fig. 5 | `attribute_boxes.png` | `plot_attribute_boxes.py` |
| Fig. 6 | `attribute_correlations.png` | `plot_attribute_correlations.py` |
| Fig. 8 | `store_cdf.png` | `plot_store_cdf.py` |
| Fig. 9 | `store_maps.png` | `plot_store_maps.py` |
| Fig. 10 | `gate_rules.png` | `plot_gate_rules.py` |
| Fig. 11 | `rule_agreement_maps.png` | `plot_rule_agreement_maps.py` |
| Fig. 12, S9 | `xai.png`, `xai_full.png` | `plot_xai.py` |
| Fig. 13, S1 | `tradeoff.png`, `tradeoff_sweep.png` | `plot_tradeoff.py` |
| Fig. 14, S11 | `regionalisation.png`, `regionalisation_scatter.png` | `plot_regionalisation.py` |
| Fig. S2 | `hydrographs.png` | `plot_hydrographs.py` |
| Fig. S3 | `attribute_bins.png` | `plot_attribute_bins.py` |
| Fig. S4 | `store_series.png` | `plot_store_series.py` |
| Fig. S5 | `snow_series_hbvc_b128_s0_gatedc3_n16_s0.png` | `plot_snow_series.py hbvc_b128_s0 gatedc3_n16_s0` |
| Fig. S6 | `gate_curves_gatedc3_n16_s0.png` | `plot_gate_curves.py` |
| Fig. S7 | `gate_seeds_gatedc3_n16.png` | `plot_gate_seeds.py` |
| Fig. S8 | `rule_agreement.png` | `plot_rule_agreement.py` |
| Fig. S10 | `gap_maps.png` | `plot_gap_maps.py` |

### Tables

| paper | file in `paper/tables/` | script (`scripts/tables/`) |
|---|---|---|
| Table 1, 2, 3; S4 to S9 | `table1_main`, `table5`, `table7`; `table10`, `table1`, `table3`, `table3b`, `table3c`, `table9` | `make_tables.py` |
| Table S1 | `tableS1_attributes` | `make_supplement_tables.py` |
| Table S2 | `table4_hbv_params` | `make_param_table.py` |

## Notes on the evaluation

- Every model is evaluated in one continuous run over the record (1970 to 2015), scored on the test decade 1998
  to 2008, separately for the 536 training and the 133 held-out catchments.
- Headline numbers are medians over catchments of the per-seed values, with the range over the five seeds.
- The LSTM is trained on 365-day windows; the HBV-based models are trained continuously, with stores carried
  between one-year chunks. `eval_continuous.py lstm_s0` shows that the LSTM scores do not depend on this.

## Licence and citation

Code: MIT (see `LICENSE`). Results, figures and tables: CC BY 4.0 (see `results/LICENSE`). CAMELS-GB and ERA5-Land
keep their own licences. If you use this work, please cite the paper and the archived release (see `CITATION.cff`).
