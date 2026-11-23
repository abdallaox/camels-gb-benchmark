# Results

One folder per run, named as in the main README. Typical files:

| file | content | written by |
|---|---|---|
| `config.yml` | the exact settings of the run | `scripts/training/train.py`, `calibrate_hbv.py` |
| `log.csv` | training loss and validation NSE per epoch | `train.py` |
| `metrics_test.csv` | test-period metrics per catchment as written at the end of training | `train.py`, `calibrate_hbv.py` |
| `metrics_test_full.csv` | test-period metrics per catchment from one continuous run over the record (used in the paper) | `scripts/evaluation/eval_full_metrics.py`, `calibrate_hbv.py` |
| `store_realism.csv` | soil and snow realism against ERA5-Land per catchment | `store_realism.py` |
| `diag_continuous.csv` | water balance of the RFL-HBV: store drift, sinks, ET/PET | `diag_continuous.py` |
| `ig.npz`, `ig_summary.csv` | integrated-gradient attribution | `xai.py` |
| `parameters_test.csv` | HBV parameters per catchment (RPL-HBV, LC-HBV) | `train.py`, `calibrate_hbv.py` |
| `sim_test.npz` (ensembles) | five-seed ensemble flow over the test period | `ensemble.py` |
| `continuous_vs_windowed.csv` (`lstm_s0`) | LSTM scored windowed and continuously | `eval_continuous.py` |

`overview/` holds the summary tables behind the paper (medians, per-catchment seed medians, store realism, rule
similarity and rates, attribution, size sweep), and `rules/` the probed flux rules of the RPL-HBV, RFL-HBV and
LC-HBV (`probe_rules.py`).

Catchments are CAMELS-GB gauge ids. `set` is `trained` or `held-out` (never seen in training).
