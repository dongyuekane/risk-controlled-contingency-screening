# Risk-controlled N−1/N−2 contingency screening

Python research code for screening connected single- and double-branch contingencies in a coupled transmission and distribution benchmark. The fixed cascade combines an event score with voltage-interface feedback from a nonlinear radial feeder calculation and uses a single Neyman–Pearson calibration of the composite score. The full coupled AC solver supplies the reference labels.

The inputs are public benchmark networks and seeded synthetic operating scenarios. The risk target is the bypass probability conditional on an unsafe case under the specified synthetic calibration population. The finite confirmation result and its confidence interval are reported separately from the calibration target.

## Start here

Use Python 3.12 on macOS or Linux. Run these commands from the repository root.

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python fetch_cases.py
python smoke.py
```

The smoke check recomputes the AC label, constraint margin and interface-feedback score for one archived confirmation case.

## Reproduce the frozen comparison

This repository packages the November 2026 submission reproducibility addendum for the fixed interface-feedback cascade. The supplied router and score scales stay fixed for the 3,000 calibration cases and the disjoint 1,000 confirmation cases. Run the replay in a working copy because it replaces generated outputs in place.

```sh
python fresh_release.py
python cascade_timing.py
python summarize_release.py
```

`fresh_release.py` fits the prescribed models from the archived training table, recalibrates thresholds, and replays the frozen AC-labeled case tables. It also generates the local model file needed by the timing benchmark. For a fresh physical simulation in a working copy, move aside the archived `confirmation_cases.csv` and `confirmation_inputs.json` pairs in both `fresh_release/calibration/` and `fresh_release/confirmation/` before running the same commands. The resulting absolute runtimes depend on the environment.

| Research output | Location |
| --- | --- |
| Frozen score specifications and experimental settings | `fresh_release/frozen_router.json`, `fresh_release/protocol.json` |
| Final confirmation outcomes and calibration thresholds | `fresh_release/confirmation_results.csv`, `fresh_release/thresholds.csv` |
| Record of individual cascade decisions and verified misses | `fresh_release/cascade_predictions.csv`, `fresh_release/independent_false_safe_checks.csv` |
| Archived serial timing | `fresh_release/final_serial_timing.csv`, `fresh_release/final_timing_summary.csv` |

The archived selective feedback result bypassed 507 of 1,000 paired cases and made four false-safe decisions among 402 unsafe cases. The resulting empirical false-safe rate was 0.995% and its one-sided 95% upper confidence limit was 2.262%. These results describe the synthetic study population and prescribed operating policy.

## Benchmark sources

The download script verifies the exact versions of `case9.m`, `case33bw.m` and `case69.m` from [MATPOWER 8.1](https://github.com/MATPOWER/matpower/tree/8.1/data). The case30 arrays are read from [PYPOWER](https://github.com/rwl/PYPOWER/blob/master/pypower/case30.py) and match the archived study arrays. The source cases are retrieved separately under their upstream terms.
