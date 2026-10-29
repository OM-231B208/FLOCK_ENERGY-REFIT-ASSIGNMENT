# Flock Energy REFIT Assignment

Reproducible REFIT data cleaning and washing-machine disaggregation analysis for the Flock Energy take-home assignment.

## Assignment sections

1. **Basic inspection and cleaning of raw data:** House 1 raw-data checks, one-minute resampling, short-gap interpolation, outage flags, and before/after plots.
2. **Exploratory analysis of washing-machine use:** official channel mapping, availability and event summaries, time-of-use plots, and household comparison.
3. **Build and evaluate an appliance-disaggregation model:** Random Forest, two-stage LightGBM, Seq2Point CNN, and causal sequence-to-sequence Temporal Convolutional Network (TCN) compared with an always-off baseline on a held-out household.
4. **Practical implications:** a utility-facing opportunity, evaluation approach, limitations, and lower-frequency data discussion.

## Data setup

Download and extract the [raw REFIT data](https://pureportal.strath.ac.uk/files/62090177/REFIT_RAW_081116.7z) and [official cleaned REFIT data](https://pureportal.strath.ac.uk/files/62090184/CLEAN_REFIT_081116.7z). Keep these large source files outside this repository.

By default, the scripts look for these extracted folders under your user Downloads directory:

- `REFIT_RAW_081116`
- `CLEAN_REFIT_081116`

To use other locations, set `REFIT_RAW_DIR` and `REFIT_CLEAN_DIR` to the corresponding absolute folder paths before running scripts. On Windows PowerShell:

```powershell
$env:REFIT_RAW_DIR = "C:\path\to\REFIT_RAW_081116"
$env:REFIT_CLEAN_DIR = "C:\path\to\CLEAN_REFIT_081116"
```

## Setup and run

Use Python 3.11 or newer and install the dependencies:

```powershell
python -m pip install -r requirements.txt
python src\section1_raw_cleaning.py
```

Run all analysis sections with:

```powershell
python src\run_analysis.py
```

Individual scripts are also available as `src\section1_raw_cleaning.py`, `src\section2_washing_machine_eda.py`, and `src\section3_disaggregation.py`. Generated CSVs are written under `data/processed/` and figures under `figures/`. Generated processed CSVs and downloaded source archives are excluded from Git.

## Project layout

- `src/`: reusable loaders and the Section 1–3 analysis scripts.
- `figures/`: generated, report-ready plots; rerun the analysis scripts to refresh them.
- `data/processed/`: generated CSV summaries and predictions (local outputs, not committed).
- `notebooks/`: reserved for notebooks. A pre-existing exploratory notebook is retained locally under `notebooks/archive/`; it has stale outputs and is not the reproducible analysis path.
- `DATA.md`, `RESULTS.md`, and `Recommendation.md`: assignment reports.

## Reproducibility and design

- Section 1 uses the raw House 1 files, keeps the last row for duplicate timestamps in sorted filename/source order (so Part 2 supersedes Part 1 in the overlap), sorts chronologically, and resamples to one-minute means.
- Only internal missing gaps up to two minutes are time-interpolated. Longer and edge gaps remain missing and are reported.
- Readings over 10 kW are flagged, not automatically removed or clipped.
- Section 2 generates seven plots: the representative day, detected episodes, cross-house comparison, usage heatmap, monthly coverage, daily energy, and normalized episode profiles. The daily-energy plot caps each panel at its 99th percentile for readability; high days are marked, and the CSV retains uncapped values.
- Section 3 compares Random Forest, two-stage LightGBM, a Seq2Point CNN, and a causal sequence-to-sequence TCN with an always-off baseline. Activity thresholds are selected on a training-side held-out household; final evaluation holds out House 5. Model predictions are not considered suitable for customer-facing estimates unless they improve on the baseline and are validated across more homes.
- Washing-machine channels are documented in `src\washing_machine_mapping.py` based on the official cleaned-data readme.

## Deliverables

- [DATA.md](./DATA.md): Section 1 coverage, quality findings, cleaning choices, and remaining concerns.
- [RESULTS.md](./RESULTS.md): washing-machine EDA and model evaluation.
- [Recommendation.md](./Recommendation.md): utility-facing use, limitations, and Section 4 answers.

## AI tools

Copilot SDK in VS Code assisted with code organization, implementation, and documentation. Dataset-specific mappings and findings must be validated against the official REFIT documentation and generated outputs; no results should be treated as official dataset facts unless verified.

## References

- [REFIT dataset page and documentation](https://pureportal.strath.ac.uk/en/datasets/refit-electrical-load-measurements/)
- [Official cleaned REFIT readme](https://pure.strath.ac.uk/ws/portalfiles/portal/62090183/CLEAN_READ_ME_081116.txt)
