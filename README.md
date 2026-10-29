# Flock Energy REFIT Assignment

This repository contains a reproducible analysis for the Flock Energy take-home assignment. It uses raw REFIT readings to demonstrate basic data inspection and cleaning, then uses the official cleaned REFIT data to explore washing-machine use and evaluate appliance disaggregation.

The work is organized around the four sections in the assignment. The scripts generate the analysis outputs; the reports explain the data choices, results, limitations, and practical implications.

## What this project does

1. **Basic inspection and cleaning of raw data:** examines House 1 raw readings, checks timestamp and value quality, resamples to one-minute intervals, interpolates only short internal gaps, flags longer outages, and plots selected channels before and after cleaning.
2. **Exploratory analysis of washing-machine use:** analyzes official cleaned one-minute washing-machine channels for Houses 1, 2, 3, and 5, including availability, detected operating episodes, usage timing, and cross-house comparisons.
3. **Build and evaluate an appliance-disaggregation model:** compares Random Forest, two-stage LightGBM, Seq2Point CNN, and a causal Temporal Convolutional Network (TCN) with an always-off baseline. House 5 is held out for final evaluation; House 3 is used for threshold validation.
4. **Practical implications:** discusses a realistic customer-insight opportunity, how to measure it, limitations and UK-to-India transferability, and what would change with 15- or 30-minute meter readings.

The model results should be interpreted cautiously: on the reported House 5 evaluation, none of the learned models outperformed the always-off baseline on MAE or RMSE. Seq2Point is the tentative learned-model candidate if average absolute power error is the priority; Random Forest is stronger on learned-model RMSE and active-minute recall, but produces many false alarms. Neither is ready for deployment. The complete comparison and its limitations are documented in [`RESULTS.md`](./RESULTS.md).

## Data required

Download and extract both REFIT archives from the University of Strathclyde:

- [Raw REFIT data](https://pureportal.strath.ac.uk/files/62090177/REFIT_RAW_081116.7z) — used for the House 1 cleaning task.
- [Official cleaned REFIT data](https://pureportal.strath.ac.uk/files/62090184/CLEAN_REFIT_081116.7z) — used for washing-machine EDA and disaggregation.

By default, the scripts expect the extracted directories in your user Downloads folder:

- `REFIT_RAW_081116`
- `CLEAN_REFIT_081116`

Keep the large source archives and extracted datasets outside Git. If your data directories are elsewhere, set these environment variables before running the scripts. In Windows PowerShell:

```powershell
$env:REFIT_RAW_DIR = "C:\path\to\REFIT_RAW_081116"
$env:REFIT_CLEAN_DIR = "C:\path\to\CLEAN_REFIT_081116"
```

The reports identify the households, channels, time coverage, units, sampling interval, and data limitations relevant to each section.

## Setup and run

Use Python 3.11 or newer. From the repository root, install the listed dependencies:

```powershell
python -m pip install -r requirements.txt
```

Run all analysis sections in order:

```powershell
python src\run_analysis.py
```

Or run a single section:

```powershell
python src\section1_raw_cleaning.py
python src\section2_washing_machine_eda.py
python src\section3_disaggregation.py
```

The scripts write generated figures to `figures/` and processed CSVs and model outputs to `data/processed/`. The processed CSVs and downloaded source data are excluded from Git; rerun the scripts to recreate them. Section 3 training and evaluation may take longer than the cleaning and EDA steps.

## Project layout

| Location | Purpose |
|---|---|
| [`src/`](./src/) | Data configuration, shared REFIT loading and analysis helpers, washing-machine channel mapping, and scripts for Sections 1–3. |
| [`figures/`](./figures/) | Generated plots for raw-data cleaning, washing-machine EDA, and model evaluation. |
| `data/processed/` | Locally generated cleaned data, summaries, predictions, and other CSV outputs; excluded from Git. |
| [`DATA.md`](./DATA.md) | Section 1 raw-data coverage, quality findings, cleaning choices, and remaining concerns. |
| [`RESULTS.md`](./RESULTS.md) | Section 2 EDA and Section 3 model evaluation, comparisons, validation, and failure analysis. |
| [`Recommendation.md`](./Recommendation.md) | Utility-facing recommendation and answers to the practical questions in Section 4. |

The scripts are the authoritative, reproducible analysis path. The `notebooks/` directory is not required to run the project.

## Reproducibility and key decisions

- **Section 1:** House 1's raw files are sorted by filename and combined. For duplicate timestamps, the last row is retained, allowing Part 2 to take precedence in overlapping timestamps. Readings are sorted chronologically and resampled to one-minute means.
- **Missing data:** only internal gaps of up to two consecutive minutes are interpolated in Section 1. Leading, trailing, and longer gaps remain missing and are reported rather than silently filled.
- **Extreme values:** readings above 10 kW in the inspected raw channels are flagged for review, not automatically removed or clipped.
- **Section 2:** the washing-machine channels are mapped from the official cleaned-data readme. The operating-episode rule and treatment of missing and issue-flagged minutes are described in [`RESULTS.md`](./RESULTS.md).
- **EDA outputs:** Section 2 produces seven figures. The daily-energy chart limits each panel to its 99th percentile for readability and marks higher values; the generated CSV retains the uncapped values.
- **Section 3:** all models are compared on the same held-out House 5 data against an always-off baseline. House 3 is used to select activity thresholds, and is part of the final training data after threshold selection. Since only one household is held out, these scores are a case study and not a reliable estimate of performance across all homes.
- **Trade-off:** the assignment favors a clear and defensible comparison over complexity. The more complex sequence models are included for comparison, but their results do not justify deployment based on this evaluation.

## Deliverables

The table maps each item requested in the take-home assignment to the corresponding repository content.

| File / deliverable | What the assignment asks for | How this repository addresses it |
|---|---|---|
| **Source code** | Notebook(s) or scripts that run from the supplied data through cleaning, EDA, model predictions, and evaluation. | Scripts in [`src/`](./src/) implement the workflow. Run [`src/run_analysis.py`](./src/run_analysis.py) for Sections 1–3 or run each section script separately. Outputs are generated under `figures/` and `data/processed/`. |
| **README.md** | What was built; setup and run instructions; assumptions, preprocessing decisions, validation design, trade-offs, AI tools used, and possible improvements. | This README describes the project, data setup, execution, reproducibility choices, validation limitations, AI assistance, and suggested next steps. |
| **DATA.md** | Raw-data coverage and quality findings, cleaning steps, gap/outlier decisions, and remaining concerns. | [`DATA.md`](./DATA.md) reports the House 1 inspection results, one-minute cleaning pipeline, missing-data and outlier handling, and data limitations. |
| **RESULTS.md** | Washing-machine EDA, plots, model results, validation approach, failure analysis, and interpretation. | [`RESULTS.md`](./RESULTS.md) documents selected homes and periods, availability and usage findings, visualizations, the baseline and model comparison, and the held-out evaluation and failure cases. |
| **Recommendation.md** | One-page utility-facing summary of suitable uses and model limitations, including answers to Section 4. | [`Recommendation.md`](./Recommendation.md) explains a potential customer insight, impact measurement, deployment and transfer limits, and the implications of lower-frequency meter data. |

## How to use the deliverables

1. **Set up and run the project:** follow this README to install dependencies, configure the REFIT data directories, and run all analysis or an individual section.
2. **Review the raw-data work:** read [`DATA.md`](./DATA.md) for House 1 data quality, cleaning decisions, and unresolved gaps or unusual readings.
3. **Review EDA and model performance:** read [`RESULTS.md`](./RESULTS.md) for washing-machine usage patterns, plots, baseline/model comparisons, validation design, and failure analysis.
4. **Review practical recommendations:** read [`Recommendation.md`](./Recommendation.md) for the utility-facing opportunity, evaluation plan, and Section 4 limitations.
5. **Reproduce results after changes:** rerun the relevant script whenever the code or input data changes. The reports describe the current analysis; they are not a substitute for regenerating outputs from changed inputs.

## AI tools

Copilot SDK in VS Code was used as a coding and writing assistant. **My role as the project author (OM-231B208)** was to define the assignment scope, choose the analysis priorities, guide model selection, review the generated results and visualizations, and decide what to include in the final comparison and reports. I directed the work toward a simple, reproducible solution, requested comparisons among Random Forest, LightGBM, Seq2Point, and TCN, and asked for additional attention to model training and EDA readability.

The AI tool supported implementation, debugging, code organization, and documentation; it did not make the project decisions or independently certify the findings. I am responsible for the submitted code, assumptions, interpretation, and final deliverables. Dataset facts and appliance-channel mappings are taken from the REFIT sources cited below. EDA statistics and model scores are outputs of this project's scripts, not results published by the REFIT authors. They should be checked by rerunning the code with the stated data and configuration.

## Possible improvements

- Evaluate on multiple held-out households or with household-grouped cross-validation, using validation homes that are kept separate from final test homes.
- Improve active-event detection and energy estimation while continuing to compare against the strong always-off baseline; do not judge usefulness by MAE alone.
- Investigate unusually high or inconsistent appliance readings against the official documentation and, where possible, the raw source data.
- Test the approach with locally collected Indian smart-meter data and appliance labels before making transfer or deployment claims.
- Re-run and update the reports and figures after any changes to source data, preprocessing, or model design.

## References

### Dataset and documentation

- REFIT Consortium. [REFIT Electrical Load Measurements dataset page and documentation](https://pureportal.strath.ac.uk/en/datasets/refit-electrical-load-measurements/). University of Strathclyde.
- REFIT Consortium. [Official cleaned REFIT readme](https://pure.strath.ac.uk/ws/portalfiles/portal/62090183/CLEAN_READ_ME_081116.txt). Used for the cleaned-data notes and appliance-channel mapping.

### Methods

- Zhong, M., Goddard, N., & Sutton, C. (2018). [Sequence-to-point learning with neural networks for nonintrusive load monitoring](https://arxiv.org/abs/1612.09106). *Proceedings of the AAAI Conference on Artificial Intelligence (AAAI-18).* The paper motivates the Seq2Point model family evaluated in Section 3; this project implements and evaluates its own model on the stated REFIT split.
- Ke, G., Meng, Q., Finley, T., Wang, T., Chen, W., Ma, W., Ye, Q., & Liu, T.-Y. (2017). [LightGBM: A highly efficient gradient boosting decision tree](https://proceedings.neurips.cc/paper/2017/hash/6449f44a102fde848669bdd9eb6b76fa-Abstract.html). *Advances in Neural Information Processing Systems, 30.* Cited for the gradient-boosted tree method used in the Section 3 comparison; its reported performance is measured here rather than assumed from the paper.
- Bai, S., Kolter, J. Z., & Koltun, V. (2018). [An empirical evaluation of generic convolutional and recurrent networks for sequence modeling](https://arxiv.org/abs/1803.01271). The causal, dilated convolutional sequence-modeling approach informed the TCN comparison; the REFIT task-specific architecture and results are implemented and evaluated in this repository.
