# RESULTS: washing-machine EDA and disaggregation model

## 2. Exploratory analysis of washing-machine use

### Homes, periods and target mapping

The official cleaned REFIT readme identifies the washing-machine channels. This analysis uses four homes with one documented washing-machine monitor each:

| House | Washing-machine channel | One-minute coverage |
|---:|---|---|
| 1 | Appliance5 | 2013-10-09 13:06 to 2015-07-10 11:56 |
| 2 | Appliance2 | 2013-09-17 22:08 to 2015-05-28 08:05 |
| 3 | Appliance6 | 2013-09-25 19:21 to 2015-06-02 10:55 |
| 5 | Appliance3 | 2013-09-26 09:56 to 2015-07-06 17:48 |

These are official cleaned files resampled to one-minute means. Any minute containing an official `Issues = 1` sample (appliance sum exceeds aggregate) is treated as unavailable for both aggregate and target. Missing values are not interpreted as zero.

### Operating-event rule

An operating episode is a continuous run at or above **10 W**, lasting at least **15 minutes**, with at least **5 observed active minutes**. Observed inactive gaps of up to **2 minutes** are joined; missing readings are never bridged. This transparent rule is a heuristic for appliance activity, not a confirmed wash-program boundary. A long wash with extended low-power pauses may be split into multiple episodes.

### Availability and usage

| House | Missing target minutes | Missing (%) | Official issue minutes excluded | Off time (<10 W, observed) | Detected episodes | Episodes / 30 valid days | Median duration (min) | Median peak (W) | Median energy (kWh) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 133,004 | 14.46% | 16,686 | 98.57% | 137 | 7.15 | 20 | 453.6 | 0.042 |
| 2 | 232,372 | 26.14% | 13,458 | 95.70% | 502 | 31.89 | 23 | 335.3 | 0.075 |
| 3 | 161,326 | 18.23% | 60,633 | 95.43% | 611 | 34.34 | 24 | 421.8 | 0.069 |
| 5 | 158,108 | 16.94% | 70,888 | 93.45% | 838 | 38.68 | 29 | 266.8 | 0.072 |

Off time is the share of nonmissing one-minute target readings below 10 W. Episodes per 30 valid days uses days with at least 90% observed target minutes, reducing bias from different coverage lengths. House 2 has the highest missing share. The difference in episode rates and cycle characteristics across homes illustrates household and appliance heterogeneity; the short detected durations and event rule mean these are not necessarily full wash cycles.

Across the four homes, normalized starts are most common in daytime hours, particularly around 08:00–12:00. Saturday has the highest average detected-episode start rate per valid day, followed by Sunday and Friday. This is descriptive only; small household counts, missing data, and the heuristic event definition limit generalization.

### Figures

- [Representative 24-hour aggregate and washing-machine trace](./figures/section2_representative_day.png) — House 1, on a day containing a detected episode. The two axes show whole-home and appliance power at one-minute resolution.
- [Detected operating episodes](./figures/section2_example_cycles.png) — first detected episode in each selected home, with surrounding aggregate and appliance load.
- [Cross-house comparison](./figures/section2_household_comparison.png) — operating-episode rate per 30 valid days and per-episode energy distributions. Normalization avoids comparing raw counts from unequal coverage.
- [Usage by hour and weekday](./figures/section2_usage_by_hour_weekday.png) — mean episode starts per valid household-day, so homes with longer records do not dominate the heatmap.

The detailed machine-readable outputs are generated under `data/processed/`: household availability summary, detected episodes, and normalized time-of-use profile.

## 3. Build and evaluate an appliance-disaggregation model

### Models and validation design

All models predict one-minute washing-machine power for House 5 and are scored on the same **525,298 valid evaluation minutes**.

- **Training households:** Houses 1, 2, and 3.
- **Threshold-validation household:** House 3. Thresholds for LightGBM and Seq2Point are selected using models fit on Houses 1 and 2; final models are then refit on Houses 1–3.
- **Held-out evaluation household:** House 5. It is not used for training or threshold selection.
- **Always-off baseline:** predict 0 W at each minute.
- **Random Forest:** tabular aggregate-now, lag, rolling-mean and time-of-week features.
- **Two-stage LightGBM:** active/inactive classifier followed by an active-only power regressor.
- **Seq2Point CNN:** one-dimensional convolutional network that maps a 31-minute trailing aggregate window to the washing-machine power at the window’s last minute. This causal form uses no future readings.

To counter the very small active class, training examples are sampled by household with all observed active windows (up to the cap) and up to ten off windows per active window, capped at 50,000 samples per home. Thresholds from 0.10 to 0.99 are compared on House 3; the one maximizing active-minute F1 is selected. All model comparisons below use the same held-out minutes, but they are not claims of broad generalization: only one home is held out.

### Metrics and results

MAE/RMSE measure power error in watts. Active precision, recall and F1 use the 10 W activity threshold. Energy error compares summed predicted with actual energy; the actual evaluation energy is **84.31 kWh**.

| Model | MAE (W) | RMSE (W) | R² | Active precision | Active recall | Active F1 | Predicted energy (kWh) | Energy error |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Always off | **9.63** | **91.63** | -0.011 | 0.000 | 0.000 | 0.000 | 0.00 | -100.0% |
| Random Forest | 18.86 | 96.27 | -0.116 | 0.087 | **0.420** | **0.144** | 115.05 | +36.5% |
| Two-stage LightGBM | 12.84 | 102.54 | -0.266 | 0.209 | 0.049 | 0.080 | 41.13 | -51.2% |
| Seq2Point CNN | 14.67 | 118.67 | -0.696 | **0.221** | 0.083 | 0.121 | 56.72 | -32.7% |

### Interpretation

Seq2Point is the temporal model most aligned with the appliance signature, and in this test it has the best energy estimate among the learned models and higher active F1 than LightGBM. It does **not** beat the Random Forest on active recall/F1, nor the always-off baseline on MAE/RMSE. The always-off result is strong because nearly all observed minutes are inactive; a low average error alone is therefore not evidence of useful appliance disaggregation. None of the tested learned models is reliable enough for customer-facing estimates.

The evaluation uses one held-out home, so its score is a case study rather than a stable estimate of performance across REFIT households. The activity threshold is tuned on one training-side validation home; more households are needed for robust threshold/model selection.

### Evaluation plot and failure cases

- [House 5 actual vs predicted washing-machine power](./figures/section3_actual_vs_predicted.png) — matched 24-hour evaluation window for Random Forest, LightGBM, Seq2Point, and actual appliance power.

Failure cases include simultaneous household loads that obscure the washing-machine signature, differences in appliance power patterns between homes, and short/low-power phases that are difficult to classify. Missing intervals and official cleaned-data processing also affect the target labels. The active class is small even after stratified sampling, and fitting across three homes does not capture all household behavior.

Further work should evaluate several held-out households, compare thresholds using multiple training-side validation homes, and test appliance-specific event features or a richer temporal network. Do not deploy the current model without that validation.
