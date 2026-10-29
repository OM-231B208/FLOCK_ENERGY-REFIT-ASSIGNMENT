# DATA: raw-data coverage, quality findings and cleaning

## 1. Basic inspection and cleaning of raw data

### Source and coverage

- Dataset: REFIT Electrical Load Measurements, University of Strathclyde.
- Raw household: House 1.
- Files: `RAW_House1_Part1.csv` and `RAW_House1_Part2.csv`.
- Observed timestamp coverage after parsing: 2013-10-09 14:06:17 to 2015-07-10 11:56:32.
- The official cleaned-data readme documents 20 households (House 1 through House 21, excluding House 14), aggregate plus nine appliance monitors, active power in watts, and a nominal 8-second sampling interval. Sensors were polled every 6–8 seconds and did not update synchronously.
- Source values are treated as power in watts. Section 1 parses raw `Time` timestamps as provided; Sections 2 and 3 use the official cleaned timestamps, which the readme says were corrected for UK daylight saving time.
- Raw data are sampled irregularly. In House 1, the median interval after sorting and deduplication is 7 seconds; 49.66% of adjacent intervals differ from that median by more than 25%. This observed distribution reflects the irregular polling and timestamp granularity.

### Inspection findings

| Check | Finding |
|---|---:|
| Raw rows across the two files | 8,518,763 |
| Invalid parsed timestamps | 0 |
| Backwards timestamp transitions in concatenated source order | 3 |
| Duplicate timestamps removed | 761,551 |
| Negative values across Aggregate, Appliance1 and Appliance2 | 0 |
| Readings above 10,000 W across those channels | 19,355 |
| Aggregate raw missing readings | 188,973 |
| Appliance1 raw missing readings | 437,890 |
| Appliance2 raw missing readings | 459,468 |

Part 1 and Part 2 overlap in time. The official readme explains that Part 1 represented unavailable appliance readings as zero, while Part 2 used NaN for unavailable sensors. Files are read in filename order and the last row at a repeated timestamp is retained, so Part 2 takes precedence in their overlap; within a file, the last repeated row is retained. The timestamp index is then sorted chronologically.

Values above 10 kW are flagged as extreme for review, not assumed to be errors: readings are neither clipped nor discarded. The maximums observed in the inspected channels were 29,159 W (Aggregate), 76,677 W (Appliance1), and 98,301 W (Appliance2). These values warrant review and may materially affect plots and downstream use.

### Cleaning decisions

1. Parse `Time` with invalid values reported and excluded from resampling.
2. Convert the selected power channels to numeric values.
3. Sort the two source files by name, keep the last row for each duplicate timestamp so Part 2 takes precedence, then sort timestamps chronologically.
4. Resample `Aggregate`, `Appliance1`, and `Appliance2` to one-minute means.
5. Interpolate only inside gaps of at most two consecutive missing minutes. Leading/trailing gaps and longer gaps remain missing; they are not silently filled.
6. Preserve and report extreme readings rather than clipping them.

After resampling and the short-gap interpolation, Aggregate has 113,647 missing minutes, Appliance1 has 114,957, and Appliance2 has 115,684. The longest remaining missing run is 59,904 minutes. These long outages are not imputed. Missing-run counts are computed separately for each channel.

### Outputs

- Reproducible script: [`src/section1_raw_cleaning.py`](./src/section1_raw_cleaning.py)
- One-minute House 1 CSV: `data/processed/house1_cleaned_1min.csv` (generated locally; not committed)
- Machine-readable inspection summary: `data/processed/house1_quality_summary.csv` (generated locally; not committed)
- Before/after plots: `figures/section1_house1_before_after.png`

### Official cleaned data notes for Sections 2 and 3

The official readme states that cleaned timestamps were corrected for UK daylight saving time, IAM values were adjusted when monitors were moved or reset, missing values were forward-filled, and IAM spikes above 4,000 W were replaced with zero. Its `Issues` flag is set when the sum of appliance channels exceeds aggregate; this analysis excludes any one-minute bin containing an `Issues = 1` sample. These official changes are useful but mean the cleaned channels are not untouched physical ground truth.

The readme's appliance list maps the washing-machine channels used here as House 1 `Appliance5`, House 2 `Appliance2`, House 3 `Appliance6`, and House 5 `Appliance3`. The broader mapping used in code is in [`src/washing_machine_mapping.py`](./src/washing_machine_mapping.py). The EDA and model use Houses 1, 2, 3, and 5, all with one documented washing-machine channel.

### Remaining concerns

The raw files have overlapping coverage, irregular sampling, a substantial number of duplicate timestamps, long outages, and extreme values. The simple cleaning above does not reproduce the official REFIT cleaning pipeline. Appliances are identified only by generic channel names here; the official readme is needed to attach appliance labels to `ApplianceN`.
