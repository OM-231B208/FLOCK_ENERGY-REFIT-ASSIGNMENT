"""Section 1: Basic inspection and cleaning of raw House 1 data."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from config import FIG_DIR, PROC_DIR, RAW_DIR

POWER_COLUMNS = ["Aggregate", "Appliance1", "Appliance2"]
MAX_SHORT_GAP_MINUTES = 2
EXTREME_POWER_WATTS = 10_000


def get_raw_files() -> list[Path]:
    files = sorted(RAW_DIR.glob("RAW_House1_Part*.csv"))
    if not files:
        raise FileNotFoundError(
            f"No House 1 raw CSVs found in {RAW_DIR}. "
            "Set REFIT_RAW_DIR to the extracted raw-data directory."
        )
    return files


def load_raw_house1() -> tuple[pd.DataFrame, dict[str, int | float | str]]:
    files = get_raw_files()
    frames = []
    file_order_checks = []
    for path in files:
        frame = pd.read_csv(path, usecols=["Time", *POWER_COLUMNS])
        parsed_times = pd.to_datetime(frame["Time"], errors="coerce").dropna()
        file_order_checks.append(
            f"{path.name}: "
            f"{parsed_times.min()} to {parsed_times.max()}, "
            f"chronological={parsed_times.is_monotonic_increasing}"
        )
        frames.append(frame)

    raw = pd.concat(frames, ignore_index=True)
    row_count = len(raw)
    raw["Time"] = pd.to_datetime(raw["Time"], errors="coerce")
    invalid_timestamps = int(raw["Time"].isna().sum())
    raw = raw.dropna(subset=["Time"]).copy()
    out_of_order_transitions = int((raw["Time"].diff().dt.total_seconds() < 0).sum())
    raw_missing = {
        f"{column}_raw_missing_readings": int(raw[column].isna().sum())
        for column in POWER_COLUMNS
    }
    for column in POWER_COLUMNS:
        raw[column] = pd.to_numeric(raw[column], errors="coerce")

    duplicate_count = int(raw["Time"].duplicated(keep="last").sum())
    raw = raw.drop_duplicates(subset="Time", keep="last")
    raw = raw.sort_values("Time", kind="stable").set_index("Time")

    interval_seconds = raw.index.to_series().diff().dt.total_seconds().dropna()
    typical_interval = float(interval_seconds.median()) if not interval_seconds.empty else 0.0
    irregular_intervals = (
        int((interval_seconds.sub(typical_interval).abs() > typical_interval * 0.25).sum())
        if typical_interval > 0
        else 0
    )
    negative_readings = int((raw[POWER_COLUMNS] < 0).sum().sum())
    extreme_readings = int((raw[POWER_COLUMNS] > EXTREME_POWER_WATTS).sum().sum())

    summary: dict[str, int | float | str] = {
        "source_files": ", ".join(path.name for path in files),
        "source_file_coverage_and_order": " | ".join(file_order_checks),
        "raw_rows": row_count,
        "invalid_timestamps": invalid_timestamps,
        "out_of_order_transitions_before_sort": out_of_order_transitions,
        "duplicate_timestamps_removed": duplicate_count,
        "duplicate_rule": (
            "Keep last encountered row (files sorted by name; stable source order), "
            "so Part2 takes precedence over Part1 in their overlap."
        ),
        "timestamp_start": str(raw.index.min()),
        "timestamp_end": str(raw.index.max()),
        "median_interval_seconds": typical_interval,
        "irregular_intervals_over_25pct_from_median": irregular_intervals,
        "irregular_interval_percent": (
            100 * irregular_intervals / len(interval_seconds)
            if len(interval_seconds)
            else 0.0
        ),
        "negative_power_readings": negative_readings,
        f"readings_over_{EXTREME_POWER_WATTS}_W": extreme_readings,
        "extremes_action": "Flagged for review; no readings clipped or removed.",
    }
    summary.update(raw_missing)
    summary.update(
        {
            f"{column}_raw_max_watts": float(raw[column].max())
            for column in POWER_COLUMNS
        }
    )

    return raw, summary


def missing_run_lengths(series: pd.Series) -> list[int]:
    missing = series.isna()
    run_ids = missing.ne(missing.shift()).cumsum()
    return (
        series[missing]
        .groupby(run_ids[missing])
        .size()
        .astype(int)
        .tolist()
    )


def clean_to_one_minute(raw: pd.DataFrame) -> pd.DataFrame:
    one_minute = raw[POWER_COLUMNS].resample("1min").mean()
    return one_minute.interpolate(
        method="time",
        limit=MAX_SHORT_GAP_MINUTES,
        limit_area="inside",
    )


def plot_before_after(raw: pd.DataFrame, cleaned: pd.DataFrame) -> Path:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    start = cleaned.index.min()
    end = start + pd.Timedelta(hours=6)
    raw_window = raw.loc[start:end]
    clean_window = cleaned.loc[start:end]
    fig, axes = plt.subplots(3, 2, figsize=(15, 10), sharex="col")

    for row, column in enumerate(POWER_COLUMNS):
        axes[row, 0].plot(raw_window.index, raw_window[column], linewidth=0.6)
        axes[row, 0].set_title(f"Before cleaning — {column}")
        axes[row, 1].plot(clean_window.index, clean_window[column], linewidth=0.8)
        axes[row, 1].set_title(f"After cleaning — {column}")
        for axis in axes[row]:
            axis.set_ylabel("Power (W)")
            axis.grid(alpha=0.3)

    axes[-1, 0].set_xlabel("Timestamp")
    axes[-1, 1].set_xlabel("Timestamp")
    fig.tight_layout()
    output = FIG_DIR / "section1_house1_before_after.png"
    fig.savefig(output, dpi=150)
    plt.close(fig)
    return output


def run() -> None:
    raw, summary = load_raw_house1()
    cleaned = clean_to_one_minute(raw)
    PROC_DIR.mkdir(parents=True, exist_ok=True)
    output_csv = PROC_DIR / "house1_cleaned_1min.csv"
    cleaned.to_csv(output_csv, index_label="Time")

    for column in POWER_COLUMNS:
        runs = missing_run_lengths(cleaned[column])
        summary[f"{column}_missing_minutes_after_short_gap_fill"] = int(
            cleaned[column].isna().sum()
        )
        summary[f"{column}_long_outage_runs"] = sum(
            run_length > MAX_SHORT_GAP_MINUTES for run_length in runs
        )
        summary[f"{column}_longest_outage_minutes"] = max(runs, default=0)

    quality_csv = PROC_DIR / "house1_quality_summary.csv"
    pd.Series(summary, name="value").to_csv(quality_csv, header=True)
    figure = plot_before_after(raw, cleaned)

    print(pd.Series(summary).to_string())
    print(f"\nCleaned one-minute CSV: {output_csv}")
    print(f"Quality summary: {quality_csv}")
    print(f"Before/after figure: {figure}")


if __name__ == "__main__":
    run()
