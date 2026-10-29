"""Shared loading and event-detection helpers for Sections 2 and 3."""

from pathlib import Path

import numpy as np
import pandas as pd

from config import clean_file
from washing_machine_mapping import WASHING_MACHINE_CHANNELS

WASHING_MACHINE_ACTIVE_THRESHOLD_W = 10.0
MIN_ACTIVE_MINUTES_PER_CYCLE = 5
MIN_CYCLE_DURATION_MINUTES = 15
MAX_INACTIVE_GAP_TO_MERGE_MINUTES = 2


def load_house_one_minute(house: int) -> pd.DataFrame:
    if house not in WASHING_MACHINE_CHANNELS:
        raise ValueError(f"No documented washing-machine channel for House {house}.")

    path: Path = clean_file(house)
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing cleaned REFIT file: {path}. "
            "Set REFIT_CLEAN_DIR to the extracted cleaned-data directory."
        )

    channels = WASHING_MACHINE_CHANNELS[house]
    required_columns = ["Time", "Aggregate", *channels]
    available_columns = pd.read_csv(path, nrows=0).columns.tolist()
    has_issues = "Issues" in available_columns
    use_columns = [*required_columns, *(["Issues"] if has_issues else [])]
    frame = pd.read_csv(path, usecols=use_columns)
    frame["Time"] = pd.to_datetime(frame["Time"], errors="coerce")
    frame = frame.dropna(subset=["Time"]).drop_duplicates(
        subset="Time", keep="first"
    )
    frame = frame.sort_values("Time", kind="stable").set_index("Time")

    issue_minutes = None
    if has_issues:
        frame["Issues"] = pd.to_numeric(frame["Issues"], errors="coerce")
        issue_minutes = (
            frame["Issues"].fillna(0).resample("1min").max().gt(0)
        )

    for column in ["Aggregate", *channels]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    minute = frame[["Aggregate", *channels]].resample("1min").mean()
    if len(channels) == 1:
        minute["WashingMachine"] = minute[channels[0]]
    else:
        minute["WashingMachine"] = minute[list(channels)].sum(
            axis=1, min_count=len(channels)
        )

    output = minute[["Aggregate", "WashingMachine"]]
    if issue_minutes is not None:
        aligned_issues = issue_minutes.reindex(output.index, fill_value=False)
        output.loc[aligned_issues, ["Aggregate", "WashingMachine"]] = np.nan
        output.attrs["issue_flagged_minutes"] = int(aligned_issues.sum())
    else:
        output.attrs["issue_flagged_minutes"] = 0
    return output


def bridge_short_observed_gaps(
    power: pd.Series,
    max_gap_minutes: int = MAX_INACTIVE_GAP_TO_MERGE_MINUTES,
) -> tuple[np.ndarray, np.ndarray]:
    values = power.to_numpy(dtype=float)
    valid = np.isfinite(values)
    active = valid & (values >= WASHING_MACHINE_ACTIVE_THRESHOLD_W)
    bridged = active.copy()

    cursor = 0
    while cursor < len(values):
        if not valid[cursor]:
            cursor += 1
            continue
        if active[cursor]:
            cursor += 1
            continue
        gap_start = cursor
        while cursor < len(values) and valid[cursor] and not active[cursor]:
            cursor += 1
        gap_length = cursor - gap_start
        if (
            gap_length > 0
            and gap_length <= max_gap_minutes
            and gap_start > 0
            and cursor < len(values)
            and active[gap_start - 1]
            and active[cursor]
        ):
            bridged[gap_start:cursor] = True

    return active, bridged


def detect_cycles(power: pd.Series) -> pd.DataFrame:
    """Find sustained active runs, bridging only short, observed off-gaps."""
    original_active, bridged_active = bridge_short_observed_gaps(power)
    transitions = np.diff(
        np.r_[False, bridged_active, False].astype(np.int8)
    )
    starts = np.flatnonzero(transitions == 1)
    stops = np.flatnonzero(transitions == -1)
    cycles = []

    for start, stop in zip(starts, stops):
        duration = int(stop - start)
        active_minutes = int(original_active[start:stop].sum())
        if (
            duration < MIN_CYCLE_DURATION_MINUTES
            or active_minutes < MIN_ACTIVE_MINUTES_PER_CYCLE
        ):
            continue

        cycle_values = power.iloc[start:stop]
        cycles.append(
            {
                "start": power.index[start],
                "end": power.index[stop - 1],
                "duration_minutes": duration,
                "active_minutes": active_minutes,
                "peak_power_w": float(cycle_values.max()),
                "energy_kwh": float(cycle_values.sum() / 60.0 / 1000.0),
            }
        )

    return pd.DataFrame(
        cycles,
        columns=[
            "start",
            "end",
            "duration_minutes",
            "active_minutes",
            "peak_power_w",
            "energy_kwh",
        ],
    )
