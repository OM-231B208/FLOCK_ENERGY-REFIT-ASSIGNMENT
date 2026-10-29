"""Section 2: Exploratory analysis of washing-machine use."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from config import FIG_DIR, PROC_DIR
from refit_analysis import (
    MAX_INACTIVE_GAP_TO_MERGE_MINUTES,
    MIN_CYCLE_DURATION_MINUTES,
    WASHING_MACHINE_ACTIVE_THRESHOLD_W,
    detect_cycles,
    load_house_one_minute,
)
from washing_machine_mapping import SELECTED_HOUSES, WASHING_MACHINE_CHANNELS


def quality_summary(
    house: int, data: pd.DataFrame, cycles: pd.DataFrame
) -> dict[str, int | float | str]:
    target = data["WashingMachine"]
    observed = target.notna()
    daily_coverage = observed.groupby(data.index.normalize()).mean()
    valid_days = daily_coverage[daily_coverage >= 0.90].index
    valid_day_cycles = (
        cycles[cycles["start"].dt.normalize().isin(valid_days)]
        if not cycles.empty
        else cycles
    )
    off = target[observed] < WASHING_MACHINE_ACTIVE_THRESHOLD_W

    return {
        "house": house,
        "washing_machine_channel": ", ".join(WASHING_MACHINE_CHANNELS[house]),
        "start": str(data.index.min()),
        "end": str(data.index.max()),
        "minutes_in_coverage": len(data),
        "missing_aggregate_minutes": int(data["Aggregate"].isna().sum()),
        "missing_washing_machine_minutes": int(target.isna().sum()),
        "issue_flagged_minutes_discarded": int(
            data.attrs.get("issue_flagged_minutes", 0)
        ),
        "washing_machine_missing_percent": (
            100 * target.isna().mean() if len(target) else 0.0
        ),
        "observed_washing_machine_minutes": int(observed.sum()),
        "days_with_at_least_90pct_target_coverage": len(valid_days),
        "off_minutes_below_10_w": int(off.sum()),
        "off_percent_of_observed_minutes": (
            100 * off.mean() if len(off) else 0.0
        ),
        "detected_cycles": len(cycles),
        "cycles_on_valid_coverage_days": len(valid_day_cycles),
        "cycles_per_30_valid_coverage_days": (
            30 * len(valid_day_cycles) / len(valid_days) if len(valid_days) else 0.0
        ),
        "median_cycle_duration_minutes": (
            float(cycles["duration_minutes"].median()) if not cycles.empty else 0.0
        ),
        "median_cycle_peak_power_w": (
            float(cycles["peak_power_w"].median()) if not cycles.empty else 0.0
        ),
        "median_cycle_energy_kwh": (
            float(cycles["energy_kwh"].median()) if not cycles.empty else 0.0
        ),
    }


def choose_representative_day(data: pd.DataFrame, cycles: pd.DataFrame) -> pd.Timestamp:
    if not cycles.empty:
        return cycles.iloc[0]["start"].normalize()

    coverage = data["WashingMachine"].notna().groupby(data.index.date).mean()
    usable_days = coverage[coverage >= 0.95]
    if usable_days.empty:
        return data.index.min().normalize()
    return pd.Timestamp(usable_days.index[0])


def plot_representative_day(house_data: dict[int, pd.DataFrame]) -> Path:
    house = SELECTED_HOUSES[0]
    data = house_data[house]
    cycles = detect_cycles(data["WashingMachine"])
    day_start = choose_representative_day(data, cycles)
    day = data.loc[day_start : day_start + pd.Timedelta(days=1)]

    fig, axis = plt.subplots(figsize=(13, 5))
    axis.plot(day.index, day["Aggregate"], color="tab:blue", label="Aggregate")
    axis.set_ylabel("Aggregate power (W)", color="tab:blue")
    axis.set_xlabel("Local timestamp")
    axis.grid(alpha=0.25)
    secondary = axis.twinx()
    secondary.plot(
        day.index,
        day["WashingMachine"],
        color="tab:orange",
        label="Washing machine",
    )
    secondary.set_ylabel("Washing-machine power (W)", color="tab:orange")
    axis.set_title(f"House {house}: representative 24-hour trace from {day_start.date()}")
    lines, labels = axis.get_legend_handles_labels()
    second_lines, second_labels = secondary.get_legend_handles_labels()
    axis.legend(lines + second_lines, labels + second_labels, loc="upper right")
    fig.tight_layout()
    output = FIG_DIR / "section2_representative_day.png"
    fig.savefig(output, dpi=150)
    plt.close(fig)
    return output


def plot_example_cycles(
    house_data: dict[int, pd.DataFrame],
    cycles_by_house: dict[int, pd.DataFrame],
) -> Path:
    fig, axes = plt.subplots(
        len(SELECTED_HOUSES), 1, figsize=(13, 3 * len(SELECTED_HOUSES))
    )
    for axis, house in zip(axes, SELECTED_HOUSES):
        cycles = cycles_by_house[house]
        if cycles.empty:
            axis.text(0.5, 0.5, "No cycle met the detection rule", ha="center")
            axis.set_axis_off()
            continue
        cycle = cycles.iloc[0]
        start = cycle["start"] - pd.Timedelta(minutes=30)
        end = cycle["end"] + pd.Timedelta(minutes=30)
        data = house_data[house].loc[start:end]
        axis.plot(data.index, data["Aggregate"], label="Aggregate", alpha=0.8)
        axis.plot(
            data.index,
            data["WashingMachine"],
            label="Washing machine",
            color="tab:orange",
        )
        axis.axvspan(cycle["start"], cycle["end"], color="tab:orange", alpha=0.12)
        axis.set_title(
            f"House {house}: {cycle['start']} "
            f"({cycle['duration_minutes']} min, {cycle['energy_kwh']:.2f} kWh)"
        )
        axis.set_ylabel("Power (W)")
        axis.grid(alpha=0.25)
        axis.legend(loc="upper right")
    axes[-1].set_xlabel("Local timestamp")
    fig.tight_layout()
    output = FIG_DIR / "section2_example_cycles.png"
    fig.savefig(output, dpi=150)
    plt.close(fig)
    return output


def plot_household_comparison(
    summary: pd.DataFrame, cycles_by_house: dict[int, pd.DataFrame]
) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    sns.barplot(
        data=summary,
        x="house",
        y="cycles_per_30_valid_coverage_days",
        ax=axes[0],
        color="steelblue",
    )
    axes[0].set_title("Detected operating episodes per 30 valid days")
    axes[0].set_xlabel("House")
    axes[0].set_ylabel("Episodes / 30 valid days")

    cycle_rows = [
        {"house": house, **cycle}
        for house, cycles in cycles_by_house.items()
        for cycle in cycles.to_dict("records")
    ]
    cycle_frame = pd.DataFrame(cycle_rows)
    if not cycle_frame.empty:
        sns.boxplot(data=cycle_frame, x="house", y="energy_kwh", ax=axes[1])
    else:
        axes[1].text(0.5, 0.5, "No detected cycles", ha="center")
    axes[1].set_title("Per-cycle washing-machine energy")
    axes[1].set_xlabel("House")
    axes[1].set_ylabel("Energy (kWh)")
    fig.tight_layout()
    output = FIG_DIR / "section2_household_comparison.png"
    fig.savefig(output, dpi=150)
    plt.close(fig)
    return output


def build_usage_profile(
    house_data: dict[int, pd.DataFrame],
    cycles_by_house: dict[int, pd.DataFrame],
) -> pd.DataFrame:
    weekday_names = [
        "Monday", "Tuesday", "Wednesday", "Thursday",
        "Friday", "Saturday", "Sunday",
    ]
    household_rates = []
    for house in SELECTED_HOUSES:
        data = house_data[house]
        coverage = data["WashingMachine"].notna().groupby(
            data.index.normalize()
        ).mean()
        valid_days = coverage[coverage >= 0.90].index
        valid_day_numbers = pd.Series(valid_days).dt.dayofweek
        exposure = valid_day_numbers.value_counts()
        cycles = cycles_by_house[house]
        eligible_starts = (
            cycles[cycles["start"].dt.normalize().isin(valid_days)]["start"]
            if not cycles.empty
            else pd.Series(dtype="datetime64[ns]")
        )
        counts = pd.crosstab(
            eligible_starts.dt.dayofweek, eligible_starts.dt.hour
        ).reindex(index=range(7), columns=range(24), fill_value=0)

        for weekday in range(7):
            eligible_days = int(exposure.get(weekday, 0))
            if eligible_days == 0:
                continue
            for hour in range(24):
                household_rates.append(
                    {
                        "house": house,
                        "weekday": weekday_names[weekday],
                        "hour": hour,
                        "cycles_per_valid_day": counts.loc[weekday, hour]
                        / eligible_days,
                    }
                )

    return pd.DataFrame(household_rates)


def plot_start_time_heatmap(profile: pd.DataFrame) -> Path:
    fig, axis = plt.subplots(figsize=(12, 4))
    if profile.empty:
        axis.text(0.5, 0.5, "No detected cycles", ha="center")
    else:
        weekday_order = [
            "Monday", "Tuesday", "Wednesday", "Thursday",
            "Friday", "Saturday", "Sunday",
        ]
        heatmap_data = (
            profile.groupby(["weekday", "hour"])["cycles_per_valid_day"]
            .mean()
            .unstack("hour")
            .reindex(index=weekday_order, columns=range(24), fill_value=0)
        )
        sns.heatmap(
            heatmap_data,
            cmap="Blues",
            ax=axis,
            cbar_kws={"label": "Mean episodes per valid day"},
        )
    axis.set_title("Operating-episode starts per valid day, by weekday and hour")
    axis.set_xlabel("Hour of day")
    axis.set_ylabel("Day of week")
    fig.tight_layout()
    output = FIG_DIR / "section2_usage_by_hour_weekday.png"
    fig.savefig(output, dpi=150)
    plt.close(fig)
    return output


def run() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    PROC_DIR.mkdir(parents=True, exist_ok=True)
    house_data = {
        house: load_house_one_minute(house) for house in SELECTED_HOUSES
    }
    cycles_by_house = {
        house: detect_cycles(data["WashingMachine"])
        for house, data in house_data.items()
    }

    summary = pd.DataFrame(
        [
            quality_summary(house, data, cycles_by_house[house])
            for house, data in house_data.items()
        ]
    )
    usage_profile = build_usage_profile(house_data, cycles_by_house)
    cycle_frames = [
        cycles.assign(house=house)
        for house, cycles in cycles_by_house.items()
        if not cycles.empty
    ]
    cycles = pd.concat(cycle_frames, ignore_index=True) if cycle_frames else pd.DataFrame()
    summary_path = PROC_DIR / "section2_household_summary.csv"
    cycles_path = PROC_DIR / "section2_detected_cycles.csv"
    usage_path = PROC_DIR / "section2_usage_profile.csv"
    summary.to_csv(summary_path, index=False)
    cycles.to_csv(cycles_path, index=False)
    usage_profile.to_csv(usage_path, index=False)

    plots = [
        plot_representative_day(house_data),
        plot_example_cycles(house_data, cycles_by_house),
        plot_household_comparison(summary, cycles_by_house),
        plot_start_time_heatmap(usage_profile),
    ]
    print("Cycle rule: >=10 W for at least 5 observed minutes,")
    print(
        f"minimum event duration {MIN_CYCLE_DURATION_MINUTES} minutes; "
        f"bridge <= {MAX_INACTIVE_GAP_TO_MERGE_MINUTES}-minute observed off-gaps."
    )
    print(summary.to_string(index=False))
    print(f"\nSummary: {summary_path}")
    print(f"Detected cycles: {cycles_path}")
    print(f"Normalized usage profile: {usage_path}")
    print("Figures:")
    for plot in plots:
        print(f"- {plot}")


if __name__ == "__main__":
    run()
