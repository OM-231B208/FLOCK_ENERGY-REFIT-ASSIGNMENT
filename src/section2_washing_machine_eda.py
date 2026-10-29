"""Section 2: Exploratory analysis of washing-machine use."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
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
    active = observed & target.ge(WASHING_MACHINE_ACTIVE_THRESHOLD_W)
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
        "active_washing_machine_minutes": int(active.sum()),
        "active_percent_of_observed_minutes": (
            100 * active.sum() / observed.sum() if observed.sum() else 0.0
        ),
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
        "p95_active_power_w": (
            float(target[active].quantile(0.95)) if active.any() else 0.0
        ),
        "p99_active_power_w": (
            float(target[active].quantile(0.99)) if active.any() else 0.0
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
        sns.boxplot(
            data=cycle_frame,
            x="house",
            y="energy_kwh",
            ax=axes[1],
            showfliers=False,
            color="lightsteelblue",
        )
        generator = np.random.default_rng(42)
        for position, house in enumerate(SELECTED_HOUSES):
            energies = cycle_frame.loc[
                cycle_frame["house"] == house, "energy_kwh"
            ].to_numpy()
            axes[1].scatter(
                generator.normal(position, 0.06, len(energies)),
                energies,
                s=9,
                alpha=0.16,
                color="black",
                linewidths=0,
            )
        axes[1].set_yscale("log")
        axes[1].set_ylabel("Energy (kWh, log scale)")
    else:
        axes[1].text(0.5, 0.5, "No detected cycles", ha="center")
    axes[1].set_title("Detected-episode washing-machine energy")
    axes[1].set_xlabel("House")
    axes[1].set_ylabel("Energy (kWh)")
    fig.tight_layout()
    output = FIG_DIR / "section2_household_comparison.png"
    fig.savefig(output, dpi=150)
    plt.close(fig)
    return output


def build_monthly_coverage(house_data: dict[int, pd.DataFrame]) -> pd.DataFrame:
    records = []
    for house, data in house_data.items():
        target = data["WashingMachine"]
        for month, values in target.resample("MS"):
            records.append(
                {
                    "house": house,
                    "month": month,
                    "expected_minutes": len(values),
                    "observed_minutes": int(values.notna().sum()),
                    "coverage_percent": (
                        100 * values.notna().mean() if len(values) else 0.0
                    ),
                }
            )
    return pd.DataFrame(records)


def build_daily_energy(house_data: dict[int, pd.DataFrame]) -> pd.DataFrame:
    records = []
    for house, data in house_data.items():
        target = data["WashingMachine"]
        grouped = target.resample("D")
        daily = pd.DataFrame(
            {
                "observed_minutes": grouped.count(),
                "energy_kwh": grouped.sum(min_count=1) / 60_000.0,
                "coverage_percent": grouped.apply(
                    lambda day: 100 * day.notna().mean()
                ),
            }
        )
        daily.loc[daily["coverage_percent"] < 90.0, "energy_kwh"] = float("nan")
        for day, row in daily.iterrows():
            records.append(
                {
                    "house": house,
                    "date": day,
                    "observed_minutes": int(row["observed_minutes"]),
                    "coverage_percent": float(row["coverage_percent"]),
                    "energy_kwh": row["energy_kwh"],
                }
            )
    return pd.DataFrame(records)


def plot_monthly_coverage(monthly: pd.DataFrame) -> Path:
    fig, axis = plt.subplots(figsize=(13, 5))
    for house, values in monthly.groupby("house"):
        axis.plot(
            values["month"],
            values["coverage_percent"],
            marker="o",
            markersize=3,
            linewidth=1.5,
            label=f"House {house}",
        )
    axis.axhline(90, color="grey", linestyle="--", linewidth=1, label="90% reference")
    axis.set_ylim(0, 105)
    axis.set_title("Monthly washing-machine data availability")
    axis.set_xlabel("Month")
    axis.set_ylabel("Observed target minutes (%)")
    axis.grid(alpha=0.25)
    axis.legend(ncol=3)
    fig.tight_layout()
    output = FIG_DIR / "section2_monthly_data_coverage.png"
    fig.savefig(output, dpi=160)
    plt.close(fig)
    return output


def plot_daily_energy(daily_energy: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True)
    for axis, house in zip(axes.flat, SELECTED_HOUSES):
        data = daily_energy[daily_energy["house"] == house].set_index("date")
        valid = data["energy_kwh"].dropna()
        if valid.empty:
            axis.text(0.5, 0.5, "No days meet the coverage rule", ha="center")
            axis.set_title(f"House {house}: daily washing-machine energy")
            axis.set_ylabel("Energy (kWh/day)")
            continue

        percentile_99 = float(valid.quantile(0.99))
        y_limit = max(percentile_99 * 1.1, 0.02)
        typical_days = valid[valid <= percentile_99]
        high_days = valid[valid > percentile_99]
        axis.scatter(
            typical_days.index,
            typical_days,
            s=8,
            alpha=0.25,
            color="steelblue",
            label="Valid day",
        )
        if not high_days.empty:
            axis.scatter(
                high_days.index,
                np.full(len(high_days), y_limit * 0.97),
                marker="^",
                s=24,
                color="firebrick",
                label=f">99th percentile (n={len(high_days)})",
            )
        rolling = typical_days.rolling("14D", min_periods=5).mean()
        axis.plot(
            rolling.index,
            rolling,
            color="darkorange",
            linewidth=1.8,
            label="14-day mean (typical days)",
        )
        axis.set_title(f"House {house}: daily washing-machine energy")
        axis.set_ylabel("Energy (kWh/day)")
        axis.set_ylim(0, y_limit)
        if not high_days.empty:
            axis.text(
                0.02,
                0.94,
                f"Max observed: {valid.max():.2f} kWh/day",
                transform=axis.transAxes,
                va="top",
                fontsize=8,
            )
        axis.grid(alpha=0.2)
    for axis in axes[-1, :]:
        axis.set_xlabel("Date")
    fig.suptitle(
        "Daily energy (valid days only; blue dots <=P99, red triangles >P99; "
        "orange = 14-day typical-day mean)",
        y=1.01,
    )
    fig.tight_layout()
    output = FIG_DIR / "section2_daily_energy.png"
    fig.savefig(output, dpi=160)
    plt.close(fig)
    return output


def plot_normalized_cycle_profiles(
    house_data: dict[int, pd.DataFrame],
    cycles_by_house: dict[int, pd.DataFrame],
) -> Path:
    progress = np.linspace(0.0, 1.0, 101)
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True)
    for axis, house in zip(axes.flat, SELECTED_HOUSES):
        power = house_data[house]["WashingMachine"]
        profiles = []
        for cycle in cycles_by_house[house].itertuples(index=False):
            start = power.index.get_indexer([cycle.start])[0]
            stop = power.index.get_indexer([cycle.end])[0]
            if start < 0 or stop <= start:
                continue
            values = power.iloc[start : stop + 1].to_numpy(dtype=float)
            valid = np.isfinite(values)
            if valid.sum() < 2:
                continue
            source_progress = np.linspace(0.0, 1.0, len(values))[valid]
            profiles.append(
                np.interp(progress, source_progress, values[valid])
            )

        if profiles:
            values = np.stack(profiles)
            median = np.median(values, axis=0)
            lower, upper = np.quantile(values, [0.25, 0.75], axis=0)
            axis.fill_between(progress * 100, lower, upper, alpha=0.2)
            axis.plot(progress * 100, median, linewidth=1.8)
            axis.set_title(f"House {house}: median profile (n={len(profiles)} episodes)")
        else:
            axis.text(0.5, 0.5, "No valid operating episodes", ha="center")
            axis.set_title(f"House {house}")
        axis.set_ylabel("Washing-machine power (W)")
        axis.grid(alpha=0.2)
    for axis in axes[-1, :]:
        axis.set_xlabel("Episode progress (%)")
    fig.suptitle("Washing-machine operating-episode profiles", y=1.01)
    fig.tight_layout()
    output = FIG_DIR / "section2_normalized_cycle_profiles.png"
    fig.savefig(output, dpi=160, bbox_inches="tight")
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
    monthly_coverage = build_monthly_coverage(house_data)
    daily_energy = build_daily_energy(house_data)
    cycle_frames = [
        cycles.assign(house=house)
        for house, cycles in cycles_by_house.items()
        if not cycles.empty
    ]
    cycles = pd.concat(cycle_frames, ignore_index=True) if cycle_frames else pd.DataFrame()
    summary_path = PROC_DIR / "section2_household_summary.csv"
    cycles_path = PROC_DIR / "section2_detected_cycles.csv"
    usage_path = PROC_DIR / "section2_usage_profile.csv"
    monthly_path = PROC_DIR / "section2_monthly_coverage.csv"
    daily_energy_path = PROC_DIR / "section2_daily_energy.csv"
    summary.to_csv(summary_path, index=False)
    cycles.to_csv(cycles_path, index=False)
    usage_profile.to_csv(usage_path, index=False)
    monthly_coverage.to_csv(monthly_path, index=False)
    daily_energy.to_csv(daily_energy_path, index=False)

    plots = [
        plot_representative_day(house_data),
        plot_example_cycles(house_data, cycles_by_house),
        plot_household_comparison(summary, cycles_by_house),
        plot_start_time_heatmap(usage_profile),
        plot_monthly_coverage(monthly_coverage),
        plot_daily_energy(daily_energy),
        plot_normalized_cycle_profiles(house_data, cycles_by_house),
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
    print(f"Monthly coverage: {monthly_path}")
    print(f"Daily energy: {daily_energy_path}")
    print("Figures:")
    for plot in plots:
        print(f"- {plot}")


if __name__ == "__main__":
    run()
