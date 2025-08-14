import logging

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from pandas import DataFrame, Categorical, date_range

from io_operation.input.definitions import DataKind
from parameteric_evaluation.definitions import CombinedMetricEnum, PhysicalMetric
from parameteric_evaluation.physical import SharedEnergy
from utility.enum_definitions import convert_enum_to_value

logger = logging.getLogger(__name__)


def plot_shared_energy(input_da, n_fam, bess_size):
    sns.set_style("whitegrid")
    palette = sns.color_palette("Set2")

    fig, axes = plt.subplots(ncols=2, nrows=2, figsize=(20, 17), sharey=True)

    # Daily base
    daily = input_da.groupby("time.dayofyear").sum()
    daily, _ = SharedEnergy.calculate(daily)
    daily = daily.sel({DataKind.CALCULATED.value: PhysicalMetric.SHARED_ENERGY})

    agg_levels = ["15min", "hour", "season", "year"]
    for i, (agg_level, ax) in enumerate(zip(agg_levels, axes.flat)):
        logger.info(f"Plotting shared energy for: {agg_level}")
        if agg_level == "hour":
            aggregated = input_da.resample(time='1h').mean()
        elif agg_level == "15min":
            aggregated = input_da
        elif agg_level == "season":
            aggregated = input_da.resample(time="QE", closed="left", label="right").mean()
        else:
            aggregated = input_da.resample(time="YE", closed="left", label="right").mean()

        aggregated, _ = SharedEnergy.calculate(aggregated)

        if agg_level in ["season", "year"]:

            # Create a daily time index covering the same span as aggregated
            daily_time = date_range(start=aggregated.time.min().values, end=aggregated.time.max().values, freq="D")

            # Reindex and interpolate
            aggregated_daily = aggregated.reindex(time=daily_time)
            aggregated = aggregated_daily.interpolate_na(dim="time", method="zero")
        else:
            aggregated = aggregated.resample(time="1d")
            aggregated = aggregated.mean()
        aggregated = aggregated.sel({DataKind.CALCULATED.value: PhysicalMetric.SHARED_ENERGY})

        diff_vals = aggregated.data[:365] - daily.data
        sorted_vals = sorted(diff_vals)

        axt = ax.twinx()
        ax.plot(np.diff(sorted_vals), label=f"{agg_level}-daily", color="grey", linestyle="-", linewidth=2.5)
        axt.plot(sorted_vals, label=f"{agg_level}", color=palette[i], linestyle="-", linewidth=2.5)

        # Labels & titles
        ax.set_ylabel('Difference between aggregates (kWh)', fontsize=12)
        axt.set_ylabel('Sorted difference values (kWh)', fontsize=12)
        ax.set_xlabel('Day of Year', fontsize=12)
        ax.set_title(f"Shared energy for: {agg_level}-daily", fontsize=12)
        # Combine legends from both axes
        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = axt.get_legend_handles_labels()
        ax.legend(lines + lines2, labels + labels2, loc="upper right", frameon=True)

    fig.suptitle(f"Shared Energy Gap\nFamilies = {int(n_fam)}, Battery Size = {bess_size} kWh", fontsize=14,
                 weight="bold")



    plt.tight_layout()
    plt.savefig("shared_energy.png", dpi=300)
    plt.show()
    plt.close()


def plot_results(results, n_fam, bess_size):
    df = DataFrame(columns=["time_resolution", "metric", "value"])
    for label in results.metric:
        label = label.values.item()
        if not isinstance(label, CombinedMetricEnum):
            continue
        df.loc[len(df), :] = [label.first, label.second,
                              results.sel(metric=label, battery_size=bess_size, number_of_families=n_fam).values[0]]

    df = df.map(convert_enum_to_value)

    palette = {"Self sufficiency": "#1f77b4", "Self consumption": "#ff7f0e", "Self production": "#2ca02c",
               "Grid liability": "#d62728"}

    order = ["15min", "hour", "dayofyear", "month", "season", "year"]  # logical ordering
    df["time_resolution"] = Categorical(df["time_resolution"], categories=order, ordered=True)

    plt.figure(figsize=(8, 5))
    ax = sns.scatterplot(data=df, x="time_resolution", y="value", hue="metric", style="metric", palette=palette, s=200,
                         alpha=0.8, hue_order=palette.keys())

    # Add reference lines
    for y in [0, 0.5, 1]:
        ax.axhline(y, ls="--", color="gray", lw=0.7)

    # Value labels
    for _, row in df.iterrows():
        ax.text(row["time_resolution"], row["value"] + 0.02, f"{row['value']:.2f}", ha='center', fontsize=8)

    plt.xlabel('Time resolution')
    plt.ylabel('Metric value')
    plt.title(f'Number of families={n_fam}, bess_size={bess_size} kWh')
    plt.legend(title="Metric type")
    plt.tight_layout()
    plt.show()
