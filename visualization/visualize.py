from os import makedirs
from os.path import join

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

from data_processing_pipeline.definitions import Stage
from data_processing_pipeline.pipeline_stage import PipelineStage
from data_storage.omnes_data_array import OmnesDataArray
from utility.configuration import config


class Visualize(PipelineStage):
    stage = Stage.VISUALIZE
    _name = "visualize"

    def __init__(self, name=_name, *args, **kwargs):
        super().__init__(name, *args, **kwargs)
        self.vis_function = args[0]

    def execute(self, dataset: OmnesDataArray | None, *args, **kwargs) -> OmnesDataArray | None:
        self.vis_function(dataset, *args, **kwargs)
        return dataset


def plot_target_metrics_evaluation(results_da: OmnesDataArray, **kwargs):
    """
    Visualize target metrics evaluation results in an informative and elegant way.

    The visualization shows for each metric:
    - Target values vs. achieved metric values (line plot)
    - Number of families required to achieve each target (bar plot)

    Parameters:
    -----------
    results_da : OmnesDataArray
        Data array with dimensions: ['metric', 'target', 'result_type']
        where result_type contains 'number_of_families' and 'metric_realized'
    """
    sns.set_style("whitegrid")

    # Get metrics from data
    if "metric" not in results_da.dims:
        metrics = [results_da.coords.get("metric", "Unknown").values.item()]
        results_da = results_da.expand_dims({"metric": metrics})
    else:
        metrics = results_da.coords["metric"].values

    # Filter metrics that have valid data
    valid_metrics = []
    for metric in metrics:
        metric_data = results_da.sel(metric=metric)
        n_families = metric_data.sel(result_type="number_of_families").values.flatten()
        metric_realized = metric_data.sel(result_type="metric_realized").values.flatten()
        valid_mask = ~np.isnan(n_families) & ~np.isnan(metric_realized)
        if any(valid_mask):
            valid_metrics.append(metric)

    if len(valid_metrics) == 0:
        return  # No valid data to plot

    n_metrics = len(valid_metrics)

    # Create figure with subplots only for valid metrics
    fig, axes = plt.subplots(
        nrows=n_metrics,
        ncols=2,
        figsize=(14, 4 * n_metrics),
        squeeze=False
    )

    # Color palette
    palette = sns.color_palette("husl", n_metrics)

    for i, metric in enumerate(valid_metrics):
        metric_data = results_da.sel(metric=metric)
        targets = metric_data.coords["target"].values

        # Extract data
        n_families = metric_data.sel(result_type="number_of_families").values.flatten()
        metric_realized = metric_data.sel(result_type="metric_realized").values.flatten()

        # Filter out NaN values
        valid_mask = ~np.isnan(n_families) & ~np.isnan(metric_realized)
        targets_valid = targets[valid_mask]
        n_families_valid = n_families[valid_mask]
        metric_realized_valid = metric_realized[valid_mask]

        color = palette[i]

        # Left plot: Target vs Realized metric values
        ax1 = axes[i, 0]

        # Plot diagonal reference line (perfect achievement)
        min_val = min(targets_valid.min(), metric_realized_valid.min()) * 0.95
        max_val = max(targets_valid.max(), metric_realized_valid.max()) * 1.05
        ax1.plot([min_val, max_val], [min_val, max_val],
                 'k--', alpha=0.5, label='Perfect achievement', linewidth=1.5)

        # Plot realized values
        ax1.scatter(targets_valid, metric_realized_valid,
                   color=color, s=100, zorder=5, edgecolor='white', linewidth=1.5)
        ax1.plot(targets_valid, metric_realized_valid,
                color=color, alpha=0.7, linewidth=2)

        # Fill area between target and realized
        ax1.fill_between(targets_valid, targets_valid, metric_realized_valid,
                        alpha=0.2, color=color)

        ax1.set_xlabel('Target Value', fontsize=11)
        ax1.set_ylabel('Realized Value', fontsize=11)
        ax1.set_title(f'{metric}\nTarget vs. Realized', fontsize=12, fontweight='bold')
        ax1.legend(loc='lower right', frameon=True)
        ax1.set_xlim(min_val, max_val)
        ax1.set_ylim(min_val, max_val)

        # Right plot: Number of families required
        ax2 = axes[i, 1]

        bars = ax2.bar(range(len(targets_valid)), n_families_valid,
                      color=color, alpha=0.8, edgecolor='white', linewidth=1.5)

        # Add value labels on bars
        for j, (bar, nf, realized) in enumerate(zip(bars, n_families_valid, metric_realized_valid)):
            height = bar.get_height()
            ax2.annotate(f'{int(nf)}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9, fontweight='bold')
            # Add realized value annotation
            ax2.annotate(f'({realized:.2f})',
                        xy=(bar.get_x() + bar.get_width() / 2, height / 2),
                        ha='center', va='center', fontsize=8, color='white', fontweight='bold')

        ax2.set_xticks(range(len(targets_valid)))
        ax2.set_xticklabels([f'{t:.2f}' for t in targets_valid], rotation=45, ha='right')
        ax2.set_xlabel('Target Value', fontsize=11)
        ax2.set_ylabel('Number of Families', fontsize=11)
        ax2.set_title(f'{metric}\nFamilies Required per Target', fontsize=12, fontweight='bold')

        # Add trend line
        if len(targets_valid) > 1:
            z = np.polyfit(range(len(targets_valid)), n_families_valid, 1)
            p = np.poly1d(z)
            x_line = np.linspace(0, len(targets_valid) - 1, 100)
            ax2.plot(x_line, p(x_line), '--', color='darkgrey',
                    alpha=0.7, linewidth=2, label='Trend')
            ax2.legend(loc='upper left', frameon=True)

    # Add overall title
    fig.suptitle('Target Metrics Evaluation Results',
                fontsize=14, fontweight='bold', y=1.02)

    plt.tight_layout()

    # Save figure
    figures_path = config.get("path", "figures")
    makedirs(figures_path, exist_ok=True)
    plt.savefig(join(figures_path, "target_metrics_evaluation.png"),
                dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()


def plot_target_metrics_summary(results_da: OmnesDataArray, **kwargs):
    """
    Create a summary visualization comparing all metrics on a single plot.

    Parameters:
    -----------
    results_da : OmnesDataArray
        Data array with dimensions: ['metric', 'target', 'result_type']
    """
    sns.set_style("whitegrid")

    # Get metrics from data
    if "metric" not in results_da.dims:
        metrics = [results_da.coords.get("metric", "Unknown").values.item()]
        results_da = results_da.expand_dims({"metric": metrics})
    else:
        metrics = results_da.coords["metric"].values

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Color palette for metrics
    palette = {
        "Self consumption": "#1f77b4",
        "Self sufficiency": "#ff7f0e",
        "Self production": "#2ca02c",
        "Grid liability": "#d62728"
    }
    default_palette = sns.color_palette("husl", len(metrics))

    legend_handles = []

    for i, metric in enumerate(metrics):
        metric_str = str(metric)
        color = palette.get(metric_str, default_palette[i])

        metric_data = results_da.sel(metric=metric)
        targets = metric_data.coords["target"].values
        n_families = metric_data.sel(result_type="number_of_families").values.flatten()
        metric_realized = metric_data.sel(result_type="metric_realized").values.flatten()

        # Filter NaN
        valid_mask = ~np.isnan(n_families) & ~np.isnan(metric_realized)
        if not any(valid_mask):
            continue

        targets_valid = targets[valid_mask]
        n_families_valid = n_families[valid_mask]
        metric_realized_valid = metric_realized[valid_mask]

        # Left plot: All metrics comparison
        ax1.plot(n_families_valid, metric_realized_valid,
                marker='o', color=color, linewidth=2.5, markersize=8,
                label=metric_str, alpha=0.8)
        ax1.scatter(n_families_valid, metric_realized_valid,
                   color=color, s=80, zorder=5, edgecolor='white', linewidth=1)

        # Right plot: Target achievement gap
        gap = metric_realized_valid - targets_valid
        ax2.bar(np.arange(len(targets_valid)) + i * 0.2, gap,
               width=0.18, color=color, alpha=0.8, label=metric_str)

        legend_handles.append(Line2D([0], [0], color=color, linewidth=2.5,
                                     marker='o', markersize=8, label=metric_str))

    # Configure left plot
    ax1.set_xlabel('Number of Families', fontsize=12)
    ax1.set_ylabel('Metric Value Achieved', fontsize=12)
    ax1.set_title('Metric Achievement vs. Community Size', fontsize=13, fontweight='bold')
    ax1.legend(handles=legend_handles, loc='lower right', frameon=True)
    ax1.axhline(y=1.0, color='grey', linestyle='--', alpha=0.5, linewidth=1)
    ax1.axhline(y=0.5, color='grey', linestyle=':', alpha=0.5, linewidth=1)

    # Configure right plot
    ax2.axhline(y=0, color='black', linewidth=1)
    ax2.set_xlabel('Target Index', fontsize=12)
    ax2.set_ylabel('Achievement Gap (Realized - Target)', fontsize=12)
    ax2.set_title('Target Achievement Gap', fontsize=13, fontweight='bold')
    ax2.legend(loc='best', frameon=True)

    plt.tight_layout()

    # Save figure
    figures_path = config.get("path", "figures")
    makedirs(figures_path, exist_ok=True)
    plt.savefig(join(figures_path, "target_metrics_summary.png"),
                dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()
