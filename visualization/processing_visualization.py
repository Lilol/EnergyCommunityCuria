import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from pandas import DataFrame

from parameteric_evaluation.definitions import CombinedMetricEnum
from utility.enum_definitions import convert_enum_to_value


def plot_shared_energy(sh1, sh2, n_fam):
    plt.figure()
    plt.plot(np.diff(sorted(sh2 - sh1)), label=f"{n_fam}", color='lightgrey')
    plt.yticks([])
    plt.xlabel('Number of day of year')
    plt.twinx().plot(sorted(sh2 - sh1), label=f"{n_fam}")
    plt.ylabel('Gap between hourly and daily shared energy (kWh)')
    plt.gca().yaxis.set_label_position("left")
    plt.gca().yaxis.tick_left()
    plt.title(f"Number of families: {int(n_fam)}")
    plt.tight_layout()
    plt.savefig("shared_energy.png")
    plt.show()
    plt.close()


def plot_results(results, n_fam, bess_size):
    plt.figure()
    df = DataFrame(columns=["time_resolution", "metric", "value"], index=[])
    for label in results.metric:
        label = label.values.item()
        if not isinstance(label, CombinedMetricEnum):
            continue
        df.loc[len(df), :] = label.first, label.second, \
            results.sel(metric=label, battery_size=bess_size, number_of_families=n_fam).values[0]
    df = df.map(convert_enum_to_value)
    sns.scatterplot(data=df, x="time_resolution", y="value", hue="metric", style="metric",
                    palette=sns.color_palette("colorblind"))
    plt.xlabel('Time resolution')
    plt.ylabel('Metric type')
    plt.title(f'Number of families={n_fam}, bess_size={bess_size}')
    plt.legend()
    plt.show()
    plt.close()
