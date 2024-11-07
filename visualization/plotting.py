# ----------------------------------------------------------------------------
# Import statement

# Data management
import os

# Plotting
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import cm
from matplotlib import gridspec as gs
from matplotlib import patheffects as pe

# common variables
from input.definitions import ColumnName
from input.reader import BillsReader
from visualization.plotting_utils import get_colors_from_map, pie_chart

if __name__ == '__main__':
    # ----------------------------------------------------------------------------
    # General setup for plots
    fig_width = 16 / 2.54  # inch (SETUP)
    fig_height = 8 / 2.54  # inch (SETUP)
    fontsize = 16  # (SETUP)
    alpha = 0.8  # (SETUP)
    mpl.rcParams.update({'font.size': fontsize, 'figure.figsize': (fig_width, fig_height)})
    divide_energy = 1000
    if divide_energy == 1:
        unit_energy = 'kWh'
    elif divide_energy == 1000:
        unit_energy = 'MWh'
    else:
        unit_energy = '?'

    # ----------------------------------------------------------------------------
    # Setup and data loading

    # Directory with files
    directory_data = 'DatiProcessati'

    # Names of the files to load
    file_plants = "data_plants.csv"  # list of plants
    file_plants_tou = "data_plants_tou.csv"  # monthly production data
    file_plants_year = "data_plants_year.csv"  # one-year hourly production data
    file_users = "data_users.csv"  # list of end users
    file_users_tou = "data_users_tou.csv"  # monthly consumption data
    file_users_year = "data_users_year.csv"  # one-year hourly consumption data
    file_results = "results.csv"  # results of evaluation

    # Load data
    data_plants = pd.read_csv(os.path.join(directory_data, file_plants), sep=';')
    data_plants_tou = pd.read_csv(os.path.join(directory_data, file_plants_tou), sep=';')
    data_plants_year = pd.read_csv(os.path.join(directory_data, file_plants_year), sep=';')
    data_users = pd.read_csv(os.path.join(directory_data, file_users), sep=';')
    data_users_tou = pd.read_csv(os.path.join(directory_data, file_users_tou), sep=';')
    data_users_year = pd.read_csv(os.path.join(directory_data, file_users_year), sep=';')
    data_results = pd.read_csv(os.path.join(directory_data, file_results), sep=';')

    #
    data_users[ColumnName.USER_TYPE] = data_users[ColumnName.USER_TYPE].str.upper()
    data_plants[ColumnName.USER_TYPE] = data_plants[ColumnName.USER_TYPE].str.upper()
    new_cols = {col: f"F{f}" for f, col in enumerate(BillsReader._time_of_use_energy_column_names)}
    data_users = data_users.rename(columns=new_cols)
    data_plants = data_plants.rename(columns=new_cols)
    data_users_tou = data_users_tou.rename(columns=new_cols)
    data_plants_tou = data_plants_tou.rename(columns=new_cols)
    BillsReader._time_of_use_energy_column_names = list(new_cols.values())

    data_plants_tou[ColumnName.ANNUAL_ENERGY] = data_plants_tou[BillsReader._time_of_use_energy_column_names].sum(axis=1)

    #
    data_users[[*BillsReader._time_of_use_energy_column_names, ColumnName.ANNUAL_ENERGY]] /= divide_energy
    data_plants[[*BillsReader._time_of_use_energy_column_names, ColumnName.ANNUAL_ENERGY]] /= divide_energy
    data_users_tou[[*BillsReader._time_of_use_energy_column_names, ColumnName.ANNUAL_ENERGY]] /= divide_energy
    data_plants_tou[[*BillsReader._time_of_use_energy_column_names, ColumnName.ANNUAL_ENERGY]] /= divide_energy

    # ----------------------------------------------------------------------------
    # %% We analyze the users and plants data sets

    # We first define some useful variables
    municipalities = sorted(set(data_users[ColumnName.MUNICIPALITY]) | set(data_plants[ColumnName.MUNICIPALITY]))
    n_municipalities = len(municipalities)
    types = list(set(data_users[ColumnName.USER_TYPE]) | set(data_plants[ColumnName.USER_TYPE]))

    # Setup for these plots
    nrows, ncols = 2, 2
    figsize = (fig_width, fig_height)
    gridspec_kw = dict(height_ratios=[0.9, 0.1], width_ratios=[0.5, 0.5])  # (SETUP)
    pie_chart_kw = dict(labels_pos='legend', autopct="absolute", wedgeprops=dict(alpha=alpha), )
    bar_chart_kw = dict(alpha=alpha)
    legend_kw = dict(loc='center', ncol=n_municipalities)  # (SETUP)
    cmap_municipality = 'cool'  # (SETUP)
    cmap_type = 'winter'  # (SETUP)
    months_labels = ['Gen', 'Feb', 'Mar', 'Apr', 'Mag', 'Giu', 'Lug', 'Ago', 'Set', 'Ott', 'Nov', 'Dic']

    # Define colors dict to assign to each municipality
    colors_municipality = dict(zip(municipalities, get_colors_from_map(n_municipalities, cmap_municipality)))

    # Define colors dict to assign to each type of end-user/plant
    colors_types = dict(zip(types, get_colors_from_map(len(types), cmap_type)))

    # Define colors dict to assign to each ToU tariff
    colors_tou = dict(F0=cm.get_cmap('nipy_spectral')(0.2), F1=cm.get_cmap('nipy_spectral')(0.85),
                      F2=cm.get_cmap('nipy_spectral')(0.70), F3=cm.get_cmap('nipy_spectral')(0.55))

    # Here, we make two side-by-side subplots showing the distribution of
    # end users and plants by municipality

    # Setup for this plot
    titles = ["Utenze comunali", "Impianti rinnovabili"]  # (SETUP)
    subplots_adjust_kw = dict(left=0, right=1, bottom=0.05, top=0.9, hspace=0)  # (SETUP)

    # Select data and labels to plot
    datas = [data_users, data_plants]
    cols = [ColumnName.MUNICIPALITY, ColumnName.MUNICIPALITY]
    pies_counts, pies_labels = [], []
    for i, data in enumerate(datas):
        col = cols[i]
        counts, labels = zip(*[(count, label) for label, count in data[col].value_counts().sort_index().items()])
        pies_counts.append(counts)
        pies_labels.append(labels)

    # Make figure
    fig = plt.figure(figsize=figsize)
    spec = gs.GridSpec(nrows=nrows, ncols=ncols, **gridspec_kw)
    axes = [fig.add_subplot(spec[0, 0]), fig.add_subplot(spec[0, 1])]
    ax_leg = fig.add_subplot(spec[1, :])

    # Plot data
    for i, (counts, labels) in enumerate(zip(pies_counts, pies_labels)):
        # Select axis
        ax = axes[i]
        # Plot pie chart
        this_pie_chart_kw = pie_chart_kw.copy()
        if this_pie_chart_kw['autopct'] == 'absolute':
            this_pie_chart_kw['autopct'] = lambda s: f"{int(s / 100 * sum(counts))}"
        pie_chart(counts, labels=labels, ax=ax, colors=colors_municipality, **this_pie_chart_kw)

    # Add titles
    for ax, title in zip(axes, titles):
        ax.set_title(title)

    # Fix the legend to be on the dedicated subplots
    handles, labels = [], []
    for i, ax in enumerate(axes):
        for handle, label in zip(*ax.get_legend_handles_labels()):
            if label not in labels:
                labels.append(label)
                handles.append(handle)
        ax.legend().remove()
    ax_leg.axis('off')
    ax_leg.legend(handles, labels, **legend_kw)

    # Adjust layout
    fig.subplots_adjust(**subplots_adjust_kw)

    plt.show()

    # Here, we make two side-by-side subplots showing the distribution of
    # yearly energy consumed and produced by municipality

    # Setup for this plots
    titles = [f'Consumo annuo ({unit_energy})', f'Produzione annua ({unit_energy})']  # (SETUP)
    subplots_adjust_kw = dict(left=0, right=1, bottom=0.05, top=0.9, hspace=0)  # (SETUP)

    # Select data and labels to plot
    datas = [data_users, data_plants]
    cols_label = [ColumnName.MUNICIPALITY, ColumnName.MUNICIPALITY]
    cols_data = [ColumnName.ANNUAL_ENERGY, ColumnName.ANNUAL_ENERGY]
    pies_counts = []
    pies_labels = []
    for i, data in enumerate(datas):
        counts, labels = zip(
            *[(count, label) for label, count in data.groupby(cols_label[i]).sum()[cols_data[i]].sort_index().items()])
        pies_counts.append(list(counts))
        pies_labels.append(list(labels))

    # Make figure
    fig = plt.figure(figsize=figsize)
    spec = gs.GridSpec(nrows=nrows, ncols=ncols, **gridspec_kw)
    axes = [fig.add_subplot(spec[0, 0]), fig.add_subplot(spec[0, 1])]
    ax_leg = fig.add_subplot(spec[1, :])

    # Plot data
    for i, (counts, labels) in enumerate(zip(pies_counts, pies_labels)):
        # Select axis
        ax = axes[i]
        # Plot pie chart
        this_pie_chart_kw = pie_chart_kw.copy()
        if this_pie_chart_kw['autopct'] == 'absolute':
            this_pie_chart_kw['autopct'] = lambda s: f"{int(s / 100 * sum(counts))}"
        pie_chart(counts, labels=labels, ax=ax, colors=colors_municipality, **this_pie_chart_kw)

    # Add titles
    for ax, title in zip(axes, titles):
        ax.set_title(title)

    # Fix the legend to be on the dedicated subplots
    handles, labels = [], []
    for i, ax in enumerate(axes):
        for handle, label in zip(*ax.get_legend_handles_labels()):
            if label not in labels:
                labels.append(label)
                handles.append(handle)
        ax.legend().remove()
    ax_leg.axis('off')
    ax_leg.legend(handles, labels, **legend_kw)

    # Adjust layout
    fig.subplots_adjust(**subplots_adjust_kw)

    plt.show()

    # Here we show how the end users and plants are distributed in classes
    # of power grouped by municipality

    # Setup for this plots
    nrows, ncols = 2, 3
    gridspec_kw = {**gridspec_kw, 'width_ratios': [0.05, 0.45, 0.45]}
    titles = ['Potenza utenze (kW)', 'Potenza impianti (kW)']  # (SETUP)
    tickparams_kw = dict(axis='x', rotation=20)  # (SETUP)
    subplots_adjust_kw = dict(left=0.01, right=0.98, bottom=0.05, top=0.9, hspace=0.5)  # (SETUP)
    ylabel = "Numero"  # (SETUP)
    grid_kw = dict(axis='y')  # (SETUP)

    # Select data and labels to plot
    datas = [data_users, data_plants]
    cols_group = [ColumnName.MUNICIPALITY, ColumnName.MUNICIPALITY]
    cols_data = [ColumnName.POWER, ColumnName.POWER]
    binss = [[0, 3, 6, 10, 16.5, data_users[ColumnName.POWER].max()],  # (SETUP)
             [0, 20, data_plants[ColumnName.POWER].max()]]  # (SETUP)
    labels_from_bins = lambda bins: [
        f'<{b}' if j == 0 else f'{bins[j]}-{b}' if j < len(bins) - 2  # else f'{int(bins[j])}$\leq${int(b)}'
        else f'$\geq${bins[j]}' for j, b in enumerate(bins[1:])]  # (SETUP)
    bars_counts, bars_labels, legend_labels = [], [], []
    for i, data in enumerate(datas):
        bins = binss[i]
        col_group = cols_group[i]
        col_data = cols_data[i]
        counts = []
        labels = []
        if isinstance(bins, int):
            _, bins = np.histogram(data[col_data], bins=bins)
        for group, df in data.groupby(col_group):
            labels.append(group)
            these_counts, _ = np.histogram(df[col_data], bins=bins)
            counts.append(list(these_counts))
        bars_counts.append(counts)
        bars_labels.append(labels_from_bins(bins))
        legend_labels.append(labels)

    # Make figure
    fig = plt.figure(figsize=figsize)
    spec = gs.GridSpec(nrows=nrows, ncols=ncols, **gridspec_kw)
    axes = [fig.add_subplot(spec[0, 1]), fig.add_subplot(spec[0, 2])]
    ax_leg = fig.add_subplot(spec[1, :])
    ax_label = fig.add_subplot(spec[0, 0])

    # Plot data
    for i, (counts, labels, legend) in enumerate(zip(bars_counts, bars_labels, legend_labels)):
        # Select axis
        ax = axes[i]
        # Plot bars
        bottom = [0] * len(labels)
        for count, legend in zip(counts, legend):
            ax.bar(range(len(labels)), count, label=legend, bottom=bottom, color=colors_municipality[legend])
            bottom = [b + c for b, c in zip(bottom, count)]

        # Set ticks
        ax.set_xticks(range(len(labels)), labels)
        ax.tick_params(**tickparams_kw)

        # Show grid
        ax.grid(**grid_kw)

    # Add titles
    for ax, title in zip(axes, titles):
        ax.set_title(title)

    # Add ylabel
    ax_label.axis('off')
    ax_label.text(0.5, 0.5, ylabel, rotation=90, ha='center', va='center')

    # Fix the legend to be on the dedicated subplots
    handles, labels = [], []
    for i, ax in enumerate(axes):
        for handle, label in zip(*ax.get_legend_handles_labels()):
            if label not in labels:
                labels.append(label)
                handles.append(handle)
        ax.legend().remove()
    ax_leg.axis('off')
    ax_leg.legend(handles, labels, **legend_kw)

    # Adjust layout
    fig.subplots_adjust(**subplots_adjust_kw)

    plt.show()

    # Here we show how the end users and plants are distributed in classes
    # of power grouped by type

    # Setup for this plots
    titles = ['Potenza utenze (kW)', 'Potenza impianti (kW)']  # (SETUP)
    tickparams_kw = dict(axis='x', rotation=20)  # (SETUP)
    subplots_adjust_kw = dict(left=0.01, right=0.98, bottom=0.05, top=0.9, hspace=0.5)  # (SETUP)
    grid_kw = dict(axis='y')  # (SETUP)

    # Select data and labels to plot
    datas = [data_users, data_plants]
    cols_group = [ColumnName.USER_TYPE, ColumnName.USER_TYPE]
    cols_data = [ColumnName.POWER, ColumnName.POWER]
    bars_counts, bars_labels, legend_labels = [], [], []
    for i, data in enumerate(datas):
        bins = binss[i]
        col_group = cols_group[i]
        col_data = cols_data[i]
        counts = []
        labels = []
        if isinstance(bins, int):
            _, bins = np.histogram(data[col_data], bins=bins)
        for group, df in data.groupby(col_group):
            labels.append(group)
            these_counts, _ = np.histogram(df[col_data], bins=bins)
            counts.append(list(these_counts))
        bars_counts.append(counts)
        bars_labels.append(labels_from_bins(bins))
        legend_labels.append(labels)

    # Make figure
    fig = plt.figure(figsize=figsize)
    spec = gs.GridSpec(nrows=nrows, ncols=ncols, **gridspec_kw)
    axes = [fig.add_subplot(spec[0, 1]), fig.add_subplot(spec[0, 2])]
    axes_leg = [fig.add_subplot(spec[1, 1]), fig.add_subplot(spec[1, 2])]
    ax_label = fig.add_subplot(spec[0, 0])

    # Plot data
    for i, (counts, labels, legend) in enumerate(zip(bars_counts, bars_labels, legend_labels)):
        # Select axis
        ax = axes[i]
        # Plot bars
        bottom = [0] * len(labels)
        for count, legend in zip(counts, legend):
            ax.bar(range(len(labels)), count, label=legend, bottom=bottom, color=colors_types[legend])
            bottom = [b + c for b, c in zip(bottom, count)]

        # Set ticks
        ax.set_xticks(range(len(labels)), labels)
        ax.tick_params(**tickparams_kw)

        # Show grid
        ax.grid(**grid_kw)

    # Add titles
    for ax, title in zip(axes, titles):
        ax.set_title(title)

    # Add ylabel
    ax_label.axis('off')
    ax_label.text(0.5, 0.5, ylabel, rotation=90, ha='center', va='center')

    # Fix the legend to be on the dedicated subplots
    for ax, ax_leg in zip(axes, axes_leg):
        handles, labels = [], []
        for handle, label in zip(*ax.get_legend_handles_labels()):
            if label not in labels:
                labels.append(label)
                handles.append(handle)
        ax.legend().remove()
        ax_leg.axis('off')
        ax_leg.legend(handles, labels, **legend_kw)

    # Adjust layout
    fig.subplots_adjust(**subplots_adjust_kw)

    plt.show()

    # Here, we make two side-by-side subplots showing the energy consumed/
    # produced in each tou tariff and by type of end-user/plant

    titles = ['Consumo fasce orarie', 'Produzione fasce orarie']  # (SETUP)
    tickparams_kw = dict(axis='y', rotation=20)  # (SETUP)
    subplots_adjust_kw = dict(left=0.01, right=0.98, bottom=0.02, top=0.9, wspace=0.25, hspace=0.25)  # (SETUP)
    ylabel = f"Energia ({unit_energy})"  # (SETUP)
    grid_kw = dict(axis='y')  # (SETUP)

    # Select data and labels to plot
    datas = [data_users, data_plants]
    cols_group = [ColumnName.USER_TYPE, ColumnName.USER_TYPE]
    cols_data = [BillsReader._time_of_use_energy_column_names, BillsReader._time_of_use_energy_column_names]
    bars_counts, bars_labels, legend_labels = [], [], []
    for i, data in enumerate(datas):
        col_group = cols_group[i]
        col_data = cols_data[i]
        counts, labels = [], []
        for _, df in data.groupby(col_group):
            counts_labels = df[col_data].sum()
            counts.append(list(counts_labels))
            labels = list(counts_labels.index)
        bars_counts.append(counts)
        bars_labels.append(labels)
        legend_labels.append(list(data.groupby(col_group).groups.keys()))

    # Make figure
    fig = plt.figure(figsize=figsize)
    spec = gs.GridSpec(nrows=nrows, ncols=ncols, **gridspec_kw)
    axes = [fig.add_subplot(spec[0, 1]), fig.add_subplot(spec[0, 2])]
    axes_leg = [fig.add_subplot(spec[1, 1]), fig.add_subplot(spec[1, 2])]
    ax_label = fig.add_subplot(spec[0, 0])

    # Plot data
    for i, (counts, labels, legend) in enumerate(zip(bars_counts, bars_labels, legend_labels)):
        # Select axis
        ax = axes[i]

        # Do not plot zeros
        new_labels = [l for j, l in enumerate(labels) if any([count[j] > 0 for count in counts])]
        for j, l in enumerate(labels):
            if not any([count[j] > 0 for count in counts]):
                labels.pop(j)
                for count in counts:
                    count.pop(j)

        # Plot bars
        bottom = [0] * len(labels)
        for count, legend in zip(counts, legend):
            ax.bar(labels, count, label=legend, bottom=bottom, color=colors_types[legend])
            bottom = [b + c for b, c in zip(bottom, count)]

        # Set ticks
        ax.set_xticks(range(len(labels)), labels)
        ax.tick_params(**tickparams_kw)

        # Show grid
        ax.grid(**grid_kw)

    # Add titles
    for ax, title in zip(axes, titles):
        ax.set_title(title)

    # Add ylabel
    ax_label.axis('off')
    ax_label.text(0.5, 0.5, ylabel, rotation=90, ha='center', va='center')

    # Fix the legend to be on the dedicated subplots
    for ax, ax_leg in zip(axes, axes_leg):
        handles, labels = [], []
        for handle, label in zip(*ax.get_legend_handles_labels()):
            if label not in labels:
                labels.append(label)
                handles.append(handle)
        ax.legend().remove()
        ax_leg.axis('off')
        ax_leg.legend(handles, labels, **legend_kw)

    # Adjust layout
    fig.subplots_adjust(**subplots_adjust_kw)

    plt.show()

    # Here, we do not divide any more by municipality / type because we focus
    # on the time variations (month/hour) of the consumption and production

    # Setup for these plots
    ylabel = f"Energia ({unit_energy})"
    xlabel = ''
    xticks = list(range(12)), months_labels
    grid_kw = dict(axis='y')  # (SETUP)
    ylabel = f"Energia ({unit_energy})"

    # Here we show how the end users and plants are distributed in classes
    # of power grouped by municipality

    # Setup for this plots
    figsize = (fig_width, fig_height)

    # Select data and labels to plot
    datas = [data_users_tou, data_plants_tou]
    cols_group = [ColumnName.MONTH, ColumnName.MONTH]
    cols_data = [BillsReader._time_of_use_energy_column_names, BillsReader._time_of_use_energy_column_names]
    bars_counts, legend_labels = [], []
    bars_labels = [list(range(12)), list(range(12))]
    for i, data in enumerate(datas):
        col_group = cols_group[i]
        col_data = cols_data[i]
        counts, labels = [], []
        for col, series in data.groupby(col_group).sum()[col_data].items():
            counts.append(list(series))
            labels.append(col)
        bars_counts.append(counts)
        legend_labels.append(labels)

    # Make figures and plot data
    for i, (counts, labels, legend) in enumerate(zip(bars_counts, bars_labels, legend_labels)):
        fig, ax = plt.subplots(figsize=figsize)

        # Do not plot zeros
        new_labels = [l for j, l in enumerate(legend) if any([count[j] > 0 for count in counts])]
        for j, l in enumerate(legend):
            if not any([count[j] > 0 for count in counts]):
                legend.pop(j)
                for count in counts:
                    count.pop(j)

        # Plot bars
        bottom = [0] * len(labels)
        for count, legend in zip(counts, legend):
            ax.bar(labels, count, label=legend, bottom=bottom, color=[colors_tou[legend]])
            bottom = [b + c for b, c in zip(bottom, count)]

        # Set ticks
        ax.set_xticks(*xticks)

        # Set labels
        ax.set_ylabel(ylabel)

        # Show grid
        ax.grid(**grid_kw)

        plt.show()

    # Here we use the same data to make one plot for each ToU tariff, to better
    # highlight how production and consumption are similar or not

    # Setup for this plots
    figsize = (fig_width, 2 * fig_height)
    width = 0.8
    align = 'center'
    ncols = 2
    nrows = sum([any([any([c > 0 for c in count[i]]) for count in bars_counts]) for i in
                 range(len(BillsReader._time_of_use_energy_column_names))])
    gridspec_kw = dict(width_ratios=[0.5, 0.5], height_ratios=[1 / nrows] * nrows)
    tickparams_kw = dict(axis='x', rotation=60)
    subplots_adjust_kw = dict(left=0.1, top=0.95, right=0.95, bottom=0.15, wspace=0.05, hspace=0.05)
    ylabel_kw = dict(x=0.03, y=0.5, s=ylabel, ha='center', va='center', rotation=90)
    legend_kw = dict(loc='lower center', ncol=4)
    titles = ['Consumo', 'Produzione']

    # Make figure
    fig = plt.figure(figsize=figsize)
    spec = gs.GridSpec(nrows, ncols, **gridspec_kw)
    axes = []
    for i in range(nrows):
        for j in range(ncols):
            if i == 0:
                sharex = None
                sharey = None if j == 0 else axes[-1]
            elif j == 0:
                sharex, sharey = axes[-2], None
            else:
                sharex, sharey = axes[-2], axes[-1]
            axes.append(fig.add_subplot(spec[i, j], sharex=sharex, sharey=sharey))

    # Plot data
    for j, (counts, labels, legends) in enumerate(zip(bars_counts, bars_labels, legend_labels)):
        k = 0
        for i, (count, legend) in enumerate(zip(counts, legends)):

            # Skip if empty
            if not any([any([c > 0 for c in cc[i]]) for cc in bars_counts]):
                continue

            # Select axis
            ax = axes[2 * k + j]

            # Plot bars
            ax.bar(labels, count, label=legend, align=align, width=width, color=colors_tou[legend])

            # Adjust ticks
            ax.tick_params(**tickparams_kw)
            ax.set_xticks(*xticks)
            if k < nrows - 1:
                ax.tick_params(axis='x', length=0)
                plt.setp(ax.get_xticklabels(), visible=False)
            if j == 1:
                ax.tick_params(axis='y', length=0)
                plt.setp(ax.get_yticklabels(), visible=False)
            ax.grid(**grid_kw)

            # Add title
            if k == 0:
                ax.set_title(titles[j])

            k += 1

    # Add ylabel to figure
    fig.text(**ylabel_kw)

    # Add legend to figure
    handles, labels = [], []
    for ax in axes:
        these_handles, these_labels = ax.get_legend_handles_labels()
        for handle, label in zip(these_handles, these_labels):
            if label not in labels:
                labels.append(label)
                handles.append(handle)
    fig.legend(handles, labels, **legend_kw)

    # Adjust subplots
    fig.subplots_adjust(**subplots_adjust_kw)

    plt.show()

    # ----------------------------------------------------------------------------
    # %% We analyze the hourly profiles

    # Setup for these plots
    titles = ['Inverno', 'Primavera', 'Estate', 'Autunno']
    lw = 2
    path_effect = lambda lw: [pe.Stroke(linewidth=1.5 * lw, foreground='w'), pe.Normal()]
    c_users = 'tab:blue'
    c_plants = 'tab:red'
    c_shared = 'tab:green'
    c_togrid = 'tab:red'
    c_fromgrid = 'tab:blue'
    ylabel = 'Potenza (kW)'
    xlabel = 'Tempo (h)'

    # Here we show some hourly profiles in typical weeks

    # Setup for this plot (SETUP)
    nrows = 2
    ncols = 2
    figsize = (fig_width, 1.5 * fig_height)
    xticks = range(0, 168, 24)
    tickparams_kw = dict(axis='x', rotation=0)
    subplots_adjust_kw = dict(top=0.92, right=0.98, bottom=0.15, left=0.12, wspace=0.05, hspace=0.25)
    ylabel_kw = dict(x=0.03, y=0.5, s=ylabel, ha='center', va='center', rotation=90)
    xlabel_kw = dict(x=0.5, y=0.03, s=xlabel, ha='center', va='center', )

    # Select data
    profiles_plants = {}
    data = data_plants_year.groupby(
        [ColumnName.SEASON, ColumnName.MONTH, ColumnName.DAY_OF_MONTH]).sum().reset_index().groupby(
        [ColumnName.SEASON, ColumnName.DAY_OF_WEEK]).mean()
    for season, df in data.groupby(ColumnName.SEASON):
        profile = df.loc[:, '0':].values.flatten()
        profiles_plants[season] = profile
    profiles_users = {}
    data = data_users_year.groupby(
        [ColumnName.SEASON, ColumnName.MONTH, ColumnName.DAY_OF_MONTH]).sum().reset_index().groupby(
        [ColumnName.SEASON, ColumnName.DAY_OF_WEEK]).mean()
    for season, df in data.groupby(ColumnName.SEASON):
        profile = df.loc[:, '0':].values.flatten()
        profiles_users[season] = profile

    c = 0
    for profiles in (profiles_plants, profiles_users):

        color = c_plants if c == 0 else c_users
        c += 1

        # Make figure
        fig, axes = plt.subplots(nrows, ncols, figsize=figsize, sharey=True, sharex=True)

        # Plot data
        for i, (season, profile) in enumerate(profiles.items()):
            # Select axis
            ax = axes.flatten()[i]
            # Plot profile
            ax.plot(profile, color=color, lw=lw, path_effects=path_effect(lw))
            # Set title
            ax.set_title(titles[i])
            # Set ticks
            ax.tick_params(**tickparams_kw)
            ax.set_xticks(xticks)
            # Add grid
            ax.grid()

        # Add labels
        fig.text(**xlabel_kw)
        fig.text(**ylabel_kw)

        # Adjust subplots
        fig.subplots_adjust(**subplots_adjust_kw)

        plt.show()

    # Here, we show the match between production and consumption in typical days

    # Setup for this plot (SETUP)
    nrows, ncols = 2, 2
    gridspec_kw = dict(width_ratios=[0.375, 0.375, 0.25])
    alpha = 0.5
    xticks = range(0, 24, 4)
    legend_kw = dict(loc='center', ncol=1)
    subplots_adjust_kw = dict(top=0.92, right=0.98, bottom=0.12, left=0.12, wspace=0.08, hspace=0.2)
    ylabel_kw = dict(x=0.03, y=0.5, s=ylabel, ha='center', va='center', rotation=90)
    xlabel_kw = dict(x=0.45, y=0.03, s=xlabel, ha='center', va='center', )

    # Select data
    profiles_plants = {}
    data = data_plants_year.groupby([ColumnName.SEASON, ColumnName.MONTH, ColumnName.DAY_OF_MONTH]).sum().reset_index()
    for season, df in data.groupby(ColumnName.SEASON):
        profile = df.mean().loc['0':].values
        profiles_plants[season] = profile
    profiles_users = {}
    data = data_users_year.groupby([ColumnName.SEASON, ColumnName.MONTH, ColumnName.DAY_OF_MONTH]).sum().reset_index()
    for season, df in data.groupby(ColumnName.SEASON):
        profile = df.mean().loc['0':].values
        profiles_users[season] = profile

    # Make figure
    fig = plt.figure(figsize=figsize)
    spec = gs.GridSpec(nrows, ncols + 1, **gridspec_kw)
    axes = []
    for i in range(len(titles)):
        if i == 0:
            axes.append(fig.add_subplot(spec[0, 0]))
        else:
            axes.append(fig.add_subplot(spec[i // ncols, i % ncols], sharex=axes[-1], sharey=axes[-1]))
    ax_leg = fig.add_subplot(spec[:, -1])

    # Plot data
    for i, season in enumerate(profiles_plants):

        # Select axis
        ax = axes[i]

        # Select data
        profile_plants = profiles_plants[season]
        profile_users = profiles_users[season]

        # Interpolate for graphiucal reasons
        t = np.arange(len(profile_plants))
        t0, tf = t[[0, -1]]
        t_plot = np.linspace(t0, tf, 10 * len(t))
        p_plants = np.interp(t_plot, t, profile_plants)
        p_users = np.interp(t_plot, t, profile_users)

        # Plot profiles
        ax.plot(t_plot, p_plants, label='$P_\mathrm{impianti}$', color=c_plants, lw=lw, path_effects=path_effect(lw))
        ax.plot(t_plot, p_users, label='$P_\mathrm{utenze}$', color=c_users, lw=lw, path_effects=path_effect(lw))
        ax.fill_between(t_plot, p_plants, p_users, where=p_plants < p_users, label='$E_\mathrm{\\leftarrow rete}$',
                        color=c_fromgrid, alpha=alpha)
        ax.fill_between(t_plot, p_plants, p_users, where=p_plants > p_users, label='$E_\mathrm{\\rightarrow rete}$',
                        color=c_togrid, alpha=alpha)
        ax.fill_between(t_plot, np.minimum(p_plants, p_users), label='$E_\mathrm{condivisa}$', color=c_shared, alpha=alpha)

        # Set ticks
        ax.tick_params(**tickparams_kw)
        ax.set_xticks(xticks)
        if i // ncols < 1:
            ax.tick_params(axis='x', length=0)
            plt.setp(ax.get_xticklabels(), visible=False)
        if i % ncols == 1:
            ax.tick_params(axis='y', length=0)
            plt.setp(ax.get_yticklabels(), visible=False)

        # Set title
        ax.set_title(titles[i])

        # Set grid
        ax.grid()

    # Make legend
    handles, labels = [], []
    for ax in axes:
        these_handles, these_labels = ax.get_legend_handles_labels()
        for handle, label in zip(these_handles, these_labels):
            if label in labels:
                continue
            labels.append(label)
            handles.append(handle)
    ax_leg.axis('off')
    ax_leg.legend(handles, labels, **legend_kw)

    # Add labels
    fig.text(**xlabel_kw)
    fig.text(**ylabel_kw)

    # Adjust subplots
    fig.subplots_adjust(**subplots_adjust_kw)

    plt.show()

    # ----------------------------------------------------------------------------
    # %% We analyze the results of the evaluation

    # Here, we show the initial analysis with the number of families

    # Select data
    data = data_results.loc[data_results['bess_size'] == 0]

    # Setup for this plot (SETUP)
    figsize = (fig_width, fig_height)
    x = list(data['n_fam'])
    y = [1 * round(sc * 100 / 1) for sc in list(data['sc'])]
    hlines = [h if y_ >= h else y_ for y_, h in zip(y, data['sc_target'] * 100)]
    vlines = x.copy()
    xticks = x.copy()
    yticks = hlines
    plot_kw = dict(lw=3, path_effects=path_effect(lw), zorder=1)
    s_min, s_max = 100, 500
    scatter_kw = dict(marker='s', s=[s_min + (s_max - s_min) * (x_ - min(x)) / (max(x) - min(x)) for x_ in x], zorder=2)
    line_kw = dict(color='lightgrey', lw=1.5, ls='--', zorder=3)
    arrow_kw = dict(color='lightgrey', lw=0.0005, ls='--', zorder=3)
    xlabel = 'Numero famiglie'
    ylabel = 'Autoconsumo (%)'
    subplots_adjust_kw = dict(top=0.98, right=0.98, left=0.11, bottom=0.19)

    # Make figure
    fig, ax = plt.subplots(figsize=figsize)

    # Plot data
    ax.plot(x, y, **plot_kw)
    ax.scatter(x, y, **scatter_kw, )

    # Adjust limits
    ax.set_ylim((None, ax.get_ylim()[1] * 1.05))  # (SETUP)

    # Set ticks
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)

    # Set labels
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    # Set grid
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()
    for i, (hline, vline) in enumerate(zip(hlines, vlines)):
        if i > 0:
            ax.plot((x_min, vline), (hline, hline), **line_kw)
            ax.plot((vline, vline), (y_min, hline), **line_kw)
            ax.annotate("", (vline, y_min), (vline, y_min + 0.01 * y_min), arrowprops=arrow_kw)
    ax.set_xlim((x_min, x_max))
    ax.set_ylim((y_min, y_max))

    # Adjust subplots
    fig.subplots_adjust(**subplots_adjust_kw)

    plt.show()

    # Here, we make two side-by-side plots showing the other two indicators, i.e.,
    # self-sufficiency and emissions reduction

    # Setup for this plot (SETUP)
    figsize = (fig_width, fig_height)
    nrows, ncols = 1, 2
    x = list(data['n_fam'])
    ys = [[1 * round(ss * 100 / 1) for ss in list(data['ss'])], [1 * round(esr * 100 / 1) for esr in list(data['esr'])], ]
    y_right = [1 * round(sc * 100 / 1) for sc in list(data['sc'])]
    xticks = x.copy()
    plot_kw = dict(lw=3, path_effects=path_effect(lw), zorder=1)
    s_min, s_max = 50, 200
    scatter_kw = dict(marker='s', s=[s_min + (s_max - s_min) * (x_ - min(x)) / (max(x) - min(x)) for x_ in x], zorder=2)
    line_kw = dict(color='tab:red', ls='--', lw=1.5)
    plot_right_kw = dict(lw=3, color='lightgrey', alpha=0.8, marker='s', )
    xlabel = 'Numero famiglie'
    ylabel = 'Indicatore (%)'
    titles = ['Autosufficienza', 'Riduzione emissioni CO$_\mathrm{2}$', ]
    tickparams_kw = dict(axis='x', rotation=60)
    xlabel_kw = dict(x=0.5, y=0.035, s=xlabel, ha='center', va='center')
    ylabel_kw = dict(x=0.025, y=0.55, s=ylabel, ha='center', va='center', rotation=90)
    subplots_adjust_kw = dict(top=0.9, right=0.98, left=0.1, bottom=0.23, wspace=0.2)

    # Make figure
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)

    # Plot data
    for i, y in enumerate(ys):
        # Select axis
        ax = axes[i]
        # ax = axtw.twinx()
        # ax.yaxis.set_label_position("left")
        # ax.yaxis.tick_left()
        # axtw.yaxis.set_label_position("right")
        # axtw.yaxis.tick_right()
        # ax.spines['right'].set_color(plot_right_kw['color'])

        # Plot lines
        ax.plot(x, y, **plot_kw)
        ax.scatter(x, y, **scatter_kw, )

        # # Plot SC on right axis
        # # axtw = ax.twinx()
        # axtw.plot(x, y_right, **plot_right_kw)
        # if i == 0:
        #     axtw.set_yticks([])

        # Adjust limits
        ax.set_ylim((None, ax.get_ylim()[1] * 1.05))  # (SETUP)

        # Set ticks
        ax.tick_params(**tickparams_kw)
        ax.set_xticks(xticks)

        # Set title
        ax.set_title(titles[i])

        # Set grid
        ylim = ax.get_ylim()
        ax.axhline(0, **line_kw)
        ax.set_ylim(ylim)
        ax.grid(zorder=-1)

    # Set labels
    fig.text(**xlabel_kw)
    fig.text(**ylabel_kw)

    # Adjust subplots
    fig.subplots_adjust(**subplots_adjust_kw)

    plt.show()

    # %% Here, we make two side-by-side plots showing two quantities, i.e.,
    # shared energy and capex

    # Setup for this plot (SETUP)
    figsize = (fig_width, fig_height)
    nrows, ncols = 1, 2
    x = range(len(data))
    ys = [list(data['e_sh'] / 1000), list(data['capex'] / 1000), ]
    xticks = x, [f"{int(n_fam)}" for n_fam in data['n_fam']]
    xlabel = 'Numero famiglie'
    ylabels = ['Energia condivisa (MWh)', 'CAPEX (k€)', ]
    tickparams_kw = dict(axis='x', rotation=60)
    xlabel_kw = dict(x=0.5, y=0.035, s=xlabel, ha='center', va='center')
    subplots_adjust_kw = dict(top=0.98, right=0.98, left=0.11, bottom=0.23, wspace=0.37)

    # Make figure
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)

    # Plot data
    for i, y in enumerate(ys):
        # Select axis
        ax = axes[i]

        # Plot lines
        ax.bar(x, y, )

        # Set ticks
        ax.tick_params(**tickparams_kw)
        ax.set_xticks(*xticks)

        # Set label
        ax.set_ylabel(ylabels[i])

        # Set grid
        ax.grid()

    # Set labels
    fig.text(**xlabel_kw)

    # Adjust subplots
    fig.subplots_adjust(**subplots_adjust_kw)

    plt.show()
    plt.close(fig)