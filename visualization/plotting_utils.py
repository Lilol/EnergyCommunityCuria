from collections.abc import Iterable

import matplotlib
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Patch

alpha = 0.8
fig_width, fig_height = 0, 0
fontsize = 16


# General setup for plots
def init_plot_properties():
    cm_to_inch = 2.54
    global fig_width, fig_height
    fig_width = 16 / cm_to_inch  # inch
    fig_height = 8 / cm_to_inch  # inch
    global alpha
    alpha = 0.8
    matplotlib.rcParams.update({'font.size': fontsize, 'figure.figsize': (fig_width, fig_height)})


# Check if colormap is discrete
def is_cmap_discrete(cmap):
    """Check if a colormap is discrete or continuous."""
    return cmap.N < 256


# Get colors from colormap
def get_colors_from_map(n_colors, cmap):
    """Generate a list of colors from a colormap."""

    # Get colormap if not already provided
    if isinstance(cmap, str):
        cmap = plt.cm.get_cmap(cmap)

    # Check if colormap is discrete and get starting/end points for sampling
    if is_cmap_discrete(cmap):
        start, end = 0.5 / cmap.N, (n_colors - 0.5) / cmap.N
    else:
        start, end = 0, 1

    # Uniformly sample the colormaps between 0 and n_colors/n_thresh or 1
    colors = [cmap(i) for i in np.linspace(start, end, n_colors)]

    return colors


def get_colors(labels, colors):
    """
    Assigns colors to labels based on the given input.

    Args:
        labels (iterable): List-like object containing labels.
        colors (dict, str, or iterable): Colors mapping for labels.
            It can be a dictionary, string, or iterable.

    Returns:
        list: List of colors assigned to labels.

    Raises:
        ValueError: If the input is invalid or the number of colors doesn't
            match the number of labels.
    """

    # Manage cases for colors
    if isinstance(colors, dict):
        # Check if colors is a dictionary
        if labels is None:
            raise ValueError("Labels cannot be None when colors is a dict")
        if not set(labels).issubset(colors.keys()):
            raise ValueError("Labels should be a subset of keys in colors")
        colors = [colors[label] for label in labels]
    elif isinstance(colors, str):
        # Check if colors is a string
        pass  # Add additional validation logic here if needed
    elif isinstance(colors, Iterable):
        # Check if colors is an iterable (e.g., list, tuple)
        labels = (labels,) if not isinstance(labels, Iterable) else labels
        if len(colors) != len(labels):
            raise ValueError("Number of colors should match number of labels")
    else:
        raise ValueError("Invalid colors format")

    return colors


# Make a figure with one or two subplot(s)
def make_fig(make_legend=False, figsize=(12, 10), legend_loc='column', gridspec_kw=None):
    """
    Create a figure with one or two subplots based on the specified parameters.

    Args:
        make_legend (bool, optional): Flag to create a legend subplot.
            Default is False.
        figsize (tuple, optional): Figure size.
            It is specified as a tuple (width, height). Default is (12, 10).
        legend_loc (str, optional): Location of the legend subplots.
            Only used if make_legend is True. Can be 'column' or 'row'.
            Default is 'column'.
        gridspec_kw (dict, optional): Additional keyword arguments.
            Passed to the subplots creation. Default is an empty dictionary.

    Returns:
        fig (Figure): The created figure.
        ax (Axes or array-like of Axes): The main subplot(s).
        ax_leg (Axes or None): The legend subplot if make_legend is True,
            otherwise None.

    Raises:
        ValueError: If an invalid legend_loc value is provided.

    """
    # Determine the number of rows and columns for subplots based on
    # make_legend and legend_loc parameters
    if make_legend:
        if legend_loc == 'column':
            nrows, ncols = 1, 2
        elif legend_loc == 'row':
            nrows, ncols = 2, 1
        else:
            raise ValueError("Invalid legend_loc value. Should be 'column' or 'row'.")
    else:
        nrows, ncols = 1, 1

    # Create the figure and subplots based on the specified parameters
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize, gridspec_kw=gridspec_kw)

    # If there are multiple subplots, assign the first subplot to ax and the
    # second subplot to ax_leg
    if nrows * ncols > 1:
        ax, ax_leg = ax[0], ax[1]
    else:
        ax_leg = None

    return fig, ax, ax_leg


# Personalized pie/doughnut chart
def pie_chart(counts, labels=None, autopct='', labels_pos=None, pcts_pos=None, colors=None, ax=None, ax_leg=None,
              plot_zeros=False, **kwargs):
    # Manage optional parameters
    # # Fontsize for everything
    # fontsize = kwargs.pop('fontsize', 15)
    # Colormap
    cmap = kwargs.pop('cmap', None)
    #
    makefig_kw = {'figsize': (12, 10), 'make_legend': False, **kwargs.pop('makefig_kw', dict())}
    #
    # Annotation kwargs
    annotate_props = {'arrowprops': dict(arrowstyle='-'),  # 'fontsize': fontsize,
                      'bbox': dict(boxstyle="square,pad=0.3", fc='w', ec='k', lw=0.72), 'zorder': 0, 'va': 'center',
                      **kwargs.pop('annotate_props', dict())}
    # Legend kwargs
    legend_props = {'loc': 'center',  # 'fontsize': fontsize,
                    **kwargs.pop('legend_props', dict())}

    # Pie kwargs
    # kwargs = {'textprops': dict(fontsize=fontsize), **kwargs}

    # Colors
    if colors is not None:
        colors = get_colors(labels, colors)
    elif cmap is not None:
        colors = get_colors_from_map(len(labels), cmap=cmap)

    # Create axis
    if ax is None:
        fig, ax, ax_leg = make_fig(**makefig_kw)

    # Select data
    if not plot_zeros:
        counts, labels = zip(*[(count, label) for count, label in zip(counts, labels) if count > 0])

    # Make pie plot
    wedges, texts, autotexts = ax.pie(counts, labels=labels, colors=colors, autopct=autopct, **kwargs)

    #
    if pcts_pos == labels_pos is not None:
        for t, at in zip(texts, autotexts):
            t.set_text(' '.join((t.get_text(), at.get_text())))
            at.remove()
            pcts_pos = None

    #
    for text_list, pos in zip([texts, autotexts], [labels_pos, pcts_pos]):
        if pos is None:
            continue

        #
        elif pos == 'annotate':
            for i, p in enumerate(wedges):
                ang = (p.theta2 - p.theta1) / 2. + p.theta1
                y = np.sin(np.deg2rad(ang))
                x = np.cos(np.deg2rad(ang))
                horizontalalignment = {-1: "right", 1: "left"}[int(np.sign(x))]
                connectionstyle = f"angle,angleA=0,angleB={ang}"
                annotate_props["arrowprops"].update({"connectionstyle": connectionstyle})
                ax.annotate(text_list[i].get_text(), xy=(x, y), xytext=(1.35 * np.sign(x), 1.4 * y),
                            horizontalalignment=horizontalalignment, **annotate_props)

        #
        elif pos == 'legend':
            legend_elements = [Patch(label=text.get_text(), color=wedge.get_facecolor()) for text, wedge in
                               zip(text_list, wedges)]
            if ax_leg is None:
                ax.legend(handles=legend_elements, **legend_props)
            else:
                ax_leg.legend(handles=legend_elements, **legend_props)
                ax_leg.axis('off')

        #
        for text in text_list:
            text.remove()

    return ax

# #%%
# ax = pie_chart(sizes1,
#               #  labels=None,
#               # labels_pos='annotate', pcts_pos='legend',
#               # autopct=None,#lambda s: f"{int(s)}",
#               # radius=0.6,
#               # wedgeprops=dict(width=0.4)
# )
# ax = pie_chart(sizes2, labels=labels2,
#                ax=ax,
#               labels_pos='annotate', pcts_pos='annotate',
#               autopct=lambda s: f"({int(s)})",
#               radius=1,
#               wedgeprops=dict(width=0.2),
#                legend_props=dict(loc='center left'),
#                cmap='Greens'
# )
# plt.show()
# # pie_chart(sizes2, labels=labels2, autopct='', ax=ax, labels_pos='legend',
# #           radius=1, wedgeprops=dict(width=0.8))
# # plt.show()

# # ----------------------------------------------------------------------------
# # Data visualization
#
# # Setup of plotting parameters
# figsize = (10, 4)  # size of the figures (inch)
# fontsize = 12  # size of the font
# cmap = 'cool'
# color_map = {
#     'tab:blue': 'Blues',
#     'tab:red': 'Reds',
#     'tab:green': 'Greens',
#     'tab:orange': 'Oranges',
#     'tab:purple': 'Purples'
# }
# def_colors = list(color_map.keys())
# def_cmaps = list(color_map.values())
#
# #%%
# # Here, We get some insights on the data divided by municipality
#
# # Dictionary with color for each municipality
# colors = dict(zip(municipalities, get_colors_from_map(len(municipalities), cmap)))
#
# # Settings for this plot
# start_angle = 90  # angle at which the pie chart starts
# gridspec_kw = {'width_ratios': [1, 1, 0.5]}  # specifications for the grid
# datas = [data_users, data_plants]
# columns = [ColumnName.MUNICIPALITY, ColumnName.MUNICIPALITY]
# titles = ['Utenze comunali', 'Impianti PV']
#
#
# # Here we plot the number of end users and plants for each municipality
# fig, axes = plt.subplots(1, 3, figsize=figsize, gridspec_kw=gridspec_kw)
# axes, ax_leg = axes[:-1], axes[-1]
#
# # Plot the two pie charts
# for i, data in enumerate(datas):
#     # Evaluate data
#     label_counts = data[columns[i]].value_counts().sort_index()
#     labels, counts = list(label_counts.index), list(label_counts)
#     # Select axis
#     ax = axes[i]
#     # Create the pie chart
#     colorz = [colors[label] for label in labels]
#     autopct = lambda p: '{:.0f}'.format((p/100) * sum(counts))
#     patches, texts, _= ax.pie(counts, labels=labels, autopct=autopct,
#                           colors=colorz, startangle=start_angle)
#
#     # Hide the labels
#     for text in texts:
#         text.remove()
#
#     # Add title
#     ax.set_title(titles[i], fontsize=fontsize)
#
# # Add a legend
# ax_leg.axis('off')
# legend_elements = [Patch(label=label, color=colors[label]) for label in labels]
# ax_leg.legend(handles=legend_elements, fontsize=fontsize, loc='center')
#
# # Here we plot the number of end users and plants for each municipality
#
# # Settings for this plot
# start_angle = 90  # angle at which the pie chart starts
# gridspec_kw = {'width_ratios': [1, 1, 0.5]}  # specifications for the grid
# datas = [data_users, data_plants]
# columns = [col_user_energy, col_plant_energy]
# titles = ['Consumo', 'Produzione']
# divide_by = 1000
#
#
# fig, axes = plt.subplots(1, 3, figsize=figsize, gridspec_kw=gridspec_kw)
# axes, ax_leg = axes[:-1], axes[-1]
#
# # Plot the two pie charts
# for i, data in enumerate(datas):
#     col = columns[i]
#     # Evaluate data
#     label_counts = dict(data.groupby(ColumnName.MUNICIPALITY)[col].sum() / divide_by)
#     labels, counts = list(label_counts.keys()), list(label_counts.values())
#     # Select axis
#     ax = axes[i]
#     # Create the pie chart
#     colorz = [colors[municipality] for municipality in labels]
#     autopct = lambda p: '{:.0f}'.format((p/100) * sum(counts))
#     patches, texts, _= ax.pie(counts, labels=labels, autopct=autopct,
#                           colors=colorz, startangle=start_angle)
#
#     # Hide the labels
#     for text in texts:
#         text.remove()
#
#     # Add title
#     ax.set_title(titles[i], fontsize=fontsize)
#
# # Add a legend
# ax_leg.axis('off')
# legend_elements = [Patch(label=label, color=colors[label]) for label in labels]
# ax_leg.legend(handles=legend_elements, fontsize=fontsize, loc='center')
#
# plt.show()
#
# #%%
# # Here, We get some insights on the data
#
# # Settings for this plot
# start_angle = 90  # angle at which the pie chart starts
# gridspec_kw = {}  # specifications for the grid
# titles = ['Utenze comunali', 'Impianti PV']
#
#
# # Here we plot the number of end users and plants for each municipality
# fig, axes = plt.subplots(1, 2, figsize=figsize, gridspec_kw=gridspec_kw)
# axes, ax_leg = axes[:-1], axes[-1]
#
# # Plot the two pie charts
#
# # Evaluate data
# label_counts = data_users[col_user_type].value_counts()
# labels, counts = list(label_counts.index), list(label_counts)
# # Select axis
# ax = axes[0]
# # Create the pie chart
# # colorz = [colors[label] for label in labels]
# autopct = lambda p: '{:.0f}'.format((p/100) * sum(counts))
# patches, texts, _= ax.pie(counts, labels=labels, autopct=autopct,
#                           startangle=start_angle)
#
#
# # Add title
# ax.set_title(titles[0], fontsize=fontsize)
#
# plt.show()
