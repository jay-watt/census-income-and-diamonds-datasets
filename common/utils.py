import math
import warnings

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib import ticker
from scipy.stats import kurtosis, skew
from tabulate import tabulate

from common.config import UNITS

warnings.filterwarnings('ignore', category=FutureWarning)
sns.set_theme(style='darkgrid', palette='deep')


# Table creator
def display_table(title, data):
    print(f'{title}')
    index_name = data.index.name.capitalize() if data.index.name else 'Feature'
    print(
        tabulate(
            data,
            headers=[index_name]
            + [header.capitalize() for header in data.columns],
            tablefmt='fancy_grid',
            showindex='always',
        )
    )
    print()


# Plot helpers
def num_tick_formatter(tick_label, pos):
    if abs(tick_label - round(tick_label)) < 1e-5:
        return '{:,.0f}'.format(tick_label)
    else:
        return (
            '{:,.0f}'.format(tick_label)
            if tick_label >= 1000
            else '{:.2f}'.format(tick_label)
        )


def format_num_ticks(plot, axis_str):
    locator = ticker.FixedLocator(getattr(plot, f'get_{axis_str}ticks')())
    formatter = ticker.FuncFormatter(num_tick_formatter)
    getattr(plot, f'{axis_str}axis').set_major_locator(locator)
    getattr(plot, f'{axis_str}axis').set_major_formatter(formatter)


def format_text_ticks(plot, axis_str):
    locator = ticker.FixedLocator(getattr(plot, f'get_{axis_str}ticks')())
    getattr(plot, f'{axis_str}axis').set_major_locator(locator)
    getter = getattr(plot, f'get_{axis_str}ticklabels')
    labels = [label.get_text().capitalize() for label in getter()]
    setter = getattr(plot, f'set_{axis_str}ticklabels')
    setter(labels, rotation=45 if axis_str == 'x' else 0)


def format_plot(plot, title, xlabel, ylabel, x_isnum, y_isnum):
    plot.set_title(title, fontsize=11)
    plot.set_xlabel(xlabel, labelpad=5, fontsize=10)
    plot.set_ylabel(ylabel, labelpad=10, fontsize=10)
    format_num_ticks(plot, 'x') if x_isnum else format_text_ticks(plot, 'x')
    format_num_ticks(plot, 'y') if y_isnum else format_text_ticks(plot, 'y')
    plot.tick_params(labelsize=9)


def create_subplot_layout(num_plots):
    num_cols = 3 if (num_plots >= 9 or num_plots % 3 == 0) else 2
    num_rows = math.ceil(num_plots / num_cols)

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(10, num_rows * 4))

    if num_plots == 1:
        return [axes]

    flat_axes = axes.ravel()

    for i in range(num_plots + 1, num_rows * num_cols):
        flat_axes[i].axis('off')

    fig.subplots_adjust(
        left=0.1,
        right=0.98,
        top=1 - 0.1 / num_rows,
        bottom=0.2 / num_rows,
        hspace=0.15 * num_rows,
        wspace=0.15 * num_cols,
    )
    return flat_axes


# Plots
def plot_distribution_barplot(feature, data, ax=None):
    plot = sns.barplot(x=data.index, y=data.values, ax=ax)

    title = f'Barplot: {feature.capitalize()} Distribution'
    format_plot(plot, title, feature.capitalize(), 'Count', False, True)


def plot_corr_barplot(data, y_name, ax):
    plot = sns.barplot(x=data['feature'], y=data[y_name], ax=ax)

    title = 'Barplot: Feature Correlations with Class'
    format_plot(plot, title, 'Feature', y_name, False, True)


def plot_distribution_boxplot(feature, data, ax):
    plot = sns.boxplot(data=data, ax=ax)

    title = f'Boxplot: {feature.capitalize()} Distribution'
    ylabel = (
        f'{feature.capitalize()} ({UNITS[feature]})'
        if feature in UNITS
        else feature.capitalize()
    )
    format_plot(plot, title, '', ylabel, True, True)


def plot_heatmap(data):
    plot = sns.heatmap(
        data,
        annot=True,
        annot_kws={'size': 9},
        linewidths=0.5,
    )

    title = 'Heatmap: Feature-to-Feature Correlations'
    format_plot(plot, title, '', '', False, False)

    cbar = plot.collections[0].colorbar
    cbar.ax.tick_params(labelsize=9)


def plot_distribution_histogram(feature, data, ax):
    bins = (
        data.nunique()
        if data.nunique() < 10
        else len(np.histogram_bin_edges(data, bins='doane')) - 1
    )
    plot = sns.histplot(data=data, bins=bins, ax=ax)

    title = f'Histogram: {feature.capitalize()} Distribution'
    xlabel = feature.capitalize() + (
        f' ({UNITS[feature]})' if feature in UNITS else ''
    )
    ylabel = 'Frequency'
    format_plot(plot, title, xlabel, ylabel, True, True)
    return skew(data), kurtosis(data)


def plot_feature_to_class_scatter(feature, data, ax):
    class_ = data.columns[-1]
    plot = sns.scatterplot(x=feature, y=class_, data=data, ax=ax)

    feature_str = feature.capitalize()
    class_str = class_.capitalize()
    title = f'Scatterplot: {feature_str} against {class_str}'
    xlabel = feature_str + (f' ({UNITS[feature]})' if feature in UNITS else '')
    format_plot(plot, title, xlabel, class_str, True, True)


def plot_feature_to_class_boxplot(feature, data, ax):
    class_ = data.columns[-1]
    plot = sns.boxplot(x=feature, y=class_, data=data, ax=ax)

    feature_str = feature.capitalize()
    title = f'Boxplot: {feature_str}'
    xlabel = feature_str + (f' ({UNITS[feature]})' if feature in UNITS else '')
    format_plot(plot, title, xlabel, class_.capitalize(), False, True)


def plot_countplot(x, hue, data, ax):
    plot = sns.countplot(x=x, hue=hue, data=data, ax=ax)

    x_str = x.capitalize()
    title = f'Countplot: {x_str}'
    xlabel = x_str + (f' ({UNITS[x]})' if x in UNITS else '')
    format_plot(plot, title, xlabel, hue.capitalize(), False, True)
