import math
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from common.config import CLEANED_DATA_DIR, UNITS
from matplotlib import ticker
from scipy.stats import kurtosis, skew
from tabulate import tabulate

warnings.filterwarnings('ignore', category=FutureWarning)
sns.set_theme(style="darkgrid", palette="deep")


# Handling data
def write_cleaned_data(data, filename):
    data.reset_index(drop=True, inplace=True)
    data.index = data.index + 1
    data.to_csv(f'{CLEANED_DATA_DIR}/{filename}.csv', index=True)


def load_cleaned_data(class_name, filename):
    df = pd.read_csv(f'{CLEANED_DATA_DIR}/{filename}.csv', index_col=0)
    return df.drop(columns=class_name), df[class_name]


# Textual table creator
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


# Plots
def plot_barplot(data):
    abs_data = data.abs()
    sorted_data = abs_data.sort_values(ascending=False)
    plot = sns.barplot(x=sorted_data.index, y=sorted_data.values)

    title = "Barplot: Pearson Correlation Coefficients with Class"
    xlabel = 'Feature'
    ylabel = "Correlation Coefficient"
    format_plot(plot, title, xlabel, ylabel, False, True)


def plot_boxplot(feature, data, ax):
    plot = sns.boxplot(y=data, ax=ax)

    title = f'Boxplot: {feature.capitalize()} Distribution'
    ylabel = (
        f'{feature.capitalize()} ({UNITS[feature]})'
        if feature in UNITS
        else feature.capitalize()
    )
    format_plot(plot, title, '', ylabel, True, True)

    q1 = data.quantile(0.25)
    median = data.median()
    q3 = data.quantile(0.75)

    offset_x = 0.025
    offset_y = (q3 - q1) * 0.05

    ax.text(
        offset_x,
        q1 + offset_y,
        f'Q1: {q1:,.2f}',
        verticalalignment='bottom',
        horizontalalignment='left',
        fontsize=10,
    )
    ax.text(
        offset_x,
        median + offset_y,
        f'Median: {median:.2f}',
        verticalalignment='bottom',
        horizontalalignment='left',
        fontsize=10,
    )
    ax.text(
        offset_x,
        q3 - offset_y,
        f'Q3: {q3:,.2f}',
        verticalalignment='top',
        horizontalalignment='left',
        fontsize=10,
    )


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


def plot_histogram(feature, data, ax):
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


def plot_scatterplot(feature, class_name, data, ax):
    plot = sns.scatterplot(x=feature, y=class_name, data=data, ax=ax)

    feature_str = feature.capitalize()
    title = f"Scatterplot: {feature_str} against {class_name.capitalize()}"
    xlabel = feature_str + (f' ({UNITS[feature]})' if feature in UNITS else '')
    ylabel = class_name.capitalize()
    format_plot(plot, title, xlabel, ylabel, True, True)


# Plotting helpers
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
    locator = ticker.FixedLocator(getattr(plot, f"get_{axis_str}ticks")())
    formatter = ticker.FuncFormatter(num_tick_formatter)
    getattr(plot, f"{axis_str}axis").set_major_locator(locator)
    getattr(plot, f"{axis_str}axis").set_major_formatter(formatter)


def format_text_ticks(plot, axis_str):
    setter = getattr(plot, f"set_{axis_str}ticklabels")
    getter = getattr(plot, f"get_{axis_str}ticklabels")
    setter(
        [label.get_text().capitalize() for label in getter()],
        rotation=45 if axis_str == 'x' else 0,
    )


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


# Distribution helpers
def interpret_skew(x):
    if x < -0.5:
        return 'Left skew'
    elif -0.5 <= x <= 0.5:
        return 'Symmetric'
    else:
        return 'Right skew'


def interpret_kurt(x):
    if x < -0.5:
        return 'Platykurtic'
    elif -0.5 <= x <= 0.5:
        return 'Mesokurtic'
    else:
        return 'Leptokurtic'
