import math
import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from config import (
    CLEANED_DATA,
    DATA_DIR,
    KURT_THRESHOLD,
    ORIGINAL_DATA,
    PLOTS_DIR,
    SEED,
    SKEW_THRESHOLD,
    UNITS,
)
from matplotlib import ticker
from scipy.stats import kurtosis, skew
from sklearn.model_selection import train_test_split
from tabulate import tabulate

# Handle warnings
warnings.filterwarnings('ignore', category=FutureWarning)

# Apply seaborn settings
sns.set_theme(style="darkgrid", palette="deep")


# Data handlers
def load_original_data():
    return pd.read_csv(os.path.join(DATA_DIR, ORIGINAL_DATA), index_col=0)


def split_data(class_name, data):
    train_X, test_X, train_y, test_y = train_test_split(
        data.drop(columns=class_name),
        data[class_name],
        test_size=0.3,
        random_state=SEED,
    )
    train = pd.concat([train_X, train_y], axis=1)
    test = pd.concat([test_X, test_y], axis=1)
    return train, test


def write_cleaned_data(data, data_name):
    data.reset_index(drop=True, inplace=True)
    data.index = data.index + 1
    data.to_csv(os.path.join(DATA_DIR, CLEANED_DATA[data_name]), index=True)


def load_cleaned_data(class_name, data_name):
    data = pd.read_csv(os.path.join(DATA_DIR, CLEANED_DATA[data_name]))
    return data.drop(columns=class_name), data[class_name]


# Textual table creator
def display_table(title, data):
    print(f'{title}')
    print(
        tabulate(
            data,
            headers=[header.capitalize() for header in data.keys()],
            tablefmt='fancy_grid',
            showindex=True,
        )
    )
    print()


# Ploting helpers
def format_num_ticks(tick_label, pos):
    if abs(tick_label - round(tick_label)) < 1e-5:
        return '{:,.0f}'.format(tick_label)
    else:
        return (
            '{:,.0f}'.format(tick_label)
            if tick_label >= 1000
            else '{:.2f}'.format(tick_label)
        )


def format_axis(ax, data):
    ax.xaxis.set_major_locator(ticker.FixedLocator(ax.get_xticks()))
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(format_num_ticks))
    ax.yaxis.set_major_locator(ticker.FixedLocator(ax.get_yticks()))
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(format_num_ticks))


def format_plot(
    plot, title, xlabel, ylabel, xticklabels=None, yticklabels=None
):
    plot.set_title(title, fontsize=11)
    plot.set_xlabel(xlabel, labelpad=5, fontsize=10)
    plot.set_ylabel(ylabel, labelpad=10, fontsize=10)
    plot.tick_params(labelsize=9)

    if xticklabels is not None:
        plot.set_xticklabels(xticklabels, rotation=45)

    if yticklabels is not None:
        plot.set_yticklabels(yticklabels)


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
def plot_barplot(data):
    abs_data = data.abs()
    sorted_data = abs_data.sort_values(ascending=False)
    plt.figure(figsize=(10, 6))
    plot = sns.barplot(x=sorted_data.index, y=sorted_data.values)

    title = "Barplot: Pearson Correlation Coefficients with Class"
    ylabel = "Correlation Coefficient"
    xticklabels = [
        label.get_text().capitalize() for label in plot.get_xticklabels()
    ]
    format_plot(plot, title, '', ylabel, xticklabels)

    plt.tight_layout()

    plt.savefig(os.path.join(PLOTS_DIR, 'part1_class_correlations.png'))


def plot_boxplot(feature, data, ax):
    plot = sns.boxplot(y=data, ax=ax)

    title = f'Boxplot: {feature.capitalize()} Distribution'
    xlabel = ''
    ylabel = feature.capitalize() + (
        f' ({UNITS[feature]})' if feature in UNITS else ''
    )
    format_plot(plot, title, xlabel, ylabel)
    format_axis(ax, data)

    # Add annotations for Q1, median, Q3
    q1 = data.quantile(0.25)
    median = data.median()
    q3 = data.quantile(0.75)

    offset_x = 0.05
    offset_y = (q3 - q1) * 0.05

    ax.text(
        offset_x,
        q1 + offset_y,
        f'Q1: {q1:,.0f}',
        verticalalignment='bottom',
        horizontalalignment='left',
        fontsize=10,
    )
    ax.text(
        offset_x,
        median + offset_y,
        f'Median: {median:,.0f}',
        verticalalignment='bottom',
        horizontalalignment='left',
        fontsize=10,
    )
    ax.text(
        offset_x,
        q3 + offset_y,
        f'Q3: {q3:,.0f}',
        verticalalignment='bottom',
        horizontalalignment='left',
        fontsize=10,
    )


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
    format_plot(plot, title, xlabel, ylabel)
    format_axis(ax, data)

    return skew(data), kurtosis(data)


def plot_scatterplot(feature, class_name, data, ax):
    plot = sns.scatterplot(x=data[feature], y=data[class_name], ax=ax)

    feature_str = feature.capitalize()
    class_str = class_name.capitalize()
    title = f'Scatterplot: {feature_str} against {class_str}'
    xlabel = feature_str + (f' ({UNITS[feature]})' if feature in UNITS else '')
    ylabel = class_str + (f' ({UNITS[class_name]})')
    format_plot(plot, title, xlabel, ylabel)
    format_axis(ax, data)


# Distribution helpers
def interpret_skew(x):
    if x < -SKEW_THRESHOLD:
        return 'Left skew'
    elif -SKEW_THRESHOLD <= x <= SKEW_THRESHOLD:
        return 'Symmetric'
    else:
        return 'Right skew'


def interpret_kurt(x):
    if x < -KURT_THRESHOLD:
        return 'Platykurtic'
    elif -KURT_THRESHOLD <= x <= KURT_THRESHOLD:
        return 'Mesokurtic'
    else:
        return 'Leptokurtic'
