import math
import matplotlib.pyplot as plt
from matplotlib import ticker
import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis
import seaborn as sns
from sklearn.model_selection import train_test_split
from tabulate import tabulate
import warnings


warnings.filterwarnings('ignore', category=FutureWarning)
sns.set_theme(style="darkgrid", palette="deep")

DATA_PATH = '/home/j/machine-learning/a4/Part1/data/original/diamonds.csv'
SEED = 309
UNITS = {
    'price': 'USD',
    'carat': 'ct',
    'depth': '% width',
    'table': '% width',
    'x': 'mm',
    'y': 'mm',
    'z': 'mm',
    'volume': 'mm3',
}


# Handling data
def load_data():
    return pd.read_csv(DATA_PATH, index_col=0)


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
    data.to_csv(f'data/cleaned/{data_name}.csv', index=True)


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


# Plots
def plot_histogram(feature, data, ax):
    bins = (
        data.nunique()
        if data.nunique() < 10
        else len(np.histogram_bin_edges(data, bins='doane')) - 1
    )
    plot = sns.histplot(data=data, bins=bins, ax=ax)

    # Format

    plot.set_title(
        f'Histogram: {feature.capitalize()} Distribution', fontsize=11
    )
    xlabel = feature.capitalize() + (
        f' ({UNITS[feature]})' if feature in UNITS else ''
    )
    plot.set_xlabel(xlabel, labelpad=5, fontsize=10)
    plot.set_ylabel('Frequency', labelpad=10, fontsize=10)
    plot.xaxis.set_major_locator(ticker.FixedLocator(plot.get_xticks()))
    plot.xaxis.set_major_formatter(ticker.FuncFormatter(format_num_ticks))
    plot.yaxis.set_major_locator(ticker.FixedLocator(plot.get_yticks()))
    plot.yaxis.set_major_formatter(ticker.FuncFormatter(format_num_ticks))
    plot.tick_params(labelsize=9)

    return skew(data), kurtosis(data)


def plot_boxplot(feature, data, ax):
    plot = sns.boxplot(y=data, ax=ax)

    # Format
    plot.set_title(
        f'Boxplot: {feature.capitalize()} Distribution', fontsize=11
    )
    plot.set_xlabel('')
    plot.set_ylabel(
        f'{feature.capitalize()} ({UNITS[feature]})', labelpad=10, fontsize=10
    )
    plot.yaxis.set_major_locator(ticker.FixedLocator(plot.get_yticks()))
    plot.yaxis.set_major_formatter(ticker.FuncFormatter(format_num_ticks))
    plot.tick_params(labelsize=9)

    # Add annotations for Q1, median, Q3
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
    plt.figure(figsize=(10, 8))

    plot = sns.heatmap(
        data,
        annot=True,
        annot_kws={'size': 9},
        linewidths=0.5,
    )

    # Format
    plot.set_title('Heatmap: Feature-to-Feature Correlations', fontsize=11)
    plot.set_xticklabels(
        [label.get_text().capitalize() for label in plot.get_xticklabels()],
        rotation=45,
    )
    plot.set_yticklabels(
        [label.get_text().capitalize() for label in plot.get_yticklabels()],
        rotation=0,
    )
    plot.tick_params(labelsize=9)
    cbar = plot.collections[0].colorbar
    cbar.ax.tick_params(labelsize=9)
    plt.tight_layout()

    plt.savefig('../Reports/plots/part1_feature_correlations.png')


def plot_barplot(data):
    abs_data = data.abs()
    sorted_data = abs_data.sort_values(ascending=False)
    plt.figure(figsize=(10, 6))
    plot = sns.barplot(x=sorted_data.index, y=sorted_data.values)

    # Format
    plot.set_title(
        "Barplot: Pearson Correlation Coefficients with Class", fontsize=11
    )
    plot.set_ylabel("Correlation Coefficient", labelpad=10, fontsize=10)
    plot.set_xticklabels(
        [label.get_text().capitalize() for label in plot.get_xticklabels()],
        rotation=45,
    )
    plot.tick_params(labelsize=9)
    plt.tight_layout()

    plt.savefig('../Reports/plots/part1_class_correlations.png')


# Plotting helpers
def format_num_ticks(tick_label, pos):
    if abs(tick_label - round(tick_label)) < 1e-5:
        return '{:,.0f}'.format(tick_label)
    else:
        return (
            '{:,.0f}'.format(tick_label)
            if tick_label >= 1000
            else '{:.2f}'.format(tick_label)
        )


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
