import math
from matplotlib import ticker
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate
from scipy.stats import skew, kurtosis
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)


UNITS = {
    'price': 'USD',
    'carat': 'ct',
    'depth': '% total width',
    'table': '% total width',
    'x': 'mm',
    'y': 'mm',
    'z': 'mm',
}


# Stats
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


def get_numerical_stats(data):
    selected_features = data.select_dtypes(include=['number'])
    stats = selected_features.describe().T
    variance = selected_features.var()
    feature_range = selected_features.max() - selected_features.min()
    stats['variance'] = variance
    stats['range'] = feature_range
    stats.index.name = 'feature'
    stats.drop(columns=['25%', '50%', '75%'], inplace=True)
    return stats


def get_categorical_stats(data):
    selected_features = data.select_dtypes(include=['object'])
    stats = selected_features.describe().T
    stats.index.name = 'feature'
    return stats


def display_stats(data):
    numerical_stats = get_numerical_stats(data)
    numerical_stats['count'] = numerical_stats['count'].map(
        lambda x: f'{x:,.0f}'
    )
    numerical_stats.iloc[:, 1:] = numerical_stats.iloc[:, 1:].map(
        lambda x: f'{x:,.2f}'
    )
    display_table('Numerical Summary Statistics', numerical_stats)

    categorical_stats = get_categorical_stats(data)
    categorical_stats[['count', 'unique', 'freq']] = categorical_stats[
        ['count', 'unique', 'freq']
    ].map(lambda x: f'{x:,.0f}')
    display_table('Categorical Summary Statistics', categorical_stats)


# Summary
def display_summary(class_name, data):
    numerical = (
        data.drop(columns=[class_name])
        .select_dtypes(include=['number'])
        .columns
    )
    categorical = (
        data.drop(columns=[class_name])
        .select_dtypes(include=['object'])
        .columns
    )
    summary = {
        'Instances': f'{len(data):,.0f}',
        'Features': len(categorical) + len(numerical),
        'Categorical': len(categorical),
        'Numerical': len(numerical),
    }
    summary = pd.DataFrame([summary]).T
    summary.columns = ['count']
    display_table('Dataset Summary: Features and Instances', summary)
    return class_name


# Class distribution
def format_ticks(tick_label, pos):
    if abs(tick_label - round(tick_label)) < 1e-5:
        return '{:,.0f}'.format(tick_label)
    else:
        return (
            '{:,.0f}'.format(tick_label)
            if tick_label >= 1000
            else '{:.2f}'.format(tick_label)
        )


def plot_histogram(feature, data, ax):
    bins = len(np.histogram_bin_edges(data, bins="doane")) - 1
    hist = sns.histplot(data=data, bins=bins, ax=ax)

    # Format
    hist.set_title(
        f"Histogram: {feature.capitalize()} Distribution", fontsize=12
    )
    hist.set_xlabel(
        f'{feature.capitalize()} ({UNITS[feature]})', labelpad=10, fontsize=11
    )
    hist.set_ylabel("Frequency", labelpad=10, fontsize=11)
    hist.xaxis.set_major_formatter(ticker.FuncFormatter(format_ticks))
    hist.yaxis.set_major_formatter(ticker.FuncFormatter(format_ticks))
    hist.tick_params(labelsize=10)

    return skew(data), kurtosis(data)


def plot_boxplot(feature, data, ax):
    box = sns.boxplot(y=data, ax=ax)

    # Format
    box.set_title(f"Boxplot: {feature.capitalize()} Distribution", fontsize=12)
    box.set_xlabel("")
    box.set_ylabel(
        f'{feature.capitalize()} ({UNITS[feature]})', labelpad=10, fontsize=11
    )
    box.yaxis.set_major_formatter(ticker.FuncFormatter(format_ticks))
    box.tick_params(labelsize=10)

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
        fontsize=11,
    )
    ax.text(
        offset_x,
        median + offset_y,
        f'Median: {median:.2f}',
        verticalalignment='bottom',
        horizontalalignment='left',
        fontsize=11,
    )
    ax.text(
        offset_x,
        q3 - offset_y,
        f'Q3: {q3:,.2f}',
        verticalalignment='top',
        horizontalalignment='left',
        fontsize=11,
    )


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


def plot_class_distribution(class_name, class_data):
    sns.set()
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 6))
    fig.subplots_adjust(
        left=0.125, right=0.975, top=0.925, bottom=0.12, wspace=0.35
    )
    skew, kurt = plot_histogram(class_name, class_data, axes[0])
    plot_boxplot(class_name, class_data, axes[1])
    plt.savefig('../Reports/plots/part1_class_distribution.png')
    print('Class Distribution')
    print(f'Skewness value:\t{skew:.2f}\tshape:\t{interpret_skew(skew)}')
    print(f'Kurtosis value:\t{kurt:.2f}\tshape:\t{interpret_kurt(kurt)}\n')


# Numerical feature distributions
def plot_numerical_distributions(data):
    selected_features = data.select_dtypes(include=['number']).columns
    num_selected_features = len(selected_features)
    num_cols = 3 if num_selected_features >= 9 else 2
    num_rows = math.ceil(num_selected_features / num_cols)

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(10, 12))
    fig.subplots_adjust(
        left=0.125,
        right=0.975,
        top=0.965,
        bottom=0.065,
        hspace=0.4,
        wspace=0.35,
    )
    flat_axes = axes.ravel()

    distributions_results = {}
    for count, feature in enumerate(selected_features):
        skew, kurt = plot_histogram(feature, data[feature], flat_axes[count])

        distributions_results[feature] = {
            'skewness value': skew,
            'kurtosis value': kurt,
        }

    for i in range(count + 1, num_rows * num_cols):
        flat_axes[i].axis('off')

    plt.savefig('../Reports/plots/part1_numerical_distributions.png')

    return pd.DataFrame.from_dict(distributions_results, orient='index')


def display_numerical_distributions(data):
    distributions = plot_numerical_distributions(data)
    distributions['skewness shape'] = distributions['skewness value'].apply(
        interpret_skew
    )
    distributions['kurtosis shape'] = distributions['kurtosis value'].apply(
        interpret_kurt
    )
    distributions[['skewness value', 'kurtosis value']] = distributions[
        ['skewness value', 'kurtosis value']
    ].map(lambda x: f'{x:,.2f}')
    distributions = distributions[
        [
            'skewness value',
            'skewness shape',
            'kurtosis value',
            'kurtosis shape',
        ]
    ]
    display_table(
        'Numerical Distributions',
        distributions,
    )


def analyse(data):
    display_stats(data)
    class_name = data.columns[-1]
    display_summary(class_name, data)
    plot_class_distribution(class_name, data[class_name])
    display_numerical_distributions(data.drop(columns=[class_name]))
