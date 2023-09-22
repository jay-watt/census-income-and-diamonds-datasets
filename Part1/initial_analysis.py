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


def get_numerical_stats(df):
    selected_features = df.select_dtypes(include=['number'])
    stats = selected_features.describe().T
    variance = selected_features.var()
    feature_range = selected_features.max() - selected_features.min()
    stats['variance'] = variance
    stats['range'] = feature_range
    stats.index.name = 'feature'
    stats.drop(columns=['25%', '50%', '75%'], inplace=True)
    return stats


def get_categorical_stats(df):
    selected_features = df.select_dtypes(include=['object'])
    stats = selected_features.describe().T
    stats.index.name = 'feature'
    return stats


def display_stats(df):
    numerical_stats = get_numerical_stats(df)
    numerical_stats['count'] = numerical_stats['count'].map(
        lambda x: f'{x:,.0f}'
    )
    numerical_stats.iloc[:, 1:] = numerical_stats.iloc[:, 1:].map(
        lambda x: f'{x:,.2f}'
    )
    display_table('Numerical Summary Statistics', numerical_stats)

    categorical_stats = get_categorical_stats(df)
    categorical_stats[['count', 'unique', 'freq']] = categorical_stats[
        ['count', 'unique', 'freq']
    ].map(lambda x: f'{x:,.0f}')
    display_table('Categorical Summary Statistics', categorical_stats)


# Summary
def display_summary(df, class_name):
    numerical = (
        df.drop(columns=[class_name]).select_dtypes(include=['number']).columns
    )
    categorical = (
        df.drop(columns=[class_name]).select_dtypes(include=['object']).columns
    )
    summary = {
        'Instances': f'{len(df):,.0f}',
        'Features': len(categorical) + len(numerical),
        'Categorical': len(categorical),
        'Numerical': len(numerical),
    }
    summary = pd.DataFrame([summary]).T
    summary.columns = ['count']
    display_table('Dataset Summary: Features and Instances', summary)
    return class_name


# Class distribution
def format_ticks(x, pos):
    if abs(x - round(x)) < 1e-5:
        return '{:,.0f}'.format(x)
    else:
        return '{:,.0f}'.format(x) if x >= 1000 else '{:.2f}'.format(x)


def plot_histogram(feature, data, ax, bins):
    hist = sns.histplot(data=data, bins=bins, ax=ax)

    # Format
    hist.set_title(
        f"Histogram: {feature.capitalize()} Distribution", fontsize=12
    )
    hist.set_xlabel(feature.capitalize(), labelpad=10, fontsize=11)
    hist.set_ylabel("Frequency", labelpad=10, fontsize=11)
    hist.xaxis.set_major_formatter(ticker.FuncFormatter(format_ticks))
    hist.yaxis.set_major_formatter(ticker.FuncFormatter(format_ticks))
    hist.tick_params(labelsize=10)


def plot_boxplot(feature, data, ax):
    box = sns.boxplot(y=data, ax=ax)

    # Format
    box.set_title(f"Boxplot: {feature.capitalize()} Distribution", fontsize=12)
    box.set_xlabel("")
    box.set_ylabel(feature.capitalize(), labelpad=10, fontsize=11)
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


def plot_class_distribution(df, class_name):
    sns.set()
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 6))
    fig.subplots_adjust(
        left=0.1, right=0.975, top=0.925, bottom=0.12, wspace=0.35
    )
    bins = len(np.histogram_bin_edges(df[class_name], bins="fd")) - 1
    plot_histogram(class_name, df[class_name], axes[0], bins)
    plot_boxplot(class_name, df[class_name], axes[1])
    plt.savefig('../Reports/plots/part1_class_distribution.png')


# Numerical feature distributions
def plot_numerical_distributions(df, class_name):
    selected_features = df.select_dtypes(include=['number']).columns
    num_selected_features = (
        len(selected_features)
        if class_name not in selected_features
        else len(selected_features) - 1
    )
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
        bins = len(np.histogram_bin_edges(df[feature], bins="doane")) - 1
        if feature != class_name:
            plot_histogram(feature, df[feature], flat_axes[count], bins)
        else:
            count -= 1

        distributions_results[feature] = {
            'bins': bins,
            'skewness': skew(df[feature]),
            'kurtosis': kurtosis(df[feature]),
        }

    for i in range(count + 1, num_rows * num_cols):
        flat_axes[i].axis('off')
    plt.savefig('../Reports/plots/part1_numerical_distributions.png')
    return pd.DataFrame.from_dict(distributions_results, orient='index')


def display_numerical_distributions(df, class_name):
    distributions = plot_numerical_distributions(df, class_name)
    distributions['skewness shape'] = distributions['skewness'].apply(
        lambda x: 'Left skew'
        if x < -0.5
        else ('Symmetric' if -0.5 <= x <= 0.5 else 'Right skew')
    )
    distributions['kurtosis shape'] = distributions['kurtosis'].apply(
        lambda x: 'Platykurtic'
        if x < -0.5
        else ('Mesokurtic' if -0.5 <= x <= 0.5 else 'Leptokurtic')
    )
    distributions[['skewness', 'kurtosis']] = distributions[
        ['skewness', 'kurtosis']
    ].map(lambda x: f'{x:,.2f}')
    display_table(
        'Numerical Distributions: Bins, Skewness and Kurtosis',
        distributions,
    )


def analyse(df):
    display_stats(df)
    class_name = df.columns[-1]
    display_summary(df, class_name)
    plot_class_distribution(df, class_name)
    display_numerical_distributions(df, class_name)
