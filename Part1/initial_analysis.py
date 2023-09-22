import matplotlib.pyplot as plt
import pandas as pd
from utils import (
    create_subplot_layout,
    interpret_kurt,
    interpret_skew,
    plot_boxplot,
    plot_histogram,
)

from utils import display_table


# Stats
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
        'Instances': f'{data.shape[0]:,.0f}',
        'Features': data.shape[1],
        'Categorical': len(categorical),
        'Numerical': len(numerical),
    }
    summary = pd.DataFrame([summary]).T
    summary.columns = ['count']
    display_table('Dataset Summary: Features and Instances', summary)
    return class_name


# Class distribution
def plot_class_distribution(class_name, class_data):
    axes = create_subplot_layout(2)

    skew, kurt = plot_histogram(class_name, class_data, axes[0])
    plot_boxplot(class_name, class_data, axes[1])

    plt.savefig('../Reports/plots/part1_class_distribution.png')

    print('Class Distribution')
    print(f'Skewness value:\t{skew:.2f}\tshape:\t{interpret_skew(skew)}')
    print(f'Kurtosis value:\t{kurt:.2f}\tshape:\t{interpret_kurt(kurt)}\n')


# Analysis main function
def analyse(data):
    display_stats(data)
    class_name = data.columns[-1]
    display_summary(class_name, data)
    plot_class_distribution(class_name, data[class_name])
    return class_name
