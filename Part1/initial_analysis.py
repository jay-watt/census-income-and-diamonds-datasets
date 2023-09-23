import matplotlib.pyplot as plt
import pandas as pd
from config import PLOTS_DIR
from utils import (
    create_subplot_layout,
    display_table,
    interpret_kurt,
    interpret_skew,
    plot_boxplot,
    plot_histogram,
)


def get_stats_by_dtype(data, dtype):
    return data.select_dtypes(include=[dtype]).describe().T


def format_numerical_stats(stats):
    stats['count'] = stats['count'].apply(lambda x: f'{x:,.0f}')
    stats.iloc[:, 1:] = stats.iloc[:, 1:].applymap(lambda x: f'{x:,.2f}')


def format_categorical_stats(stats):
    stats[['count', 'unique', 'freq']] = stats[
        ['count', 'unique', 'freq']
    ].applymap(lambda x: f'{x:,.0f}')


def display_stats(data):
    numerical_stats = get_stats_by_dtype(data, 'number')
    numerical_stats['variance'] = numerical_stats.var()
    numerical_stats['range'] = numerical_stats.max() - numerical_stats.min()
    numerical_stats.drop(columns=['25%', '50%', '75%'], inplace=True)
    format_numerical_stats(numerical_stats)
    display_table('Numerical Summary Statistics', numerical_stats)

    categorical_stats = get_stats_by_dtype(data, 'object')
    format_categorical_stats(categorical_stats)
    display_table('Categorical Summary Statistics', categorical_stats)


def get_feature_types(data, class_name):
    numerical = (
        data.drop(columns=class_name).select_dtypes(include=['number']).columns
    )
    categorical = (
        data.drop(columns=class_name).select_dtypes(include=['object']).columns
    )
    return numerical, categorical


def display_summary(class_name, data):
    numerical, categorical = get_feature_types(data, class_name)
    summary = {
        'Instances': f'{data.shape[0]:,.0f}',
        'Features': data.shape[1] - 1,
        'Categorical': len(categorical),
        'Numerical': len(numerical),
    }
    summary = pd.DataFrame(summary, index=['count']).T
    display_table('Dataset Summary: Features and Instances', summary)


def plot_class_distribution(class_name, class_data):
    axes = create_subplot_layout(2)

    skew, kurt = plot_histogram(class_name, class_data, axes[0])
    plot_boxplot(class_name, class_data, axes[1])

    plt.savefig(f'{PLOTS_DIR}/part1_class_distribution.png')

    print('Class Distribution')
    print(f'Skewness value:\t{skew:.2f}\tshape:\t{interpret_skew(skew)}')
    print(f'Kurtosis value:\t{kurt:.2f}\tshape:\t{interpret_kurt(kurt)}\n')


def analyse(data):
    print('Initially analysing data...\n')
    display_stats(data)
    class_name = data.columns[-1]
    display_summary(class_name, data)
    plot_class_distribution(class_name, data[class_name])
    print('Initial analysis complete!\n')
    return class_name
