import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats import kurtosis, skew

from common.config import CORR_THRESHOLD, PLOTS_DIR
from common.utils import (
    create_subplot_layout,
    display_table,
    plot_distribution_barplot,
    plot_distribution_histogram,
    plot_heatmap,
)


# Data cleaning
def identify_missing_values(missing_signifier, data):
    print("***Imputing")
    cleaned_data = data.copy()
    cleaned_data.replace(missing_signifier, np.nan, inplace=True)
    display_missing_values(cleaned_data)
    return cleaned_data


def remove_duplicates(data):
    print("***Removing duplicates")
    cleaned_data = data.copy()
    print(f"Data shape before removing duplicates: {cleaned_data.shape}")
    duplicate_rows = cleaned_data[cleaned_data.duplicated()]
    print("Number of duplicate rows:", duplicate_rows.shape[0])
    cleaned_data.drop_duplicates(inplace=True)
    print(f"Data shape after removing duplicates: {cleaned_data.shape}\n")
    cleaned_data.reset_index(drop=True, inplace=True)
    return cleaned_data


def display_missing_values(data):
    missing = data.isnull().sum()
    missing = pd.DataFrame(missing[missing > 0])
    missing.columns = ['count']
    missing['percentage'] = (missing['count'] * 100 / len(data)).map(
        lambda x: f'{x:.2f}'
    )
    missing['count'] = missing['count'].apply(lambda x: f'{x:,.0f}')
    display_table('Missing Values Summary by Feature', missing)


def impute(data):
    cleaned_data = data.copy()
    imputed_features = []
    for feature in cleaned_data.columns:
        if cleaned_data[feature].isna().any():
            if np.issubdtype(cleaned_data[feature].dtype, np.number):
                cleaned_data[feature].fillna(
                    cleaned_data[feature].mean(), inplace=True
                )
            else:
                mode_val = cleaned_data[feature].mode().iloc[0]
                cleaned_data[feature].fillna(mode_val, inplace=True)
            imputed_features.append(feature)
    print(f'Features imputed: {", ".join(imputed_features)}\n')
    return cleaned_data


# Initial analysis
def display_summary(data):
    numerical, categorical = get_feature_types(data)
    summary = {
        'Instances': f'{data.shape[0]:,.0f}',
        'Features': data.shape[1] - 1,
        'Categorical': len(categorical),
        'Numerical': len(numerical),
    }
    summary = pd.DataFrame(summary, index=['count']).T
    summary.index.name = 'attribute'
    display_table('Dataset Summary: Features and Instances', summary)


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
    numerical_stats['variance'] = numerical_stats['std'] ** 2
    numerical_stats['range'] = numerical_stats['max'] - numerical_stats['min']
    numerical_stats.drop(columns=['25%', '50%', '75%'], inplace=True)
    format_numerical_stats(numerical_stats)
    display_table('Numerical Summary Statistics', numerical_stats)

    categorical_stats = get_stats_by_dtype(data, 'object')
    format_categorical_stats(categorical_stats)
    display_table('Categorical Summary Statistics', categorical_stats)


def get_feature_types(data):
    class_ = data.columns[-1]
    numerical = (
        data.drop(columns=class_).select_dtypes(include=['number']).columns
    )
    categorical = (
        data.drop(columns=class_).select_dtypes(include=['object']).columns
    )
    return numerical, categorical


def summarise(data):
    print('Analysing data...')
    display_summary(data)
    display_stats(data)


# Univariate analysis
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


def plot_numerical_shapes(features, data, filename):
    axes = create_subplot_layout(len(features))

    shape_results = []
    for count, feature in enumerate(features):
        feature_data = data[feature]
        plot_distribution_histogram(feature, feature_data, axes[count])
        shape_results.append(
            {
                'feature': feature,
                'Skewness Value': skew(feature_data),
                'Kurtosis Value': kurtosis(feature_data),
            }
        )

    plt.savefig(f'{PLOTS_DIR}/{filename}_univariate_numerical.png')
    return pd.DataFrame(shape_results)


def display_numerical_shapes(features, data, filename):
    shapes = plot_numerical_shapes(features, data, filename)
    shapes.set_index('feature', inplace=True)
    shapes['Skewness Shape'] = shapes['Skewness Value'].apply(interpret_skew)
    shapes['Kurtosis Shape'] = shapes['Kurtosis Value'].apply(interpret_kurt)
    shapes[['Skewness Value', 'Kurtosis Value']] = shapes[
        ['Skewness Value', 'Kurtosis Value']
    ].applymap('{:,.2f}'.format)
    shapes = shapes[
        [
            'Skewness Value',
            'Skewness Shape',
            'Kurtosis Value',
            'Kurtosis Shape',
        ]
    ]
    display_table('Numerical Feature Distributions', shapes)


def plot_categorical_shapes(features, data, filename):
    axes = create_subplot_layout(len(features))

    for count, feature in enumerate(features):
        plot_distribution_barplot(feature, data[feature], axes[count])

    plt.savefig(f'{PLOTS_DIR}/{filename}_univariate_categorical.png')


def display_feature_shapes(data, filename):
    numerical, categorical = get_feature_types(data)
    display_numerical_shapes(numerical, data, filename)
    plot_categorical_shapes(categorical, data, filename)


# Multivariate analysis
def get_corr_pairs(corr_matrix):
    corr_pairs = [
        (feature1, feature2, correlation_value)
        for i, feature1 in enumerate(corr_matrix.columns)
        for j, feature2 in enumerate(corr_matrix.columns[i + 1 :], start=i + 1)
        if abs(correlation_value := corr_matrix.iloc[i, j]) > CORR_THRESHOLD
    ]
    corr_pairs_df = pd.DataFrame(
        corr_pairs, columns=['feature 1', 'feature 2', 'correlation']
    )
    sorted_pairs = corr_pairs_df.sort_values(by='correlation', ascending=False)
    sorted_pairs = sorted_pairs.reset_index(drop=True)
    sorted_pairs.index = sorted_pairs.index + 1
    sorted_pairs.index.name = 'ranking'

    sorted_pairs['correlation'] = sorted_pairs['correlation'].apply(
        '{:,.2f}'.format
    )
    return sorted_pairs


def plot_feature_corrs(data, filename):
    plt.subplots(figsize=(10, 8))
    plot_heatmap(data)
    plt.tight_layout()
    plt.savefig(f'{PLOTS_DIR}/{filename}_multivariate.png')


def display_feature_corrs(data, filename):
    numerical, _ = get_feature_types(data)
    corr_matrix = data[numerical].corr()
    corr_pairs = get_corr_pairs(corr_matrix)

    plot_feature_corrs(corr_matrix, filename)

    if len(corr_pairs) > 0:
        title = f'Correlated Feature Pairs (> {CORR_THRESHOLD})'
        display_table(title, corr_pairs)
