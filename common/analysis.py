import pandas as pd
from common.utils import display_table


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
    return data.columns[-1]


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
    summary.index.name = 'attribute'
    display_table('Dataset Summary: Features and Instances', summary)


def display_missing_values(data):
    missing = data.isnull().sum()
    missing = pd.DataFrame(missing[missing > 0])
    missing.columns = ['count']
    missing['percentage'] = (missing['count'] * 100 / len(data)).map(
        lambda x: f'{x:.2f}'
    )
    missing['count'] = missing['count'].apply(lambda x: f'{x:,.0f}')
    display_table('Missing Values Summary by Feature', missing)


def summarise(data):
    class_name = display_stats(data)
    display_summary(class_name, data)
    display_missing_values(data)
    return class_name
