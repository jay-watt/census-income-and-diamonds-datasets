import matplotlib.pyplot as plt
import pandas as pd
from sklearn.discriminant_analysis import StandardScaler
from utils import (
    create_subplot_layout,
    display_table,
    interpret_skew,
    interpret_kurt,
    plot_heatmap,
    plot_histogram,
    plot_barplot,
    write_cleaned_data,
)


# Preprocessing
def encode(data):
    ordinal_mappings = {
        'cut': {
            'Fair': 1,
            'Good': 2,
            'Very Good': 3,
            'Premium': 4,
            'Ideal': 5,
        },
        'color': {'J': 1, 'I': 2, 'H': 3, 'G': 4, 'F': 5, 'E': 6, 'D': 7},
        'clarity': {
            'I1': 1,
            'SI2': 2,
            'SI1': 3,
            'VS2': 4,
            'VS1': 5,
            'VVS2': 6,
            'VVS1': 7,
            'IF': 8,
        },
    }
    for feature in ordinal_mappings:
        data[feature] = (
            data[feature].map(ordinal_mappings[feature]).astype(int)
        )
    return data


def remove_outliers(class_name, data):
    initial_instances = data.shape[0]
    for feature in data.drop(columns=[class_name]):
        Q1 = data[feature].quantile(0.25)
        Q3 = data[feature].quantile(0.75)
        IQR = Q3 - Q1

        # Defining bounds for outliers
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # Filter data
        data = data[
            (data[feature] >= lower_bound) & (data[feature] <= upper_bound)
        ]
    instances_removed = initial_instances - data.shape[0]
    print(
        f"Number of rows removed due to outliers:\t{instances_removed:,.0f}\n"
    )
    return data


def scale(class_name, data):
    class_data = data[class_name].reset_index(drop=True)
    scaler = StandardScaler()
    scaled = scaler.fit_transform(data.drop(columns=[class_name]))
    column_names = data.columns[data.columns != class_name]
    scaled_data = pd.DataFrame(scaled, columns=column_names).reset_index(
        drop=True
    )
    return pd.concat([scaled_data, class_data], axis=1)


# EDA
def plot_distributions(features, data):
    axes = create_subplot_layout(len(features))

    distributions_results = {}
    for count, feature in enumerate(features):
        skew, kurt = plot_histogram(feature, data[feature], axes[count])

        distributions_results[feature] = {
            'skewness value': skew,
            'kurtosis value': kurt,
        }

    plt.savefig('../Reports/plots/part1_distributions.png')

    return pd.DataFrame.from_dict(distributions_results, orient='index')


def display_distributions(features, data):
    distributions = plot_distributions(features, data)

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
    display_table('Numerical Distributions', distributions)


def display_correlation(data):
    corr_matrix = data.corr()
    corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i + 1, len(corr_matrix.columns)):
            feature1 = corr_matrix.columns[i]
            feature2 = corr_matrix.columns[j]
            correlation_value = corr_matrix.iloc[i, j]
            corr_pairs.append((feature1, feature2, correlation_value))

    plot_heatmap(corr_matrix)

    corr_pairs.sort(key=lambda x: abs(x[2]), reverse=True)
    corr_pairs = pd.DataFrame(
        corr_pairs, columns=['feature 1', 'feature 2', 'correlation']
    )
    corr_pairs['correlation'] = corr_pairs['correlation'].apply(
        lambda x: f'{x:,.2f}'
    )
    corr_pairs.index = corr_pairs.index + 1
    display_table('Highly Correlated Feature Pairs (>0.5)', corr_pairs.head(5))


def display_influence(class_name, data):
    corr = data.drop(columns=[class_name]).corrwith(data[class_name])
    plot_barplot(corr)

    corr = corr.sort_values(ascending=False, key=lambda x: abs(x))
    corr_df = pd.DataFrame(
        {'feature': corr.index, 'pearson coefficient': corr.values}
    )
    corr_df = corr_df[corr_df['pearson coefficient'].abs() > 0.5]
    corr_df['pearson coefficient'] = corr_df['pearson coefficient'].apply(
        lambda x: f'{x:.2f}'
    )
    corr_df.index = corr_df.index + 1
    display_table('Features Highly Correlated to Class (>0.5)', corr_df)


# Further preprocessing
def feature_engineer(data):
    data['volume'] = data['x'] * data['y'] * data['z']
    return data.drop(columns=['x', 'y', 'z'], inplace=True)


# Preprocessing and EDA main function
def preprocess(class_name, data, data_name):
    # Preprocess
    encoded_data = encode(data)
    display_distributions(data.drop(columns=[class_name]), data)
    inlier_data = remove_outliers(class_name, encoded_data)
    scaled_data = scale(class_name, inlier_data)

    # EDA
    if data_name == 'train':
        display_correlation(data.drop(columns=[class_name]))
        display_influence(class_name, data)

    # Further preprocessing
    print('transforming')
    transformed_data = feature_engineer(data)
    scaled_data = scale(class_name, transformed_data)

    # EDA
    if data_name == 'train':
        display_correlation(scaled_data.drop(columns=[class_name]))
        display_influence(class_name, scaled_data)

    write_cleaned_data(scaled_data, data_name)
