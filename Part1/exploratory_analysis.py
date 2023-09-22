import matplotlib.pyplot as plt
import pandas as pd
from utils import (
    create_subplot_layout,
    display_table,
    interpret_skew,
    interpret_kurt,
    plot_heatmap,
    plot_histogram,
    plot_barplot,
)


# Feature distributions
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


# Feature correlations
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


# Feature influence
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


# Analysis main function
def analyse_cleaned_data(class_name, data):
    display_distributions(data.drop(columns=[class_name]).columns, data)
    display_correlation(data.drop(columns=[class_name]))
    display_influence(class_name, data)
