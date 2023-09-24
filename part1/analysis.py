import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import f_oneway

from common.analysis import (
    display_feature_corrs,
    display_feature_shapes,
    get_feature_types,
    identify_missing_values,
    impute,
    interpret_kurt,
    interpret_skew,
    remove_duplicates,
    summarise,
)
from common.config import ORIGINAL_DATA_DIR, PLOTS_DIR
from common.utils import (
    create_subplot_layout,
    display_table,
    plot_corr_barplot,
    plot_countplot,
    plot_distribution_boxplot,
    plot_distribution_histogram,
    plot_feature_to_class_boxplot,
)
from part1.config import DATA_FILENAME, PLOTS_FILENAME


# Data cleaning
def clean_data(data):
    print('Cleaning data...')
    cleaned_data = data.copy()
    cleaned_data = remove_duplicates(cleaned_data)
    cleaned_data = identify_missing_values(0, cleaned_data)
    cleaned_data = impute(cleaned_data)
    print('Data cleaning complete!\n')
    return cleaned_data


def load_original_data():
    data = pd.read_csv(f'{ORIGINAL_DATA_DIR}/{DATA_FILENAME}.csv', index_col=0)
    return clean_data(data)


# Initial analysis
def plot_class_shape(data):
    class_ = data.columns[-1]
    class_data = data[data.columns[-1]]
    axes = create_subplot_layout(2)

    skew, kurt = plot_distribution_histogram(class_, class_data, axes[0])
    plot_distribution_boxplot(class_, class_data, axes[1])

    plt.savefig(f'{PLOTS_DIR}/{PLOTS_FILENAME}_class_distribution.png')

    print('Class Distribution')
    print(f'Skewness value:\t{skew:.2f}\tshape:\t{interpret_skew(skew)}')
    print(f'Kurtosis value:\t{kurt:.2f}\tshape:\t{interpret_kurt(kurt)}\n')


# Bivariate analysis
def plot_numerical_to_class(features, data, filename):
    axes = create_subplot_layout(len(features))

    for count, feature in enumerate(features):
        plot_feature_to_class_boxplot(feature, data, axes[count])

    plt.savefig(f'{PLOTS_DIR}/{filename}_bivariate_numerical.png')


def plot_categorical_to_class(features, data, filename):
    axes = create_subplot_layout(len(features))

    for count, feature in enumerate(features):
        plot_countplot(feature, data.columns[-1], data, axes[count])

    plt.savefig(f'{PLOTS_DIR}/{filename}_bivariate_categorical.png')


def plot_features_to_class(data, filename):
    numerical, categorical = get_feature_types(data)
    plot_numerical_to_class(numerical, data, filename)
    plot_categorical_to_class(categorical, data, filename)


# Multivariate analysis
def get_numerical_class_corrs(features, data):
    feature_data = data[features]
    pearson = feature_data.corrwith(data[data.columns[-1]])

    pearson = pd.DataFrame(
        {
            'feature': pearson.index,
            'Pearson Coefficient': pearson.values,
        }
    )
    pearson['Pearson Coefficient'] = pearson['Pearson Coefficient'].abs()
    return pearson.sort_values(by="Pearson Coefficient", ascending=False)


def display_numerical_class_corrs(corrs):
    corrs['Pearson Coefficient'] = corrs['Pearson Coefficient'].apply(
        '{:.2f}'.format
    )
    corrs.index = range(1, len(corrs) + 1)
    corrs.index.name = 'ranking'
    display_table('Numerical Feature Correlations with Class', corrs)


def get_categorical_class_corrs(features, data):
    class_data = data[data.columns[-1]]

    anova_results = []
    for feature in features:
        categories = data[feature].unique()
        samples = [
            class_data[data[feature] == category] for category in categories
        ]
        f_stat, p_value = f_oneway(*samples)

        anova_results.append(
            {'feature': feature, 'F Statistic': f_stat, 'p-Value': p_value}
        )

    anova_results = pd.DataFrame(anova_results)
    return anova_results.sort_values(by='F Statistic', ascending=False)


def display_categorical_class_corrs(corrs):
    corrs.index = range(1, len(corrs) + 1)
    corrs.index.name = 'ranking'
    display_table(
        'Categorical Feature Correlations with Class',
        corrs,
    )


def plot_class_corrs(numerical_data, categorical_data, filename):
    axes = create_subplot_layout(2)
    plot_corr_barplot(numerical_data, 'Pearson Coefficient', axes[0])
    plot_corr_barplot(categorical_data, 'F Statistic', axes[1])
    plt.savefig(f'{PLOTS_DIR}/{filename}_numerical_class_corrs.png')


def display_class_corrs(data, filename):
    numerical, categorical = get_feature_types(data)
    numerical_corrs = get_numerical_class_corrs(numerical, data)
    categorical_corrs = get_categorical_class_corrs(categorical, data)
    plot_class_corrs(numerical_corrs, categorical_corrs, filename)
    display_numerical_class_corrs(numerical_corrs)
    display_categorical_class_corrs(categorical_corrs)


def display_corrs(data, filename):
    display_feature_corrs(data, filename)
    display_class_corrs(data, filename)


# Main analysis function
def run_analysis():
    # Data cleaning
    data = load_original_data()

    # Initial analysis
    summarise(data)

    # Univariate analysis
    plot_class_shape(data)
    display_feature_shapes(data, PLOTS_FILENAME)

    # Bivariate analysis
    plot_features_to_class(data, PLOTS_FILENAME)

    # Multivariate analysis
    display_corrs(data, PLOTS_FILENAME)
    print('Data analysis complete!\n')
