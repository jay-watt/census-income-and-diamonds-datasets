import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import chi2_contingency, pointbiserialr

from common.analysis import (
    display_feature_corrs,
    display_feature_shapes,
    get_feature_types,
    identify_missing_values,
    impute,
    remove_duplicates,
    summarise,
)
from common.config import ORIGINAL_DATA_DIR, PLOTS_DIR
from common.utils import (
    create_subplot_layout,
    display_table,
    plot_corr_barplot,
    plot_countplot,
    plot_distribution_barplot,
    plot_feature_to_class_scatter,
)
from part2.config import COLUMN_NAMES, DATA_FILENAMES, PLOTS_FILENAME


# Data cleaning
def remove_whitespaces(data):
    return data.applymap(lambda x: x.strip() if isinstance(x, str) else x)


def clean_data(data):
    cleaned_data = data.copy()
    cleaned_data = remove_whitespaces(cleaned_data)
    cleaned_data = remove_duplicates(cleaned_data)
    cleaned_data = identify_missing_values('?', cleaned_data)
    cleaned_data = impute(cleaned_data)
    return cleaned_data


def load_original_data():
    train = pd.read_csv(
        f'{ORIGINAL_DATA_DIR}/{DATA_FILENAMES["train"]}',
        header=None,
        names=COLUMN_NAMES,
    )
    test = pd.read_csv(
        f'{ORIGINAL_DATA_DIR}/{DATA_FILENAMES["test"]}',
        header=None,
        names=COLUMN_NAMES,
        skiprows=1,
    )

    print('Cleaning data...')
    train = train.applymap(lambda x: x.strip() if isinstance(x, str) else x)
    test = test.applymap(lambda x: x.strip() if isinstance(x, str) else x)
    train = clean_data(train)
    test = clean_data(test)
    print('Data cleaning complete!\n')


# Initial analysis
def plot_class_shape(data):
    class_ = data.columns[-1]
    plt.figure(figsize=(10, 6))
    plot_distribution_barplot(class_, data[class_])

    plt.savefig(f'{PLOTS_DIR}/{PLOTS_FILENAME}_class_distribution.png')


# Bivariate analysis
def plot_numerical_to_class(features, data, filename):
    axes = create_subplot_layout(len(features))

    for count, feature in enumerate(features):
        plot_countplot(data.columns[-1], feature, data, axes[count])

    plt.savefig(f'{PLOTS_DIR}/{filename}_bivariate_numerical.png')


def plot_categorical_to_class(features, data, filename):
    axes = create_subplot_layout(len(features))

    for count, feature in enumerate(features):
        plot_feature_to_class_scatter(feature, data[feature], axes[count])

    plt.savefig(f'{PLOTS_DIR}/{filename}_bivariate_categorical.png')


def plot_features_to_class(data, filename):
    numerical, categorical = get_feature_types(data)
    plot_numerical_to_class(numerical, data, filename)
    plot_categorical_to_class(categorical, data, filename)


# Multivariate analysis
def get_numerical_class_corrs(features, data):
    target_variable = data[data.columns[-1]]
    corrs = [
        pointbiserialr(data[feature], target_variable) for feature in features
    ]
    biserial = pd.DataFrame(
        {
            'feature': features,
            'Correlation Coefficient': [corr[0] for corr in corrs],
            'p-Value': [corr[1] for corr in corrs],
        }
    )
    return (
        biserial.set_index('feature')
        .abs()
        .sort_values(by='Correlation Coefficient')
    )


def display_numerical_class_corrs(corrs):
    corrs['Correlation Coefficient'] = corrs['Correlation Coefficient'].apply(
        '{:.2f}'.format
    )
    corrs.index = corrs.index + 1
    display_table('Numerical Feature Correlations with Class', corrs)


def get_categorical_class_corrs(features, data):
    target_variable = data[data.columns[-1]]
    chi2_results = []
    for feature in features:
        contingency_table = pd.crosstab(data[feature], target_variable)
        chi2, p, _, _ = chi2_contingency(contingency_table)
        chi2_results.append(
            {'feature': feature, 'Chi2 Value': chi2, 'p-Value': p}
        )
    chi2_results = pd.DataFrame(chi2_results)
    return chi2_results.sort_values(by='Chi2 Value', ascending=False)


def display_categorical_class_corrs(corrs):
    corrs.index = range(1, len(corrs) + 1)
    corrs.index.name = 'ranking'
    display_table('Categorical Feature Correlations with Class', corrs)


def plot_class_corrs(numerical_data, categorical_data, filename):
    axes = create_subplot_layout(2)
    plot_corr_barplot(
        numerical_data.set_index('feature')['Correlation Coefficient'], axes[0]
    )
    plot_corr_barplot(
        categorical_data.set_index('feature')['Chi2 Value'], axes[1]
    )
    plt.savefig(f'{PLOTS_DIR}/{filename}_numerical_class_corrs.png')


def display_class_corrs(data, filename):
    numerical, categorical = get_feature_types(data)
    numerical_corrs = get_numerical_class_corrs(numerical, data)
    categorical_corrs = get_categorical_class_corrs(categorical, data)
    display_numerical_class_corrs(numerical_corrs)
    display_categorical_class_corrs(categorical_corrs)
    plot_class_corrs(numerical_corrs, categorical_corrs, filename)


def display_corrs(data, filename):
    display_feature_corrs(data, filename)
    display_class_corrs(data, filename)


# Main analysis function
def run_analysis():
    # Data cleaning
    data, _ = load_original_data()

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
