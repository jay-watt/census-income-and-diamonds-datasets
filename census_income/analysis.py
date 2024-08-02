import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.stats import pointbiserialr

from common.preprocessing import (calculate_cramers_v, condense_tables,
                                  create_plot_layout, export_and_show_plot,
                                  format_plot_axes, get_feature_types, export_and_print_table,
                                  run_feature_correlation_analysis,
                                  run_univariate_numerical_analysis,
                                  summarise)
from common.config import PALETTE
from census_income.cleaning import balance_class_categories, load_original_data

sns.set_theme(style='darkgrid', palette=PALETTE)


# Category reduction functions
def get_top_n_categories(df, feature, n):
    counts = df[feature].value_counts()
    top_categories = counts.head(n)

    if len(counts) > n:
        top_categories.loc['Other'] = counts.tail(len(counts) - n).sum()

    return top_categories.sort_values(ascending=False)


def reduce_to_n_categories(df, feature, n):
    reduced_df = df.copy()

    # Get top categories
    top_categories = get_top_n_categories(df, feature, n - 1)

    # Replace categories not in top_categories with 'Other'
    reduced_df[feature] = reduced_df[feature].apply(
        lambda x: x if x in top_categories else 'Other'
    )
    return reduced_df


# Univariate class analysis
def plot_class_distribution(df, class_):
    plt.subplots(1, 1, figsize=(4, 4), tight_layout=True)

    # Plot and format countplot
    countplot = sns.countplot(data=df, x=class_)
    format_plot_axes(countplot, class_, 'frequency')

    export_and_show_plot('class_distribution')


# Univariate categorical analysis functions
def plot_categorical_distributions(df):
    _, categorical = get_feature_types(df)

    axes = create_plot_layout(len(categorical), 2)

    # Plot and format barplots with reduced features
    for count, feature in enumerate(categorical):
        # Reduce categories
        top_feature_categories = get_top_n_categories(df, feature, 10)

        barplot = sns.barplot(
            x=top_feature_categories.values,
            y=top_feature_categories.index,
            ax=axes[count],
        )
        format_plot_axes(barplot, 'frequency', feature)

    export_and_show_plot('categorical_distributions')


# Bivariate feature-to-class analysis functions
def plot_numerical_to_class(df, numerical):
    class_ = df.columns[-1]
    balanced_df = balance_class_categories(df)

    axes = create_plot_layout(len(numerical))

    # Plot and format boxplots
    for count, feature in enumerate(numerical):
        boxplot = sns.boxplot(x=class_, y=feature,
                              data=balanced_df, ax=axes[count])
        format_plot_axes(boxplot, class_, feature)

    export_and_show_plot('numerical_to_class')


def plot_categorical_to_class(df, categorical):
    class_ = df.columns[-1]
    balanced_df = balance_class_categories(df)

    axes = create_plot_layout(len(categorical), 2)

    for count, feature in enumerate(categorical):
        # Plot and format countplots
        countplot = sns.countplot(
            data=reduce_to_n_categories(balanced_df, feature, 10),
            y=feature,
            hue=class_,
            ax=axes[count],
        )
        format_plot_axes(countplot, class_, feature)

    export_and_show_plot('categorical_to_class')


def run_bivariate_class_analysis(df):
    numerical, categorical = get_feature_types(df)
    plot_numerical_to_class(df, numerical)
    plot_categorical_to_class(df, categorical)


# Class correlation analysis functions
def get_numerical_correlations(df, numerical):
    class_ = df.columns[-1]

    # Check if class is already binary converted
    if not pd.api.types.is_numeric_dtype(df[class_].dtype):
        # Save original class column
        original_class_df = df[class_].copy()

        # Convert class column to binary representation
        df[class_] = df[class_].map({'>50K': 1, '<=50K': 0})
    else:
        original_class_df = df[class_]

    # Initialise list to store results
    results = []

    # Calculate Point-Biserial between numerical features and class
    for feature in numerical:
        coefficient, p_value = pointbiserialr(df[feature], df[df.columns[-1]])

        # Append the result as a dictionary to the results list
        results.append(
            {
                'feature': feature,
                'correlation': coefficient,
                'p_value': p_value,
            }
        )

    # Restore original class column
    df[class_] = original_class_df

    # Create a new DataFrame from the results list
    point_biserial = pd.DataFrame(results)
    point_biserial.set_index('feature', inplace=True)

    # Sort by absolute value of Point-Biserial coefficient
    return point_biserial.sort_values(
        by='correlation', key=lambda x: x.abs(), ascending=False
    )


def get_categorical_correlations(df, categorical):
    class_ = df.columns[-1]

    # Initialise list to store Cramer's V results
    cramers_v = []

    # Calculate Cramér's V between categorical features and class
    for feature in categorical:
        coefficient, p_value = calculate_cramers_v(df[feature], df[class_])

        cramers_v.append(
            {
                'feature': feature,
                'correlation': coefficient,
                'p_value': p_value,
            }
        )

    # Create a dataframe of correlation results
    cramers_v_df = pd.DataFrame(cramers_v)
    cramers_v_df.set_index('feature', inplace=True)

    # Convert all values to numeric
    cramers_v_df = cramers_v_df.apply(pd.to_numeric, errors='coerce')

    # Sort by coefficient
    return cramers_v_df.sort_values(by='correlation', ascending=False)


def plot_class_correlations(numerical_df, categorical_df):
    axes = create_plot_layout(2)

    # Plot and format barplot for numerical features
    numerical_barplot = sns.barplot(
        data=numerical_df,
        x='correlation',
        y=numerical_df.index,
        ax=axes[0],
    )
    format_plot_axes(
        numerical_barplot, 'point-biserial coefficient', 'feature'
    )

    # Plot and format barplot for categorical features
    categorical_barplot = sns.barplot(
        data=categorical_df,
        x='correlation',
        y=categorical_df.index,
        ax=axes[1],
    )
    format_plot_axes(categorical_barplot, "cramer's v coefficient", 'feature')

    export_and_show_plot('class_correlations')


def display_class_correlations(corrs, feature_type):
    corrs.index += 1
    corrs.index.name = 'ranking'
    export_and_print_table(
        f'{feature_type} feature correlations with class',
        corrs,
    )


def run_class_correlation_analysis(df):
    numerical, categorical = get_feature_types(df)

    numerical_corrs = get_numerical_correlations(df, numerical)
    export_and_print_table(
        'numerical correlations with class', numerical_corrs
    )

    categorical_corrs = get_categorical_correlations(df, categorical)
    export_and_print_table(
        'categorical correlations with class', categorical_corrs
    )

    plot_class_correlations(numerical_corrs, categorical_corrs)


# Main analysis function
def run_analysis():
    # Data cleaning
    df, _ = load_original_data('analysis')

    # Initial analysis
    summarise(df)

    # Remove missing values for EDA
    df = df.dropna()

    # Univariate analysis
    plot_class_distribution(df, df.columns[-1])
    run_univariate_numerical_analysis(df)
    plot_categorical_distributions(df)

    # Bivariate analysis (feature-to-class)
    run_bivariate_class_analysis(df)

    # Feature correlation analysis
    run_feature_correlation_analysis(df)

    # Class correlation analysis
    run_class_correlation_analysis(df)

    # Export tables
    condense_tables()

    print('\n--- Data analysis complete! ---\n')
