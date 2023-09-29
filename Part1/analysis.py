import pandas as pd
import seaborn as sns
from scipy.stats import f_oneway, pearsonr

from Part1.cleaning import load_original_data
from Part1_2_Common.analysis import (create_plot_layout, export_and_show_plot,
                                     format_plot_axes, get_feature_types,
                                     run_feature_correlation_analysis,
                                     run_univariate_numerical_analysis,
                                     summarise)
from Part1_2_Common.cleaning import export_and_print_table
from Part1_2_Common.config import KDE_SAMPLE_SIZE, PALETTE

sns.set_theme(style='darkgrid', palette=PALETTE)


# Univariate class analysis
def plot_class_distribution(df, class_):
    axes = create_plot_layout(2)

    # Plot and format histogram
    histogram = sns.histplot(data=df, x=class_, bins='doane', ax=axes[0])
    format_plot_axes(histogram, class_, 'frequency')

    # Plot and format boxplot
    boxplot = sns.boxplot(data=df, y=class_, ax=axes[1])
    format_plot_axes(boxplot, 'frequency', class_)

    export_and_show_plot('class_distribution')


# Univariate categorical analysis
def plot_categorical_distributions(df):
    _, categorical = get_feature_types(df)
    axes = create_plot_layout(len(categorical))

    # Plot and format countplots
    for count, feature in enumerate(categorical):
        countplot = sns.countplot(data=df, y=feature, ax=axes[count])
        format_plot_axes(
            countplot,
            'frequency',
            feature,
        )

    export_and_show_plot('categorical_distributions')


# Bivariate feature-to-class analysis functions
def plot_numerical_to_class(df, numerical):
    class_ = df.columns[-1]

    # Sample data to decrease computation load
    sample_df = df.sample(n=KDE_SAMPLE_SIZE)

    axes = create_plot_layout(len(numerical))

    # Plot and format KDE plots
    for count, feature in enumerate(numerical):
        kde = sns.kdeplot(
            data=sample_df, x=feature, y=class_, ax=axes[count], fill=True
        )
        format_plot_axes(kde, feature, class_)

    export_and_show_plot('numerical_to_class')


def plot_categorical_to_class(df, categorical):
    class_ = df.columns[-1]

    axes = create_plot_layout(len(categorical))

    # Plot and format boxplots
    for count, feature in enumerate(categorical):
        boxplot = sns.boxplot(
            data=df,
            x=class_,
            y=feature,
            ax=axes[count],
        )
        format_plot_axes(boxplot, class_, feature)

    export_and_show_plot('categorical_to_class')


def run_bivariate_class_analysis(df):
    numerical, categorical = get_feature_types(df)
    plot_numerical_to_class(df, numerical)
    plot_categorical_to_class(df, categorical)


# Class correlation analysis functions
def get_numerical_correlations(df, numerical):
    feature_df = df[numerical]

    # Initialise list to store Pearson results
    pearson = []

    # Calculate Pearson between numerical features and class
    for feature in feature_df.columns:
        correlation, p_value = pearsonr(
            feature_df[feature], df[df.columns[-1]]
        )
        pearson.append(
            {
                'feature': feature,
                'correlation': correlation,
                'p_value': p_value,
            }
        )

    pearson_df = pd.DataFrame(pearson)
    pearson_df.set_index('feature', inplace=True)

    # Sort by absolute value of Pearson coefficient
    return pearson_df.sort_values(
        by='correlation', key=lambda x: x.abs(), ascending=False
    )


def get_categorical_correlations(df, categorical):
    class_df = df[df.columns[-1]]

    # Initialise list to store ANOVA results
    anova = []

    # Perform ANOVA test between categorical features and class
    for feature in categorical:
        categories = df[feature].unique()
        samples = [
            class_df[df[feature] == category] for category in categories
        ]
        f_stat, p_value = f_oneway(*samples)

        anova.append(
            {'feature': feature, 'correlation': f_stat, 'p_value': p_value}
        )

    # Create a dataframe of correlation results
    anova_df = pd.DataFrame(anova)
    anova_df.set_index('feature', inplace=True)

    # Sort by F-statistic
    return anova_df.sort_values(by='correlation', ascending=False)


def plot_class_correlations(numerical_df, categorical_df):
    axes = create_plot_layout(2)

    # Plot and format barplot for numerical features
    numerical_barplot = sns.barplot(
        data=numerical_df,
        x='correlation',
        y=numerical_df.index,
        ax=axes[0],
    )
    format_plot_axes(numerical_barplot, 'pearson coefficient', 'feature')

    # Plot and format barplot for categorical features
    categorical_barplot = sns.barplot(
        data=categorical_df,
        x='correlation',
        y=categorical_df.index,
        ax=axes[1],
    )
    format_plot_axes(categorical_barplot, 'f-statistic', 'feature')

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
        'numerical feature correlations with class', numerical_corrs
    )

    categorical_corrs = get_categorical_correlations(df, categorical)
    export_and_print_table(
        'categorical feature correlations with class', categorical_corrs
    )

    plot_class_correlations(numerical_corrs, categorical_corrs)


# Main analysis function
def run_analysis():
    # Data cleaning
    df = load_original_data('analysis')

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

    print('\n--- Data analysis complete! ---\n')
