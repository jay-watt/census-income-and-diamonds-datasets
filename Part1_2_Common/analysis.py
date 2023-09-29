import math
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import chi2_contingency, kurtosis, skew

from Part1_2_Common.cleaning import export_and_print_table
from Part1_2_Common.config import (CORR_THRESHOLD, EXCESS_KURT_THRESHOLD,
                                   HIGH_SKEW_THRESHOLD, PLOTS_DIR,
                                   SKEW_THRESHOLD, TABLES_DIR, UNITS)


# Function to convert all saved analysis spreadsheets into a single file
def condense_tables():
    output_filename = 'analysis.xlsx'
    output_filepath = os.path.join(TABLES_DIR, output_filename)

    with pd.ExcelWriter(output_filepath, engine='xlsxwriter') as writer:
        for file_name in os.listdir(TABLES_DIR):
            if file_name.endswith('.xlsx') and file_name != output_filename:
                file_path = os.path.join(TABLES_DIR, file_name)

                try:
                    data = pd.read_excel(file_path, engine='openpyxl')
                except Exception as e:
                    print(f"Error reading {file_name}: {e}")
                    continue

                def format_cell(x):
                    if isinstance(x, str) and not x.isupper():
                        return x.capitalize()
                    elif isinstance(x, float):
                        return '{:,.2f}'.format(x)
                    elif isinstance(x, int):
                        return '{:,}'.format(x)
                    return x

                data = data.applymap(format_cell)

                # Format the headers (column names)
                data.columns = [col.capitalize() for col in data.columns]

                data.to_excel(
                    writer, sheet_name=f'{file_name[:-5]}', index=False)

                try:
                    os.remove(file_path)
                except Exception as e:
                    print(f"Error deleting {file_name}: {e}")


# Helper functions
def print_process_heading(process):
    print('=' * 50)
    print(f'{process.upper()}')
    print('=' * 50)


def get_feature_types(df):
    class_ = df.columns[-1]
    numerical = (
        df.drop(columns=class_).select_dtypes(include=['number']).columns
    )
    categorical = (
        df.drop(columns=class_).select_dtypes(include=['object']).columns
    )
    return numerical, categorical


# Initial analysis functions
def display_summary(df):
    numerical, categorical = get_feature_types(df)
    summary = {
        'instances': df.shape[0],
        'features': df.shape[1] - 1,
        'categorical': len(categorical),
        'numerical': len(numerical),
    }
    summary_df = pd.DataFrame({'count': summary})

    summary_df.index.name = 'attribute'
    export_and_print_table('dataset summary', summary_df)


def get_stats_by_dtype(df, dtype):
    return df.select_dtypes(include=[dtype]).describe().T


def display_stats(df):
    # Get numerical stats
    numerical_stats = get_stats_by_dtype(df, 'number')

    # Calculate variance and range
    numerical_stats['variance'] = numerical_stats['std'] ** 2
    numerical_stats['range'] = numerical_stats['max'] - numerical_stats['min']

    # Drop redundant columns
    numerical_stats.drop(columns=['25%', '50%', '75%'], inplace=True)

    numerical_stats.index.name = 'feature'

    export_and_print_table('numerical statistics', numerical_stats)

    # Get categorical stats
    categorical_stats = get_stats_by_dtype(df, 'object')

    # Get unique values
    categorical_stats['values'] = [
        ', '.join(map(str, df[column].unique()))
        for column in categorical_stats.index
    ]

    categorical_stats.index.name = 'feature'
    export_and_print_table('categorical statistics', categorical_stats)


# Run initial analysis
def summarise(df):
    print_process_heading('analysing data')
    display_summary(df)
    display_stats(df)


# Plotting helper functions
def create_plot_layout(num_plots, num_cols=None):
    # Calculate number columns
    if num_cols is None:
        # No specified value, will set to 2 or 3
        num_cols = 2 if num_plots in [2, 4] else 3

    # Calculate number of rows
    num_rows = math.ceil(num_plots / num_cols)

    # Calculate figure height to fit A4 page
    if num_cols < 3:
        fig_height = 4 * num_rows if num_rows < 10.5 / 4 else 10.5
    else:
        fig_height = 2.5 * num_rows if num_rows < 10.5 / 2.5 else 10.5

    # Create layout
    _, axes = plt.subplots(
        num_rows, num_cols, figsize=(8, fig_height), tight_layout=True
    )

    # Convert axes to 1D if layout has mutiple rows and columns
    axes_1d = axes.ravel() if num_rows > 1 and num_cols > 1 else axes

    # Clean axes by removing any unused positions
    for i in range(num_plots, num_rows * num_cols):
        axes_1d[i].axis('off')

    # Return 1D axes
    return axes_1d


def format_plot_axes(plot, xlabel, ylabel):
    # Add units to axis labels if applicable
    xlabel = xlabel if xlabel.istitle() else ' '.join(
        word.capitalize() for word in xlabel.split())
    xlabel += f' ({UNITS[xlabel]})' if xlabel in UNITS else ''

    ylabel = ylabel if ylabel.istitle() else ' '.join(
        word.capitalize() for word in ylabel.split())
    ylabel += f' ({UNITS[ylabel]})' if ylabel in UNITS else ''

    # Set axis labels
    plot.set_xlabel(xlabel, fontsize=10)
    plot.set_ylabel(ylabel, fontsize=10)

    # Set tick size
    plot.tick_params(labelsize=9)


def export_and_show_plot(filename):
    plt.savefig(f'{PLOTS_DIR}/{filename}.png')
    # plt.show()


# Univariate feature analysis functions
def plot_numerical_distributions(df, numerical):
    axes = create_plot_layout(len(numerical))
    # Plot and format histograms
    for count, feature in enumerate(numerical):
        feature_df = df[feature]
        # Calculate bins
        bins = (
            # Discrete features
            feature_df.nunique()
            if feature_df.nunique() < 10
            # Continuous features
            else len(np.histogram_bin_edges(feature_df, bins='doane')) - 1
        )

        histogram = sns.histplot(
            data=df, x=feature, bins=bins, ax=axes[count])  # type: ignore
        format_plot_axes(histogram, feature, 'frequency')

    export_and_show_plot('numerical_distributions')


def interpret_skewness(skew):
    if skew <= -HIGH_SKEW_THRESHOLD:
        return 'Very long left tail'
    elif -HIGH_SKEW_THRESHOLD < skew < -SKEW_THRESHOLD:
        return 'Long left tail'
    elif -SKEW_THRESHOLD <= skew <= SKEW_THRESHOLD:
        return 'Approximately symmetric'
    elif SKEW_THRESHOLD < skew <= HIGH_SKEW_THRESHOLD:
        return 'Long right tail'
    else:
        return 'Very long right tail'


def interpret_kurtosis(kurt):

    excess_kurt = kurt - 3

    if excess_kurt < -EXCESS_KURT_THRESHOLD:
        return 'Light tails, platykurtic'
    elif excess_kurt > EXCESS_KURT_THRESHOLD:
        return 'Heavy tails, leptokurtic'
    else:
        return 'Approximately normal tails, mesokurtic'


def calculate_skewness_and_kurtosis(df, numerical):
    # Loop through numerical features and calculate skewness and kurtosis
    skew_and_kurt = [
        {
            'feature': feature,
            'skewness': skew(df[feature]),
            'kurtosis': kurtosis(df[feature]),
        }
        for feature in numerical
    ]

    skew_and_kurt_df = pd.DataFrame(skew_and_kurt)
    skew_and_kurt_df = skew_and_kurt_df.set_index('feature')

    # Get value interpretations
    skew_and_kurt_df['skewness_shape'] = skew_and_kurt_df['skewness'].apply(
        interpret_skewness
    )
    skew_and_kurt_df['kurtosis_shape'] = skew_and_kurt_df['kurtosis'].apply(
        interpret_kurtosis
    )

    export_and_print_table(
        'skewness and kurtosis', skew_and_kurt_df
    )


# Run numerical univariate feature analysis
def run_univariate_numerical_analysis(df):
    numerical, _ = get_feature_types(df)

    plot_numerical_distributions(df, numerical)
    calculate_skewness_and_kurtosis(df, numerical)


# Feature correlation analysis functions
def plot_correlation_matrix(corr_matrix, feature_type):
    fig_width = min(max(len(corr_matrix.index), 4), 8)
    plt.subplots(1, 1, figsize=(fig_width, fig_width / 1.2), tight_layout=True)

    # Plot and format correlation heatmap
    heatmap = sns.heatmap(
        corr_matrix,
        annot=True,
        annot_kws={'size': 9},
        linewidths=1,
        cmap="Blues",
    )
    format_plot_axes(heatmap, '', '')

    # Format colour bar
    cbar = heatmap.collections[0].colorbar
    cbar.ax.tick_params(labelsize=9)

    export_and_show_plot(f'{feature_type}_correlations')


def get_feature_pair_correlations(corr_matrix, feature_type):
    # Get list of pairs of features
    pairs = [
        (feature1, feature2)
        for i, feature1 in enumerate(corr_matrix.columns)
        for j, feature2 in enumerate(corr_matrix.columns[i + 1:], start=i + 1)
    ]

    # Get correlation values for each pair
    correlation_values = [corr_matrix.at[feature1, feature2]
                          for feature1, feature2 in pairs]

    # Create dataframe of feature pairs along with their correlation values
    pairs_df = pd.DataFrame(
        pairs,
        columns=['feature1', 'feature2']
    )
    pairs_df['correlation'] = correlation_values

    # Filter pairs by the absolute correlation threshold
    pairs_df = pairs_df[pairs_df['correlation'].abs() >= CORR_THRESHOLD]

    # Sort by absolute value of correlation coefficient
    pairs_df = pairs_df.sort_values(
        by='correlation', key=lambda x: x.abs(), ascending=False
    ).reset_index(drop=True)
    # Name the index 'ranking'
    pairs_df.index.name = 'ranking'
    pairs_df.index += 1

    # Export and display feature pairs
    if len(pairs_df) > 0:
        export_and_print_table(
            f'{feature_type} feature pair correlations',
            pairs_df,
        )


# Categorical correlation analysis function
def calculate_cramers_v(x, y):
    # Create contingency table
    confusion_matrix = pd.crosstab(x, y)

    # Calculate the chi-squared statistic
    chi2, p_value = chi2_contingency(confusion_matrix)[:2]

    # Calculate the total number of observations
    n = confusion_matrix.sum().sum()

    # Calculate the phi-squared value
    phi2 = chi2 / n

    # Calculate the number of rows and columns in the contingency table
    r, k = confusion_matrix.shape

    # Apply a bias correction to the phi-squared value
    phi2corr = max(0, phi2 - ((k - 1) * (r - 1)) / (n - 1))

    # Apply bias correction to the number of rows and columns
    rcorr = r - ((r - 1) ** 2) / (n - 1)
    kcorr = k - ((k - 1) ** 2) / (n - 1)

    # Calculate and return Cramér's V statistic
    return np.sqrt(phi2corr / min((kcorr - 1), (rcorr - 1))), p_value


# Run feature correlation analysis
def run_feature_correlation_analysis(df):
    numerical, categorical = get_feature_types(df)

    # Calculate Pearson
    numerical_corr_matrix = df[numerical].corr(method='pearson')

    plot_correlation_matrix(numerical_corr_matrix, 'numerical')
    get_feature_pair_correlations(numerical_corr_matrix, 'numerical')

    # Initialise dataframe to store Cramer's V results
    categorical_corr_matrix = pd.DataFrame(
        index=categorical, columns=categorical
    )

    # Calculate Cramér's V
    for feature1 in categorical:
        for feature2 in categorical:
            (
                categorical_corr_matrix.loc[feature1, feature2],
                _,
            ) = calculate_cramers_v(df[feature1], df[feature2])

    # Convert all values to numeric
    categorical_corr_matrix = categorical_corr_matrix.applymap(
        pd.to_numeric, errors='coerce'
    )

    plot_correlation_matrix(categorical_corr_matrix, 'categorical')
    get_feature_pair_correlations(categorical_corr_matrix, 'categorical')
