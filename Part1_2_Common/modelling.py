import pandas as pd
import seaborn as sns

from Part1_2_Common.analysis import (create_plot_layout, export_and_show_plot,
                                     format_plot_axes)
from Part1_2_Common.cleaning import export_and_print_table
from Part1_2_Common.config import CLEANED_DATA_DIR


# Preparation function
def load_cleaned_data(filename):
    # Load from cleaned CSV
    df = pd.read_csv(f'{CLEANED_DATA_DIR}/{filename}.csv', index_col=0)

    # Seperate and return features and class dataframes
    class_ = df.columns[-1]
    return df.drop(columns=class_), df[class_]


# Results functions
def plot_model_metrics(df):
    axes = create_plot_layout(df.shape[1], 2)

    # Plot and format barplots
    for count, metric in enumerate(df.columns):
        metric_df = df[metric].sort_values(ascending=False)
        barplot = sns.barplot(x=metric_df, y=metric_df.index, ax=axes[count])
        format_plot_axes(barplot, metric, 'algorithm')

    export_and_show_plot('modelling_metrics')


def display_modelling_results(df):
    plot_model_metrics(df)

    export_and_print_table('algorithm comparison results', df)

    print('\n--- Modelling complete! ---')
