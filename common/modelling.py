import pandas as pd
import seaborn as sns

from common.config import CLEANED_DATA_DIR
from common.preprocessing import (
    create_plot_layout,
    export_and_show_plot,
    format_plot_axes,
    save_and_print_table,
)


# Preparation function
def load_cleaned_data(filename):
    # Load from cleaned CSV
    df = pd.read_csv(f"{CLEANED_DATA_DIR}/{filename}.csv", index_col=0)

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
        format_plot_axes(barplot, metric, "algorithm")

    export_and_show_plot("modelling_metrics")


def display_modelling_results(df):
    plot_model_metrics(df)

    save_and_print_table("model comparison results", df)

    print("\n--- Modelling complete! ---")
