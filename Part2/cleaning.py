import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from Part1_2_Common.analysis import (export_and_show_plot, format_plot_axes,
                                     print_process_heading)
from Part1_2_Common.cleaning import Cleaner
from Part1_2_Common.config import ORIGINAL_DATA_DIR
from Part2.config import COLUMN_NAMES, DATA_FILENAMES


def balance_class_categories(df):
    class_ = df.columns[-1]
    min_class_size = df[class_].value_counts().min()

    return (
        df.groupby(class_)
        .apply(lambda x: x.sample(min_class_size))
        .reset_index(drop=True)
    )


class Part2Cleaner(Cleaner):
    def __init__(self, df, process_str):
        super().__init__(df, process_str)
        self.top_categories = {}

    def set_data_name(self, data_name):
        self.data_name = data_name
        print_process_heading(
            f'cleaning {self.data_name} data for {self.process_str}'
        )

    def plot_missingness(self):
        # Get features with missing values
        missing = self.df.isnull().any()
        features = missing[missing].index.tolist()

        # Initialise a dataframe to store missingness information
        missing_df = pd.DataFrame()

        # Balance class categories
        balanced_df = balance_class_categories(self.df)

        # Reduce categories
        for feature in features:
            # Calculate proportion of missing values per class category
            grouped_df = balanced_df.groupby(self.class_)[feature]
            missing_df[feature] = grouped_df.apply(
                lambda x: x.isna().mean()
            )

        # Transpose the dataframe for easier plotting
        missing_df = missing_df.T.reset_index()
        missing_df = missing_df.melt(
            id_vars='index', var_name='class', value_name='missing_proportion'
        )

        # Plot the stacked bar plot
        plt.figure(figsize=(6, 4))
        plt.tight_layout()

        barplot = sns.barplot(
            data=missing_df,
            x='index',
            y='missing_proportion',
            hue='class',
        )
        format_plot_axes(
            barplot, 'feature', 'missing value proportion'
        )

        export_and_show_plot('missingness_to_class')

    def clean(self):
        self.remove_duplicates()
        self.identify_missing_values('?')
        self.plot_missingness()
        print('\n--- Cleaning complete! ---\n')


# Data handling functions
def load_original_data(process_str):
    training = pd.read_csv(
        f'{ORIGINAL_DATA_DIR}/{DATA_FILENAMES["training"]}',
        header=None,
        names=COLUMN_NAMES,
        sep=', ',
        engine='python',
    )

    test = pd.read_csv(
        f'{ORIGINAL_DATA_DIR}/{DATA_FILENAMES["test"]}',
        header=None,
        names=COLUMN_NAMES,
        skiprows=1,
        sep=', ',
        engine='python',
    )

    cleaner = Part2Cleaner(training, process_str)
    cleaner.set_data_name('training')
    cleaner.clean()
    cleaned_training = cleaner.df

    if process_str == 'analysis':
        return cleaned_training, test

    cleaner.df = test
    cleaner.set_data_name('test')
    cleaner.clean()

    return cleaned_training, cleaner.df
