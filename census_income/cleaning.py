import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from common.preprocessing import (export_and_show_plot, format_plot_axes,
                                  print_process_heading, Preprocessor)
from common.config import ORIGINAL_DATA_DIR
from census_income.config import COLUMN_NAMES, DATA_FILENAMES


def balance_class_categories(df):
    class_ = df.columns[-1]
    min_class_size = df[class_].value_counts().min()

    return (
        df.groupby(class_)
        .apply(lambda x: x.sample(min_class_size))
        .reset_index(drop=True)
    )


class CensusIncomePreprocessor(Preprocessor):
    def __init__(self, train, test, class_):
        super().__init__(train, test, class_)
        self.top_categories = {}

    def plot_missingness(self, set_type):
        # Get features with missing values
        missing = self.dfs[set_type].isnull().any()
        features = missing[missing].index.tolist()

        # Initialise a dataframe to store missingness information
        missing_df = pd.DataFrame()

        # Balance class categories
        balanced_df = balance_class_categories(self.dfs[set_type])

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

    def clean(self, set_type):
        self.remove_duplicates(set_type)
        self.identify_missing_values('?', set_type)
        self.plot_missingness(set_type)
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

    class_ = training.columns[-1]
    test.columns[-1] = class_
    preprocessor = CensusIncomePreprocessor(training, test, class_)
    preprocessor.clean('training')
    cleaned_training = preprocessor.dfs["training"]

    if process_str == 'analysis':
        return cleaned_training, test

    preprocessor.clean('test')

    cleaned_test = preprocessor.dfs["test"]
    return cleaned_training, cleaned_test
