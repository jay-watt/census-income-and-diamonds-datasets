import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split

from Part1_2_Common.analysis import (
    create_plot_layout,
    export_and_show_plot,
    format_plot_axes,
    print_process_heading,
)
from Part1_2_Common.cleaning import Cleaner
from Part1_2_Common.config import ORIGINAL_DATA_DIR, SEED
from Part1.config import DATA_FILENAME


class Part1Cleaner(Cleaner):
    def plot_missingness(self):
        class_ = self.df.columns[-1]

        # Get features with missing values
        missing = self.df.isnull().any()
        features = missing[missing].index.tolist()

        axes = create_plot_layout(len(features))

        # Plot and format boxplots
        for count, feature in enumerate(features):
            x_labels = (
                self.df[feature]
                .isnull()
                .map({True: 'Missing', False: 'Non-Missing'})
            )
            boxplot = sns.boxplot(
                x=x_labels,
                y=self.df[class_],
                ax=axes[count],
            )
            format_plot_axes(
                boxplot, feature.capitalize(), class_.capitalize()
            )

        export_and_show_plot('missingness_to_class')

    # Run cleaning
    def clean(self):
        print_process_heading(f'cleaning data for {self.process_str}')
        self.remove_duplicates()
        self.identify_missing_values(0)
        self.plot_missingness()
        print('\n--- Cleaning complete! ---\n')


# Data handling functions
def split_data(df):
    class_ = df.columns[-1]

    # Split data
    train_X, test_X, train_y, test_y = train_test_split(
        df.drop(columns=class_),
        df[class_],
        test_size=0.3,
        random_state=SEED,
    )

    # Combine and return X and y dataframes for each dataset
    train = pd.concat([train_X, train_y], axis=1)
    test = pd.concat([test_X, test_y], axis=1)
    return train, test


def load_original_data(process_str):
    df = pd.read_csv(f'{ORIGINAL_DATA_DIR}/{DATA_FILENAME}.csv', index_col=0)

    cleaner = Part1Cleaner(df, process_str)
    cleaner.clean()
    return cleaner.df
