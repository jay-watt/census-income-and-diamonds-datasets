import numpy as np
import pandas as pd
from tabulate import tabulate

from Part1_2_Common.config import MAX_COL_WIDTH, TABLES_DIR


# Helper functions
def print_results(attribute, process, num_before, num_after):
    before = f'{attribute} before {process}: {num_before}'
    after = f'{attribute} after {process}: {num_after}'
    print('  - ', before)
    print('  - ', after)


def export_and_print_table(title, df):
    # Create sheet name with <= 31 chars
    title_words = title.split()
    while True:
        sheet_name = '_'.join(title_words)

        if len(sheet_name) <= 31:
            break

        title_words = title_words[:-1]

    # Write dataframe to a spreadsheet
    df.to_excel(f'{TABLES_DIR}/{sheet_name}.xlsx')

    # Print dataframe as a text table
    formatted_title = ' '.join(word.capitalize() for word in title.split())
    formatted_table = tabulate(
        df.round(2),
        headers=[df.index.name] + list(df.columns),
        tablefmt='fancy_grid',
        showindex='always',
        maxcolwidths=[MAX_COL_WIDTH] * (len(df.columns) + 1),
    )
    print(f'\n{formatted_title}')
    print(formatted_table, '\n')


class Cleaner:
    def __init__(self, df, process_str):
        self.df = df
        self.class_ = df.columns[-1]

        # If 'analysis', get detailed information while cleaning
        self.process_str = process_str

    def remove_duplicates(self):
        print('\nRemoving duplicates')

        # Count duplicates before removal
        before = self.df.duplicated().sum()

        # Remove duplicates
        self.df.drop_duplicates(inplace=True)

        # Count duplicates after removal
        after = self.df.duplicated().sum()

        print_results('duplicates', 'removal', before, after)

    def get_missing_values(self):
        # Get missing value counts of all features
        missing = self.df.isnull().sum()

        # Create dataframe of features with counts over 0
        missing_df = pd.DataFrame(missing[missing > 0], columns=['count'])
        missing_df.index.name = 'feature'

        # Calculate percentage of missing values out of total instances
        missing_df['percentage'] = missing_df['count'] * 100 / self.df.shape[0]

        # Sort by highest number of missing values
        missing_df = missing_df.sort_values(by='count', ascending=False)

        export_and_print_table(
            'missing values',
            missing_df,
        )

    def identify_missing_values(self, missing_placeholder):
        print(f'\nIdentifying {missing_placeholder}s as missing values')

        # Count missing values before replacement
        before = self.df.isnull().sum().sum()

        # Replace missing value placeholder with NaN
        self.df.replace(missing_placeholder, np.nan, inplace=True)

        # Count missing values after replacement
        after = self.df.isnull().sum().sum()

        print_results('missing values', 'identification', before, after)

        # Get detailed missing value information
        if self.process_str == 'analysis':
            self.get_missing_values()
