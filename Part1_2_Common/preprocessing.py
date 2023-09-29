import numpy as np
import pandas as pd
from scipy.stats import zscore
from sklearn.preprocessing import StandardScaler

from Part1_2_Common.analysis import get_feature_types, print_process_heading
from Part1_2_Common.cleaning import print_results
from Part1_2_Common.config import CLEANED_DATA_DIR, Z_THRESHOLD


class Preprocessor:
    def __init__(self, df):
        self.class_ = df.columns[-1]
        self.df = df

        # Initialise scaling model
        self.scaler = StandardScaler()

    def set_data_name(self, data_name):
        self.data_name = data_name
        self.class_ = self.df.columns[-1]
        print_process_heading(f'preprocessing {self.data_name} data')

    def map_ordinal_features(self, mappings):
        print("\nMapping categorical features")
        features_to_map = list(mappings.keys())

        # Count categorical features before mapping
        _, categorical = get_feature_types(self.df)
        before = len(categorical)

        # Perform mapping
        self.df.replace(mappings, inplace=True)
        self.df[features_to_map] = self.df[features_to_map].astype(int)

        # Count categorical features after mapping
        _, categorical = get_feature_types(self.df)
        after = len(categorical)

        print_results('categorical features', 'mapping', before, after)

    @staticmethod
    def filter_outliers(feature):
        z_scores = zscore(feature)
        abs_z_scores = np.abs(z_scores)
        return abs_z_scores < Z_THRESHOLD

    def remove_outliers(self):
        print("\nRemoving outliers")
        numerical, _ = get_feature_types(self.df)
        numerical_feature_data = self.df[numerical]

        # Count instances before outlier removal
        before = self.df.shape[0]

        # Remove outliers
        mask = numerical_feature_data.apply(self.filter_outliers).all(axis=1)
        self.df = self.df[mask]

        # Count instances after outlier removal
        after = self.df.shape[0]

        print_results('instances', 'removal', before, after)

    def remove_redundant_features(self, features_to_reduce):
        print(f"\nRemoving redundant features {', '.join(features_to_reduce)}")
        features_to_remove = [
            feature
            for feature in features_to_reduce
            if feature in self.df.columns
        ]

        # Count features before removal
        before = self.df.shape[1] - 1

        # Remove redundant features
        self.df.drop(
            columns=features_to_remove,
            inplace=True,
        )

        # Count features after removal
        after = self.df.shape[1] - 1

        print_results('features', 'removal', before, after)

    def scale(self):
        print("\nScaling")

        # Split data into features and class
        X = self.df.drop(columns=self.class_)
        y = self.df[self.class_]

        # Fit scaler if current dataset is training dataset
        if self.data_name == 'training':
            self.scaler.fit(X)

        # Scale
        scaled_X = self.scaler.transform(X)

        # Create dataframe from scaled data
        self.df = pd.concat(
            [
                pd.DataFrame(
                    scaled_X,
                    columns=X.columns,
                ),
                y.reset_index(drop=True),
            ],
            axis=1,
        )

    def write_cleaned_data(self, filename):
        self.df.reset_index(drop=True, inplace=True)
        self.df.index += 1

        self.df.to_csv(f'{CLEANED_DATA_DIR}/{filename}.csv', index=True)

        print(f'\n--- Preprocessing of {self.data_name} data complete! ---\n')
