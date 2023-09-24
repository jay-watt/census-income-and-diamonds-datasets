import pandas as pd
from sklearn.preprocessing import StandardScaler

from common.config import CLEANED_DATA_DIR


class Preprocessor:
    def __init__(self, class_, data, data_name):
        self.class_ = class_
        self.data = data
        self.data_name = data_name
        print(f'Preprocessing {data_name} data...\n')

    def reassign_processed_data(self, processed_data, results_str):
        self.data = processed_data
        print(results_str, '\n')
        return processed_data

    def map_ordinal_features(self, mappings):
        print("***Mapping ordinal and binary categorical features")
        mapped_data = self.data.copy()
        selected_features = list(mappings.keys())

        mapped_data.replace(mappings, inplace=True)
        mapped_data[selected_features] = mapped_data[selected_features].astype(
            int
        )

        results_str = f'Features mapped: {", ".join(selected_features)}'
        return self.reassign_processed_data(mapped_data, results_str)

    @staticmethod
    def filter_outliers(feature):
        Q1 = feature.quantile(0.25)
        Q3 = feature.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        return feature.between(lower_bound, upper_bound)

    def remove_outliers(self):
        print("***Removing outliers")
        cleaned_data = self.data.copy()

        feature_data = cleaned_data.drop(columns=[self.class_])
        mask = feature_data.apply(self.filter_outliers).all(axis=1)
        cleaned_data = cleaned_data[mask]

        rows_removed = self.data.shape[0] - cleaned_data.shape[0]
        results_str = f'Outlier instances removed: {rows_removed:,.0f}'
        return self.reassign_processed_data(cleaned_data, results_str)

    def remove_redundant_features(self, selected_features):
        print("***Removing redundant features")
        cleaned_data = self.data.copy()
        features_to_remove = [
            feature
            for feature in selected_features
            if feature in cleaned_data.columns
        ]

        cleaned_data.drop(
            columns=features_to_remove,
            inplace=True,
        )

        results_str = (
            f'Redundant features removed: {", ".join(features_to_remove)}'
        )
        return self.reassign_processed_data(cleaned_data, results_str)

    def scale(self):
        print("***Scaling")
        scaled_data = self.data.copy()
        scaler = StandardScaler()

        scaled_feature_data = scaler.fit_transform(
            scaled_data.drop(columns=self.class_)
        )
        scaled_data = pd.concat(
            [
                pd.DataFrame(
                    scaled_feature_data,
                    columns=scaled_data.columns[
                        scaled_data.columns != self.class_
                    ],
                ),
                scaled_data[self.class_].reset_index(drop=True),
            ],
            axis=1,
        )
        print(
            f'Final features selected: {", ".join(list(scaled_data.columns))}'
        )
        return self.reassign_processed_data(
            scaled_data, 'Scaler used: StandardScaler'
        )

    def write_cleaned_data(self, filename):
        self.data.reset_index(drop=True, inplace=True)
        self.data.index = self.data.index + 1
        self.data.to_csv(f'{CLEANED_DATA_DIR}/{filename}.csv', index=True)
        print(f'Preprocessing of {self.data_name} data complete!\n')
