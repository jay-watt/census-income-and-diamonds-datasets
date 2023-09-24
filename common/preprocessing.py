import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from common.config import (
    CORR_THRESHOLD,
    KURT_THRESHOLD,
    PLOTS_DIR,
    SKEW_THRESHOLD,
)
from common.utils import (
    create_subplot_layout,
    display_table,
    interpret_kurt,
    interpret_skew,
    plot_histogram,
    plot_scatterplot,
)
from scipy.stats import boxcox, kurtosis, skew
from sklearn.preprocessing import StandardScaler


class Preprocessor:
    def __init__(self, class_name, data):
        self.class_name = class_name
        self.data = data

    def reassign_processed_data(self, processed_data, results_str):
        self.data = processed_data
        print(results_str, '\n')
        return processed_data

    def impute(self):
        print("***Imputing missing values")
        imputed_data = self.data.copy()

        imputed_features = []
        for feature in imputed_data.columns:
            if imputed_data[feature].isna().any():
                if np.issubdtype(imputed_data[feature].dtype, np.number):
                    imputed_data[feature].fillna(
                        imputed_data[feature].mean(), inplace=True
                    )
                else:
                    mode_val = imputed_data[feature].mode().iloc[0]
                    imputed_data[feature].fillna(mode_val, inplace=True)
                imputed_features.append(feature)

        results_str = f'Features imputed: {", ".join(imputed_features)}'
        return self.reassign_processed_data(imputed_data, results_str)

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

    def find_abnormal_features(self):
        skewed_features = self.data.drop(columns=self.class_name).apply(
            lambda x: skew(x)
        )
        high_skew = skewed_features[
            abs(skewed_features) > SKEW_THRESHOLD
        ].index

        kurt_features = self.data.drop(columns=self.class_name).apply(
            lambda x: kurtosis(x)
        )
        high_kurt = kurt_features[kurt_features > KURT_THRESHOLD].index

        return high_skew.union(high_kurt)

    def transform_abnormal_features(self):
        print("***Transforming abnormal features")
        transformed_data = self.data.copy()
        features_to_transform = self.find_abnormal_features()

        for feature in features_to_transform:
            transformed_data[feature], _ = boxcox(
                transformed_data[feature] + 0.01
            )

        results_str = (
            f'Features transformed: {", ".join(features_to_transform)}'
        )
        return self.reassign_processed_data(transformed_data, results_str)

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

        feature_data = cleaned_data.drop(columns=[self.class_name])
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
            scaled_data.drop(columns=self.class_name)
        )
        scaled_data = pd.concat(
            [
                pd.DataFrame(
                    scaled_feature_data,
                    columns=scaled_data.columns[
                        scaled_data.columns != self.class_name
                    ],
                ),
                scaled_data[self.class_name].reset_index(drop=True),
            ],
            axis=1,
        )
        print(
            f'Final features selected: {", ".join(list(scaled_data.columns))}'
        )
        return self.reassign_processed_data(
            scaled_data, 'Scaler used: StandardScaler'
        )


class EDA:
    def __init__(self, class_name, data, data_name, part_str):
        self.class_name = class_name
        self.data = data
        self.data_name = data_name
        self.part_str = part_str

    def plot_distributions(self, process_str, features):
        axes = create_subplot_layout(len(features))

        distributions_results = {}
        for count, feature in enumerate(features):
            skew, kurt = plot_histogram(
                feature, self.data[feature], axes[count]
            )
            distributions_results[feature] = {
                'skewness value': skew,
                'kurtosis value': kurt,
            }

        plt.savefig(
            f'{PLOTS_DIR}/{self.part_str}_distributions_{process_str}.png'
        )
        return pd.DataFrame.from_dict(distributions_results, orient='index')

    def display_distributions(self, process_str, features):
        if self.data_name == 'train':
            distributions = self.plot_distributions(process_str, features)
            distributions['skewness shape'] = distributions[
                'skewness value'
            ].apply(interpret_skew)
            distributions['kurtosis shape'] = distributions[
                'kurtosis value'
            ].apply(interpret_kurt)
            distributions[
                ['skewness value', 'kurtosis value']
            ] = distributions[['skewness value', 'kurtosis value']].applymap(
                '{:,.2f}'.format
            )
            distributions = distributions[
                [
                    'skewness value',
                    'skewness shape',
                    'kurtosis value',
                    'kurtosis shape',
                ]
            ]
            display_table(f'Distributions {process_str}', distributions)

    def plot_scatterplots(self, process_str, features):
        if self.data_name == 'train':
            axes = create_subplot_layout(len(features))
            for count, feature in enumerate(features):
                plot_scatterplot(
                    feature, self.class_name, self.data, axes[count]
                )
            plt.savefig(
                f'{PLOTS_DIR}/{self.part_str}_scatter_{process_str}.png'
            )

    @staticmethod
    def correlation_pairs(corr_matrix):
        corr_pairs = [
            (feature1, feature2, correlation_value)
            for i, feature1 in enumerate(corr_matrix.columns)
            for j, feature2 in enumerate(
                corr_matrix.columns[i + 1 :], start=i + 1
            )
            if abs(correlation_value := corr_matrix.iloc[i, j])
            > CORR_THRESHOLD
        ]
        corr_pairs_df = pd.DataFrame(
            corr_pairs, columns=['feature 1', 'feature 2', 'correlation']
        )
        sorted_pairs = corr_pairs_df.sort_values(
            by='correlation', ascending=False
        )
        sorted_pairs = sorted_pairs.reset_index(drop=True)
        sorted_pairs.index = sorted_pairs.index + 1

        sorted_pairs['correlation'] = sorted_pairs['correlation'].apply(
            '{:,.2f}'.format
        )
        return sorted_pairs

    def display_correlations(self, process_str):
        corr_matrix = self.data.drop(columns=self.class_name).corr()
        corr_pairs = self.correlation_pairs(corr_matrix)
        if len(corr_pairs) > 0:
            corr_pairs.index.name = 'ranking'
            title = f'Correlated Feature Pairs (> {CORR_THRESHOLD})'
            display_table(title, corr_pairs)

    def broadly_analyse(self, process_str):
        if self.data_name == 'train':
            self.plot_distributions(
                process_str, self.data.drop(columns=self.class_name).columns
            )
            self.plot_scatterplots(
                process_str, self.data.drop(columns=self.class_name).columns
            )
