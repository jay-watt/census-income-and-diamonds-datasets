import matplotlib.pyplot as plt
import pandas as pd
from config import KURT_THRESHOLD, ORDINAL_MAPPINGS, PLOTS_PATH, SKEW_THRESHOLD
from scipy.stats import boxcox, kurtosis, skew
from sklearn.preprocessing import StandardScaler
from utils import (
    create_subplot_layout,
    display_table,
    interpret_kurt,
    interpret_skew,
    plot_barplot,
    plot_histogram,
    plot_scatterplot,
    write_cleaned_data,
)


class Preprocessor:
    def __init__(self, class_name, data):
        self.class_name = class_name
        self.data = data

    def encode(self):
        self.data.replace(ORDINAL_MAPPINGS, inplace=True)
        return self.data

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
        features_to_transform = self.find_abnormal_features()
        features_str = ', '.join(features_to_transform)
        print(f'Boxcox transformation applied to: {features_str}\n')

        for feature in features_to_transform:
            self.data[feature], _ = boxcox(self.data[feature] + 0.01)
        return self.data

    def remove_outliers(self, abnormal_features):
        rows_before_removal = self.data.shape[0]
        self.data = self.data[
            self.data[abnormal_features]
            .apply(self.filter_outliers, axis=0)
            .all(1)
        ]
        rows_removed = rows_before_removal - self.data.shape[0]
        features_str = ', '.join(abnormal_features)
        print(f"Abnormal features inspected: {features_str}")
        print(f"Rows removed using modified IQR method: {rows_removed:,.0f}\n")
        return self.data

    @staticmethod
    def filter_outliers(feature):
        Q1 = feature.quantile(0.25)
        Q3 = feature.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        return feature.between(lower_bound, upper_bound)

    def transform_correlated_features(self):
        print('Transforming highly correlated features: x, y, z, carat\n')
        transformed_data = self.data.copy()
        transformed_data['volume'] = (
            transformed_data['x']
            * transformed_data['y']
            * transformed_data['z']
        )
        transformed_data['carat_to_volume'] = (
            transformed_data['carat'] / transformed_data['volume']
        )
        self.data = transformed_data
        return self.data

    def remove_redundant_features(self):
        cleaned_data = self.data.copy()
        cleaned_data.drop(
            columns=['depth', 'x', 'y', 'z', 'carat', 'carat_to_volume'],
            inplace=True,
        )
        self.data = cleaned_data
        return self.data

    def scale(self):
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(
            self.data.drop(columns=self.class_name)
        )
        self.data = pd.concat(
            [
                pd.DataFrame(
                    scaled_features,
                    columns=self.data.columns[
                        self.data.columns != self.class_name
                    ],
                ),
                self.data[self.class_name].reset_index(drop=True),
            ],
            axis=1,
        )
        return self.data


class EDA:
    def __init__(self, class_name, data):
        self.class_name = class_name
        self.data = data

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

        plt.savefig(f'{PLOTS_PATH}/part1_distributions_{process_str}.png')
        return pd.DataFrame.from_dict(distributions_results, orient='index')

    def display_distributions(self, process_str, features):
        distributions = self.plot_distributions(process_str, features)
        distributions['skewness shape'] = distributions[
            'skewness value'
        ].apply(interpret_skew)
        distributions['kurtosis shape'] = distributions[
            'kurtosis value'
        ].apply(interpret_kurt)
        distributions[['skewness value', 'kurtosis value']] = distributions[
            ['skewness value', 'kurtosis value']
        ].applymap('{:,.2f}'.format)
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
        axes = create_subplot_layout(len(features))
        for count, feature in enumerate(features):
            plot_scatterplot(feature, self.class_name, self.data, axes[count])
        plt.savefig(f'{PLOTS_PATH}/part1_scatter_{process_str}.png')

    @staticmethod
    def correlation_pairs(corr_matrix):
        corr_pairs = [
            (feature1, feature2, correlation_value)
            for i, feature1 in enumerate(corr_matrix.columns)
            for j, feature2 in enumerate(
                corr_matrix.columns[i + 1 :], start=i + 1
            )
            if abs(correlation_value := corr_matrix.iloc[i, j]) > 0.5
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
        display_table(
            f'Highly Correlated Feature Pairs (> 0.5) {process_str}',
            corr_pairs,
        )

    def display_influence(self, process_str):
        feature_data = self.data.drop(columns=self.class_name)
        pearson = feature_data.corrwith(self.data[self.class_name])
        plot_barplot(pearson)

        pearson = pearson.abs().sort_values(ascending=False)
        pearson = pd.DataFrame(
            {
                'feature': pearson.index,
                'pearson coefficient': pearson.values,
            }
        )
        pearson['pearson coefficient'] = pearson['pearson coefficient'].apply(
            '{:.2f}'.format
        )
        pearson.index = pearson.index + 1
        display_table(
            f'Feature Correlations with Class {process_str}', pearson
        )

    def analyse_correlations(self, process_str):
        self.display_correlations(process_str)
        self.display_influence(process_str)

    def broadly_analyse(self, process_str):
        self.plot_distributions(
            process_str, self.data.drop(columns=self.class_name).columns
        )
        self.plot_scatterplots(
            process_str, self.data.drop(columns=self.class_name).columns
        )


def preprocess_and_eda(class_name, data, data_name):
    preprocessor = Preprocessor(class_name, data)
    eda = EDA(class_name, data)

    print(f"Preprocessing {data_name} data...\n")
    print("***Encoding categorical features")
    eda.data = preprocessor.encode()

    eda.broadly_analyse('before_preprocessing')

    print("***Transforming abnormal features")
    abnormal_features = preprocessor.find_abnormal_features()
    eda.display_distributions('before_transform', abnormal_features)
    eda.data = preprocessor.transform_abnormal_features()
    eda.display_distributions('after_transform', abnormal_features)

    print("***Removing outliers")
    abnormal_features = preprocessor.find_abnormal_features()
    eda.plot_scatterplots('before_outlier_removal', abnormal_features)
    eda.data = preprocessor.remove_outliers(abnormal_features)
    eda.plot_scatterplots('after_outlier_removal', abnormal_features)

    print("***Transforming highly correlated features")
    eda.analyse_correlations('before_transform')
    eda.data = preprocessor.transform_correlated_features()
    eda.analyse_correlations('after_transform')

    print("***Removing redundant features")
    eda.data = preprocessor.remove_redundant_features()
    eda.analyse_correlations('after_feature_removal')

    print("***Scaling")
    eda.data = preprocessor.scale()

    eda.broadly_analyse('after_preprocessing')

    print("***Writing cleaned data")
    write_cleaned_data(preprocessor.data, data_name)

    print("\nPreprocessing and EDA completed successfully!\n")
