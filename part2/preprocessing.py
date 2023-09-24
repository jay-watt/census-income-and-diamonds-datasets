import matplotlib.pyplot as plt
import pandas as pd
from common.config import PLOTS_DIR
from common.preprocessing import EDA, Preprocessor
from common.utils import display_table, plot_barplot, write_cleaned_data
from part2.config import CHI2_THRESHOLD, DATA_FILENAMES, MAPPINGS
from scipy.stats import chi2_contingency


class Part2Preprocessor(Preprocessor):
    def encode_nominal_features(self):
        print("***Encoding nominal categorical features")
        encoded_data = self.data.copy()
        selected_features = (
            encoded_data.drop(columns=self.class_name)
            .select_dtypes(include=['object'])
            .columns
        )
        encoded_data = pd.get_dummies(
            encoded_data, columns=selected_features, drop_first=True
        )
        one_hot_columns = encoded_data.columns.difference(self.data.columns)
        encoded_data[one_hot_columns] = encoded_data[one_hot_columns].astype(
            int
        )
        results_str = (
            f'Features one-hot encoded: {", ".join(selected_features)}'
        )
        return self.reassign_processed_data(encoded_data, results_str)

    def transform_correlated_features(self):
        print("***Transforming and removing highly correlated features")
        transformed_data = self.data.copy()
        selected_features = ['race_White', 'race_Black']

        transformed_data['race'] = transformed_data.apply(
            lambda x: 1
            if x['race_Black'] == 1
            else 2
            if x['race_White'] == 1
            else 0,
            axis=1,
        )
        transformed_data.drop(columns=selected_features, inplace=True)
        transformed_data.drop(columns=['education-num'], inplace=True)

        print('Correlated feature removed: education-num')
        results_str = (
            f'Correlated features transformed: {", ".join(selected_features)}'
        )
        return self.reassign_processed_data(
            transformed_data,
            results_str,
        )


class Part2EDA(EDA):
    def display_influence(self, chi2_results, process_str):
        plot_barplot(chi2_results.set_index('feature')['chi2'])
        fig_path = f'{PLOTS_DIR}/{self.part_str}_influences_{process_str}.png'
        plt.savefig(fig_path)

        chi2_results['chi2'] = chi2_results['chi2'].apply('{:,.2f}'.format)
        chi2_results['p-value'] = chi2_results['p-value'].apply(
            '{:.2e}'.format
        )
        chi2_results.index.name = 'ranking'
        display_table(
            f'Feature Influence on Class (Chi2 > {CHI2_THRESHOLD})',
            chi2_results,
        )

    def get_redundant_features(self, process_str):
        feature_data = self.data.drop(columns=self.class_name)
        class_data = self.data[self.class_name]

        chi2_results = []
        for feature in feature_data.columns:
            contingency_table = pd.crosstab(feature_data[feature], class_data)
            chi2, p, _, _ = chi2_contingency(contingency_table)
            chi2_results.append(
                {'feature': feature, 'chi2': chi2, 'p-value': p}
            )

        chi2_results = pd.DataFrame(chi2_results)
        chi2_results = chi2_results.sort_values(by='chi2', ascending=False)
        redundant_features = chi2_results[
            chi2_results['chi2'] <= CHI2_THRESHOLD
        ]['feature'].tolist()
        chi2_results = chi2_results[chi2_results['chi2'] > CHI2_THRESHOLD]
        chi2_results.index = range(1, len(chi2_results) + 1)

        if self.data_name == 'train':
            self.display_influence(chi2_results, process_str)

        return redundant_features

    def analyse_correlations(self, process_str):
        if self.data_name == 'train':
            self.display_correlations(process_str)
        return self.get_redundant_features(process_str)


def preprocess_and_eda(
    class_name, data, data_name, part_str, redundant_features
):
    preprocessor = Part2Preprocessor(class_name, data)
    eda = Part2EDA(class_name, data, data_name, part_str)

    print(f"Preprocessing {data_name} data...\n")
    eda.data = preprocessor.impute()
    eda.data = preprocessor.map_ordinal_features(MAPPINGS)
    eda.data = preprocessor.encode_nominal_features()

    if data_name == 'train':
        redundant_features = eda.analyse_correlations('before_feature_removal')
    eda.data = preprocessor.transform_correlated_features()
    eda.data = preprocessor.remove_redundant_features(redundant_features)

    eda.data = preprocessor.scale()

    write_cleaned_data(preprocessor.data, DATA_FILENAMES[data_name])

    if data_name == 'train':
        print("Preprocessing and EDA complete!\n")
    else:
        print("Preprocessing complete!\n")

    return redundant_features
