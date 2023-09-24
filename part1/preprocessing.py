import matplotlib.pyplot as plt
import pandas as pd
from common.config import PLOTS_DIR
from common.preprocessing import EDA, Preprocessor
from common.utils import display_table, plot_barplot, write_cleaned_data
from part1.config import DATA_FILENAME, MAPPINGS


class Part1Preprocessor(Preprocessor):
    def transform_correlated_features(self):
        print("***Transforming and removing highly correlated features")
        transformed_data = self.data.copy()
        selected_features = ['x', 'y', 'z']

        transformed_data['volume'] = (
            transformed_data['x']
            * transformed_data['y']
            * transformed_data['z']
        )
        transformed_data.drop(columns=selected_features, inplace=True)
        transformed_data.drop(columns=['carat'], inplace=True)

        print('Correlated feature removed: carat')
        results_str = (
            f'Correlated features transformed: {", ".join(selected_features)}'
        )
        return self.reassign_processed_data(transformed_data, results_str)


class Part1EDA(EDA):
    def display_influence(self, process_str):
        feature_data = self.data.drop(columns=self.class_name)
        pearson = feature_data.corrwith(self.data[self.class_name])

        plot_barplot(pearson)
        plt.savefig(f'{PLOTS_DIR}/{self.part_str}_class_correlations.png')

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
        display_table('Feature Correlations with Class', pearson)

    def analyse_correlations(self, process_str):
        if self.data_name == 'train':
            self.display_correlations(process_str)
            self.display_influence(process_str)


def preprocess_and_eda(class_name, data, data_name, part_str):
    preprocessor = Part1Preprocessor(class_name, data)
    eda = Part1EDA(class_name, data, data_name, part_str)

    print(f"Preprocessing {data_name} data...\n")
    eda.data = preprocessor.impute()
    eda.data = preprocessor.map_ordinal_features(MAPPINGS)
    eda.broadly_analyse('before_preprocessing')

    abnormal_features = preprocessor.find_abnormal_features()
    eda.display_distributions('before_transform', abnormal_features)
    eda.data = preprocessor.transform_abnormal_features()
    eda.display_distributions('after_transform', abnormal_features)

    eda.plot_scatterplots(
        'before_outlier_removal', eda.data.drop(columns=class_name).columns
    )
    eda.data = preprocessor.remove_outliers()
    eda.plot_scatterplots(
        'after_outlier_removal',
        eda.data.drop(columns=eda.class_name).columns,
    )

    eda.analyse_correlations('before_transform')
    eda.data = preprocessor.transform_correlated_features()
    eda.analyse_correlations('after_transform')

    eda.data = preprocessor.scale()

    eda.broadly_analyse('after_preprocessing')

    write_cleaned_data(preprocessor.data, f'{DATA_FILENAME}_{data_name}')

    if data_name == 'train':
        print("Preprocessing and EDA complete!\n")
    else:
        print("Preprocessing complete!\n")
