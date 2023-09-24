import matplotlib.pyplot as plt
import pandas as pd
from common.config import PLOTS_DIR
from common.preprocessing import EDA, Preprocessor
from common.utils import display_table, plot_barplot, write_cleaned_data
from part1.config import DATA_FILENAME


class Part1Preprocessor(Preprocessor):
    def transform_correlated_features(self):
        print("***Transforming highly correlated features")
        transformed_data = self.data.copy()

        transformed_data['volume'] = (
            transformed_data['x']
            * transformed_data['y']
            * transformed_data['z']
        )

        return self.reassign_and_return(
            transformed_data,
            'Highly correlated features transformed: x, y, z into volume',
        )

    def remove_redundant_features(self):
        print("***Removing redundant features")
        cleaned_data = self.data.copy()

        cleaned_data.drop(
            columns=['x', 'y', 'z', 'depth', 'carat'],
            inplace=True,
        )

        return self.reassign_and_return(
            cleaned_data, 'Redundant features removed: x, y, z, carat, depth'
        )


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
        display_table(
            f'Feature Correlations with Class {process_str}', pearson
        )

    def analyse_correlations(self, process_str):
        if self.data_name == 'train':
            self.display_correlations(process_str)
            self.display_influence(process_str)


def preprocess_and_eda(class_name, data, data_name, part_str):
    preprocessor = Part1Preprocessor(class_name, data)
    eda = Part1EDA(class_name, data, data_name, part_str)

    print(f"Preprocessing {data_name} data...\n")
    eda.data = preprocessor.encode()
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

    eda.data = preprocessor.remove_redundant_features()
    eda.analyse_correlations('after_feature_removal')

    eda.data = preprocessor.scale()

    eda.broadly_analyse('after_preprocessing')

    write_cleaned_data(preprocessor.data, DATA_FILENAME, data_name)

    if data_name == 'train':
        print("Preprocessing and EDA complete!\n")
    else:
        print("Preprocessing complete!\n")
