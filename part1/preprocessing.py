import pandas as pd
from sklearn.model_selection import train_test_split

from common.config import SEED
from common.preprocessing import Preprocessor
from part1.analysis import load_original_data
from part1.config import DATA_FILENAME, MAPPINGS


def split_data():
    data = load_original_data()
    class_ = data.columns[-1]
    train_X, test_X, train_y, test_y = train_test_split(
        data.drop(columns=class_),
        data[class_],
        test_size=0.3,
        random_state=SEED,
    )
    train = pd.concat([train_X, train_y], axis=1)
    test = pd.concat([test_X, test_y], axis=1)
    return train, test, class_


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

    def preprocess(self):
        self.map_ordinal_features(MAPPINGS)
        self.remove_outliers()
        self.transform_correlated_features()
        self.scale()
        self.write_cleaned_data(f'{DATA_FILENAME}_{self.data_name}')


def run_preprocessing():
    print('Prepare for preprocessing...')
    train, test, class_ = split_data()
    train_preprocessor = Part1Preprocessor(class_, train, 'train')
    train_preprocessor.preprocess()
    test_preprocessor = Part1Preprocessor(class_, test, 'test')
    test_preprocessor.preprocess()
