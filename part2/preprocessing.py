import pandas as pd
from sklearn.feature_selection import RFECV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from common.config import SEED
from common.preprocessing import Preprocessor
from part2.analysis import load_original_data
from part2.config import DATA_FILENAMES, MAPPINGS


class Part2Preprocessor(Preprocessor):
    def encode_nominal_features(self):
        print("***Encoding nominal categorical features")
        encoded_data = self.data.copy()
        selected_features = (
            encoded_data.drop(columns=self.class_)
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

    def cross_validate_feature_reduction(self, X, y):
        estimator = LogisticRegression()
        min_features_to_select = 1
        cv = StratifiedKFold(5, random_state=SEED)
        selector = RFECV(
            estimator,
            step=1,
            cv=cv,
            scoring="accuracy",
            min_features_to_select=min_features_to_select,
        )
        return selector.fit(X, y)

    def reduce_features(self):
        reduced_data = self.data.copy()
        X = reduced_data.drop(self.class_, axis=1)
        y = reduced_data[self.class_]

        self.selector = (
            self.cross_validate_feature_reduction(X, y)
            if self.data_name == 'train'
            else self.selector
        )

        print(f"Optimal number of features: {self.selector.n_features_}")
        self.reduction_features = self.selector.support_

        X_reduced = X.iloc[:, self.reduction_features]
        reduced_data = pd.concat([X_reduced, y], axis=1)

        results_str = (
            f'Selected features: {", ".join(list(X_reduced.columns))}'
        )
        return self.reassign_processed_data(reduced_data, results_str)

    def preprocess(self):
        self.map_ordinal_features(MAPPINGS)
        self.encode_nominal_features()
        self.transform_correlated_features()
        self.reduce_features()
        self.scale()
        self.write_cleaned_data(f'{DATA_FILENAMES[self.data_name]}')


def run_preprocessing():
    print('Prepare for preprocessing...')
    train, test = load_original_data()
    class_ = train.columns[-1]
    train_preprocessor = Part2Preprocessor(class_, train, 'train')
    train_preprocessor.preprocess()
    test_preprocessor = Part2Preprocessor(class_, test, 'test')
    test_preprocessor.preprocess()
