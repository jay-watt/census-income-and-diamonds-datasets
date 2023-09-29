import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

from Part1.cleaning import load_original_data
from Part1.config import DATA_FILENAME, MAPPINGS
from Part1_2_Common.cleaning import print_results
from Part1_2_Common.config import SEED
from Part1_2_Common.preprocessing import Preprocessor


def split_data(df):
    class_ = df.columns[-1]
    train_X, test_X, train_y, test_y = train_test_split(
        df.drop(columns=class_),
        df[class_],
        test_size=0.3,
        random_state=SEED,
    )
    train = pd.concat([train_X, train_y], axis=1)
    test = pd.concat([test_X, test_y], axis=1)

    return train, test


class Part1Preprocessor(Preprocessor):
    def __init__(self, df):
        super().__init__(df)

        # Initialise imputing model
        self.imputer = IterativeImputer(
            estimator=LinearRegression(),
            random_state=SEED,
        )

    def impute(self):
        print('\nImputing')

        # Count missing values before imputation
        before = self.df.isnull().sum().sum()

        # Split data into features and class
        X = self.df.drop(columns=[self.class_])
        y = self.df[self.class_]

        # Define features with linear relationships to class
        linear_features = ['carat', 'x', 'y', 'z']
        non_linear_features = [
            feature for feature in X.columns if feature not in linear_features]

        # Fit imputer if current dataset is training dataset
        if self.data_name == 'training':
            linear_X = X[linear_features]
            self.imputer.fit(linear_X)

        # Impute
        imputed_linear_X_array = self.imputer.transform(X[linear_features])

        # Create dataframe from imputed data
        imputed_linear_X = pd.DataFrame(
            imputed_linear_X_array,
            columns=linear_features,
            index=X.index,
        )
        # Concatenate imputed linear features with non-linear features
        imputed_X = pd.concat(
            [imputed_linear_X, X[non_linear_features]], axis=1)

        self.df = pd.concat([imputed_X, y], axis=1)

        # Count missing values after imputation
        after = self.df.isnull().sum().sum()

        print_results('missing values', 'imputation', before, after)

    def transform_correlated_features(self):
        print("\nTransforming and/or removing highly correlated features")

        # Get columns before transformation and removal
        before = ', '.join(list(self.df.columns))

        # Create new volume feature
        self.df['volume'] = self.df['x'] * self.df['y'] * self.df['z']

        # Remove all highly correlated features
        self.df.drop(columns=['x', 'y', 'z', 'carat'], inplace=True)

        # Move class column to end of dataframe
        self.df = self.df[
            [col for col in self.df if col != self.class_] + [self.class_]
        ]

        # Get columns after transformation and removal
        after = ', '.join(list(self.df.drop(columns=self.class_).columns))

        print_results('features', 'transformation', before, after)

    def preprocess(self):
        self.map_ordinal_features(MAPPINGS)
        self.impute()
        self.transform_correlated_features()
        self.remove_outliers()
        self.scale()
        self.write_cleaned_data(f'{DATA_FILENAME}_{self.data_name}')


# Main preprocessing function
def run_preprocessing():
    data = load_original_data('preprocessing')
    train, test = split_data(data)

    # Preprocess training data
    preprocessor = Part1Preprocessor(train)
    preprocessor.set_data_name('training')
    preprocessor.preprocess()

    # Preprocess test data
    preprocessor.df = test
    preprocessor.set_data_name('test')
    preprocessor.preprocess()
