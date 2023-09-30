import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from Part1_2_Common.analysis import get_feature_types
from Part1_2_Common.preprocessing import print_results
from Part1_2_Common.config import SEED
from Part1_2_Common.analysis import Preprocessor
from Part2.analysis import get_top_n_categories, load_original_data
from Part2.config import (DATA_FILENAMES, MAPPINGS, RFE_SAMPLE_FRACTION,
                          VAR_THRESHOLD)


class Part2Preprocessor(Preprocessor):
    def __init__(self, df):
        super().__init__(df)

        # Initialise low variance categories list
        self.categories_to_drop = {}

        # Initialise one-hot encoder and column names
        self.encoder = OneHotEncoder(drop='first', sparse=False)
        self.encoded_columns = []

        # Initialise reduction variables
        self.reduction_model = RandomForestClassifier(
            random_state=SEED
        )
        self.selector = RFE(estimator=self.reduction_model)
        self.n_features = None

    # Preprocessing functions
    def impute(self):
        print('\nImputing')

        # Count missing values before imputation
        before = self.df.isnull().sum().sum()

        # Impute occupation and workclass with Undisclosed
        self.df['occupation'].fillna(value='Undisclosed', inplace=True)
        self.df['workclass'].fillna(value='Undisclosed', inplace=True)

        # Impute native-country with mode
        mode_native_country = self.df['native-country'].mode()[0]
        self.df['native-country'].fillna(
            value=mode_native_country, inplace=True
        )

        # Count missing values after imputation
        after = self.df.isnull().sum().sum()

        print_results('missing values', 'imputation', before, after)

    def convert_to_binary(self):
        print('\nConverting categorical columns to binary')

        # Get columns before conversion
        categorical = [
            'native-country', 'sex', 'income']
        before = '\n     ' + ", ".join(categorical)

        new_cols = []

        # Convert categorical features to binary
        for feature in categorical:
            top_category = get_top_n_categories(
                self.df, feature, 1
            ).index.tolist()[0]

            # Create a new binary feature
            new_col_name = f"{feature}_{top_category}"
            self.df[new_col_name] = (
                self.df[feature] == top_category).astype(int)
            new_cols.append(new_col_name)

            # Drop original column
            self.df.drop(columns=feature, inplace=True)

            if feature == self.class_:
                self.class_ = new_col_name

        # Move class column to end of dataframe
        self.df = self.df[
            [col for col in self.df if col != self.class_] + [self.class_]
        ]

        # Get columns after conversion
        after = '\n     ' + ', '.join(new_cols)

        process_str = 'binary conversion'
        print_results('features and class', process_str, before, after)

    def identify_low_variance_categories(self):
        features = ['relationship', 'marital-status', 'occupation']

        for feature in features:
            # Group by the feature and calculate the variance
            category_variances = self.df.groupby(
                feature)[self.class_].var()

            # Identify categories with variance below the threshold
            low_variance_categories = category_variances[
                category_variances < VAR_THRESHOLD].index

            self.categories_to_drop[feature] = low_variance_categories.tolist()

    def remove_low_variance_categories(self):
        if self.data_name == 'training':
            self.identify_low_variance_categories()

        print(f"\nRemoving categories with variance below {VAR_THRESHOLD}")

        # Count instances before removal
        before = self.df.shape[0]

        for feature, categories in self.categories_to_drop.items():
            self.df = self.df[~self.df[feature].isin(categories)]

        # Count instances after outlier removal
        after = self.df.shape[0]

        print_results('instances', 'category removal', before, after)

    def transform_correlated_features(self):
        print("\nTransforming highly correlated features")
        # Get columns before transformation and removal
        before = len(self.df.columns[:-1])

        # Combine relationship and marital status
        self.df['relationship_marital_status'] = self.df[
            'relationship'] + "_" + self.df['marital-status']
        self.df.drop(columns=['relationship', 'marital-status'], inplace=True)

        pairs = [
            ('relationship_marital_status', 'sex_Male'),
            ('occupation', 'sex_Male'),
        ]

        for categorical_feature, binary_feature in pairs:
            # Create new features representing the binary feature interaction
            for category in self.df[categorical_feature].unique():
                new_col_name = f"{category}_{binary_feature}"

                condition1 = self.df[categorical_feature] == category
                condition2 = self.df[binary_feature] == 1

                self.df[new_col_name] = (condition1 & condition2).astype(int)

        self.df.drop(columns='sex_Male', inplace=True)

        # Move class column to end of dataframe
        self.df = self.df[
            [col for col in self.df if col != self.class_] + [self.class_]
        ]

        # Get columns after transformation and removal
        after = len(self.df.columns[:-1])

        print_results('features', 'transformation', before, after)

    def encode_nominal_features(self):
        print("\nEncoding nominal categorical features")
        print(self.df.columns)

        # Count categorical features before encoding
        _, categorical = get_feature_types(self.df)
        before = len(categorical)

        # Fit encoder on training dataset
        if self.data_name == 'training':
            self.encoder.fit(self.df[categorical])

        # Encode
        encoded_df = self.encoder.transform(self.df[categorical])
        encoded_features = self.encoder.get_feature_names_out(
            categorical).astype(str)

        # Replace the original categorical columns
        self.df.drop(columns=categorical, inplace=True)
        self.df = pd.concat(
            [
                self.df,
                pd.DataFrame(
                    encoded_df,
                    columns=encoded_features,
                    index=self.df.index,
                ),
            ],
            axis=1,
        )

        # Save encoded columns from training
        if self.data_name == 'training':
            self.encoded_columns = self.df.columns
        else:
            # Add missing columns to test and set to 0
            for column in self.encoded_columns:
                if column not in self.df.columns:
                    self.df[column] = 0
            # Remove additional columns from test
            columns_to_drop = [
                column for column in self.df.columns
                if column not in self.encoded_columns]
            self.df.drop(columns=columns_to_drop, inplace=True)

            # Place columns in same order as training
            self.df = self.df[self.encoded_columns]

        # Move class column to end of dataframe
        self.df = self.df[
            [col for col in self.df if col != self.class_] + [self.class_]
        ]

        # Count categorical features after encoding
        _, categorical = get_feature_types(self.df)
        after = len(categorical)

        print_results('categorical features', 'encoding', before, after)

    def cross_validate_reduction(self):
        # Use sample of data for RFE cross-validation
        sample_df = self.df.sample(frac=RFE_SAMPLE_FRACTION, random_state=SEED)
        sample_X = sample_df.drop(columns=self.class_)
        sample_y = sample_df[self.class_]

        # Initialise the highest score and early stopping counter
        highest_score = 0
        no_improve = 0
        early_stopping_rounds = 5

        # Determine the number of features to retain
        for i in range(47, 50):
            selector = RFE(self.reduction_model, n_features_to_select=i)
            pipeline = Pipeline(steps=[('rfe', selector), (
                'model', self.reduction_model)])
            score = cross_val_score(
                pipeline,
                sample_X,
                sample_y,
                cv=5,
                n_jobs=-1,
                scoring="accuracy"
            ).mean()

            if score > highest_score:
                highest_score = score
                self.n_features = i
                no_improve = 0  # reset counter
            else:
                no_improve += 1  # increment counter
            if no_improve >= early_stopping_rounds:
                break

    def reduce_features(self):
        print("\nReducing features with RFE")

        if self.data_name == 'training':
            # Cross-validate to find best number of features
            self.cross_validate_reduction()

        # Separate features and class data
        X = self.df.drop(columns=self.class_)
        y = self.df[self.class_]

        # Count features before reduction
        before = len(X.columns)

        # Fit RFE with CV n_features to training dataset
        if self.data_name == 'training':
            self.selector = RFE(
                self.reduction_model, n_features_to_select=self.n_features)
            self.selector = self.selector.fit(X, y)

        # Reduce
        X_reduced = X.loc[:, self.selector.support_]

        # Concatenate the reduced data with class
        self.df = pd.concat([X_reduced, y.reset_index(drop=True)], axis=1)

        # Count features after reduction
        after = len(self.df.columns[:-1])

        print_results('features', 'reduction', before, after)

    def preprocess(self):
        self.impute()
        self.remove_outliers()
        self.map_ordinal_features(MAPPINGS)
        self.remove_redundant_features(['education-num', 'fnlwgt'])
        self.convert_to_binary()
        self.remove_low_variance_categories()
        self.transform_correlated_features()
        self.encode_nominal_features()
        self.reduce_features()
        self.scale()
        self.write_cleaned_data(f'{DATA_FILENAMES[self.data_name]}')


def run_preprocessing():
    train, test = load_original_data('preprocessing')

    # Preprocess training data
    preprocessor = Part2Preprocessor(train)
    preprocessor.set_data_name('training')
    preprocessor.preprocess()

    # Preprocess test data
    preprocessor.df = test
    preprocessor.set_data_name('test')
    preprocessor.preprocess()
