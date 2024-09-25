import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from common.preprocessing import print_processing_results, Preprocessor, get_feature_types
from common.config import SEED, Z_THRESHOLD
from census_income.analysis import get_top_n_categories, load_original_data
from census_income.config import MAPPINGS, RFE_SAMPLE_FRACTION, VAR_THRESHOLD


class CensusIncomePreprocessor(Preprocessor):
    def __init__(self, train, test, class_):
        super().__init__(train, test, class_)

        # Initialise low variance categories list
        self.categories_to_drop = {}

        # Initialise one-hot encoder and column names
        self.encoder = OneHotEncoder(drop='first', sparse_output=False)
        self.encoded_columns = []

        # Initialise reduction variables
        self.reduction_model = RandomForestClassifier(
            random_state=SEED
        )
        self.selector = RFE(estimator=self.reduction_model)
        self.n_features = None

    # Preprocessing functions
    def impute(self, set_type):
        print('\nImputing')

        # Count missing values before imputation
        before = self.dfs[set_type].isnull().sum().sum()

        # Impute occupation and workclass with Undisclosed
        self.dfs[set_type]['occupation'].fillna(
            value='Undisclosed', inplace=True)
        self.dfs[set_type]['workclass'].fillna(
            value='Undisclosed', inplace=True)

        # Impute native-country with mode
        mode_native_country = self.dfs[set_type]['native-country'].mode()[0]
        self.dfs[set_type]['native-country'].fillna(
            value=mode_native_country, inplace=True
        )

        # Count missing values after imputation
        after = self.dfs[set_type].isnull().sum().sum()

        print_processing_results('missing values', 'imputation', before, after)

    def convert_to_binary(self, set_type):
        print('\nConverting categorical columns to binary')

        # Get columns before conversion
        categorical = [
            'native-country', 'sex', 'income']
        before = '\n     ' + ", ".join(categorical)

        new_cols = []

        # Convert categorical features to binary
        for feature in categorical:
            top_category = get_top_n_categories(
                self.dfs[set_type], feature, 1
            ).index.tolist()[0]
            top_category = top_category.rstrip('.')

            # Create a new binary feature
            new_col_name = f"{feature}_{top_category}"
            self.dfs[set_type][new_col_name] = (
                self.dfs[set_type][feature] == top_category).astype(int)
            new_cols.append(new_col_name)

            # Drop original column
            self.dfs[set_type].drop(columns=feature, inplace=True)
            print(f"Original class: {self.class_}")
            if feature == self.class_:
                self.class_ = new_col_name
                print(f"Updated class: {self.class_}")

        # Move class column to end of dataframe
        print(self.dfs[set_type].columns)
        self.dfs[set_type] = self.dfs[set_type][
            [col for col in self.dfs[set_type]
                if col != self.class_] + [self.class_]
        ]

        # Get columns after conversion
        after = '\n     ' + ', '.join(new_cols)

        process_str = 'binary conversion'
        print_processing_results('features and class',
                                 process_str, before, after)

    def identify_low_variance_categories(self, set_type):
        features = ['relationship', 'marital-status', 'occupation']

        for feature in features:
            # Group by the feature and calculate the variance
            category_variances = self.dfs[set_type].groupby(
                feature)[self.class_].var()

            # Identify categories with variance below the threshold
            low_variance_categories = category_variances[
                category_variances < VAR_THRESHOLD].index

            self.categories_to_drop[feature] = low_variance_categories.tolist()

    def remove_low_variance_categories(self, set_type):
        if set_type == 'training':
            self.identify_low_variance_categories(set_type)

        print(f"\nRemoving categories with variance below {VAR_THRESHOLD}")

        # Count instances before removal
        before = self.dfs[set_type].shape[0]

        for feature, categories in self.categories_to_drop.items():
            self.dfs[set_type] = self.dfs[set_type][~self.dfs[set_type]
                                                    [feature].isin(categories)]

        # Count instances after outlier removal
        after = self.dfs[set_type].shape[0]

        print_processing_results(
            'instances', 'category removal', before, after)

    def transform_correlated_features(self, set_type):
        print("\nTransforming highly correlated features")
        # Get columns before transformation and removal
        before = len(self.dfs[set_type].columns[:-1])

        # Combine relationship and marital status
        self.dfs[set_type]['relationship_marital_status'] = self.dfs[set_type][
            'relationship'] + "_" + self.dfs[set_type]['marital-status']
        self.dfs[set_type].drop(
            columns=['relationship', 'marital-status'], inplace=True)

        pairs = [
            ('relationship_marital_status', 'sex_Male'),
            ('occupation', 'sex_Male'),
        ]

        for categorical_feature, binary_feature in pairs:
            # Create new features representing the binary feature interaction
            for category in self.dfs[set_type][categorical_feature].unique():
                new_col_name = f"{category}_{binary_feature}"

                condition1 = self.dfs[set_type][categorical_feature] == category
                condition2 = self.dfs[set_type][binary_feature] == 1

                self.dfs[set_type][new_col_name] = (
                    condition1 & condition2).astype(int)

        self.dfs[set_type].drop(columns='sex_Male', inplace=True)

        # Move class column to end of dataframe
        self.dfs[set_type] = self.dfs[set_type][
            [col for col in self.dfs[set_type]
                if col != self.class_] + [self.class_]
        ]

        # Get columns after transformation and removal
        after = len(self.dfs[set_type].columns[:-1])

        print_processing_results('features', 'transformation', before, after)

    def encode_nominal_features(self, set_type):
        print("\nEncoding nominal categorical features")
        print(self.dfs[set_type].columns)

        # Count categorical features before encoding
        _, categorical = get_feature_types(self.dfs[set_type])
        before = len(categorical)

        # Fit encoder on training dataset
        if set_type == 'training':
            self.encoder.fit(self.dfs[set_type][categorical])

        # Encode
        encoded_df = self.encoder.transform(self.dfs[set_type][categorical])
        encoded_features = self.encoder.get_feature_names_out(
            categorical).astype(str)

        # Replace the original categorical columns
        self.dfs[set_type].drop(columns=categorical, inplace=True)
        self.dfs[set_type] = pd.concat(
            [
                self.dfs[set_type],
                pd.DataFrame(
                    encoded_df,
                    columns=encoded_features,
                    index=self.dfs[set_type].index,
                ),
            ],
            axis=1,
        )

        # Save encoded columns from training
        if set_type == 'training':
            self.encoded_columns = self.dfs[set_type].columns
        else:
            # Add missing columns to test and set to 0
            for column in self.encoded_columns:
                if column not in self.dfs[set_type].columns:
                    self.dfs[set_type][column] = 0
            # Remove additional columns from test
            columns_to_drop = [
                column for column in self.dfs[set_type].columns
                if column not in self.encoded_columns]
            self.dfs[set_type].drop(columns=columns_to_drop, inplace=True)

            # Place columns in same order as training
            self.dfs[set_type] = self.dfs[set_type][self.encoded_columns]

        # Move class column to end of dataframe
        self.dfs[set_type] = self.dfs[set_type][
            [col for col in self.dfs[set_type] if col != self.class_] + [self.class_]
        ]

        # Count categorical features after encoding
        _, categorical = get_feature_types(self.dfs[set_type])
        after = len(categorical)

        print_processing_results(
            'categorical features', 'encoding', before, after)

    def cross_validate_reduction(self, set_type):
        # Use sample of data for RFE cross-validation
        sample_df = self.dfs[set_type].sample(
            frac=RFE_SAMPLE_FRACTION, random_state=SEED)
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

    def reduce_features(self, set_type):
        print("\nReducing features with RFE")

        if set_type == 'training':
            # Cross-validate to find best number of features
            self.cross_validate_reduction(set_type)

        # Separate features and class data
        X = self.dfs[set_type].drop(columns=self.class_)
        y = self.dfs[set_type][self.class_]

        # Count features before reduction
        before = len(X.columns)

        # Fit RFE with CV n_features to training dataset
        if set_type == 'training':
            self.selector = RFE(
                self.reduction_model, n_features_to_select=self.n_features)
            self.selector = self.selector.fit(X, y)

        # Reduce
        X_reduced = X.loc[:, self.selector.support_]

        # Concatenate the reduced data with class
        self.dfs[set_type] = pd.concat(
            [X_reduced, y.reset_index(drop=True)], axis=1)

        # Count features after reduction
        after = len(self.dfs[set_type].columns[:-1])

        print_processing_results('features', 'reduction', before, after)

    def preprocess(self, set_type):
        self.impute(set_type)
        self.remove_outliers(set_type, Z_THRESHOLD)
        self.map_ordinal_features(set_type, MAPPINGS)
        self.remove_redundant_features(set_type, ['education-num', 'fnlwgt'])
        self.convert_to_binary(set_type)
        self.remove_low_variance_categories(set_type)
        self.transform_correlated_features(set_type)
        self.encode_nominal_features(set_type)
        self.reduce_features(set_type)
        self.scale(set_type)
        self.write_cleaned_data(set_type)


def run_preprocessing():
    train, test = load_original_data('preprocessing')
    class_ = train.columns[-1]

    # Preprocess training data
    preprocessor = CensusIncomePreprocessor(train, test, class_)
    preprocessor.preprocess('training')

    # Preprocess test data
    preprocessor.df = test
    preprocessor.preprocess('test')
