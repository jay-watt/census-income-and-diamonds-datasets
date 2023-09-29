import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from Part1_2_Common.analysis import (get_feature_pair_correlations,
                                     get_feature_types)
from Part1_2_Common.cleaning import export_and_print_table, print_results
from Part1_2_Common.config import SEED
from Part1_2_Common.preprocessing import Preprocessor
from Part2.analysis import (get_numerical_correlations, get_top_n_categories,
                            load_original_data)
from Part2.config import DATA_FILENAMES, MAPPINGS, PCA_SAMPLE_FRACTION


class Part2Preprocessor(Preprocessor):
    def __init__(self, df):
        super().__init__(df)

        # Initialise capital feature thresholds
        self.binary_thresholds = {}

        # Initialise one-hot encoder
        self.encoder = OneHotEncoder(drop='first', sparse=False)

        # Initialise feature reduction hyperparameter and model
        self.n_components = 0
        self.pca = PCA()
        self.reduction_model = RandomForestClassifier(
            random_state=SEED, n_jobs=-1
        )

    # Preprocessing assessment functions
    def assess_class_correlations(self, process_str):
        numerical, _ = get_feature_types(self.df)
        numerical_corrs = get_numerical_correlations(self.df, numerical)
        export_and_print_table(
            f'{process_str} assessment', numerical_corrs)

    def assess_feature_correlations(self):
        numerical, _ = get_feature_types(self.df)

        # Calculate Pearson
        numerical_corr_matrix = self.df[numerical].corr(method='pearson')
        get_feature_pair_correlations(numerical_corr_matrix, 'numerical')

    # Preprocessing functions
    def percentile_reduce(self):
        features_to_reduce = ['age', 'hours-per-week']
        
        for feature in features_to_reduce:
            # Calculate the 0.5th and 99.5th percentiles
            lower_limit = df[feature].quantile(0.005)
            upper_limit = df[feature].quantile(0.995)

            # Filter the DataFrame to retain only rows within the percentile limits
            self.df = self.df[(self.df[feature] >= lower_limit) & (self.df[feature] <= upper_limit)]
        
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

    def calculate_threshold(self, feature):
        sorted_df = self.df.sort_values(by=[self.class_, feature])

        # Find the 95th percentile for the lower category
        lower_category = sorted_df[self.class_].min()
        lower_category_df = sorted_df[sorted_df[self.class_] == lower_category]
        threshold = (lower_category_df[feature].quantile(
            0.99))
        self.binary_thresholds[feature] = int(threshold)

    def convert_continuous_to_binary(self, feature):
        # Calculate the threshold with training dataset
        if self.data_name == 'training':
            self.calculate_threshold(feature)

        # Create a new binary categorical feature based on the threshold
        self.df[f"{feature}_>{self.binary_thresholds[feature]}"] = self.df[
            feature
        ].apply(lambda x: 1 if x > self.binary_thresholds[feature] else 0)

        self.df.drop(columns=feature, inplace=True)

    def convert_to_binary(self):
        numerical = ['capital-gain', 'capital-loss']
        categorical = [
            'native-country', 'sex', 'income']

        print(
            f'\nConverting categorical columns {", ".join(categorical)},')
        print(
            f'and numerical columns {", ".join(numerical)} to binary')

        # Get columns before conversion
        before = ', '.join(list(self.df.columns))

        # Convert numerical features to binary
        for feature in numerical:
            self.convert_continuous_to_binary(feature)

        # Convert categorical features to binary
        for feature in categorical:
            top_category = get_top_n_categories(
                self.df, feature, 1
            ).index.tolist()[0]

            # Create a new binary feature
            new_col_name = f"{feature}_{top_category}"
            self.df[new_col_name] = (
                self.df[feature] == top_category).astype(int)

            # Drop original column
            self.df.drop(columns=feature, inplace=True)

            if feature == self.class_:
                self.class_ = new_col_name

        # Move class column to end of dataframe
        self.df = self.df[
            [col for col in self.df if col != self.class_] + [self.class_]
        ]

        # Get columns after conversion
        after = ', '.join(list(self.df.columns))

        process_str = 'binary conversion'
        print_results('features and class', process_str, before, after)
        self.assess_class_correlations(process_str)

    def transform_correlated_features(self):
        print("\nTransforming highly correlated features")

        # Get columns before transformation and removal
        before = len(self.df.columns[:-1])

        # Define the pairs of categorical and binary numeric features
        pairs = [
            ('relationship', 'sex_Male'),
            ('marital-status', 'sex_Male'),
            ('race', 'native-country_United-States')
        ]

        # Loop through each pair and create interaction terms
        for categorical_feature, binary_feature in pairs:
            # Create new features representing the binary feature interaction
            for category in self.df[categorical_feature].unique():
                new_col_name = f"{category}_{binary_feature}"

                condition1 = self.df[categorical_feature] == category
                condition2 = self.df[binary_feature] == 1

                self.df[new_col_name] = (condition1 & condition2).astype(int)

        # Remove all highly correlated features
        # self.df.drop(columns=['x', 'y', 'z', 'carat'], inplace=True)

        # Get columns after transformation and removal
        after = len(self.df.columns[:-1])

        print_results('features', 'transformation', before, after)
        self.assess_feature_correlations()

    def encode_nominal_features(self):
        print("\nEncoding nominal categorical features")

        # Count categorical features before encoding
        _, categorical = get_feature_types(self.df)
        before = len(categorical)

        # Fit encoder on training dataset
        if self.data_name == 'training':
            self.encoder.fit(self.df[categorical])

        # Encode
        encoded_df = self.encoder.transform(self.df[categorical])
        encoded_features = self.encoder.get_feature_names_out(categorical)

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

        # Move class column to end of dataframe
        self.df = self.df[
            [col for col in self.df if col != self.class_] + [self.class_]
        ]

        # Count categorical features after encoding
        _, categorical = get_feature_types(self.df)
        after = len(categorical)

        print_results('categorical features', 'encoding', before, after)

    def cross_validate_reduction(self):
        # Use sample of data for PCA cross-validation
        sample_df = self.df.sample(frac=PCA_SAMPLE_FRACTION, random_state=SEED)
        sample_X = sample_df.drop(columns=self.class_)
        sample_y = sample_df[self.class_]

        # Initialise the explained variance ratio and early stopping counter
        explained_variance_ratio = 0
        no_improve = 0
        early_stopping_rounds = 5

        # Determine the number of components to retain
        for i in range(1, sample_X.shape[1] + 1, 5):
            pca = PCA(n_components=i, random_state=SEED)
            pipeline = Pipeline(
                steps=[('pca', pca), ('model', self.reduction_model)]
            )
            score = cross_val_score(
                pipeline,
                sample_X,
                sample_y,
                cv=5,
                n_jobs=-1,
                scoring="accuracy",
            ).mean()

            if score > explained_variance_ratio:
                explained_variance_ratio = score
                self.n_components = i
                no_improve = 0  # reset counter
            else:
                no_improve += 1  # increment counter
            if no_improve >= early_stopping_rounds:
                break

    def reduce_features(self):
        print("\nReducing features with PCA")

        if self.data_name == 'training':
            # Cross-validate to find best n_components
            self.cross_validate_reduction()

        # Separate features and class data
        X = self.df.drop(columns=self.class_)
        y = self.df[self.class_]

        # Count features before reduction
        before = len(X.columns)

        # Fit PCA with CV n_components to training dataset
        if self.data_name == 'training':
            self.pca = PCA(n_components=self.n_components, random_state=SEED)
            self.pca.fit(X)

        # Reduce
        X_reduced = self.pca.transform(X)

        # Convert the reduced data to dataframe and concatenate with class
        X_reduced = pd.DataFrame(
            X_reduced,
            columns=[f"PC{i}" for i in range(1, self.n_components + 1)],
        )
        self.df = pd.concat([X_reduced, y.reset_index(drop=True)], axis=1)

        # Count features after reduction
        after = len(self.df.columns[:-1])

        print_results('features', 'reduction', before, after)

    def preprocess(self):
        self.percentile_reduce()
        self.impute()
        self.map_ordinal_features(MAPPINGS)
        self.assess_class_correlations('ordinal mapping')
        self.remove_redundant_features(['education-num', 'fnlwgt'])
        self.convert_to_binary()
        self.transform_correlated_features()
        self.remove_outliers()
        self.encode_nominal_features()
        # self.reduce_features()
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
