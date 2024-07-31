import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from scipy.stats import pearsonr, zscore
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

from Part1_2_Common.config import KDE_SAMPLE_SIZE, ORIGINAL_DATA_DIR, SEED
from Part1_2_Common.preprocessing import (
    Preprocessor,
    calculate_skewness_and_kurtosis,
    condense_and_export_tables,
    create_plot_layout,
    export_and_show_plot,
    format_plot_axes,
    get_feature_pair_correlations,
    get_feature_types,
    get_missing_values,
    plot_correlation_matrix,
    plot_numerical_distributions,
    print_process_heading,
    print_processing_results,
    save_and_print_table,
    summarise,
)

DATA_FILENAME = "diamonds"

MAPPINGS = {
    "cut": {
        "Fair": 1,
        "Good": 2,
        "Very Good": 3,
        "Premium": 4,
        "Ideal": 5,
    },
    "color": {"J": 1, "I": 2, "H": 3, "G": 4, "F": 5, "E": 6, "D": 7},
    "clarity": {
        "I1": 1,
        "SI2": 2,
        "SI1": 3,
        "VS2": 4,
        "VS1": 5,
        "VVS2": 6,
        "VVS1": 7,
        "IF": 8,
    },
}

Z_THRESHOLD = 9


# Data handling
def split_data(df):
    class_ = df.columns[-1]
    train_X, test_X, train_y, test_y = train_test_split(
        df.drop(columns=class_),
        df[class_],
        test_size=0.3,
        random_state=SEED,
    )

    # Concatenate train and test datasets
    train = pd.concat([train_X, train_y], axis=1)
    test = pd.concat([test_X, test_y], axis=1)

    return train, test, class_


# IDA
def plot_missingness(df, class_):
    # Get features with missing values
    missing = df.isnull().any()
    features = missing[missing].index.tolist()

    # Plot boxplots
    axes = create_plot_layout(len(features))
    for count, feature in enumerate(features):
        x_labels = (
            df[feature].isnull().map({True: "Missing", False: "Non-Missing"})
        )
        boxplot = sns.boxplot(
            x=x_labels,
            y=df[class_],
            ax=axes[count],
        )
        format_plot_axes(boxplot, feature.capitalize(), class_.capitalize())

    # Results
    export_and_show_plot("missingness_to_class")


def plot_outliers(df, class_):
    # Calculate z-score for numerical features
    numerical, _ = get_feature_types(df)
    z_scores = np.abs(zscore(df[numerical]))

    # Plot z-score scatterplots
    axes = create_plot_layout(len(numerical), 2)
    for count, feature in enumerate(numerical):
        scatter = sns.scatterplot(
            # Convert single column dataframes to series
            x=df[feature].squeeze(),
            y=df[class_].squeeze(),
            hue=z_scores[feature].squeeze(),
            ax=axes[count],
            palette="viridis",
        )
        format_plot_axes(scatter, feature, class_)

    # Results
    export_and_show_plot("outliers")


def plot_duplicates(df, class_):
    # Separate duplicate rows from rest of dataset
    duplicates = df[df.duplicated()]
    non_duplicates = df.drop_duplicates()

    # Plot KDE plots
    plt.figure(figsize=(10, 6))
    sns.kdeplot(
        duplicates[class_],
        shade=True,
        color="r",
        label="Duplicates",
    )
    sns.kdeplot(
        non_duplicates[class_],
        shade=True,
        color="b",
        label="Non-Duplicates",
    )
    plt.legend()

    # Results
    export_and_show_plot("duplicates")


# EDA
def plot_class_distribution(df, class_):
    axes = create_plot_layout(2)

    # Plot and format histogram
    histogram = sns.histplot(data=df, x=class_, bins="doane", ax=axes[0])
    format_plot_axes(histogram, class_, "frequency")

    # Plot and format boxplot
    boxplot = sns.boxplot(data=df, y=class_, ax=axes[1])
    format_plot_axes(boxplot, "frequency", class_)

    export_and_show_plot("class_distribution")


def plot_numerical_to_class(df, numerical):
    class_ = df.columns[-1]

    # Sample data to decrease computation load
    sample_df = df.sample(n=KDE_SAMPLE_SIZE)

    axes = create_plot_layout(len(numerical))

    # Plot and format KDE plots
    for count, feature in enumerate(numerical):
        kde = sns.kdeplot(
            data=sample_df, x=feature, y=class_, ax=axes[count], fill=True
        )
        format_plot_axes(kde, feature, class_)

    export_and_show_plot("numerical_to_class")


def get_numerical_correlations(df, numerical):
    feature_df = df[numerical]

    # Initialise list to store Pearson results
    pearson = []

    # Calculate Pearson between numerical features and class
    for feature in feature_df.columns:
        correlation, p_value = pearsonr(
            feature_df[feature], df[df.columns[-1]]
        )
        pearson.append(
            {
                "feature": feature,
                "correlation": correlation,
                "p_value": p_value,
            }
        )

    pearson_df = pd.DataFrame(pearson)
    pearson_df.set_index("feature", inplace=True)

    # Sort by absolute value of Pearson coefficient
    return pearson_df.sort_values(
        by="correlation", key=lambda x: x.abs(), ascending=False
    )


def plot_class_correlations(corrs_df):
    axes = create_plot_layout(1)

    # Plot and format barplot for numerical features
    numerical_barplot = sns.barplot(
        data=corrs_df,
        x="correlation",
        y=corrs_df.index,
        ax=axes[0],
    )
    format_plot_axes(numerical_barplot, "pearson coefficient", "feature")

    export_and_show_plot("class_correlations")


# Data processing and analysis
class Part1Preprocessor(Preprocessor):
    def __init__(self, train, test, class_):
        super().__init__(train, test, class_)

        # Initialise imputing model
        self.imputer = IterativeImputer(
            estimator=LinearRegression(),
            random_state=SEED,
        )

    # Cleaning
    def impute(self, set_type):
        print("\nImputing")

        # Count missing values before imputation
        before = self.dfs[set_type].isnull().sum().sum()

        # Split data into features and class
        X = self.dfs[set_type].drop(columns=[self.class_])
        y = self.dfs[set_type][self.class_]

        # Define features with linear relationships to class
        linear_features = ["carat", "x", "y", "z"]
        non_linear_features = [
            feature for feature in X.columns if feature not in linear_features
        ]

        # Fit imputer on training set
        if set_type == "training":
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
            [imputed_linear_X, X[non_linear_features]], axis=1
        )

        # Concatenate feature data with class data
        self.dfs[set_type] = pd.concat([imputed_X, y], axis=1)

        # Count missing values after imputation
        after = self.dfs[set_type].isnull().sum().sum()
        print_processing_results("missing values", "imputation", before, after)

    def run_cleaning(self):
        for set_type in ["training", "test"]:
            print_process_heading(f"cleaning {set_type} data")

            # Replace missing value signifiers with NaN
            self.identify_missing_values(0, set_type)

            # Analyse training set missing values
            if set_type == "training":
                get_missing_values(self.dfs[set_type])
                plot_missingness(self.dfs[set_type], self.class_)

            # Impute
            self.impute(set_type)
            print(f"\n--- Cleaning {set_type} data complete! ---\n")

    def run_further_cleaning(self):
        for set_type in ["training", "test"]:
            print_process_heading(f"cleaning {set_type} data further")

            # Analyse training set outliers
            if set_type == "training":
                plot_outliers(self.dfs[set_type], self.class_)

            # Remove outliers using z-score method
            self.remove_outliers(set_type, Z_THRESHOLD)

            # Analyse training set duplicates
            if set_type == "training":
                plot_duplicates(self.dfs[set_type], self.class_)

            # Remove duplicates
            # self.remove_duplicates(set_type)

            # Convert all features to numerical
            self.map_ordinal_features(set_type, MAPPINGS)
            print(f"\n--- Further cleaning of {set_type} data complete! ---\n")

    # EDA
    def run_eda(self):
        print_process_heading("analysing train data")

        # Univariate analysis - class
        plot_class_distribution(self.dfs["training"], self.class_)

        # Univariate analysis - features
        numerical, _ = get_feature_types(self.dfs["training"])
        plot_numerical_distributions(self.dfs["training"], numerical)
        calculate_skewness_and_kurtosis(self.dfs["training"], numerical)

        # Bivariate analysis - feature-to-class
        plot_numerical_to_class(self.dfs["training"], numerical)

        # Multivariate analysis - feature-to-feature correlations
        numerical_corr_matrix = self.dfs["training"][numerical].corr(
            method="pearson"
        )
        plot_correlation_matrix(numerical_corr_matrix, "numerical")
        get_feature_pair_correlations(numerical_corr_matrix, "numerical")

        # Multivariate analysis - feature-to-class correlations
        numerical_corrs = get_numerical_correlations(
            self.dfs["training"], numerical
        )
        save_and_print_table(
            "numerical correlations with class", numerical_corrs
        )

        # Export tables
        condense_and_export_tables()
        print("\n--- Analysis of train data complete! ---\n")

    # Preprocessing
    def transform_correlated_features(self, set_type):
        print("\nTransforming and/or removing highly correlated features")

        # Get columns before transformation and removal
        before = "\n     " + ", ".join(list(self.dfs[set_type].columns))

        # Create new volume feature
        self.dfs[set_type]["volume"] = (
            self.dfs[set_type]["x"]
            * self.dfs[set_type]["y"]
            * self.dfs[set_type]["z"]
        )

        # Remove all highly correlated features
        self.dfs[set_type].drop(columns=["x", "y", "z", "carat"], inplace=True)

        # Move class column to end of dataframe
        self.dfs[set_type] = self.dfs[set_type][
            [col for col in self.dfs[set_type] if col != self.class_]
            + [self.class_]
        ]

        # Get columns after transformation and removal
        after = "\n     " + ", ".join(
            list(self.dfs[set_type].drop(columns=self.class_).columns)
        )

        print_processing_results("features", "transformation", before, after)

    def run_preprocessing(self):
        for set_type in ["training", "test"]:
            print_process_heading(f"preprocessing {set_type} data")

            # Transform and remove highly correlated features
            self.transform_correlated_features(set_type)

            # Scale
            self.scale(set_type)

            # Write cleaned data to CSV
            self.write_cleaned_data(set_type)


# Main
def run_preprocessing_and_analysis():
    # Loading
    df = pd.read_csv(f"{ORIGINAL_DATA_DIR}/{DATA_FILENAME}.csv", index_col=0)

    # Splitting
    train, test, class_ = split_data(df)

    # Cleaning and analysis
    preprocessor = Part1Preprocessor(train, test, class_)
    preprocessor.run_cleaning()

    # IDA
    summarise(preprocessor.dfs["training"])

    # Further cleaning and analysis
    preprocessor.run_further_cleaning()

    # EDA
    preprocessor.run_eda()

    # Preprocessing
    preprocessor.run_preprocessing()
