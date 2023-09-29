import time

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge, SGDRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR, LinearSVR
from sklearn.tree import DecisionTreeRegressor

from Part1.config import CV_SAMPLE_FRACTION, DATA_FILENAME
from Part1_2_Common.analysis import print_process_heading
from Part1_2_Common.config import SEED
from Part1_2_Common.modelling import (display_modelling_results,
                                      load_cleaned_data)


# Preparation functions
def initialise_models():
    return [
        ('Linear', LinearRegression()),
        ('K-Neighbors', KNeighborsRegressor()),
        ('Ridge', Ridge(random_state=SEED)),
        ('Decision Tree', DecisionTreeRegressor(random_state=SEED)),
        (
            'Random Forest',
            RandomForestRegressor(random_state=SEED),
        ),
        (
            'Gradient Boosting',
            GradientBoostingRegressor(random_state=SEED),
        ),
        ('SGD', SGDRegressor(random_state=SEED)),
        ('Support Vector', SVR()),
        ('Linear SVR', LinearSVR(random_state=SEED)),
        (
            'Multi-layer Perceptron',
            MLPRegressor(random_state=SEED),
        ),
    ]


def prepare_for_modelling():
    print_process_heading('modelling')

    X_train, y_train = load_cleaned_data(f'{DATA_FILENAME}_training')
    X_test, y_test = load_cleaned_data(f'{DATA_FILENAME}_test')
    return X_train, X_test, y_train, y_test, initialise_models()


# Assessment functions
def calculate_metrics(y_test, predictions):
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    rse = np.sum((predictions - y_test) ** 2) / np.sum(
        (np.mean(y_test) - y_test) ** 2
    )
    mae = mean_absolute_error(y_test, predictions)
    return mse, rmse, rse, mae


def cross_validate(model, X_train, y_train):
    subset_size = int(len(X_train) * CV_SAMPLE_FRACTION)
    rmse_scores = []

    for _ in range(5):
        X_subset, _, y_subset, _ = train_test_split(
            X_train, y_train, train_size=subset_size, shuffle=True
        )

        cv_scores = cross_val_score(
            model, X_subset, y_subset, scoring='neg_mean_squared_error'
        )
        rmse_scores.append(np.sqrt(-cv_scores.mean()))
    return np.mean(rmse_scores)


def get_model_metrics(model, results, X_train, y_train, y_test, predictions):
    mse, rmse, rse, mae = calculate_metrics(y_test, predictions)
    cv_rmse = cross_validate(model, X_train, y_train)

    results[-1].update(
        {
            'MSE': mse,
            'RMSE': rmse,
            'RSE': rse,
            'MAE': mae,
            'CV RMSE': cv_rmse,
        }
    )

    return results


def assess_model(name, model, results, X_train, X_test, y_train, y_test):
    start_time = time.time()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    results.append(
        {'algorithm': name, 'execution_time': time.time() - start_time}
    )

    return get_model_metrics(
        model,
        results,
        X_train,
        y_train,
        y_test,
        predictions,
    )


def run_assessment(X_train, X_test, y_train, y_test, models):
    comparison_results = []
    for name, model in models:
        comparison_results = assess_model(
            name, model, comparison_results, X_train, X_test, y_train, y_test
        )

    comparison_results_df = pd.DataFrame(comparison_results)
    return comparison_results_df.set_index('algorithm')


# Main modelling function
def run_modelling():
    # Preparation
    X_train, X_test, y_train, y_test, models = prepare_for_modelling()

    # Assessment
    comparison_results = run_assessment(
        X_train, X_test, y_train, y_test, models
    )

    # Results
    display_modelling_results(comparison_results)
