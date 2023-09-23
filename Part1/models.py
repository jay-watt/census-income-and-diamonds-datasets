import time

import numpy as np
import pandas as pd
from config import MODELS
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import cross_val_score
from utils import display_table, load_cleaned_data


def assess_model(name, model, results, X_train, X_test, y_train, y_test):
    start_time = time.time()

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    rse = np.sum((predictions - y_test) ** 2) / np.sum(
        (np.mean(y_test) - y_test) ** 2
    )
    mae = mean_absolute_error(y_test, predictions)
    execution_time = time.time() - start_time

    cv_scores = cross_val_score(
        model, X_train, y_train, cv=5, scoring='neg_mean_squared_error'
    )
    cv_rmse = np.sqrt(-cv_scores.mean())

    results.append(
        {
            'Model': name,
            'MSE': round(mse, 2),
            'RMSE': round(rmse, 2),
            'RSE': round(rse, 2),
            'MAE': round(mae, 2),
            'Execution Time (s)': round(execution_time, 2),
            'CV RMSE': round(cv_rmse, 2),
        }
    )

    return results


def model(class_name):
    X_train, y_train = load_cleaned_data(class_name, 'train')
    X_test, y_test = load_cleaned_data(class_name, 'test')

    results = []
    for name, model in MODELS:
        results = assess_model(
            name, model, results, X_train, X_test, y_train, y_test
        )
    results_df = pd.DatFrame(results)
    display_table('Model Results', results_df)
