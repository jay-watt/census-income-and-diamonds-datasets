import time

import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import (AdaBoostClassifier, GradientBoostingClassifier,
                              RandomForestClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from common.preprocessing import print_process_heading
from common.config import SEED
from common.modelling import (display_modelling_results,
                              load_cleaned_data)

PARAM_GRIDS = {
    'KNN': {'n_neighbors': [3, 5, 7]},
    'Naive Bayes': {},
    'SVM': {'C': [0.1, 1.0]},
    'Decision Tree': {'max_depth': [None, 5, 10]},
    'Random Forest': {'n_estimators': [50, 100], 'max_depth': [None, 5]},
    'AdaBoost': {'n_estimators': [50, 100]},
    'Gradient Boosting': {'n_estimators': [50, 100], 'learning_rate': [0.05, 0.1]},
    'Linear Discriminant Analysis': {},
    'Multi-layer Perceptron': {'hidden_layer_sizes': [(50,), (100,)]},
    'Logistic Regression': {'C': [0.1, 1.0]},
}

# Preparation functions
def initialise_models():
    return [
        ('KNN', KNeighborsClassifier()),
        ('Naive Bayes', GaussianNB()),
        ('SVM', SVC(probability=True)),
        ('Decision Tree', DecisionTreeClassifier(random_state=SEED)),
        ('Random Forest', RandomForestClassifier(random_state=SEED)),
        ('AdaBoost', AdaBoostClassifier(random_state=SEED)),
        ('Gradient Boosting', GradientBoostingClassifier(random_state=SEED)),
        ('Linear Discriminant Analysis', LinearDiscriminantAnalysis()),
        ('Multi-layer Perceptron', MLPClassifier(random_state=SEED)),
        ('Logistic Regression', LogisticRegression(random_state=SEED)),
    ]


def prepare_for_modelling():
    print_process_heading('modelling')

    X_train, y_train = load_cleaned_data('training')
    X_test, y_test = load_cleaned_data('test')
    return X_train, X_test, y_train, y_test, initialise_models()


# Assessment functions
def calculate_metrics(y_test, predictions, probabilities):
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)
    auc = roc_auc_score(y_test, probabilities[:, 1])
    return accuracy, precision, recall, f1, auc


def get_model_metrics(
    results,
    y_test,
    predictions,
    probabilities,
):
    accuracy, precision, recall, f1, auc = calculate_metrics(
        y_test, predictions, probabilities
    )

    results[-1].update({
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'F1 score': f1,
        'AUC': auc,
    })

    return results


def assess_model(name, model, results, X_train, X_test, y_train, y_test):
    start_time = time.time()
    param_grid = PARAM_GRIDS.get(name, {})
    if param_grid:
        grid = GridSearchCV(model, param_grid, cv=3, scoring='accuracy', n_jobs=-1)
        grid.fit(X_train, y_train)
        best_model = grid.best_estimator_
        print(f"Best params for {name}: {grid.best_params_}")
    else:
        best_model = model

    start_time = time.time()
    best_model.fit(X_train, y_train)
    predictions = best_model.predict(X_test)
    probabilities = best_model.predict_proba(X_test)
    execution_time = time.time() - start_time

    results.append(
        {'algorithm': name, 'execution_time': execution_time}
    )

    return get_model_metrics(
        results,
        y_test,
        predictions,
        probabilities,
    )


def run_assessment(X_train, X_test, y_train, y_test, models):
    comparison_results = []
    for name, model in models:
        comparison_results_df = assess_model(
            name, model, comparison_results, X_train, X_test, y_train, y_test
        )
    comparison_results_df = pd.DataFrame(comparison_results)
    return comparison_results_df.set_index('algorithm')


def run_modelling():
    # Preparation
    X_train, X_test, y_train, y_test, models = prepare_for_modelling()

    # Assessment
    comparison_results = run_assessment(
        X_train, X_test, y_train, y_test, models
    )

    # Results
    display_modelling_results(comparison_results)
