import time

import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import (AdaBoostClassifier, GradientBoostingClassifier,
                              RandomForestClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from Part1_2_Common.analysis import print_process_heading
from Part1_2_Common.config import SEED
from Part1_2_Common.modelling import (display_modelling_results,
                                      load_cleaned_data)
from Part2.config import DATA_FILENAMES


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

    X_train, y_train = load_cleaned_data(DATA_FILENAMES['training'])
    X_test, y_test = load_cleaned_data(DATA_FILENAMES['test'])
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
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)

    results.append(
        {'algorithm': name, 'execution_time': time.time() - start_time}
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
