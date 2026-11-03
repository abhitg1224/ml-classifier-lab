"""Classifier pipelines and training-only cross-validation."""

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from data import FEATURES, LABELS, RANDOM_STATE, TARGET, validate_data


MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "svm.joblib",
}


def make_models() -> dict[str, Pipeline]:
    svm_pipeline = Pipeline([
        ("scale", StandardScaler()),
        ("svm", SVC(C=1.0, kernel="rbf", random_state=RANDOM_STATE)),
    ])
    return {
        "Logistic Regression": Pipeline([
            ("scale", StandardScaler()),
            ("classifier", LogisticRegression(C=1.0, max_iter=2000)),
        ]),
        "Decision Tree": Pipeline([
            ("classifier", DecisionTreeClassifier(random_state=RANDOM_STATE)),
        ]),
        "KNN": Pipeline([
            ("scale", StandardScaler()),
            ("classifier", KNeighborsClassifier(n_neighbors=5)),
        ]),
        "Naive Bayes": Pipeline([
            ("classifier", GaussianNB()),
        ]),
        "Support Vector Machine": Pipeline([
            ("classifier", CalibratedClassifierCV(
                estimator=svm_pipeline,
                method="sigmoid",
                cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE),
                ensemble=False,
            )),
        ]),
    }


def compare_training_folds(training: pd.DataFrame) -> dict:
    clean = validate_data(training)
    if set(clean[TARGET]) != set(LABELS) or clean[TARGET].value_counts().min() < 6:
        raise ValueError("Cross-validation requires all seven classes and six rows per class.")
    features = clean.loc[:, list(FEATURES)]
    target = clean[TARGET]
    folds = list(StratifiedKFold(
        n_splits=3, shuffle=True, random_state=RANDOM_STATE
    ).split(features, target))
    results = {}
    for name, model in make_models().items():
        scores = cross_validate(
            model, features, target, cv=folds,
            scoring={"accuracy": "accuracy", "f1_macro": "f1_macro"},
            return_train_score=True, error_score="raise",
        )
        results[name] = {
            "training_accuracy": float(np.mean(scores["train_accuracy"])),
            "validation_accuracy": float(np.mean(scores["test_accuracy"])),
            "validation_macro_f1": float(np.mean(scores["test_f1_macro"])),
            "validation_macro_f1_std": float(np.std(scores["test_f1_macro"])),
            "fold_macro_f1": scores["test_f1_macro"].tolist(),
        }
    return {
        "folds": 3,
        "random_state": RANDOM_STATE,
        "selection_metric": "training-only validation macro F1",
        "selected_model": max(results, key=lambda name: results[name]["validation_macro_f1"]),
        "models": results,
    }
