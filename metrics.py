"""Shared six-metric evaluation for fitted classifiers and labeled test data."""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, matthews_corrcoef,
    precision_recall_fscore_support, roc_auc_score,
)
from sklearn.pipeline import Pipeline

from data import FEATURES, LABELS, MAX_TEST_ROWS, TARGET, validate_data


METRIC_NAMES = ("Accuracy", "AUC", "Precision", "Recall", "F1", "MCC")


def evaluate(model: Pipeline, observations: pd.DataFrame) -> dict:
    clean = validate_data(observations, max_rows=MAX_TEST_ROWS)
    features = clean.loc[:, list(FEATURES)]
    actual = clean[TARGET]
    classes = np.asarray(model.classes_)
    if classes.shape != (len(LABELS),) or set(classes) != set(LABELS):
        raise ValueError("The classifier must have all seven original classes.")
    predicted = np.asarray(model.predict(features))
    if predicted.shape != (len(clean),) or not set(predicted).issubset(LABELS):
        raise ValueError("The classifier returned invalid predictions.")
    probabilities = np.asarray(model.predict_proba(features))
    if probabilities.shape != (len(clean), len(LABELS)):
        raise ValueError("The classifier returned an invalid probability matrix shape.")
    if (
        not np.isfinite(probabilities).all()
        or (probabilities < 0).any()
        or (probabilities > 1).any()
        or not np.allclose(probabilities.sum(axis=1), 1.0)
    ):
        raise ValueError("The classifier returned invalid probabilities.")
    probabilities = pd.DataFrame(
        probabilities, columns=classes
    ).loc[:, list(LABELS)].to_numpy()

    precision, recall, f1, _ = precision_recall_fscore_support(
        actual, predicted, labels=list(LABELS), average="macro", zero_division=0
    )
    missing = set(LABELS) - set(actual)
    auc = None
    explanation = None
    if missing:
        explanation = (
            "Macro one-vs-rest AUC requires all seven classes. "
            f"Missing from the test data: {', '.join(sorted(missing))}."
        )
    else:
        auc = float(roc_auc_score(
            actual, probabilities, labels=list(LABELS), multi_class="ovr", average="macro"
        ))
    return {
        "metrics": {
            "Accuracy": float(accuracy_score(actual, predicted)),
            "AUC": auc,
            "Precision": float(precision),
            "Recall": float(recall),
            "F1": float(f1),
            "MCC": (
                0.0 if actual.nunique() < 2 or len(set(predicted)) < 2
                else float(matthews_corrcoef(actual, predicted))
            ),
        },
        "auc_explanation": explanation,
        "confusion_matrix": confusion_matrix(actual, predicted, labels=list(LABELS)).tolist(),
        "classification_report": classification_report(
            actual, predicted, labels=list(LABELS), output_dict=True, zero_division=0
        ),
    }


def comparison_table(results: dict) -> pd.DataFrame:
    table = pd.DataFrame.from_dict(
        {name: result["metrics"] for name, result in results.items()}, orient="index"
    ).loc[:, list(METRIC_NAMES)]
    table.index.name = "Model"
    return table
