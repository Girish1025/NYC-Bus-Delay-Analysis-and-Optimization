import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)


def evaluate_regression_models(models, X_test, y_test):
    rows = []

    for name, model in models.items():
        y_pred = model.predict(X_test)
        rows.append({
            "model": name,
            "mae": mean_absolute_error(y_test, y_pred),
            "mse": mean_squared_error(y_test, y_pred),
            "rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
            "r2": r2_score(y_test, y_pred),
        })

    return pd.DataFrame(rows).sort_values("mae")


def evaluate_classification_models(models, X_test, y_test):
    rows = []
    reports = {}

    for name, model in models.items():
        y_pred = model.predict(X_test)
        rows.append({
            "model": name,
            "accuracy": accuracy_score(y_test, y_pred),
            "precision_macro": precision_score(y_test, y_pred, average="macro", zero_division=0),
            "recall_macro": recall_score(y_test, y_pred, average="macro", zero_division=0),
            "f1_macro": f1_score(y_test, y_pred, average="macro", zero_division=0),
        })
        reports[name] = {
            "classification_report": classification_report(y_test, y_pred, zero_division=0),
            "confusion_matrix": confusion_matrix(y_test, y_pred),
        }

    return pd.DataFrame(rows).sort_values("f1_macro", ascending=False), reports


def feature_importance_table(model, feature_names, top_n=20):
    if hasattr(model, "feature_importances_"):
        values = model.feature_importances_
    elif hasattr(model, "coef_"):
        values = np.ravel(model.coef_)
    else:
        return None

    return (
        pd.DataFrame({"feature": feature_names, "importance": values})
        .assign(abs_importance=lambda d: d["importance"].abs())
        .sort_values("abs_importance", ascending=False)
        .head(top_n)
    )
