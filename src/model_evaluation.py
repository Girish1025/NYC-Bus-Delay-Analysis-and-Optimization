import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, f1_score,
    mean_absolute_error, mean_squared_error, precision_score, r2_score,
    recall_score, roc_auc_score,
)


def evaluate_regression_models(models, X_test, y_test, variant="Base"):
    rows = []
    for name, model in models.items():
        prediction = model.predict(X_test)
        mse = mean_squared_error(y_test, prediction)
        rows.append({
            "model": name,
            "variant": variant,
            "mae": mean_absolute_error(y_test, prediction),
            "mse": mse,
            "rmse": np.sqrt(mse),
            "r2": r2_score(y_test, prediction),
        })
    return pd.DataFrame(rows).sort_values("mae")


def save_regression_comparison_plot(results, output_path):
    """Save one figure comparing base and tuned regression model performance."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plot_data = results.copy()
    plot_data["label"] = plot_data["model"] + " (" + plot_data["variant"] + ")"

    error_data = plot_data.melt(
        id_vars=["label"], value_vars=["mae", "rmse"],
        var_name="metric", value_name="score",
    )
    figure, axes = plt.subplots(1, 2, figsize=(18, 7))
    sns.barplot(data=error_data, y="label", x="score", hue="metric", ax=axes[0])
    axes[0].set_title("Regression Error Comparison (Lower Is Better)")
    axes[0].set_xlabel("Minutes")
    axes[0].set_ylabel("")

    sns.barplot(data=plot_data, y="label", x="r2", hue="variant", dodge=False, ax=axes[1])
    axes[1].axvline(0, color="black", linewidth=0.8)
    axes[1].set_title("Regression R² Comparison (Higher Is Better)")
    axes[1].set_xlabel("R²")
    axes[1].set_ylabel("")
    axes[1].legend_.remove()

    figure.suptitle("Base vs Tuned Regression Models", fontsize=16)
    figure.tight_layout()
    figure.savefig(output_path, dpi=250, bbox_inches="tight")
    plt.close(figure)


def evaluate_classification_models(models, X_test, y_test):
    rows, reports = [], {}
    for name, model in models.items():
        prediction = model.predict(X_test)
        row = {
            "model": name,
            "accuracy": accuracy_score(y_test, prediction),
            "precision_weighted": precision_score(y_test, prediction, average="weighted", zero_division=0),
            "recall_weighted": recall_score(y_test, prediction, average="weighted", zero_division=0),
            "f1_weighted": f1_score(y_test, prediction, average="weighted", zero_division=0),
        }
        if hasattr(model, "predict_proba"):
            try:
                row["roc_auc_ovr"] = roc_auc_score(
                    y_test, model.predict_proba(X_test), multi_class="ovr"
                )
            except ValueError:
                row["roc_auc_ovr"] = np.nan
        rows.append(row)
        reports[name] = {
            "classification_report": classification_report(y_test, prediction, zero_division=0),
            "confusion_matrix": confusion_matrix(y_test, prediction),
        }
    return pd.DataFrame(rows).sort_values("f1_weighted", ascending=False), reports


def feature_importance_table(model, feature_names, top_n=20):
    if hasattr(model, "feature_importances_"):
        values = model.feature_importances_
    elif hasattr(model, "coef_"):
        values = np.ravel(model.coef_)
    else:
        return None
    return (
        pd.DataFrame({"feature": feature_names, "importance": values})
        .assign(abs_importance=lambda frame: frame["importance"].abs())
        .sort_values("abs_importance", ascending=False).head(top_n)
    )


def permutation_importance_table(model, X_test, y_test, top_n=20):
    result = permutation_importance(
        model, X_test, y_test, scoring="neg_mean_squared_error",
        n_repeats=5, random_state=42, n_jobs=1,
    )
    return (
        pd.DataFrame({"feature": X_test.columns, "importance": result.importances_mean})
        .sort_values("importance", ascending=False).head(top_n)
    )
