import argparse

import pandas as pd

from src.config import DATA_PATH, FIGURES_DIR, TABLES_DIR, VENDOR_MAPPING_PATH
from src.data_cleaning import prepare_clean_data
from src.data_loader import load_data
from src.eda import create_eda_summary_tables, save_notebook_eda_plots
from src.feature_engineering import engineer_features
from src.feature_selection import create_feature_selection_tables
from src.hyperparameter_tuning import tune_lasso_regressor, tune_xgboost_regressor
from src.model_evaluation import (
    evaluate_classification_models,
    evaluate_regression_models,
    feature_importance_table,
    permutation_importance_table,
    save_regression_comparison_plot,
)
from src.model_training import train_classification_models, train_regression_models
from src.optimization_insights import identify_high_delay_segments
from src.preprocessing import (
    prepare_classification_data,
    prepare_regression_data,
    select_model_features,
)


def save_tables(tables):
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        table.to_csv(TABLES_DIR / f"{name}.csv", index=False)


def run_workflow(data_path=DATA_PATH, create_plots=False, tune_models=False, sample_rows=None):
    """Run the notebook workflow in modular, reproducible stages."""
    print("1. Loading data")
    raw_data = load_data(data_path, nrows=sample_rows)

    print("2. Cleaning and validating data")
    vendor_mapping = VENDOR_MAPPING_PATH if VENDOR_MAPPING_PATH.exists() else None
    clean_data = prepare_clean_data(raw_data, vendor_mapping_path=vendor_mapping)

    print("3. Engineering route and time features")
    final_data = engineer_features(clean_data)

    print("4. Creating EDA and operational optimization tables")
    summary_tables = {
        **create_eda_summary_tables(final_data),
        **identify_high_delay_segments(final_data),
    }
    save_tables(summary_tables)
    if create_plots:
        save_notebook_eda_plots(final_data, FIGURES_DIR)

    print("5. Running correlation, association, outlier, and VIF analysis")
    model_df = select_model_features(final_data)
    feature_selection_tables = create_feature_selection_tables(model_df)
    save_tables(feature_selection_tables)

    print("6. Preparing and training regression models")
    X_train, X_test, y_train, y_test, _, _ = prepare_regression_data(final_data)
    regression_models = train_regression_models(X_train, y_train)
    regression_results = evaluate_regression_models(regression_models, X_test, y_test)
    interpretation_tables = {}
    for name, model in regression_models.items():
        importance = feature_importance_table(model, X_train.columns)
        if importance is not None:
            interpretation_tables[f"{name.lower().replace(' ', '_')}_importance"] = importance
    if "XGBoost Regressor" in regression_models:
        interpretation_tables["xgboost_permutation_importance"] = permutation_importance_table(
            regression_models["XGBoost Regressor"], X_test, y_test
        )
    comparison_results = regression_results.copy()

    if tune_models:
        print("7. Tuning Lasso and XGBoost regression models")
        tuned_models = {}
        tuned_lasso, lasso_params = tune_lasso_regressor(X_train, y_train)
        tuned_models["Lasso Regression"] = tuned_lasso
        print("Best Lasso parameters:", lasso_params)
        try:
            tuned_xgboost, xgboost_params = tune_xgboost_regressor(X_train, y_train)
            tuned_models["XGBoost Regressor"] = tuned_xgboost
            print("Best XGBoost parameters:", xgboost_params)
        except ImportError as error:
            print(error)
        tuned_results = evaluate_regression_models(tuned_models, X_test, y_test, variant="Tuned")
        comparison_results = pd.concat([regression_results, tuned_results], ignore_index=True)
        print("\nBase vs Tuned Regression Results")
        print(comparison_results.sort_values("mae"))
    else:
        print("7. Hyperparameter tuning skipped; use --tune to run it")

    save_tables({
        "regression_results": regression_results,
        "regression_base_vs_tuned_results": comparison_results,
        **interpretation_tables,
    })
    save_regression_comparison_plot(
        comparison_results,
        FIGURES_DIR / "regression_base_vs_tuned_comparison.png",
    )
    if not tune_models:
        print(regression_results)

    print("8. Preparing and training delay-severity classification models")
    X_train_c, X_test_c, y_train_c, y_test_c, _, _ = prepare_classification_data(final_data)
    classification_models = train_classification_models(X_train_c, y_train_c)
    classification_results, reports = evaluate_classification_models(
        classification_models, X_test_c, y_test_c
    )
    classification_tables = {"classification_results": classification_results}
    for name, model in classification_models.items():
        importance = feature_importance_table(model, X_train_c.columns)
        if importance is not None:
            classification_tables[f"{name.lower().replace(' ', '_')}_importance"] = importance
        classification_tables[f"{name.lower().replace(' ', '_')}_confusion_matrix"] = (
            pd.DataFrame(reports[name]["confusion_matrix"])
        )
    save_tables(classification_tables)
    print(classification_results)
    return final_data, regression_results, classification_results


def parse_args():
    parser = argparse.ArgumentParser(description="Run the ASDS6304 notebook workflow modularly.")
    parser.add_argument("--data-path", default=DATA_PATH, help="Path to the source CSV dataset.")
    parser.add_argument("--plots", action="store_true", help="Save the notebook's principal EDA plots.")
    parser.add_argument("--tune", action="store_true", help="Run computationally expensive tuning.")
    parser.add_argument("--sample-rows", type=int, help="Run only the first N rows for validation.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_workflow(args.data_path, args.plots, args.tune, args.sample_rows)
