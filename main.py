from src.config import DATA_PATH, FIGURES_DIR
from src.data_loader import load_data
from src.data_cleaning import prepare_clean_data
from src.feature_engineering import engineer_features
from src.eda import create_eda_summary_tables
from src.preprocessing import prepare_regression_data, prepare_classification_data
from src.model_training import train_regression_models, train_classification_models
from src.model_evaluation import evaluate_regression_models, evaluate_classification_models
from src.optimization_insights import identify_high_delay_segments


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    raw_data = load_data(DATA_PATH)

    print("Cleaning data...")
    clean_data = prepare_clean_data(raw_data)

    print("Engineering features...")
    final_data = engineer_features(clean_data)

    print("Creating EDA and optimization summary tables...")
    eda_tables = create_eda_summary_tables(final_data)
    priority_tables = identify_high_delay_segments(final_data)

    for name, table in {**eda_tables, **priority_tables}.items():
        print(f"\n{name}")
        print(table.head())

    print("\nPreparing regression data...")
    X_train, X_test, y_train, y_test, _, _ = prepare_regression_data(final_data)

    print("Training regression models...")
    regression_models = train_regression_models(X_train, y_train)
    regression_results = evaluate_regression_models(regression_models, X_test, y_test)
    print("\nRegression Results")
    print(regression_results)

    print("\nPreparing classification data...")
    X_train_c, X_test_c, y_train_c, y_test_c, _, _ = prepare_classification_data(final_data)

    print("Training classification models...")
    classification_models = train_classification_models(X_train_c, y_train_c)
    classification_results, reports = evaluate_classification_models(
        classification_models,
        X_test_c,
        y_test_c,
    )

    print("\nClassification Results")
    print(classification_results)

    for model_name, report in reports.items():
        print(f"\n{model_name} Classification Report")
        print(report["classification_report"])


if __name__ == "__main__":
    main()
