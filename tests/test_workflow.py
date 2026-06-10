import pandas as pd
from sklearn.dummy import DummyRegressor

from src.data_cleaning import clean_delay, prepare_clean_data
from src.feature_engineering import engineer_features
from src.feature_selection import create_feature_selection_tables
from src.model_evaluation import evaluate_regression_models, save_regression_comparison_plot
from src.preprocessing import prepare_classification_data, prepare_regression_data, select_model_features


def sample_data(rows=100):
    records = []
    delays = ["5 min", "15 minutes", "20-40", "1 hour", "90"]
    for index in range(rows):
        occurred = pd.Timestamp("2024-01-01 07:00") + pd.Timedelta(hours=index)
        created = occurred + pd.Timedelta(minutes=10)
        records.append({
            "School Year": "2023-2024",
            "Bus No": f"B{index % 8}",
            "Run Type": None if index == 0 else f"Run {index % 3}",
            "Route Number": f"Q{index % 12}",
            "Reason": f"Reason {index % 4}",
            "Schools Serviced": "A, B",
            "Occurred On": occurred,
            "Created On": created,
            "Informed On": created,
            "Last Updated On": created,
            "Boro": "Queens",
            "Bus Company Name": f"Vendor {index % 5}",
            "How Long Delayed": delays[index % len(delays)],
            "Number Of Students On The Bus": str(index % 50),
            "Has Contractor Notified Schools": "Yes",
            "Has Contractor Notified Parents": "No",
            "Have You Alerted OPT": "Yes",
            "Breakdown or Running Late": "Running Late",
            "School Age or PreK": "School-Age",
        })
    return pd.DataFrame(records)


def test_delay_cleaning_matches_notebook_rules():
    assert clean_delay("1 hour 15 minutes") == 75
    assert clean_delay("20-40") == 30
    assert clean_delay("unknown") != clean_delay("unknown")


def test_modular_notebook_workflow_prepares_models():
    cleaned = prepare_clean_data(sample_data())
    final = engineer_features(cleaned)
    model_df = select_model_features(final)

    assert {"route_prefix_group", "route_number_group", "time_block", "weekday"}.issubset(final)
    assert "route_number" not in model_df
    assert cleaned["report_delay_minutes"].eq(10).all()

    regression = prepare_regression_data(final)
    classification = prepare_classification_data(final)
    assert regression[0].shape[1] == regression[1].shape[1]
    assert classification[0].shape[1] == classification[1].shape[1]

    tables = create_feature_selection_tables(model_df)
    assert set(tables) == {
        "high_numeric_correlations",
        "high_categorical_associations",
        "iqr_outlier_counts",
        "variance_inflation_factors",
    }


def test_base_vs_tuned_comparison_plot(tmp_path):
    X = pd.DataFrame({"feature": [0, 1, 2, 3]})
    y = pd.Series([1, 2, 3, 4])
    base = DummyRegressor(strategy="mean").fit(X, y)
    tuned = DummyRegressor(strategy="median").fit(X, y)
    results = pd.concat([
        evaluate_regression_models({"Dummy": base}, X, y),
        evaluate_regression_models({"Dummy": tuned}, X, y, variant="Tuned"),
    ])
    output = tmp_path / "comparison.png"
    save_regression_comparison_plot(results, output)
    assert output.exists()
