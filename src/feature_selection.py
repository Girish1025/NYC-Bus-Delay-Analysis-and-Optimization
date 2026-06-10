import itertools

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from statsmodels.stats.outliers_influence import variance_inflation_factor


def high_numeric_correlations(df, threshold=0.8):
    numeric = df.select_dtypes(include=[np.number])
    correlations = numeric.corr(method="pearson").abs()
    upper = correlations.where(np.triu(np.ones(correlations.shape), k=1).astype(bool))
    pairs = upper.stack().reset_index()
    pairs.columns = ["feature_1", "feature_2", "correlation"]
    return pairs[pairs["correlation"] > threshold].sort_values("correlation", ascending=False)


def cramers_v(x, y):
    table = pd.crosstab(x, y)
    if table.empty or min(table.shape) <= 1:
        return np.nan
    chi2 = chi2_contingency(table)[0]
    denominator = table.to_numpy().sum() * min(table.shape[0] - 1, table.shape[1] - 1)
    return np.sqrt(chi2 / denominator) if denominator else np.nan


def high_categorical_associations(df, threshold=0.8):
    categorical = df.select_dtypes(include=["object", "category", "string"]).columns
    rows = []
    for first, second in itertools.combinations(categorical, 2):
        score = cramers_v(df[first], df[second])
        if pd.notna(score) and score > threshold:
            rows.append({"feature_1": first, "feature_2": second, "cramers_v": score})
    return pd.DataFrame(rows, columns=["feature_1", "feature_2", "cramers_v"]).sort_values(
        "cramers_v", ascending=False
    )


def iqr_outlier_counts(df):
    numeric = df.select_dtypes(include=[np.number])
    first_quartile = numeric.quantile(0.25)
    third_quartile = numeric.quantile(0.75)
    iqr = third_quartile - first_quartile
    mask = numeric.lt(first_quartile - 1.5 * iqr) | numeric.gt(third_quartile + 1.5 * iqr)
    return (
        mask.sum().sort_values(ascending=False).rename("outlier_count")
        .rename_axis("feature").reset_index()
    )


def variance_inflation_table(df):
    numeric = df.select_dtypes(include=[np.number]).dropna()
    numeric = numeric.loc[:, numeric.nunique() > 1]
    if numeric.shape[1] < 2:
        return pd.DataFrame(columns=["feature", "vif"])
    rows = [
        {"feature": column, "vif": variance_inflation_factor(numeric.values, index)}
        for index, column in enumerate(numeric.columns)
    ]
    return pd.DataFrame(rows).sort_values("vif", ascending=False)


def create_feature_selection_tables(model_df):
    """Return the notebook's correlation, association, outlier, and VIF analyses."""
    return {
        "high_numeric_correlations": high_numeric_correlations(model_df),
        "high_categorical_associations": high_categorical_associations(model_df),
        "iqr_outlier_counts": iqr_outlier_counts(model_df),
        "variance_inflation_factors": variance_inflation_table(
            model_df.drop(columns=["how_long_delayed"], errors="ignore")
        ),
    }
