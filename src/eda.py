import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def save_countplot(df, column, output_path=None, title=None):
    plt.figure(figsize=(10, 5))
    order = df[column].value_counts().index
    sns.countplot(data=df, y=column, order=order)
    plt.title(title or f"Distribution of {column}")
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def average_delay_by_group(df, group_col, top_n=10):
    return (
        df.groupby(group_col)["how_long_delayed"]
        .agg(["count", "mean", "median"])
        .sort_values("mean", ascending=False)
        .head(top_n)
        .reset_index()
    )


def create_eda_summary_tables(df):
    """Return important EDA tables used for business interpretation."""
    tables = {}

    for col in ["run_type", "reason", "service_area", "time_block", "vendor_cleaned"]:
        if col in df.columns:
            tables[f"avg_delay_by_{col}"] = average_delay_by_group(df, col)

    if {"run_type", "route_number"}.issubset(df.columns):
        tables["top_routes_by_run_type"] = (
            df.groupby(["run_type", "route_number"])["how_long_delayed"]
            .mean()
            .reset_index(name="avg_delay")
            .sort_values(["run_type", "avg_delay"], ascending=[True, False])
            .groupby("run_type")
            .head(3)
        )

    if {"service_area", "vendor_cleaned"}.issubset(df.columns):
        tables["top_vendors_by_borough"] = (
            df.groupby(["service_area", "vendor_cleaned"])["how_long_delayed"]
            .mean()
            .reset_index(name="avg_delay")
            .sort_values(["service_area", "avg_delay"], ascending=[True, False])
            .groupby("service_area")
            .head(3)
        )

    return tables
