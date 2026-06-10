from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def average_delay_by_group(df, group_col, top_n=10, ascending=False):
    return (
        df.groupby(group_col)["how_long_delayed"]
        .agg(incident_count="count", avg_delay="mean", median_delay="median")
        .sort_values("avg_delay", ascending=ascending)
        .head(top_n)
        .reset_index()
    )


def _top_group_combinations(df, first, second, top_n=3):
    return (
        df.groupby([first, second])["how_long_delayed"].mean()
        .reset_index(name="avg_delay")
        .sort_values([first, "avg_delay"], ascending=[True, False])
        .groupby(first).head(top_n)
    )


def create_eda_summary_tables(df):
    """Create the principal univariate, trend, bivariate, and vendor tables."""
    tables = {}
    for col in ["run_type", "reason", "service_area", "time_block", "vendor_cleaned"]:
        if col in df.columns:
            tables[f"avg_delay_by_{col}"] = average_delay_by_group(df, col)

    for time_col in ["hour", "weekday", "year", "period"]:
        if time_col in df.columns:
            tables[f"delay_trend_by_{time_col}"] = (
                df.groupby(time_col)["how_long_delayed"].agg(["count", "mean", "median"]).reset_index()
            )

    combinations = [
        ("run_type", "vendor_cleaned"),
        ("run_type", "route_number"),
        ("reason", "route_number"),
        ("reason", "vendor_cleaned"),
        ("service_area", "route_number"),
        ("service_area", "vendor_cleaned"),
        ("time_block", "route_number"),
        ("weekday", "vendor_cleaned"),
    ]
    for first, second in combinations:
        if {first, second}.issubset(df.columns):
            tables[f"top_{second}_by_{first}"] = _top_group_combinations(df, first, second)

    if "vendor_cleaned" in df.columns:
        aggregations = {"avg_delay": ("how_long_delayed", "mean")}
        if "report_delay_minutes" in df.columns:
            aggregations["avg_report_delay"] = ("report_delay_minutes", "mean")
        if "num_schools" in df.columns:
            aggregations["avg_schools_serviced"] = ("num_schools", "mean")
        tables["vendor_summary"] = (
            df.groupby("vendor_cleaned").agg(**aggregations).sort_values("avg_delay", ascending=False)
            .reset_index()
        )
    return tables


def save_notebook_eda_plots(df, output_dir):
    """Save a compact modular equivalent of the notebook's principal EDA plots."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    numeric = ["report_delay_minutes", "num_schools", "how_long_delayed", "number_of_students"]
    for column in numeric:
        if column not in df.columns:
            continue
        plt.figure(figsize=(10, 5))
        sns.histplot(df[column], kde=True, bins=40)
        plt.title(f"Distribution of {column}")
        plt.tight_layout()
        plt.savefig(output_dir / f"distribution_{column}.png", dpi=200)
        plt.close()

    for column in ["hour", "weekday", "year", "period", "time_block"]:
        if column not in df.columns:
            continue
        trend = df.groupby(column)["how_long_delayed"].mean().reset_index()
        plt.figure(figsize=(12, 5))
        sns.barplot(data=trend, x=column, y="how_long_delayed")
        plt.xticks(rotation=45, ha="right")
        plt.title(f"Average Delay by {column}")
        plt.tight_layout()
        plt.savefig(output_dir / f"average_delay_by_{column}.png", dpi=200)
        plt.close()
