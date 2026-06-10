import pandas as pd


def identify_high_delay_segments(df, top_n=10):
    """Create prioritization tables for operational optimization."""
    outputs = {}

    segment_columns = [
        "route_number",
        "vendor_cleaned",
        "service_area",
        "reason",
        "time_block",
    ]

    for col in segment_columns:
        if col in df.columns:
            outputs[f"high_delay_{col}"] = (
                df.groupby(col)["how_long_delayed"]
                .agg(incident_count="count", avg_delay="mean", median_delay="median")
                .query("incident_count >= 10")
                .sort_values(["avg_delay", "incident_count"], ascending=[False, False])
                .head(top_n)
                .reset_index()
            )

    if {"route_number", "time_block"}.issubset(df.columns):
        outputs["route_timeblock_priority"] = (
            df.groupby(["route_number", "time_block"])["how_long_delayed"]
            .agg(incident_count="count", avg_delay="mean")
            .query("incident_count >= 5")
            .sort_values(["avg_delay", "incident_count"], ascending=[False, False])
            .head(top_n)
            .reset_index()
        )

    if {"vendor_cleaned", "has_contractor_notified_schools_bin", "has_contractor_notified_parents_bin"}.issubset(df.columns):
        outputs["vendor_notification_gaps"] = (
            df.groupby("vendor_cleaned")
            .agg(
                incident_count=("vendor_cleaned", "size"),
                pct_notified_schools=("has_contractor_notified_schools_bin", "mean"),
                pct_notified_parents=("has_contractor_notified_parents_bin", "mean"),
                avg_delay=("how_long_delayed", "mean"),
            )
            .assign(
                pct_notified_schools=lambda d: d["pct_notified_schools"] * 100,
                pct_notified_parents=lambda d: d["pct_notified_parents"] * 100,
            )
            .query("incident_count >= 10")
            .sort_values(["avg_delay", "incident_count"], ascending=[False, False])
            .head(top_n)
            .reset_index()
        )

    return outputs
