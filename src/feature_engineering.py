import numpy as np
import pandas as pd
from .config import DELAY_BINS, DELAY_LABELS


def get_time_block(dt):
    """Map event timestamp to a school transportation time block."""
    if pd.isna(dt):
        return "Unknown"

    hour = dt.hour
    if 5 <= hour < 7:
        return "Early Morning (5-7)"
    if 7 <= hour < 9:
        return "Morning Peak (7-9)"
    if 9 <= hour < 14:
        return "Midday (9-14)"
    if 14 <= hour < 16:
        return "Afternoon Peak (14-16)"
    if 16 <= hour < 19:
        return "Evening (16-19)"
    return "Off-Hours"


def engineer_features(df):
    """Create date-time, reporting delay, notification, and delay category features."""
    df = df.copy()

    if "occurred_on" in df.columns:
        df["hour"] = df["occurred_on"].dt.hour
        df["weekday"] = df["occurred_on"].dt.day_name()
        df["year"] = df["occurred_on"].dt.year
        df["period"] = df["occurred_on"].dt.strftime("%p")
        df["time_block"] = df["occurred_on"].apply(get_time_block)

    if {"occurred_on", "informed_on"}.issubset(df.columns):
        df["report_delay_minutes"] = (
            (df["informed_on"] - df["occurred_on"]).dt.total_seconds() / 60
        )
        df["report_delay_minutes"] = df["report_delay_minutes"].clip(lower=0)

    for col in ["has_contractor_notified_schools", "has_contractor_notified_parents"]:
        if col in df.columns:
            new_col = col + "_bin"
            df[new_col] = df[col].map({"Yes": 1, "No": 0}).fillna(0)

    df["delay_category"] = pd.cut(
        df["how_long_delayed"],
        bins=DELAY_BINS,
        labels=DELAY_LABELS,
        include_lowest=True,
    )

    return df
