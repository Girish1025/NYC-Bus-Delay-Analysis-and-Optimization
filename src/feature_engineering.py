import pandas as pd

from .config import DELAY_BINS, DELAY_LABELS


def get_time_block(dt):
    """Map event timestamps to the notebook's school transportation blocks."""
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


def engineer_route_features(df):
    """Standardize routes and group rare prefixes/numbers as in the notebook."""
    if "route_number" not in df.columns:
        return df

    df = df.copy()
    df["clean_route"] = (
        df["route_number"].astype(str).str.upper().str.strip()
        .str.replace(r"[^A-Z0-9]", "", regex=True)
    )
    df = df[~df["clean_route"].str.isnumeric()]
    df["route_prefix"] = df["clean_route"].str.extract(r"^([A-Z]+)", expand=False)
    df = df.dropna(subset=["route_prefix"])
    df["route_number_std"] = df["clean_route"].str.extract(r"(\d+)", expand=False).fillna("Other")

    prefix_counts = df["route_prefix"].value_counts()
    route_counts = df["route_number_std"].value_counts()
    df["route_prefix_group"] = df["route_prefix"].where(
        df["route_prefix"].map(prefix_counts).ge(10), "OTHER"
    )
    df["route_number_group"] = df["route_number_std"].where(
        df["route_number_std"].map(route_counts).ge(5), "OTHER"
    )
    return df


def engineer_features(df):
    """Create the route, time, notification, and target features used by the notebook."""
    df = engineer_route_features(df.copy())

    if "occurred_on" in df.columns:
        df["hour"] = df["occurred_on"].dt.hour
        df["weekday"] = df["occurred_on"].dt.day_name()
        df["year"] = df["occurred_on"].dt.year
        df["period"] = df["occurred_on"].dt.strftime("%p")
        df["time_block"] = df["occurred_on"].apply(get_time_block)

    for col in ["has_contractor_notified_schools", "has_contractor_notified_parents"]:
        if col in df.columns:
            df[f"{col}_bin"] = df[col].map({"Yes": 1, "No": 0}).fillna(0)

    df["delay_category"] = pd.cut(
        df["how_long_delayed"],
        bins=DELAY_BINS,
        labels=DELAY_LABELS,
        include_lowest=True,
    )
    return df
