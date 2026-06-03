import re
import numpy as np
import pandas as pd


def clean_column_names(df):
    """Standardize column names for Python modeling."""
    df = df.copy()
    df.columns = (
        df.columns.str.strip()
        .str.lower()
        .str.replace(" ", "_", regex=False)
        .str.replace("-", "_", regex=False)
        .str.replace("/", "_", regex=False)
    )
    return df


def clean_vendor(name):
    """Normalize vendor/company names."""
    if pd.isna(name):
        return np.nan
    name = str(name).lower()
    name = re.sub(r"[^\w\s]", " ", name)
    name = re.sub(r"\s+", " ", name).strip()
    return name


def clean_delay(value):
    """Convert noisy delay strings into minutes."""
    if pd.isna(value):
        return np.nan

    x = str(value).lower().strip()
    x = x.replace("–", "-").replace("—", "-")
    x = re.sub(r"\s+", " ", x)

    garbage_words = [
        "no delay", "n/a", "na", "none", "unknown", "unk", "?", "late",
        "minutes", "mins", "min"
    ]

    if x in {"", ".", "-", "--"}:
        return np.nan

    if x in {"no", "none", "n/a", "na", "unknown"}:
        return np.nan

    if "½" in x:
        return 30

    nums = [int(n) for n in re.findall(r"\d+", x)]

    if "hr" in x or "hour" in x:
        if len(nums) == 0:
            return 60
        hours = nums[0]
        minutes = nums[1] if len(nums) > 1 else 0
        return hours * 60 + minutes

    match = re.match(r"(\d+):(\d+)", x)
    if match:
        return int(match.group(1)) * 60 + int(match.group(2))

    match = re.match(r"(\d+)/(\d+)", x)
    if match:
        denominator = float(match.group(2))
        if denominator == 0:
            return np.nan
        return int((float(match.group(1)) / denominator) * 60)

    match = re.match(r"(\d+)[- ]+(\d+)", x)
    if match:
        a, b = int(match.group(1)), int(match.group(2))
        return (a + b) // 2

    if len(nums) == 1:
        return nums[0]

    if len(nums) > 1:
        return int(np.mean(nums))

    return np.nan


def clean_service_area(value):
    """Normalize borough/service area values and remove non-NYC values later."""
    if pd.isna(value):
        return np.nan
    value = str(value).strip().title()
    replacements = {
        "Bronx": "Bronx",
        "Brooklyn": "Brooklyn",
        "Manhattan": "Manhattan",
        "Queens": "Queens",
        "Staten Island": "Staten Island",
    }
    return replacements.get(value, value)


def prepare_clean_data(df):
    """Apply core cleaning steps used before EDA and modeling."""
    df = clean_column_names(df)

    if "bus_company_name" in df.columns:
        df["vendor_cleaned"] = df["bus_company_name"].apply(clean_vendor)
    elif "vendor_name" in df.columns:
        df["vendor_cleaned"] = df["vendor_name"].apply(clean_vendor)

    if "how_long_delayed" in df.columns:
        df["how_long_delayed"] = df["how_long_delayed"].apply(clean_delay)

    for col in ["occurred_on", "created_on", "informed_on"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    if "boro" in df.columns:
        df["service_area"] = df["boro"].apply(clean_service_area)

    valid_boroughs = ["Bronx", "Brooklyn", "Manhattan", "Queens", "Staten Island"]
    if "service_area" in df.columns:
        df = df[df["service_area"].isin(valid_boroughs)]

    if "schools_serviced" in df.columns:
        df["num_schools"] = (
            df["schools_serviced"].astype(str)
            .str.split(",")
            .apply(lambda x: len([v for v in x if v.strip() and v.lower() != "nan"]))
        )

    df = df.drop_duplicates()
    df = df.dropna(subset=["how_long_delayed"])

    return df
