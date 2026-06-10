import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


DELAY_GARBAGE_WORDS = [
    "unknown", "unknow", "unk", "unsure", "dontknow", "dont know",
    "not sure", "notknown", "nothing", "none", "n/a", "na", "n-a",
    "tbd", "all day", "allday", "half day", "day", "shut down",
    "not moving", "on going", "very long", "unlimited", "not given",
    "not known", "undefined", "undefine", "undetermined", "undetermin",
    "undertermi", "undesided", "undesigned", "u/k", "u known", "unkown",
    "vary", "ins", "ha", "sdaf", "blvd", "danny", "ms. harris", "min.",
]
DELAY_PUNCTUATION_JUNK = {
    "", "-", "--", "---", "----", "------", "-------", "_", "_______",
    "----------", ".", "/", "@",
}
VALID_BOROUGHS = ["Bronx", "Brooklyn", "Manhattan", "Queens", "Staten Island"]


def clean_column_names(df):
    """Standardize source column names."""
    df = df.copy()
    df.columns = (
        df.columns.str.strip()
        .str.lower()
        .str.replace(" ", "_", regex=False)
        .str.replace("-", "_", regex=False)
        .str.replace("/", "_", regex=False)
    )
    return df


def _mode_or_nan(values):
    modes = values.mode(dropna=True)
    return modes.iloc[0] if not modes.empty else np.nan


def impute_by_hierarchy(df, column, group_levels):
    """Fill a column using successively broader grouped modes, then global mode."""
    if column not in df.columns:
        return df

    df = df.copy()
    for group_cols in group_levels:
        available = [col for col in group_cols if col in df.columns]
        if len(available) != len(group_cols):
            continue
        grouped_modes = df.groupby(available, dropna=False)[column].transform(_mode_or_nan)
        df[column] = df[column].fillna(grouped_modes)

    global_mode = df[column].mode(dropna=True)
    if not global_mode.empty:
        df[column] = df[column].fillna(global_mode.iloc[0])
    return df


def apply_notebook_imputations(df):
    """Apply the grouped-mode imputation hierarchy used in the notebook."""
    hierarchies = {
        "run_type": [["bus_no", "route_number"]],
        "bus_no": [["route_number", "run_type"]],
        "boro": [
            ["run_type", "bus_no", "route_number", "bus_company_name"],
            ["route_number"],
        ],
        "route_number": [
            ["bus_no", "run_type", "boro"],
            ["bus_no", "run_type"],
            ["bus_no"],
        ],
        "reason": [
            ["run_type", "bus_no", "route_number", "boro"],
            ["run_type", "route_number"],
            ["run_type"],
        ],
        "schools_serviced": [
            ["bus_no", "run_type", "route_number", "boro"],
            ["bus_no", "route_number"],
            ["route_number"],
        ],
        "bus_company_name": [
            ["route_number", "run_type", "boro", "bus_no"],
            ["route_number", "run_type", "bus_no"],
            ["route_number", "run_type"],
            ["route_number"],
        ],
    }
    for column, levels in hierarchies.items():
        df = impute_by_hierarchy(df, column, levels)
    return df


def clean_delay(value):
    """Convert the notebook's noisy delay strings into minutes."""
    if pd.isna(value):
        return np.nan

    value = str(value).lower().strip().replace("–", "-").replace("—", "-")
    value = re.sub(r"\s+", " ", value)
    if any(word in value for word in DELAY_GARBAGE_WORDS):
        return np.nan
    if value in DELAY_PUNCTUATION_JUNK or re.fullmatch(r"\?+", value):
        return np.nan
    if re.fullmatch(r"[a-z ]+", value):
        return np.nan
    if "½" in value:
        return 30
    if "! hour" in value:
        return 60

    numbers = [int(number) for number in re.findall(r"\d+", value)]
    if "hr" in value or "hour" in value:
        if not numbers:
            return 60
        return numbers[0] * 60 + (numbers[1] if len(numbers) > 1 else 0)

    match = re.match(r"(\d+):(\d+)", value)
    if match:
        return int(match.group(1)) * 60 + int(match.group(2))

    match = re.match(r"(\d+)/(\d+)", value)
    if match and int(match.group(2)) != 0:
        return int((int(match.group(1)) / int(match.group(2))) * 60)

    if "min" in value and numbers:
        return numbers[0]
    if len(numbers) == 1:
        return numbers[0]

    match = re.match(r"(\d+)(?:-| to | )+(\d+)", value)
    if match:
        return (int(match.group(1)) + int(match.group(2))) // 2
    if len(numbers) >= 2:
        return min(numbers)
    return np.nan


def clean_vendor(name):
    if pd.isna(name):
        return np.nan
    name = re.sub(r"[^\w\s]", " ", str(name).lower())
    return re.sub(r"\s+", " ", name).strip()


def normalize_vendor(name):
    if pd.isna(name):
        return np.nan
    name = re.sub(r"[^a-z0-9 ]", "", str(name).lower())
    return re.sub(r"\s+", " ", name).strip()


def load_vendor_mapping(mapping_path=None):
    if not mapping_path:
        return {}
    mapping_path = Path(mapping_path)
    if not mapping_path.exists():
        return {}
    with mapping_path.open(encoding="utf-8") as mapping_file:
        return json.load(mapping_file)


def prepare_clean_data(df, vendor_mapping_path=None):
    """Run notebook-equivalent cleaning before EDA and modeling."""
    df = apply_notebook_imputations(clean_column_names(df))

    if "number_of_students_on_the_bus" in df.columns:
        df = df.dropna(subset=["number_of_students_on_the_bus"])

    for col in ["occurred_on", "created_on", "informed_on", "last_updated_on"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    # The notebook identifies a source typo where 2027 should be 2007.
    if "occurred_on" in df.columns:
        typo_mask = df["occurred_on"].dt.year.eq(2027)
        df.loc[typo_mask, "occurred_on"] = df.loc[typo_mask, "occurred_on"] - pd.DateOffset(years=20)

    if {"created_on", "informed_on"}.issubset(df.columns):
        df = df[df["created_on"].eq(df["informed_on"])]

    if {"created_on", "occurred_on"}.issubset(df.columns):
        df["report_delay_minutes"] = (
            (df["created_on"] - df["occurred_on"]).dt.total_seconds() / 60
        )
        df = df[df["report_delay_minutes"].between(0, 1440, inclusive="both")]

    if "boro" in df.columns:
        df["boro"] = df["boro"].astype("string").str.strip().str.title()
        df = df[df["boro"].isin(VALID_BOROUGHS)]
        df = df.rename(columns={"boro": "service_area"})

    if "route_number" in df.columns:
        df["route_number"] = df["route_number"].astype(str).str.replace(" ", "", regex=False)

    if "schools_serviced" in df.columns:
        df["schools_serviced"] = df["schools_serviced"].astype(str).str.replace(r"\s+", "", regex=True)
        df["num_schools"] = df["schools_serviced"].str.split(",").apply(len)

    if "how_long_delayed" in df.columns:
        df["how_long_delayed"] = df["how_long_delayed"].apply(clean_delay)
        df = df.dropna(subset=["how_long_delayed"])
        df = df[df["how_long_delayed"] < 600]

    if "number_of_students_on_the_bus" in df.columns:
        df["number_of_students"] = pd.to_numeric(
            df["number_of_students_on_the_bus"].astype(str).str.replace(",", "", regex=False),
            errors="coerce",
        )
        df = df[df["number_of_students"].le(75)]

    if "bus_company_name" in df.columns:
        df["vendor"] = df["bus_company_name"].apply(clean_vendor)
        valid_vendor = df["vendor"].notna() & df["vendor"].str.len().ge(3) & ~df["vendor"].str.isdigit()
        df = df[valid_vendor]
        df["vendor_norm"] = df["vendor"].apply(normalize_vendor)
        vendor_mapping = load_vendor_mapping(vendor_mapping_path)
        df["vendor_cleaned"] = df["vendor_norm"].map(vendor_mapping).fillna(df["vendor_norm"])

    return df.drop_duplicates().copy()
