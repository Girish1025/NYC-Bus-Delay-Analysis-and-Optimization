import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler

from .config import RANDOM_STATE, TEST_SIZE


NOTEBOOK_DATA_COLUMNS = [
    "run_type", "reason", "occurred_on", "informed_on", "service_area",
    "how_long_delayed", "has_contractor_notified_schools",
    "has_contractor_notified_parents", "breakdown_or_running_late",
    "have_you_alerted_opt", "school_age_or_prek", "num_schools",
    "report_delay_minutes", "vendor_cleaned", "number_of_students",
    "route_prefix_group", "route_number_group", "route_number",
]
NOTEBOOK_MODEL_DROP_COLUMNS = [
    "occurred_on", "informed_on", "breakdown_or_running_late", "route_number",
    "school_age_or_prek", "period",
]


class MeanTargetEncoder:
    """Small target encoder that follows the notebook's train-only encoding rule."""

    def __init__(self, columns, smoothing=0.3):
        self.columns = list(columns)
        self.smoothing = smoothing
        self.global_mean = None
        self.mappings = {}

    def fit(self, X, y):
        target = pd.Series(np.asarray(y), index=X.index, dtype=float)
        self.global_mean = target.mean()
        for column in self.columns:
            stats = pd.DataFrame({"feature": X[column], "target": target}).groupby(
                "feature", dropna=False
            )["target"].agg(["mean", "count"])
            weight = stats["count"] / (stats["count"] + self.smoothing)
            self.mappings[column] = (
                weight * stats["mean"] + (1 - weight) * self.global_mean
            ).to_dict()
        return self

    def transform(self, X):
        encoded = X.copy()
        for column in self.columns:
            encoded[column] = encoded[column].map(self.mappings[column]).fillna(self.global_mean)
        return encoded

    def fit_transform(self, X, y):
        return self.fit(X, y).transform(X)


def select_notebook_data(df):
    """Select the curated intermediate dataset defined in the notebook."""
    existing = [column for column in NOTEBOOK_DATA_COLUMNS if column in df.columns]
    selected = df[existing].copy()
    for column in ["period", "time_block", "weekday"]:
        if column in df.columns:
            selected[column] = df[column]
    return selected


def select_model_features(df):
    """Apply the notebook's final model feature selection."""
    model_df = select_notebook_data(df)
    model_df = model_df.drop(columns=NOTEBOOK_MODEL_DROP_COLUMNS, errors="ignore")
    if "vendor_cleaned" in model_df.columns:
        model_df = model_df.dropna(subset=["vendor_cleaned"])
    return model_df


def _encode_and_scale(X_train, X_test, y_train, scaler):
    categorical = X_train.select_dtypes(include=["object", "category", "string"]).columns.tolist()
    numeric = X_train.select_dtypes(include=[np.number]).columns.tolist()
    encoder = MeanTargetEncoder(categorical, smoothing=0.3)
    X_train_encoded = encoder.fit_transform(X_train, y_train)
    X_test_encoded = encoder.transform(X_test)
    if numeric:
        X_train_encoded[numeric] = scaler.fit_transform(X_train_encoded[numeric])
        X_test_encoded[numeric] = scaler.transform(X_test_encoded[numeric])
    return X_train_encoded, X_test_encoded, encoder, scaler


def prepare_regression_data(df, target="how_long_delayed"):
    model_df = select_model_features(df).drop(columns=["delay_category"], errors="ignore")
    X = model_df.drop(columns=[target])
    y = model_df[target]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    X_train, X_test, encoder, scaler = _encode_and_scale(
        X_train, X_test, y_train, StandardScaler()
    )
    return X_train, X_test, y_train, y_test, encoder, scaler


def prepare_classification_data(df, target="delay_category"):
    model_df = select_model_features(df)
    y_labels = df.loc[model_df.index, target].astype(str)
    label_encoder = LabelEncoder()
    y = pd.Series(label_encoder.fit_transform(y_labels), index=model_df.index, name=target)
    X = model_df.drop(columns=["how_long_delayed", target], errors="ignore")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    X_train, X_test, encoder, scaler = _encode_and_scale(
        X_train, X_test, y_train, MinMaxScaler()
    )
    encoder.label_encoder = label_encoder
    return X_train, X_test, y_train, y_test, encoder, scaler
