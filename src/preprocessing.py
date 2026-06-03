import category_encoders as ce
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from .config import RANDOM_STATE, TEST_SIZE


def select_model_features(df):
    """Select modeling features after cleaning and feature engineering."""
    drop_cols = [
        "occurred_on",
        "created_on",
        "informed_on",
        "breakdown_or_running_late",
        "bus_no",
        "schools_serviced",
        "delay_category",
    ]
    existing_drop_cols = [c for c in drop_cols if c in df.columns]

    model_df = df.drop(columns=existing_drop_cols, errors="ignore")
    model_df = model_df.dropna(subset=["vendor_cleaned"])
    return model_df


def prepare_regression_data(df, target="how_long_delayed"):
    model_df = select_model_features(df)
    X = model_df.drop(columns=[target])
    y = model_df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    categorical_features = X_train.select_dtypes(include=["object", "category"]).columns.tolist()
    numeric_features = X_train.select_dtypes(include=["int64", "float64"]).columns.tolist()

    target_encoder = ce.TargetEncoder(cols=categorical_features, smoothing=0.3)
    X_train_encoded = target_encoder.fit_transform(X_train, y_train)
    X_test_encoded = target_encoder.transform(X_test)

    scaler = StandardScaler()
    X_train_scaled = X_train_encoded.copy()
    X_test_scaled = X_test_encoded.copy()

    if numeric_features:
        X_train_scaled[numeric_features] = scaler.fit_transform(X_train_scaled[numeric_features])
        X_test_scaled[numeric_features] = scaler.transform(X_test_scaled[numeric_features])

    return X_train_scaled, X_test_scaled, y_train, y_test, target_encoder, scaler


def prepare_classification_data(df, target="delay_category"):
    model_df = select_model_features(df)
    X = model_df.drop(columns=["how_long_delayed", target], errors="ignore")
    y = df.loc[model_df.index, target].astype(str)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    categorical_features = X_train.select_dtypes(include=["object", "category"]).columns.tolist()
    numeric_features = X_train.select_dtypes(include=["int64", "float64"]).columns.tolist()

    target_encoder = ce.TargetEncoder(cols=categorical_features, smoothing=0.3)
    X_train_encoded = target_encoder.fit_transform(X_train, y_train)
    X_test_encoded = target_encoder.transform(X_test)

    scaler = MinMaxScaler()
    if numeric_features:
        X_train_encoded[numeric_features] = scaler.fit_transform(X_train_encoded[numeric_features])
        X_test_encoded[numeric_features] = scaler.transform(X_test_encoded[numeric_features])

    return X_train_encoded, X_test_encoded, y_train, y_test, target_encoder, scaler
