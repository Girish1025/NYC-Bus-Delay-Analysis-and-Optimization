from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, SGDRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

try:
    from xgboost import XGBClassifier, XGBRegressor
except ImportError:
    XGBClassifier = XGBRegressor = None


def train_regression_models(X_train, y_train):
    models = {
        "Linear Regression": LinearRegression(),
        "SGD Regressor": make_pipeline(
            StandardScaler(),
            SGDRegressor(
                loss="squared_error", penalty="l2", alpha=0.0005,
                learning_rate="invscaling", eta0=0.001, power_t=0.25,
                max_iter=20000, tol=1e-5, early_stopping=True,
                validation_fraction=0.1, n_iter_no_change=20,
                average=True, random_state=42,
            ),
        ),
        "Lasso Regression": Lasso(alpha=0.001, max_iter=5000, random_state=42),
        "Random Forest Regressor": RandomForestRegressor(
            n_estimators=500, max_depth=20, min_samples_leaf=5,
            random_state=42, n_jobs=-1,
        ),
    }
    if XGBRegressor is not None:
        models["XGBoost Regressor"] = XGBRegressor(
            objective="reg:squarederror", n_estimators=1000, learning_rate=0.05,
            max_depth=6, subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.1, reg_lambda=1.0, random_state=1025, n_jobs=-1,
        )
    for model in models.values():
        model.fit(X_train, y_train)
    return models


def train_classification_models(X_train, y_train):
    models = {
        "Decision Tree Classifier": DecisionTreeClassifier(
            class_weight="balanced", random_state=42,
        ),
        "Random Forest Classifier": RandomForestClassifier(
            n_estimators=100, max_features="sqrt", class_weight="balanced",
            random_state=42, n_jobs=-1,
        ),
    }
    if XGBClassifier is not None:
        models["XGBoost Classifier"] = XGBClassifier(
            n_estimators=100, max_depth=3, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8, eval_metric="mlogloss",
            random_state=1025, n_jobs=-1,
        )
    for model in models.values():
        model.fit(X_train, y_train)
    return models
