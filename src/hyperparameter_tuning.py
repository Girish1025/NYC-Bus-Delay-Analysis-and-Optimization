from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

try:
    from xgboost import XGBRegressor
except ImportError:
    XGBRegressor = None


def tune_lasso_regressor(X_train, y_train):
    search = GridSearchCV(
        Lasso(max_iter=5000, random_state=42),
        {"alpha": [0.0001, 0.001, 0.01, 0.1, 1]},
        scoring="neg_mean_squared_error", cv=3, n_jobs=1,
    )
    search.fit(X_train, y_train)
    return search.best_estimator_, search.best_params_


def tune_random_forest_regressor(X_train, y_train):
    params = {
        "n_estimators": [100, 200, 300, 500],
        "max_depth": [10, 20, None],
        "min_samples_leaf": [1, 3, 5],
        "max_features": ["sqrt", 0.8, 1.0],
    }
    search = RandomizedSearchCV(
        RandomForestRegressor(random_state=42, n_jobs=-1), params, n_iter=10,
        scoring="neg_mean_absolute_error", cv=3, random_state=42, n_jobs=1,
    )
    search.fit(X_train, y_train)
    return search.best_estimator_, search.best_params_


def tune_xgboost_regressor(X_train, y_train):
    if XGBRegressor is None:
        raise ImportError("xgboost is required for XGBoost tuning")
    params = {
        "n_estimators": [200, 400, 600, 800, 1000],
        "max_depth": [4, 6, 8, 10],
        "learning_rate": [0.01, 0.03, 0.05, 0.1],
        "subsample": [0.6, 0.7, 0.8, 1.0],
        "colsample_bytree": [0.6, 0.7, 0.8, 1.0],
        "min_child_weight": [1, 3, 5, 7],
        "gamma": [0, 1, 3, 5],
        "reg_alpha": [0, 0.1, 1, 5],
        "reg_lambda": [0.1, 1, 5, 10],
    }
    search = RandomizedSearchCV(
        XGBRegressor(objective="reg:squarederror", tree_method="hist", random_state=42),
        params, n_iter=25, scoring="neg_mean_squared_error", cv=2,
        random_state=42, n_jobs=1,
    )
    search.fit(X_train, y_train)
    return search.best_estimator_, search.best_params_
