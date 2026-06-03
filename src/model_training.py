from sklearn.linear_model import LinearRegression, Lasso, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from xgboost import XGBRegressor, XGBClassifier


def train_regression_models(X_train, y_train):
    models = {
        "Linear Regression": LinearRegression(),
        "Lasso Regression": Lasso(alpha=0.001, max_iter=5000, random_state=42),
        "Decision Tree Regressor": DecisionTreeRegressor(random_state=42),
        "Random Forest Regressor": RandomForestRegressor(
            n_estimators=300,
            max_depth=20,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1,
        ),
        "XGBoost Regressor": XGBRegressor(
            objective="reg:squarederror",
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
        ),
    }

    for model in models.values():
        model.fit(X_train, y_train)

    return models


def train_classification_models(X_train, y_train):
    models = {
        "Decision Tree Classifier": DecisionTreeClassifier(
            class_weight="balanced",
            random_state=42,
        ),
        "Random Forest Classifier": RandomForestClassifier(
            n_estimators=200,
            max_features="sqrt",
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
        "XGBoost Classifier": XGBClassifier(
            n_estimators=200,
            max_depth=3,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="mlogloss",
            random_state=42,
        ),
    }

    for name, model in models.items():
        model.fit(X_train, y_train)

    return models
