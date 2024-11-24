from sklearn.ensemble import VotingRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from pytorch_tabnet.tab_model import TabNetRegressor
import torch
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.impute import SimpleImputer


class TabNetWrapper(BaseEstimator, RegressorMixin):
    def __init__(self, **kwargs):
        self.model = TabNetRegressor(**kwargs)
        self.kwargs = kwargs
        self.imputer = SimpleImputer(strategy="median")

    def fit(self, X, y):
        X_imputed = self.imputer.fit_transform(X)
        if hasattr(y, "values"):
            y = y.values

        self.model.fit(
            X_train=X_imputed,
            y_train=y.reshape(-1, 1),
            eval_metric=["mse"],
            max_epochs=200,
            patience=20,
            batch_size=1024,
            virtual_batch_size=128,
        )
        return self

    def predict(self, X):
        X_imputed = self.imputer.transform(X)
        return self.model.predict(X_imputed).flatten()


def create_model_configs():
    lgb_params = {
        "learning_rate": 0.046,
        "max_depth": 12,
        "num_leaves": 478,
        "min_data_in_leaf": 13,
        "feature_fraction": 0.893,
        "bagging_fraction": 0.784,
        "bagging_freq": 4,
        "lambda_l1": 10,
        "lambda_l2": 0.01,
        "random_state": 2024,
    }

    xgb_params = {
        "learning_rate": 0.05,
        "max_depth": 6,
        "n_estimators": 200,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 1,
        "reg_lambda": 5,
        "random_state": 2024,
    }

    tabnet_params = {
        "n_d": 64,
        "n_a": 64,
        "n_steps": 5,
        "gamma": 1.5,
        "n_independent": 2,
        "n_shared": 2,
        "lambda_sparse": 1e-4,
        "optimizer_fn": torch.optim.Adam,
        "optimizer_params": dict(lr=2e-2, weight_decay=1e-5),
        "mask_type": "entmax",
        "scheduler_params": dict(mode="min", patience=10, min_lr=1e-5, factor=0.5),
        "scheduler_fn": torch.optim.lr_scheduler.ReduceLROnPlateau,
        "verbose": 1,
        "device_name": "cuda" if torch.cuda.is_available() else "cpu",
    }

    return lgb_params, xgb_params, tabnet_params


def create_ensemble_model(cat_features=None):
    lgb_params, xgb_params, tabnet_params = create_model_configs()

    models = [
        (
            "lightgbm",
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("model", LGBMRegressor(**lgb_params)),
                ]
            ),
        ),
        (
            "xgboost",
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("model", XGBRegressor(**xgb_params)),
                ]
            ),
        ),
        (
            "catboost",
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    (
                        "model",
                        CatBoostRegressor(
                            iterations=200,
                            learning_rate=0.05,
                            depth=6,
                            cat_features=cat_features,
                            verbose=0,
                        ),
                    ),
                ]
            ),
        ),
        ("tabnet", TabNetWrapper(**tabnet_params)),
    ]

    return VotingRegressor(estimators=models, weights=[4.0, 4.0, 5.0, 4.0])
