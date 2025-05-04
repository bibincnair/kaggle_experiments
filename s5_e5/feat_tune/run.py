import json
import pandas as pd
import numpy as np
import random
from feature_engineering import (
    manual_features,
    run_featuretools_dfs,
    reduce_dimensionality,
)
from modelling import cross_validate_model
import xgboost as xgb
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
from sklearn.preprocessing import LabelEncoder
from dataclasses import dataclass, asdict


# 1) Config
@dataclass
class Config:
    # model‐agnostic
    random_seed: int = 42
    n_folds: int = 5
    early_stopping_rounds: int = 50
    use_featuretools: bool = True
    pca_variance: float = 0.95

    # XGBoost
    iterations: int = 2000
    learning_rate: float = 0.05
    depth: int = 6
    l2_leaf_reg: float = 1.0
    gamma: float = 0.0
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    tree_method: str = "gpu_hist"

    @classmethod
    def load(cls, path="data/s5_e5/xgb_best_config.json"):
        p = Path(path)
        if p.is_file():
            data = json.loads(p.read_text())
            inst = cls()
            for k, v in data.items():
                if hasattr(inst, k):
                    setattr(inst, k, v)
            return inst
        return cls()


# 2) Set seeds
seed = 42
random.seed(seed)
np.random.seed(seed)

# 3) Load
train_raw = pd.read_csv("../data/s5_e5/train.csv")
test_raw = pd.read_csv("../data/s5_e5/test.csv")

# 4) Feature Engineering
if Config().use_featuretools:
    # manual numeric only
    train_tmp = manual_features(
        train_raw, add_log_duration=False, add_temp_elevation=False
    )
    test_tmp = manual_features(
        test_raw, add_log_duration=False, add_temp_elevation=False
    )
    ft_train = run_featuretools_dfs(train_tmp, index_col="id")
    ft_test = run_featuretools_dfs(test_tmp, index_col="id")
    # align and fill
    common = ft_train.columns.intersection(ft_test.columns).tolist()
    ft_train, ft_test, pca = reduce_dimensionality(
        ft_train[common + ["Calories", "id"]],
        ft_test[common + ["id"]],
        variance_threshold=Config().pca_variance,
        random_state=seed,
    )
    X_train = ft_train.drop(columns=["id", "Calories"]).values
    y_train = np.log1p(ft_train["Calories"].values)
    X_test = ft_test.drop(columns=["id"]).values
else:
    df_tr = manual_features(train_raw)
    df_te = manual_features(test_raw)
    feats = [
        "Sex",
        "Age",
        "Duration",
        "Heart_Rate",
        "Body_Temp",
        "BMI",
        "Log_Duration",
        "Temp_Elevation",
    ]
    feats = [c for c in feats if c in df_tr]
    X_train = df_tr[feats].assign(Sex=LabelEncoder().fit_transform(df_tr["Sex"])).values
    y_train = np.log1p(df_tr["Calories"].values)
    X_test = df_te[feats].assign(Sex=LabelEncoder().fit_transform(df_te["Sex"])).values


# 5) Optuna tuning on a 50% subsample to speed up & prune
def objective(trial):
    idx = np.random.choice(len(X_train), size=int(0.5 * len(X_train)), replace=False)
    Xs, ys = X_train[idx], y_train[idx]
    params = {
        "n_estimators": trial.suggest_int("iterations", 500, 5000, step=500),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "max_depth": trial.suggest_int("depth", 3, 10),
        "reg_lambda": trial.suggest_float("l2_leaf_reg", 1e-4, 10.0, log=True),
        "gamma": trial.suggest_float("gamma", 1e-4, 1.0, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "tree_method": Config().tree_method,
        "random_state": Config().random_seed,
        "n_jobs": -1,
    }
    fit_kwargs = {
        "eval_set": [(Xs, ys)],
        "eval_metric": "rmse",
        "verbose": False,
        "early_stopping_rounds": Config().early_stopping_rounds,
    }
    models, oof, folds = cross_validate_model(
        xgb.XGBRegressor,
        params,
        Xs,
        ys,
        Config().n_folds,
        Config().random_seed,
        **fit_kwargs
    )
    trial.report(oof, step=0)
    if trial.should_prune():
        raise optuna.TrialPruned()
    return oof


suite = optuna.create_study(
    direction="minimize",
    sampler=TPESampler(seed=seed),
    pruner=MedianPruner(n_warmup_steps=5),
)
suite.optimize(objective, n_trials=20, show_progress_bar=True)

best = suite.best_trial.params
cfg = Config.load()
for k, v in best.items():
    setattr(cfg, k, v)
with open("data/s5_e5/xgb_best_config.json", "w") as f:
    json.dump(asdict(cfg), f, indent=2)

# 6) Final training on full data
fit_kwargs_full = {
    "eval_set": [],
    "eval_metric": "rmse",
    "verbose": False,
    "early_stopping_rounds": cfg.early_stopping_rounds,
}
models, oof, _ = cross_validate_model(
    xgb.XGBRegressor,
    asdict(cfg),
    X_train,
    y_train,
    cfg.n_folds,
    cfg.random_seed,
    **fit_kwargs_full
)
print("Full CV RMSLE:", oof)

# 7) Predict & save
preds_log = np.zeros(len(X_test))
for m in models:
    preds_log += m.predict(X_test) / len(models)
preds = np.expm1(preds_log).clip(min=0)
pd.DataFrame({"id": test_raw["id"], "Calories": preds}).to_csv(
    "submission.csv", index=False
)
