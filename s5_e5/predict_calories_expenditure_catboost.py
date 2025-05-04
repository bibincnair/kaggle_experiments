import catboost as cb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold

from dataclasses import dataclass
from tqdm import tqdm

import optuna
from optuna import create_study
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
from optuna.visualization import plot_optimization_history, plot_param_importances


def load_data():
    """
    Load the training and test data.
    """
    train_path = "data/s5_e5/train.csv"
    test_path = "data/s5_e5/test.csv"

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    return train_df, test_df


@dataclass
class Config:
    iterations: int = 10000
    learning_rate: float = 0.03
    depth: int = 6
    l2_leaf_reg: float = 3.0
    loss_function: str = "RMSE"
    eval_metric: str = "RMSE"
    random_seed: int = 42
    early_stopping_rounds: int = 50
    verbose: bool = False


# 1. Fix baseline_catboost to use config parameter
def baseline_catboost(train_df, test_df, config=None):
    """Train a CatBoost model on log1p(calories) to optimize RMSLE."""
    if config is None:
        config = Config()
    
    # Define features and target
    X = train_df.drop(columns=["id", "Calories"])
    y = train_df["Calories"]

    # Encode Sex
    le = LabelEncoder()
    X["Sex"] = le.fit_transform(X["Sex"])

    # Initialize CatBoost Regressor with config
    model = cb.CatBoostRegressor(
        iterations=config.iterations,
        learning_rate=config.learning_rate,
        depth=config.depth,
        l2_leaf_reg=config.l2_leaf_reg,
        loss_function=config.loss_function,
        eval_metric=config.eval_metric,
        random_seed=config.random_seed,
        verbose=100 if config.verbose else 0,
        early_stopping_rounds=config.early_stopping_rounds,
    )

    # Use cross-validation instead of a single split
    from sklearn.model_selection import KFold
    
    kf = KFold(n_splits=5, shuffle=True, random_state=config.random_seed)
    rmsle_scores = []
    
    for train_idx, val_idx in kf.split(X):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        # Log transform targets
        y_train_log = np.log1p(y_train)
        y_val_log = np.log1p(y_val)
        
        # Train model
        model.fit(
            X_train, y_train_log, 
            eval_set=(X_val, y_val_log),
            use_best_model=True,
            verbose=config.verbose
        )
        
        # Predict and calculate true RMSLE
        y_pred_log = model.predict(X_val)
        y_pred = np.expm1(y_pred_log)
        rmsle = np.sqrt(np.mean((np.log1p(y_pred) - np.log1p(y_val))**2))
        rmsle_scores.append(rmsle)
    
    # Report cross-validation results
    print(f"Cross-validation RMSLE: {np.mean(rmsle_scores):.5f} ± {np.std(rmsle_scores):.5f}")
    
    # Retrain on full dataset for final model
    y_log = np.log1p(y)
    model.fit(X, y_log)
    
    return model, le


def make_submission(model, le, test_df, submission_path="submission.csv"):
    """
    Encode test, predict, invert log1p and save id,Calories.
    """
    X_test = test_df.copy()
    ids = X_test["id"]
    X_test = X_test.drop(columns=["id"])

    # same encoding
    X_test["Sex"] = le.transform(X_test["Sex"])

    # predict log‑calories, invert
    preds_log = model.predict(X_test)
    preds = np.expm1(preds_log)

    submission = pd.DataFrame({"id": ids, "Calories": preds})
    submission.to_csv(submission_path, index=False)
    print(f"Saved submission to {submission_path}")


def tune_optuna(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    train_func: callable,
    config: Config = None,
    n_trials: int = 50,
    n_folds: int = 5
):
    """
    Optimize CatBoost hyperparameters with Optuna using cross-validation.
    Returns the best parameter dict.
    """

    def objective(trial):
        # sample hyperparameters
        params = {
            "iterations": trial.suggest_int("iterations", 1000, 10000, step=1000),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "depth": trial.suggest_int("depth", 4, 10),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-8, 10.0, log=True),
            "loss_function": "RMSE",
            "eval_metric": "RMSE",
            "random_seed": 42,
            "verbose": False,
        }
        
        # prepare data
        X = train_df.drop(columns=["id", "Calories"]).copy()
        y = train_df["Calories"]
        X["Sex"] = LabelEncoder().fit_transform(X["Sex"])
        
        # Setup cross-validation
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
        
        # Store scores from each fold
        fold_scores = []
        
        for train_idx, val_idx in tqdm(kf.split(X), total=kf.get_n_splits(), desc="Folds"):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            # Log transform targets
            y_train_log = np.log1p(y_train)
            y_val_log = np.log1p(y_val)
            
            # Train model
            model = cb.CatBoostRegressor(**params)
            model.fit(
                X_train, 
                y_train_log,
                eval_set=(X_val, y_val_log),
                early_stopping_rounds=50,
                use_best_model=True,
                verbose=100,
                # task_type="GPU",
                # devices='0'
            )
            
            # Calculate true RMSLE on validation fold
            y_pred_log = model.predict(X_val)
            y_pred = np.expm1(y_pred_log)
            rmsle = np.sqrt(np.mean((np.log1p(y_pred) - np.log1p(y_val))**2))
            fold_scores.append(rmsle)
        
        # Return mean RMSLE across all folds
        mean_rmsle = np.mean(fold_scores)
        
        # Report intermediate results
        trial.report(mean_rmsle, step=0)
        
        return mean_rmsle

    sampler = TPESampler(seed=42)
    pruner = MedianPruner()
    study = create_study(direction="minimize", sampler=sampler, pruner=pruner)
    study.optimize(objective, n_trials=n_trials)

    print("Best CV RMSLE:", study.best_value)
    print("Best params:", study.best_trial.params)
    
    # Create Config object with best parameters
    base = Config()
    best = base.__dict__.copy()
    best.update(study.best_trial.params)
    
    return Config(**best)


if __name__ == "__main__":
    # Load the data
    print("Loading data...")
    train_df, test_df = load_data()
    print("Data loaded.")
    print("Train Data Columns: ", train_df.columns)
    print("Test Data Columns: ", test_df.columns)
    
    # Train the baseline model
    print("Training baseline model...")
    model, le = baseline_catboost(train_df, test_df)

    best_config = tune_optuna(train_df, test_df, baseline_catboost)
    
    # Train the model with the best config
    model, le = baseline_catboost(train_df, test_df, best_config)
    
    make_submission(model, le, test_df, submission_path="data/s5_e5/submission.csv")
