import lightgbm as lgb
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

import json


def load_data():
    """
    Load the training and test data.
    """
    train_path = "data/s5_e5/train.csv"
    test_path = "data/s5_e5/test.csv"

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    return train_df, test_df


def engineer_features(df, is_training=True):
    """
    Create engineered features for calorie expenditure prediction.

    Parameters:
    df (pd.DataFrame): Input dataframe
    is_training (bool): Whether this is training data with Calories column

    Returns:
    pd.DataFrame: Dataframe with engineered features
    """
    # Create a copy to avoid modifying the original
    df_eng = df.copy()

    # 1. Body Composition Features
    # BMI - Body Mass Index
    df_eng["BMI"] = df_eng["Weight"] / ((df_eng["Height"] / 100) ** 2)

    # Body Surface Area (Mosteller formula)
    df_eng["BSA"] = np.sqrt((df_eng["Height"] * df_eng["Weight"]) / 3600)

    # 2. Exercise Intensity Metrics
    # Estimated Max Heart Rate
    df_eng["MaxHR"] = 220 - df_eng["Age"]

    # Heart Rate Reserve (assuming resting HR of 60)
    df_eng["HRR"] = (df_eng["Heart_Rate"] - 60) / (df_eng["MaxHR"] - 60)
    df_eng["HRR"] = df_eng["HRR"].clip(0, 1)  # Ensure values between 0-1

    # Heart Rate as percentage of max
    df_eng["HR_Percentage"] = df_eng["Heart_Rate"] / df_eng["MaxHR"]

    # Temperature Elevation from normal
    df_eng["Temp_Elevation"] = df_eng["Body_Temp"] - 37

    # Exercise Intensity Score (combining HR and temperature)
    df_eng["Intensity_Score"] = (
        df_eng["HR_Percentage"] + df_eng["Temp_Elevation"] / 4
    ) / 2

    # 3. Duration-Related Features
    # Duration Squared for non-linear relationship
    df_eng["Duration_Squared"] = df_eng["Duration"] ** 2

    # Log Duration
    df_eng["Log_Duration"] = np.log1p(df_eng["Duration"])

    # 4. Physiological Interactions
    # Duration × Heart_Rate
    df_eng["Duration_HeartRate"] = df_eng["Duration"] * df_eng["Heart_Rate"]

    # Duration × Body_Temp
    df_eng["Duration_BodyTemp"] = df_eng["Duration"] * df_eng["Temp_Elevation"]

    # Weight × Duration
    df_eng["Weight_Duration"] = df_eng["Weight"] * df_eng["Duration"]

    # Heart_Rate × Body_Temp
    df_eng["HeartRate_BodyTemp"] = df_eng["Heart_Rate"] * df_eng["Temp_Elevation"]

    # Heart_Rate × Age
    df_eng["HeartRate_Age"] = df_eng["Heart_Rate"] * df_eng["Age"]

    # 5. Metabolic Estimates
    # MET Estimate based on heart rate
    df_eng["MET_Estimate"] = 0.6 * df_eng["HR_Percentage"] * 10

    # 6. Demographic Adjustments
    # Convert Sex to numeric (1 for male, 0 for female)
    df_eng["IsMale"] = (df_eng["Sex"] == "male").astype(int)

    # Sex-based interaction terms
    df_eng["Sex_HeartRate"] = df_eng["IsMale"] * df_eng["Heart_Rate"]
    df_eng["Sex_Duration"] = df_eng["IsMale"] * df_eng["Duration"]

    # Age Groups
    df_eng["Age_Group"] = pd.cut(
        df_eng["Age"],
        bins=[0, 30, 40, 50, 60, 100],
        labels=["<30", "30-40", "40-50", "50-60", "60+"],
    )

    # Create dummy variables for Age_Group
    age_dummies = pd.get_dummies(df_eng["Age_Group"], prefix="Age_Group")
    df_eng = pd.concat([df_eng, age_dummies], axis=1)

    # Combined physiological features
    df_eng["HR_BMI"] = df_eng["Heart_Rate"] * df_eng["BMI"]
    df_eng["Duration_BMI"] = df_eng["Duration"] * df_eng["BMI"]

    # Triple interaction: Duration × Heart_Rate × Body_Temp
    df_eng["Duration_HR_Temp"] = (
        df_eng["Duration"] * df_eng["Heart_Rate"] * df_eng["Temp_Elevation"]
    )

    return df_eng


@dataclass
class Config:
    iterations: int = 17000
    learning_rate: float = 0.010576344889641518
    depth: int = 7
    l2_leaf_reg: float = 0.0005731574610766452
    num_leaves: int = 41
    loss_function: str = "RMSE"
    eval_metric: str = "RMSE"
    random_seed: int = 42
    early_stopping_rounds: int = 50
    verbose: int = -1


def baseline_lightgbm(train_df, test_df, config=None):
    """Train a LightGBM model on log1p(calories) to optimize RMSLE."""
    if config is None:
        config = Config()

    # Add engineered features to training data
    # train_df_eng = engineer_features(train_df)
    train_df_eng = train_df.copy()

    # Define features and target
    X = train_df_eng.drop(columns=["id", "Calories", "Sex", "Age_Group"])
    y = train_df_eng["Calories"]

    # Encode Sex (already handled in feature engineering)
    le = LabelEncoder()
    X["Sex"] = le.fit_transform(train_df_eng["Sex"])

    # Initialize LightGBM Regressor with config
    model = lgb.LGBMRegressor(
        n_estimators=config.iterations,
        learning_rate=config.learning_rate,
        max_depth=config.depth,
        reg_lambda=config.l2_leaf_reg,
        objective="rmse",
        random_state=config.random_seed,
        verbose=config.verbose,
        n_jobs=4,
        num_leaves=31,
    )

    # Use cross-validation instead of a single split
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
            X_train,
            y_train_log,
            eval_set=[(X_val, y_val_log)],
            eval_metric=config.eval_metric,
            callbacks=[
                lgb.early_stopping(stopping_rounds=config.early_stopping_rounds)
            ],
        )

        # Predict and calculate true RMSLE
        y_pred_log = model.predict(X_val)
        y_pred = np.expm1(y_pred_log)
        rmsle = np.sqrt(np.mean((np.log1p(y_pred) - np.log1p(y_val)) ** 2))
        rmsle_scores.append(rmsle)

    # Report cross-validation results
    print(
        f"Cross-validation RMSLE: {np.mean(rmsle_scores):.5f} ± {np.std(rmsle_scores):.5f}"
    )

    # Retrain on full dataset for final model
    y_log = np.log1p(y)
    model.fit(X, y_log)

    return model, le


def make_submission(model, le, test_df, submission_path="submission.csv"):
    """
    Encode test, predict, invert log1p and save id,Calories.
    """
    # Add engineered features to test data
    # X_test = engineer_features(test_df, is_training=False)
    X_test = test_df.copy()
    ids = X_test["id"]

    # Drop unnecessary columns
    X_test = X_test.drop(columns=["id", "Sex", "Age_Group"])

    # Apply the same encoding for Sex
    X_test["Sex"] = le.transform(test_df["Sex"])

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
    n_folds: int = 5,
):
    """
    Optimize LightGBM hyperparameters with Optuna using cross-validation.
    Returns the best parameter dict.
    """

    def objective(trial):
        # sample hyperparameters
        params = {
            "iterations": trial.suggest_int("iterations", 1000, 20000, step=1000),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "depth": trial.suggest_int("depth", 4, 16),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-8, 10.0, log=True),
            "loss_function": "RMSE",
            "eval_metric": "RMSE",
            "random_seed": 42,
            "verbose": -1,
            "num_leaves": trial.suggest_int("num_leaves", 10, 100),
            "num_jobs": 5,
        }

        train_df_eng = train_df 

        # prepare data
        X = train_df_eng.drop(columns=["id", "Calories", "Sex", "Age_Group"]).copy()
        y = train_df_eng["Calories"]

        # Encode Sex
        X["Sex"] = LabelEncoder().fit_transform(train_df_eng["Sex"])

        # Setup cross-validation
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

        # Store scores from each fold
        fold_scores = []

        for train_idx, val_idx in tqdm(
            kf.split(X), total=kf.get_n_splits(), desc="Folds"
        ):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            # Log transform targets
            y_train_log = np.log1p(y_train)
            y_val_log = np.log1p(y_val)

            # Train model
            model = lgb.LGBMRegressor(
                n_estimators=params["iterations"],
                learning_rate=params["learning_rate"],
                max_depth=params["depth"],
                reg_lambda=params["l2_leaf_reg"],
                objective="rmse",
                random_state=42,
                verbose=-1,
            )
            model.fit(
                X_train,
                y_train_log,
                eval_set=[(X_val, y_val_log)],
                eval_metric="rmse",
                callbacks=[
                    lgb.early_stopping(stopping_rounds=50),
                ],
            )

            # Calculate true RMSLE on validation fold
            y_pred_log = model.predict(X_val)
            y_pred = np.expm1(y_pred_log)
            rmsle = np.sqrt(np.mean((np.log1p(y_pred) - np.log1p(y_val)) ** 2))
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
    # feature engineering
    train_df = engineer_features(train_df)
    test_df = engineer_features(test_df, is_training=False)
    print("Data loaded.")
    print("Train Data Columns: ", train_df.columns)
    print("Test Data Columns: ", test_df.columns)

    # First, tune hyperparameters
    # best_config = tune_optuna(train_df, test_df, baseline_lightgbm)
    best_config = Config()
    # Save the best config to a JSON file
    with open("data/s5_e5/lgbm_best_config.json", "w") as f:
        json.dump(best_config.__dict__, f, indent=4)

    # Train the model with the best config
    model, le = baseline_lightgbm(train_df, test_df, best_config)

    make_submission(model, le, test_df, submission_path="data/s5_e5/submission.csv")
