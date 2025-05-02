import lightgbm as lgb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_log_error

from dataclasses import dataclass, asdict
from tqdm import tqdm
from pathlib import Path

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


def engineer_features(
    df, add_bmi=True, use_bmi_only=True, add_log_duration=True, add_temp_elevation=True
):
    """
    Create selected engineered features based on analysis.

    Parameters:
    df (pd.DataFrame): Input dataframe
    add_bmi (bool): Whether to calculate BMI.
    use_bmi_only (bool): If True and add_bmi is True, drop Height and Weight.
    add_log_duration (bool): Whether to add log(Duration + 1).
    add_temp_elevation (bool): Whether to add Body_Temp - 37.

    Returns:
    pd.DataFrame: Dataframe with *selected* engineered features
    """
    df_eng = df.copy()

    if add_temp_elevation:
        df_eng["Temp_Elevation"] = df_eng["Body_Temp"] - 37.0  # Use float

    if add_bmi:
        # Ensure height is not zero and handle potential division errors
        df_eng["Height_m"] = df_eng["Height"] / 100
        df_eng["BMI"] = df_eng["Weight"] / (
            df_eng["Height_m"] ** 2 + 1e-6
        )  # Add epsilon for stability
        df_eng.drop(columns=["Height_m"], inplace=True)  # Drop intermediate column

        if use_bmi_only:
            # Drop original Height and Weight if BMI is used exclusively
            if "Height" in df_eng.columns:
                df_eng.drop(columns=["Height"], inplace=True)
            if "Weight" in df_eng.columns:
                df_eng.drop(columns=["Weight"], inplace=True)

    if add_log_duration:
        df_eng["Log_Duration"] = np.log1p(df_eng["Duration"])

    return df_eng


@dataclass
class Config:
    iterations: int = 100
    learning_rate: float = 0.1
    depth: int = -1  # No limit on tree depth
    l2_leaf_reg: float = 0.0  # No regularization by default
    num_leaves: int = 31  # Default value in LightGBM
    loss_function: str = "RMSE"
    eval_metric: str = "RMSE"
    random_seed: int = 42
    early_stopping_rounds: int = 50
    n_folds: int = 5
    verbose: int = -1
    
    @classmethod
    def load(cls, path: str = "data/s5_e5/best_config_revised.json") -> "Config":
        p = Path(path)
        if p.is_file():
            try:
                data = json.loads(p.read_text())
                return cls(**data)
            except Exception:
                pass
        return cls()


# --- Revised Baseline Model Training ---
def train_lightgbm_cv(train_df_engineered, features_to_use, config):
    """
    Train a LightGBM model using KFold cross-validation on pre-engineered data.
    Optimizes for RMSLE by predicting log1p(target).
    Handles Sex encoding internally.
    """
    X = train_df_engineered[features_to_use].copy()
    y = train_df_engineered["Calories"]

    # --- Encode Sex ---
    # Fit the encoder on the entire training set's 'Sex' column if it exists
    le = LabelEncoder()
    if "Sex" in X.columns:
        X["Sex"] = le.fit_transform(X["Sex"])
        print("LabelEncoder fitted for 'Sex' feature.")
    else:
        le = None  # No encoder needed if 'Sex' is not a feature
        print("'Sex' feature not found in features_to_use.")

    # Log transform target
    y_log = np.log1p(y)

    # Setup cross-validation
    kf = KFold(n_splits=config.n_folds, shuffle=True, random_state=config.random_seed)
    oof_preds_log = np.zeros(len(X))
    oof_true_log = np.zeros(len(X))
    models = []
    fold_rmsle_scores = []
    feature_importances = pd.DataFrame(index=features_to_use)

    print(f"Starting {config.n_folds}-Fold Cross-Validation...")
    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y_log)):
        print(f"--- Fold {fold+1}/{config.n_folds} ---")
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train_log, y_val_log = y_log.iloc[train_idx], y_log.iloc[val_idx]

        model = lgb.LGBMRegressor(
            n_estimators=config.iterations,
            learning_rate=config.learning_rate,
            max_depth=config.depth,
            num_leaves=config.num_leaves,  # Fixed: Use config value
            reg_lambda=config.l2_leaf_reg,
            objective="rmse",  # Corresponds to config.loss_function for regression
            random_state=config.random_seed + fold,  # Vary seed per fold
            n_jobs=-1,  # Use all available cores
            # verbose=config.verbose # Controlled by callbacks now
        )

        model.fit(
            X_train,
            y_train_log,
            eval_set=[(X_val, y_val_log)],
            eval_metric=config.eval_metric,
            callbacks=[
                lgb.early_stopping(
                    stopping_rounds=config.early_stopping_rounds, verbose=False
                ),
                # lgb.log_evaluation(period=config.iterations // 10) # Optional: log progress
            ],
        )

        # Store predictions for OOF evaluation
        val_preds_log = model.predict(X_val)
        oof_preds_log[val_idx] = val_preds_log
        oof_true_log[val_idx] = y_val_log  # Store log-transformed true values

        # Calculate RMSLE for this fold (using original scale)
        y_val_pred = np.expm1(val_preds_log)
        y_val_true = np.expm1(y_val_log)
        # Clip predictions to avoid issues with negative values if they occur
        y_val_pred = np.maximum(0, y_val_pred)
        fold_rmsle = np.sqrt(mean_squared_log_error(y_val_true, y_val_pred))
        fold_rmsle_scores.append(fold_rmsle)
        print(f"Fold {fold+1} RMSLE: {fold_rmsle:.5f}")

        # Store model and feature importance
        models.append(model)
        feature_importances[f"Fold_{fold+1}"] = model.feature_importances_

    # Calculate overall OOF RMSLE
    overall_oof_rmsle = np.sqrt(
        mean_squared_log_error(np.expm1(oof_true_log), np.expm1(oof_preds_log))
    )
    print("-" * 30)
    print(f"Overall OOF RMSLE: {overall_oof_rmsle:.5f}")
    print(
        f"Mean Fold RMSLE:   {np.mean(fold_rmsle_scores):.5f} ± {np.std(fold_rmsle_scores):.5f}"
    )
    print("-" * 30)

    # --- Optional: Retrain on full data ---
    # For submission, it's common to retrain on the full dataset using found parameters
    # Or average predictions from fold models. Averaging is often more robust.
    # We will use the fold models for prediction.

    # --- Feature Importance ---
    feature_importances["mean"] = feature_importances.mean(axis=1)
    feature_importances.sort_values(by="mean", ascending=False, inplace=True)
    print("Average Feature Importances across folds:")
    print(feature_importances["mean"])
    # Plot feature importance
    plt.figure(figsize=(10, len(features_to_use) * 0.4))
    sns.barplot(x='mean', y=feature_importances.index, data=feature_importances)
    plt.title('Mean Feature Importance')
    plt.tight_layout()
    plt.show()

    return (
        models,
        le,
        overall_oof_rmsle,
    )  # Return list of models and the single fitted encoder


# --- Revised Submission Function ---
def make_submission(
    models, le, test_df_engineered, features_to_use, submission_path="submission.csv"
):
    """
    Generate predictions on the test set using an ensemble of models (from CV folds).
    Applies necessary encoding and inverse transform.
    """
    X_test = test_df_engineered[features_to_use].copy()
    ids = test_df_engineered["id"]

    # Apply the same encoding for Sex using the fitted encoder
    if le and "Sex" in X_test.columns:
        try:
            X_test["Sex"] = le.transform(X_test["Sex"])
            print("Applied fitted LabelEncoder to 'Sex' in test data.")
        except ValueError as e:
            print(
                f"Warning: Could not transform 'Sex' in test data. Maybe unseen labels? Error: {e}"
            )
            # Handle unseen labels if necessary, e.g., map to a default category or -1
            # For simplicity here, we'll assume it works or the column isn't used if error occurs
            pass  # Or handle more robustly
    elif "Sex" in features_to_use and not le:
        print(
            "Warning: 'Sex' is in features_to_use but LabelEncoder was not fitted (e.g., not present in training)."
        )

    # Predict using the ensemble of models (average predictions)
    test_preds_log = np.zeros(len(X_test))
    print(f"Predicting with {len(models)} models...")
    for model in tqdm(models, desc="Predicting Folds"):
        fold_preds_log = model.predict(X_test)
        test_preds_log += fold_preds_log / len(models)  # Average log predictions

    # Inverse transform predictions
    test_preds = np.expm1(test_preds_log)
    # Ensure predictions are non-negative
    test_preds = np.maximum(0, test_preds)

    # Create submission file
    submission = pd.DataFrame({"id": ids, "Calories": test_preds})
    submission.to_csv(submission_path, index=False)
    print(f"Saved submission with {len(submission)} rows to {submission_path}")
    print(submission.head())


# --- Revised Optuna Tuning Function ---
def tune_optuna(
    raw_train_df: pd.DataFrame,  # Takes RAW data
    feature_engineering_config: dict,  # Dict specifying how to engineer features
    base_config: Config,
    n_trials: int = 50,
):
    """
    Optimize LightGBM hyperparameters with Optuna using cross-validation.
    Applies specified feature engineering within the objective function.
    """
    print("Starting Optuna Hyperparameter Tuning...")

    def objective(trial):
        # --- Feature Engineering within Objective ---
        # Ensures tuning happens on the *exact* same feature process
        current_train_df_eng = engineer_features(
            raw_train_df, **feature_engineering_config
        )

        # --- Define Features based on Engineering ---
        current_features = list(raw_train_df.columns)
        current_features.remove("id")
        current_features.remove("Calories")
        if feature_engineering_config.get("add_temp_elevation", False):
            current_features.append("Temp_Elevation")
        if feature_engineering_config.get("add_bmi", False):
            current_features.append("BMI")
            if feature_engineering_config.get("use_bmi_only", False):
                if "Height" in current_features:
                    current_features.remove("Height")
                if "Weight" in current_features:
                    current_features.remove("Weight")
        if feature_engineering_config.get("add_log_duration", False):
            current_features.append("Log_Duration")
        # Ensure only existing columns are kept (handles cases where base columns were dropped)
        current_features = [
            f for f in current_features if f in current_train_df_eng.columns
        ]

        # --- Prepare Data ---
        X = current_train_df_eng[current_features].copy()
        y = current_train_df_eng["Calories"]
        y_log = np.log1p(y)

        # --- Encode Sex ---
        trial_le = LabelEncoder()
        if "Sex" in X.columns:
            X["Sex"] = trial_le.fit_transform(X["Sex"])

        # --- Sample Hyperparameters ---
        params = {
            "n_estimators": trial.suggest_int("iterations", 1000, 20000, step=1000),
            "learning_rate": trial.suggest_float(
                "learning_rate", 0.005, 0.05, log=True
            ),  # Adjusted range
            "max_depth": trial.suggest_int("depth", 4, 12),  # Adjusted range
            "num_leaves": trial.suggest_int(
                "num_leaves", 10, 70
            ),  # Relative to max_depth
            "reg_lambda": trial.suggest_float(
                "l2_leaf_reg", 1e-6, 10.0, log=True
            ),  # Wider range for l2
            # "reg_alpha": trial.suggest_float("l1_reg", 1e-8, 1.0, log=True), # Optional: Add L1
            # "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0), # Optional: Feature fraction
            # "subsample": trial.suggest_float("subsample", 0.6, 1.0), # Optional: Bagging fraction
            "objective": "rmse",
            "random_state": base_config.random_seed,
            "n_jobs": -1,
            # "verbose": -1, # Handled by callbacks
        }
        # Ensure num_leaves is appropriate for depth
        params["num_leaves"] = min(params["num_leaves"], 2 ** params["max_depth"] - 1)

        # --- Cross-validation within Objective ---
        kf = KFold(
            n_splits=base_config.n_folds,
            shuffle=True,
            random_state=base_config.random_seed,
        )
        fold_scores = []

        for fold, (train_idx, val_idx) in enumerate(kf.split(X, y_log)):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train_log, y_val_log = y_log.iloc[train_idx], y_log.iloc[val_idx]

            model = lgb.LGBMRegressor(**params)
            model.fit(
                X_train,
                y_train_log,
                eval_set=[(X_val, y_val_log)],
                eval_metric=base_config.eval_metric,
                callbacks=[
                    lgb.early_stopping(
                        stopping_rounds=base_config.early_stopping_rounds, verbose=False
                    )
                ],
            )

            y_pred_log = model.predict(X_val)
            y_pred = np.expm1(y_pred_log)
            y_val = np.expm1(y_val_log)
            y_pred = np.maximum(0, y_pred)  # Clip preds
            fold_rmsle = np.sqrt(mean_squared_log_error(y_val, y_pred))
            fold_scores.append(fold_rmsle)

            # Optuna Pruning Check
            trial.report(fold_rmsle, step=fold)
            if trial.should_prune():
                raise optuna.TrialPruned()

        mean_rmsle = np.mean(fold_scores)
        return mean_rmsle  # Optuna minimizes this

    # --- Run Optuna Study ---
    sampler = TPESampler(seed=base_config.random_seed)
    pruner = MedianPruner(
        n_warmup_steps=base_config.n_folds * (n_trials // 10)
    )  # Prune after ~10% trials complete a fold
    study = create_study(direction="minimize", sampler=sampler, pruner=pruner)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    print("\nOptuna Tuning Finished.")
    print("Best CV RMSLE:", study.best_value)
    print("Best params:", study.best_trial.params)

    # Create Config object with best parameters merged with base
    best_params_renamed = {  # Rename keys to match Config fields
        "iterations": study.best_trial.params.get("iterations", base_config.iterations),
        "learning_rate": study.best_trial.params.get(
            "learning_rate", base_config.learning_rate
        ),
        "depth": study.best_trial.params.get("depth", base_config.depth),
        "l2_leaf_reg": study.best_trial.params.get(
            "l2_leaf_reg", base_config.l2_leaf_reg
        ),
        "num_leaves": study.best_trial.params.get("num_leaves", base_config.num_leaves),
    }
    final_config_dict = asdict(base_config)  # Start with base config
    final_config_dict.update(best_params_renamed)  # Update with best tuned params

    return Config(**final_config_dict)


# --- Main Execution Block ---
if __name__ == "__main__":
    # --- Configuration ---
    RUN_OPTUNA = True # Set to True to run hyperparameter tuning
    N_OPTUNA_TRIALS = 20 # Increase for serious tuning
    SUBMISSION_FILENAME = "data/s5_e5/submission_simplified_features.csv" # New name
    CONFIG_SAVE_PATH = "data/s5_e5/lgbm_best_config_simplified.json" # New name

    # 1. Load Raw Data
    print("Loading data...")
    train_df_raw, test_df_raw = load_data()
    print(f"Loaded Train: {train_df_raw.shape}, Test: {test_df_raw.shape}")

    # 2. Define Feature Engineering Strategy
    # **** MODIFICATION ****
    # Based on feature importance, disable Temp_Elevation and Log_Duration
    feature_eng_config = {
        "add_bmi": True,
        "use_bmi_only": True,
        "add_log_duration": False,  # <-- Set to False
        "add_temp_elevation": False # <-- Set to False
    }
    print("Selected Feature Engineering Strategy (Simplified):")
    print(feature_eng_config)

    # 3. Apply Feature Engineering
    print("Applying feature engineering...")
    train_df_eng = engineer_features(train_df_raw, **feature_eng_config)
    test_df_eng = engineer_features(test_df_raw, **feature_eng_config)
    print(f"Engineered Train shape: {train_df_eng.shape}, Test shape: {test_df_eng.shape}")

    # 4. Define Final Feature List for Model
    # **** MODIFICATION ****
    # Must match the features created by engineer_features with the chosen config
    # Start with base features expected to remain
    features_to_use = ['Sex', 'Age', 'Duration', 'Heart_Rate', 'Body_Temp']
    # Conditionally add features based on the config
    if feature_eng_config.get('add_bmi'): features_to_use.append('BMI')
    # These lines will now correctly NOT add the features:
    if feature_eng_config.get('add_temp_elevation'): features_to_use.append('Temp_Elevation')
    if feature_eng_config.get('add_log_duration'): features_to_use.append('Log_Duration')

    # Ensure all selected features actually exist after engineering (safety check)
    features_to_use = [f for f in features_to_use if f in train_df_eng.columns]
    print("Final features for model (Simplified):", features_to_use)


    # 5. Hyperparameter Tuning (Optional)
    # (Keep this section as is - if you run Optuna, it will now tune
    # based on this simplified feature set because tune_optuna uses
    # the feature_eng_config passed to it)
    base_config = Config()
    if RUN_OPTUNA:
        # Make sure raw data is passed if tuning
        best_config = tune_optuna(
            raw_train_df=train_df_raw,
            feature_engineering_config=feature_eng_config, # Pass the simplified config
            base_config=base_config,
            n_trials=N_OPTUNA_TRIALS
        )
        print(f"Saving best config found by Optuna to {CONFIG_SAVE_PATH}")
        with open(CONFIG_SAVE_PATH, "w") as f:
            json.dump(asdict(best_config), f, indent=4)
    else:
        print("Skipping Optuna tuning. Using default config.")
        best_config = base_config
        # Optional loading logic remains the same


    # 6. Train Final Model using Cross-Validation
    print("\nTraining final model with simplified features and configuration...")
    models, label_encoder, oof_score = train_lightgbm_cv(
        train_df_engineered=train_df_eng,
        features_to_use=features_to_use, # Use the updated list
        config=best_config
    )
    print(f"Achieved OOF RMSLE with simplified features: {oof_score:.5f}") # Note the score

    # 7. Make Submission
    print("\nGenerating submission file...")
    make_submission(
        models=models,
        le=label_encoder,
        test_df_engineered=test_df_eng,
        features_to_use=features_to_use, # Use the updated list
        submission_path=SUBMISSION_FILENAME
    )

    print("\nScript finished.")