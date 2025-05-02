# --- IMPORTS ---
# Remove or comment out lightgbm import
# import lightgbm as lgb
import xgboost as xgb  # Import XGBoost
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_log_error

from dataclasses import dataclass, asdict
from tqdm import tqdm
from pathlib import Path

import optuna
from optuna import create_study
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner

# from optuna.visualization import plot_optimization_history, plot_param_importances

import json
import warnings

warnings.filterwarnings("ignore", category=UserWarning)


# --- Data Loading (Unchanged) ---
def load_data():
    """
    Load the training and test data.
    """
    train_path = "data/s5_e5/train.csv"
    test_path = "data/s5_e5/test.csv"
    # Fallback paths if needed
    # train_path = "../input/playground-series-s4e5/train.csv"
    # test_path = "../input/playground-series-s4e5/test.csv"

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    # Drop Height and Weight from train/test if using BMI only later
    # train_df.drop(columns=['Height', 'Weight'], inplace=True, errors='ignore')
    # test_df.drop(columns=['Height', 'Weight'], inplace=True, errors='ignore')

    return train_df, test_df


# --- Feature Engineering (Unchanged) ---
def engineer_features(
    df, add_bmi=True, use_bmi_only=True, add_log_duration=True, add_temp_elevation=True
):
    # (Keep the function exactly as it was in the previous step)
    df_eng = df.copy()

    if add_temp_elevation:
        df_eng["Temp_Elevation"] = df_eng["Body_Temp"] - 37.0

    if add_bmi:
        df_eng["Height_m"] = df_eng["Height"] / 100
        df_eng["BMI"] = df_eng["Weight"] / (df_eng["Height_m"] ** 2 + 1e-6)
        df_eng.drop(columns=["Height_m"], inplace=True)

        if use_bmi_only:
            if "Height" in df_eng.columns:
                df_eng.drop(columns=["Height"], inplace=True)
            if "Weight" in df_eng.columns:
                df_eng.drop(columns=["Weight"], inplace=True)

    if add_log_duration:
        df_eng["Log_Duration"] = np.log1p(df_eng["Duration"])

    return df_eng


# --- Configuration (Add XGBoost specific defaults if needed) ---
@dataclass
class Config:
    # XGBoost specific parameters (defaults adjusted)
    iterations: int = 1000  # Mapped to n_estimators
    learning_rate: float = 0.05
    depth: int = 6  # Mapped to max_depth (XGBoost default is 6)
    l2_leaf_reg: float = 1.0  # Mapped to reg_lambda (XGBoost default is 1)
    # num_leaves: int = 31 # Not directly used in XGBoost default tuning
    gamma: float = 0.0  # XGBoost specific: Minimum loss reduction required (default 0)
    subsample: float = 1.0  # XGBoost specific: Fraction of samples (default 1)
    colsample_bytree: float = 1.0  # XGBoost specific: Fraction of features (default 1)

    # Common parameters
    loss_function: str = "reg:squarederror"  # XGBoost objective for RMSE
    eval_metric: str = "rmse"  # XGBoost evaluation metric
    random_seed: int = 42
    early_stopping_rounds: int = 50
    n_folds: int = 5
    verbose: int = 0  # XGBoost verbosity (0=silent, 1=warning, 2=info, 3=debug)

    # GPU setting
    tree_method: str = "gpu_hist"  # Use 'hist' for CPU

    @classmethod
    def load(
        cls, path: str = "data/s5_e5/xgb_best_config.json"
    ) -> "Config":  # Changed default path
        p = Path(path)
        if p.is_file():
            try:
                data = json.loads(p.read_text())
                # Ensure all expected fields are present, falling back to defaults
                instance = cls()
                valid_data = {k: v for k, v in data.items() if hasattr(instance, k)}
                instance.__dict__.update(valid_data)
                return instance
            except Exception as e:
                print(f"Error loading config from {path}: {e}. Using defaults.")
                pass
        return cls()


# --- XGBoost Training Function ---
def train_xgboost_cv(train_df_engineered, features_to_use, config):
    """
    Train an XGBoost model using KFold cross-validation.
    Optimizes for RMSLE by predicting log1p(target).
    Handles Sex encoding internally. Uses GPU if available and configured.
    """
    X = train_df_engineered[features_to_use].copy()
    y = train_df_engineered["Calories"]

    # --- Encode Sex ---
    le = LabelEncoder()
    if "Sex" in X.columns:
        X["Sex"] = le.fit_transform(X["Sex"])
        print("LabelEncoder fitted for 'Sex' feature.")
    else:
        le = None
        print("'Sex' feature not found in features_to_use.")

    y_log = np.log1p(y)

    kf = KFold(n_splits=config.n_folds, shuffle=True, random_state=config.random_seed)
    oof_preds_log = np.zeros(len(X))
    oof_true_log = np.zeros(len(X))
    models = []
    fold_rmsle_scores = []
    # XGBoost feature importance doesn't require pre-defined DataFrame index usually
    # feature_importances = pd.DataFrame(index=features_to_use) # Can create later

    # --- Check GPU Availability for XGBoost ---
    actual_tree_method = config.tree_method
    if actual_tree_method == "gpu_hist":
        try:
            # A simple check: try initializing a basic XGB model with GPU
            xgb.XGBRegressor(tree_method="gpu_hist", n_estimators=1).fit(
                X.iloc[:1], y_log.iloc[:1]
            )
            print("GPU detected and accessible by XGBoost. Using 'gpu_hist'.")
        except Exception as e:
            print(
                f"Warning: Could not initialize XGBoost with GPU ('gpu_hist'). Error: {e}"
            )
            print("Falling back to CPU ('hist').")
            actual_tree_method = "hist"  # Fallback to CPU

    print(
        f"Starting {config.n_folds}-Fold Cross-Validation with XGBoost (tree_method='{actual_tree_method}')..."
    )
    all_feature_importances = {}

    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y_log)):
        print(f"--- Fold {fold+1}/{config.n_folds} ---")
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train_log, y_val_log = y_log.iloc[train_idx], y_log.iloc[val_idx]

        model = xgb.XGBRegressor(
            objective=config.loss_function,
            n_estimators=config.iterations,
            learning_rate=config.learning_rate,
            max_depth=(
                config.depth if config.depth > 0 else 0
            ),  # XGB uses 0 for no limit
            reg_lambda=config.l2_leaf_reg,
            gamma=config.gamma,
            subsample=config.subsample,
            colsample_bytree=config.colsample_bytree,
            random_state=config.random_seed + fold,
            n_jobs=-1,
            tree_method=actual_tree_method,  # Use determined method
            early_stopping_rounds=config.early_stopping_rounds,
            # verbosity=config.verbose # Control verbosity
        )

        # Note: XGBoost uses 'eval_metric' within fit, not a callback list like LightGBM for this purpose
        model.fit(
            X_train,
            y_train_log,
            eval_set=[(X_val, y_val_log)],
            eval_metric=config.eval_metric,
            verbose=False,  # Suppress verbose output during training iterations
        )

        val_preds_log = model.predict(X_val)
        oof_preds_log[val_idx] = val_preds_log
        oof_true_log[val_idx] = y_val_log

        y_val_pred = np.expm1(val_preds_log)
        y_val_true = np.expm1(y_val_log)
        y_val_pred = np.maximum(0, y_val_pred)
        fold_rmsle = np.sqrt(mean_squared_log_error(y_val_true, y_val_pred))
        fold_rmsle_scores.append(fold_rmsle)
        print(f"Fold {fold+1} RMSLE: {fold_rmsle:.5f}")

        models.append(model)
        # Store feature importances
        importances = model.get_booster().get_score(
            importance_type="weight"
        )  # or 'gain' or 'cover'
        for feature, importance in importances.items():
            if feature not in all_feature_importances:
                all_feature_importances[feature] = []
            all_feature_importances[feature].append(importance)

    overall_oof_rmsle = np.sqrt(
        mean_squared_log_error(np.expm1(oof_true_log), np.expm1(oof_preds_log))
    )
    print("-" * 30)
    print(f"Overall OOF RMSLE: {overall_oof_rmsle:.5f}")
    print(
        f"Mean Fold RMSLE:   {np.mean(fold_rmsle_scores):.5f} ± {np.std(fold_rmsle_scores):.5f}"
    )
    print("-" * 30)

    # --- Feature Importance Processing ---
    mean_importances = {
        feat: np.mean(imps) for feat, imps in all_feature_importances.items()
    }
    # Fill missing features (if a feature had 0 importance in all folds it wouldn't be in the dict)
    for feat in features_to_use:
        if feat not in mean_importances:
            mean_importances[feat] = 0.0
    feature_importance_df = pd.DataFrame.from_dict(
        mean_importances, orient="index", columns=["mean"]
    )
    feature_importance_df.sort_values(by="mean", ascending=False, inplace=True)

    print("Average Feature Importances across folds (XGBoost):")
    print(feature_importance_df["mean"])

    plt.figure(figsize=(10, max(6, len(features_to_use) * 0.4)))  # Adjust size
    sns.barplot(x="mean", y=feature_importance_df.index, data=feature_importance_df)
    plt.title("Mean Feature Importance (XGBoost)")
    plt.tight_layout()
    plt.show()

    return models, le, overall_oof_rmsle


# --- Submission Function (Unchanged - uses model.predict) ---
def make_submission(
    models, le, test_df_engineered, features_to_use, submission_path="submission.csv"
):
    # (This function remains exactly the same as before)
    X_test = test_df_engineered[features_to_use].copy()
    ids = test_df_engineered["id"]

    if le and "Sex" in X_test.columns:
        try:
            X_test["Sex"] = le.transform(X_test["Sex"])
            print("Applied fitted LabelEncoder to 'Sex' in test data.")
        except ValueError as e:
            print(f"Warning: Could not transform 'Sex' in test data. Error: {e}")
            pass
    elif "Sex" in features_to_use and not le:
        print("Warning: 'Sex' is in features_to_use but LabelEncoder was not fitted.")

    test_preds_log = np.zeros(len(X_test))
    print(f"Predicting with {len(models)} XGBoost models...")
    for model in tqdm(models, desc="Predicting Folds"):
        # XGBoost predict doesn't need specific flags here for standard prediction
        fold_preds_log = model.predict(X_test)
        test_preds_log += fold_preds_log / len(models)

    test_preds = np.expm1(test_preds_log)
    test_preds = np.maximum(0, test_preds)

    submission = pd.DataFrame({"id": ids, "Calories": test_preds})
    submission.to_csv(submission_path, index=False)
    print(f"Saved submission with {len(submission)} rows to {submission_path}")
    print(submission.head())


# --- Optuna Tuning Function for XGBoost ---
def tune_optuna_xgb(  # Renamed function
    raw_train_df: pd.DataFrame,
    feature_engineering_config: dict,
    base_config: Config,  # Pass the XGBoost config structure
    n_trials: int = 50,
):
    """
    Optimize XGBoost hyperparameters with Optuna using cross-validation.
    """
    print("Starting Optuna Hyperparameter Tuning for XGBoost...")

    # --- Determine tree_method outside objective ---
    actual_tree_method = base_config.tree_method
    if actual_tree_method == "gpu_hist":
        try:
            # Quick check if GPU is usable
            temp_df = engineer_features(
                raw_train_df.head(2), **feature_engineering_config
            )
            temp_features = [
                f for f in base_config.__dict__.keys() if f in temp_df.columns
            ]  # Get dummy features
            if not temp_features:
                temp_features = list(temp_df.columns)[1:]  # Avoid empty features
            temp_X = temp_df[temp_features]
            temp_y = np.log1p(pd.Series([1, 1]))
            if "Sex" in temp_X:
                temp_X["Sex"] = 0  # Dummy encode
            xgb.XGBRegressor(tree_method="gpu_hist", n_estimators=1).fit(temp_X, temp_y)
            print("Optuna: GPU detected and accessible by XGBoost. Using 'gpu_hist'.")
        except Exception as e:
            print(
                f"Optuna Warning: Could not initialize XGBoost with GPU ('gpu_hist'). Error: {e}"
            )
            print("Optuna: Falling back to CPU ('hist') for tuning.")
            actual_tree_method = "hist"

    def objective(trial):
        # --- Feature Engineering & Prep (same as before) ---
        current_train_df_eng = engineer_features(
            raw_train_df, **feature_engineering_config
        )
        # Define features dynamically based on config (same logic as in main)
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
        current_features = [
            f for f in current_features if f in current_train_df_eng.columns
        ]

        X = current_train_df_eng[current_features].copy()
        y = current_train_df_eng["Calories"]
        y_log = np.log1p(y)
        trial_le = LabelEncoder()
        if "Sex" in X.columns:
            X["Sex"] = trial_le.fit_transform(X["Sex"])

        # --- Sample XGBoost Hyperparameters ---
        params = {
            "n_estimators": trial.suggest_int(
                "iterations", 500, 5000, step=500
            ),  # Adjusted range maybe
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "max_depth": trial.suggest_int("depth", 3, 10),
            "reg_lambda": trial.suggest_float("l2_leaf_reg", 1e-4, 10.0, log=True),
            "gamma": trial.suggest_float("gamma", 1e-4, 1.0, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "objective": base_config.loss_function,
            "random_state": base_config.random_seed,
            "tree_method": actual_tree_method,  # Use determined method
            "n_jobs": -1,
            # "verbosity": 0, # Keep it quiet
        }

        # --- Cross-validation (same structure, different model) ---
        kf = KFold(
            n_splits=base_config.n_folds,
            shuffle=True,
            random_state=base_config.random_seed,
        )
        fold_scores = []
        for fold, (train_idx, val_idx) in enumerate(kf.split(X, y_log)):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train_log, y_val_log = y_log.iloc[train_idx], y_log.iloc[val_idx]

            model = xgb.XGBRegressor(**params)  # Use XGBoost
            model.fit(
                X_train,
                y_train_log,
                eval_set=[(X_val, y_val_log)],
                eval_metric=base_config.eval_metric,
                early_stopping_rounds=base_config.early_stopping_rounds,
                verbose=False,  # Suppress fitting verbosity
            )

            y_pred_log = model.predict(X_val)
            y_pred = np.expm1(y_pred_log)
            y_val = np.expm1(y_val_log)
            y_pred = np.maximum(0, y_pred)
            fold_rmsle = np.sqrt(mean_squared_log_error(y_val, y_pred))
            fold_scores.append(fold_rmsle)

            trial.report(fold_rmsle, step=fold)
            if trial.should_prune():
                raise optuna.TrialPruned()

        mean_rmsle = np.mean(fold_scores)
        return mean_rmsle

    # --- Run Optuna Study ---
    sampler = TPESampler(seed=base_config.random_seed)
    # Adjust warmup steps if needed, e.g., base_config.n_folds * 2
    pruner = MedianPruner(n_warmup_steps=max(5, base_config.n_folds * (n_trials // 10)))
    study = create_study(direction="minimize", sampler=sampler, pruner=pruner)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    print("\nOptuna Tuning Finished for XGBoost.")
    print("Best CV RMSLE:", study.best_value)
    print("Best params (XGBoost mapping):", study.best_trial.params)

    # Create Config object with best parameters merged with base
    final_config_dict = asdict(base_config)  # Start with base config
    # Update with best tuned params, mapping names back if needed for Config class
    final_config_dict.update(
        {
            "iterations": study.best_trial.params.get(
                "iterations", base_config.iterations
            ),
            "learning_rate": study.best_trial.params.get(
                "learning_rate", base_config.learning_rate
            ),
            "depth": study.best_trial.params.get("depth", base_config.depth),
            "l2_leaf_reg": study.best_trial.params.get(
                "l2_leaf_reg", base_config.l2_leaf_reg
            ),
            "gamma": study.best_trial.params.get("gamma", base_config.gamma),
            "subsample": study.best_trial.params.get(
                "subsample", base_config.subsample
            ),
            "colsample_bytree": study.best_trial.params.get(
                "colsample_bytree", base_config.colsample_bytree
            ),
            "tree_method": actual_tree_method,  # Keep the determined tree method
        }
    )

    return Config(**final_config_dict)


# --- Main Execution Block (Modified for XGBoost) ---
if __name__ == "__main__":
    # --- Configuration ---
    RUN_OPTUNA = True
    N_OPTUNA_TRIALS = 30  # Example trial count
    SUBMISSION_FILENAME = "data/s5_e5/submission_xgb.csv"  # New name
    CONFIG_SAVE_PATH = "data/s5_e5/xgb_best_config.json"  # New name

    # 1. Load Raw Data
    print("Loading data...")
    train_df_raw, test_df_raw = load_data()
    print(f"Loaded Train: {train_df_raw.shape}, Test: {test_df_raw.shape}")

    # 2. Define Feature Engineering Strategy (using simplified from previous step)
    feature_eng_config = {
        "add_bmi": True,
        "use_bmi_only": True,
        "add_log_duration": False,
        "add_temp_elevation": False,
    }
    print("Selected Feature Engineering Strategy:", feature_eng_config)

    # 3. Apply Feature Engineering
    print("Applying feature engineering...")
    train_df_eng = engineer_features(train_df_raw, **feature_eng_config)
    test_df_eng = engineer_features(test_df_raw, **feature_eng_config)
    print(
        f"Engineered Train shape: {train_df_eng.shape}, Test shape: {test_df_eng.shape}"
    )

    # 4. Define Final Feature List (same logic as before)
    features_to_use = ["Sex", "Age", "Duration", "Heart_Rate", "Body_Temp"]
    if feature_eng_config.get("add_bmi"):
        features_to_use.append("BMI")
    features_to_use = [f for f in features_to_use if f in train_df_eng.columns]
    print("Final features for model:", features_to_use)

    # 5. Hyperparameter Tuning (Optional) - Using XGBoost Tuner
    base_config = Config()  # Get default XGBoost config
    if RUN_OPTUNA:
        best_config = tune_optuna_xgb(  # Call XGBoost tuner
            raw_train_df=train_df_raw,
            feature_engineering_config=feature_eng_config,
            base_config=base_config,
            n_trials=N_OPTUNA_TRIALS,
        )
        print(f"Saving best XGBoost config found by Optuna to {CONFIG_SAVE_PATH}")
        with open(CONFIG_SAVE_PATH, "w") as f:
            json.dump(asdict(best_config), f, indent=4)
    else:
        print("Skipping Optuna tuning. Loading or using default XGBoost config.")
        # Try loading previously saved config
        best_config = Config.load(CONFIG_SAVE_PATH)  # Use load method

    # 6. Train Final Model using Cross-Validation - Using XGBoost Trainer
    print("\nTraining final XGBoost model with selected features and configuration...")
    models, label_encoder, oof_score = train_xgboost_cv(  # Call XGBoost trainer
        train_df_engineered=train_df_eng,
        features_to_use=features_to_use,
        config=best_config,
    )
    print(f"Achieved OOF RMSLE with XGBoost: {oof_score:.5f}")

    # 7. Make Submission (Function is compatible)
    print("\nGenerating submission file...")
    make_submission(
        models=models,
        le=label_encoder,
        test_df_engineered=test_df_eng,
        features_to_use=features_to_use,
        submission_path=SUBMISSION_FILENAME,
    )

    print("\nScript finished.")
