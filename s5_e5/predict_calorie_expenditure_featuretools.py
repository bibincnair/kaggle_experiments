# --- IMPORTS ---
# Add featuretools import
import featuretools as ft
import xgboost as xgb  # Or import lightgbm as lgb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
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
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings(
    "ignore", category=FutureWarning
)  # Featuretools can sometimes raise these


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


# --- NEW: Featuretools DFS Function ---
def run_featuretools_dfs(
    df, entity_id="exercises", index_col="id", target_entity="exercises"
):
    """
    Applies Deep Feature Synthesis using Featuretools on a single dataframe.
    Focuses on transformation primitives as there are no relationships.
    """
    print(f"\n--- Running Featuretools DFS on entity '{entity_id}' ---")
    df_ft = df.copy()

    # Ensure index_col is unique for make_index=False
    if not df_ft[index_col].is_unique:
        print(
            f"Warning: Index column '{index_col}' is not unique. Resetting index for Featuretools."
        )
        df_ft.reset_index(drop=True, inplace=True)
        # If we reset, we can't use the original 'id' as the FT index easily unless it matches the new range.
        # Let's use make_index=True instead for simplicity if id isn't unique or not the actual index.
        # However, the provided data likely has unique IDs. We assume 'id' is unique for now.
        # If 'id' IS the index already, skip make_index.
        # If 'id' is a COLUMN and unique, use it as index parameter.

    # Create an EntitySet
    es = ft.EntitySet(id="calorie_data")

    # Add the dataframe as an entity.
    # Use 'id' column as the index for the entity.
    # Specify numeric/categorical types if needed (often inferred well)
    # Make sure all intended features are numeric or categorical first.
    # Example: Convert Sex before adding if needed (though LabelEncoder handles it later)
    numeric_cols = df_ft.select_dtypes(include=np.number).columns.tolist()
    variable_types = {
        col: ft.variable_types.Numeric for col in numeric_cols if col != index_col
    }
    # Add other types if needed, e.g., categorical

    es = es.add_dataframe(
        dataframe_name=entity_id,
        dataframe=df_ft,
        index=index_col,  # Specify the column to use as the unique identifier
        # make_index=True, # Use this if you DON'T have a unique ID column already
        variable_types=variable_types,
    )

    print(f"EntitySet created with entity '{entity_id}'.")
    print(f"Available features for DFS: {list(df_ft.columns)}")

    # Define primitives - Focus on transformations for a single table
    # Can add more from ft.list_primitives()
    trans_primitives = [
        "add_numeric",
        "subtract_numeric",
        "multiply_numeric",
        "divide_numeric",
        "absolute",
        # "percentile", # Can be useful
        # Add others as needed, avoid time-based ones if no timestamp
    ]

    # Run Deep Feature Synthesis
    # max_depth=1 is sufficient for transformation primitives on a single table
    print(f"Running DFS with primitives: {trans_primitives}")
    try:
        feature_matrix, feature_defs = ft.dfs(
            entityset=es,
            target_dataframe_name=target_entity,
            agg_primitives=[],  # No aggregations without relationships
            trans_primitives=trans_primitives,
            max_depth=1,
            verbose=1,
            n_jobs=1,  # Can increase if needed, but might use lots of memory
        )
        print(f"DFS completed. Generated {feature_matrix.shape[1]} features.")
        # print("Generated feature names:", feature_defs) # Can be very long
    except Exception as e:
        print(f"Error during Featuretools DFS: {e}")
        print("Returning original dataframe.")
        return df  # Return original df on error

    # Featuretools might change dtypes, e.g., int to float, ensure consistency if needed
    # feature_matrix = feature_matrix.astype({col: df_ft[col].dtype for col in df.columns if col in feature_matrix.columns})

    return feature_matrix


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


# --- Main Execution Block (Modified to include Featuretools) ---
if __name__ == "__main__":
    # --- Configuration ---
    RUN_OPTUNA = False  # Keep False initially to test FT integration
    N_OPTUNA_TRIALS = 30
    RUN_FEATURETOOLS = True  # Flag to control Featuretools step
    SUBMISSION_FILENAME = "data/s5_e5/submission_xgb_ft.csv"  # New name
    CONFIG_SAVE_PATH = "data/s5_e5/xgb_best_config_ft.json"  # New name

    # 1. Load Raw Data
    print("Loading data...")
    train_df_raw, test_df_raw = load_data()
    print(f"Loaded Train: {train_df_raw.shape}, Test: {test_df_raw.shape}")

    # 2. Define Manual Feature Engineering Strategy (Optional - run before FT)
    feature_eng_config = {
        "add_bmi": True,
        "use_bmi_only": True,
        "add_log_duration": False,
        "add_temp_elevation": False,
    }
    print("Selected Manual Feature Engineering Strategy:", feature_eng_config)
    train_df_manual_eng = engineer_features(train_df_raw, **feature_eng_config)
    test_df_manual_eng = engineer_features(test_df_raw, **feature_eng_config)
    print(
        f"After Manual Eng - Train: {train_df_manual_eng.shape}, Test: {test_df_manual_eng.shape}"
    )

    # --- 3. Apply Featuretools DFS (Conditional) ---
    if RUN_FEATURETOOLS:
        # Important: Apply DFS to both train and test using the same primitives
        # Need to ensure consistent columns before applying DFS
        train_cols_before_ft = set(train_df_manual_eng.columns)
        test_cols_before_ft = set(test_df_manual_eng.columns)
        common_cols = list(train_cols_before_ft.intersection(test_cols_before_ft))
        target_col = "Calories"  # Assuming target is only in train before FT
        if target_col in common_cols:
            common_cols.remove(target_col)  # Don't use target in test

        # Run DFS on Training Data
        train_df_ft = run_featuretools_dfs(
            train_df_manual_eng, entity_id="exercises_train", index_col="id"
        )

        # Run DFS on Test Data (using only common columns from train before FT)
        # Ensure test data has the 'id' column needed for indexing
        test_df_ft = run_featuretools_dfs(
            test_df_manual_eng[common_cols + ["id"]],
            entity_id="exercises_test",
            index_col="id",
        )

        # Align columns after DFS - crucial! Test might not generate all features if data distribution differs.
        train_cols_after_ft = set(train_df_ft.columns)
        test_cols_after_ft = set(test_df_ft.columns)

        # Features to use are those generated in train (excluding target)
        features_generated = list(train_cols_after_ft)
        if target_col in features_generated:
            features_generated.remove(target_col)
        if "id" in features_generated:
            features_generated.remove("id")  # Don't use ID as feature

        # Ensure test set has all columns from train, fill missing with 0 or median/mean if appropriate
        print("Aligning columns after DFS...")
        missing_in_test = list(set(features_generated) - test_cols_after_ft)
        for col in missing_in_test:
            print(f"Adding missing column '{col}' to test set (filling with 0).")
            test_df_ft[col] = 0

        missing_in_train = list(
            test_cols_after_ft - set(features_generated) - {"id"}
        )  # Should ideally be empty
        if missing_in_train:
            print(
                f"Warning: Columns in test but not train after FT: {missing_in_train}. Dropping from test."
            )
            test_df_ft = test_df_ft.drop(columns=missing_in_train)

        # Ensure order is the same
        test_df_ft = test_df_ft[
            train_df_ft[features_generated].columns
        ]  # Match train column order (excluding target/id)

        train_df_final = train_df_ft
        test_df_final = test_df_ft
        features_to_use = features_generated  # Use all features generated by FT (and original ones it kept)

    else:
        print("Skipping Featuretools DFS.")
        train_df_final = train_df_manual_eng
        test_df_final = test_df_manual_eng
        # Define features based on manual engineering only (as in previous step)
        features_to_use = ["Sex", "Age", "Duration", "Heart_Rate", "Body_Temp"]
        if feature_eng_config.get("add_bmi"):
            features_to_use.append("BMI")
        features_to_use = [f for f in features_to_use if f in train_df_final.columns]

    print(f"\nFinal features count for model: {len(features_to_use)}")
    # print("Final features for model:", features_to_use) # Can be very long list

    # 4. Hyperparameter Tuning (Optional - run on the final feature set)
    base_config = Config()
    if RUN_OPTUNA and RUN_FEATURETOOLS:
        print("\nRunning Optuna on data with Featuretools features...")
        # We need to pass the *final* engineered data to Optuna if FT was run
        # Modifying tune_optuna_xgb to accept pre-engineered data might be simpler here
        # Or, ensure Optuna objective replicates *both* manual + FT steps (more complex)
        # Let's assume for now we tune *after* deciding whether to use FT features.
        # If RUN_OPTUNA is True, run it on train_df_final (which includes FT features if RUN_FEATURETOOLS was True)
        # This requires modifying tune_optuna_xgb slightly to accept engineered data directly, or adjusting its internal logic.
        # For simplicity, we'll skip Optuna re-run in this example if FT is enabled,
        # assuming we'd use a config tuned previously or the default.
        print(
            "Optuna tuning with Featuretools features enabled requires modification to tune_optuna function - Skipping tuning for now."
        )
        best_config = base_config  # Use default or load existing
        # best_config = Config.load(CONFIG_SAVE_PATH) # Load if previously saved
    elif RUN_OPTUNA and not RUN_FEATURETOOLS:
        # Run Optuna as before using tune_optuna_xgb which does manual eng internally
        best_config = tune_optuna_xgb(
            raw_train_df=train_df_raw,  # Pass raw for Optuna's internal FE
            feature_engineering_config=feature_eng_config,
            base_config=base_config,
            n_trials=N_OPTUNA_TRIALS,
        )
        print(
            f"Saving best XGBoost config (no FT) found by Optuna to {CONFIG_SAVE_PATH.replace('_ft','')}"
        )
        with open(CONFIG_SAVE_PATH.replace("_ft", ""), "w") as f:
            json.dump(asdict(best_config), f, indent=4)

    else:
        print("Skipping Optuna tuning.")
        best_config = Config.load(
            CONFIG_SAVE_PATH
            if RUN_FEATURETOOLS
            else CONFIG_SAVE_PATH.replace("_ft", "")
        )

    # 5. Train Final Model using Cross-Validation
    print("\nTraining final XGBoost model...")
    # Pass the final dataframes (potentially with FT features)
    models, label_encoder, oof_score = train_xgboost_cv(
        train_df_engineered=train_df_final,  # Use final train df
        features_to_use=features_to_use,  # Use final feature list
        config=best_config,
    )
    print(f"Achieved OOF RMSLE: {oof_score:.5f}")

    # 6. Make Submission
    print("\nGenerating submission file...")
    # Pass the final test dataframe
    make_submission(
        models=models,
        le=label_encoder,
        test_df_engineered=test_df_final,  # Use final test df
        features_to_use=features_to_use,  # Use final feature list
        submission_path=SUBMISSION_FILENAME,
    )

    print("\nScript finished.")
