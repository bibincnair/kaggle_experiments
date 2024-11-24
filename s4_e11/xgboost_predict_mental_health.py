from datetime import datetime
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple, List, Optional

import numpy as np
import pandas as pd
import optuna
import xgboost as xgb
from sklearn.impute import KNNImputer
from clearml import Task

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("logs/s4_e11_xgboost_predict_mental_health.log"),
        logging.StreamHandler(),
    ],
)


@dataclass
class Config:
    """Configuration for the model training pipeline"""

    run_validation: bool = False
    use_processed_data: bool = False
    n_trials: int = 200
    cv_folds: int = 5
    random_seed: int = 42
    model_path: str = "models/s4_e11_xgboost_model.json"
    params_path: str = "models/s4_e11_xgboost_best_params.json"
    data_path: str = "data/s4_e11/"
    early_stopping_rounds: int = 100


class DataProcessor:
    """Handles data loading and preprocessing"""

    def __init__(self, config: Config):
        self.config = config
        self.categorical_features = None
        self.numeric_features = None

    def load_processed_data(
        self,
    ) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
        """Load preprocessed data"""
        logging.info("Loading processed data")
        X_train = pd.read_pickle(f"{self.config.data_path}/processed_data/X_train.pkl")
        y_train = pd.read_pickle(f"{self.config.data_path}/processed_data/y_train.pkl")
        X_test = pd.read_pickle(f"{self.config.data_path}/processed_data/X_test.pkl")
        test_id = pd.read_pickle(f"{self.config.data_path}/processed_data/test_id.pkl")
        return X_train, y_train, X_test, test_id

    def load_raw_data(self) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
        """Load and preprocess raw data"""
        logging.info("Loading raw data")
        train_data = pd.read_csv(f"{self.config.data_path}/train.csv")
        test_data = pd.read_csv(f"{self.config.data_path}/test.csv")

        # Extract IDs and target
        test_id = test_data["id"].copy()
        X_train = train_data.drop(["Depression", "id", "Name"], axis=1)
        y_train = train_data["Depression"]
        X_test = test_data.drop(["id", "Name"], axis=1)

        # Handle student/professional logic
        X_train, X_test = self._handle_student_professional(X_train, X_test)

        # Identify features
        self.categorical_features = X_train.select_dtypes(
            include=["object"]
        ).columns.tolist()
        self.numeric_features = X_train.select_dtypes(
            include=["float64", "int64"]
        ).columns.tolist()

        # Handle missing values
        X_train, X_test = self._handle_missing_values(X_train, X_test)

        return X_train, y_train, X_test, test_id

    def _handle_student_professional(
        self, X_train: pd.DataFrame, X_test: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Handle student/professional specific logic"""
        logging.info("Handling student/professional specific logic")
        for df in [X_train, X_test]:
            student_mask = df["Working Professional or Student"] == "Student"
            professional_mask = (
                df["Working Professional or Student"] == "Working Professional"
            )

            # Handle student records
            df.loc[student_mask, ["Work Pressure", "Job Satisfaction"]] = df.loc[
                student_mask, ["Work Pressure", "Job Satisfaction"]
            ].fillna(0)
            df.loc[student_mask, "Profession"] = df.loc[
                student_mask, "Profession"
            ].fillna("Student")

            # Handle professional records
            df.loc[
                professional_mask, ["CGPA", "Study Satisfaction", "Academic Pressure"]
            ] = df.loc[
                professional_mask, ["CGPA", "Study Satisfaction", "Academic Pressure"]
            ].fillna(
                0
            )
            df.loc[professional_mask, "Profession"] = df.loc[
                professional_mask, "Profession"
            ].fillna("Others")

        return X_train.drop("Working Professional or Student", axis=1), X_test.drop(
            "Working Professional or Student", axis=1
        )

    def _handle_missing_values(
        self, X_train: pd.DataFrame, X_test: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Handle missing values using KNN imputation for numerical features and mode for categorical"""
        logging.info("Handling missing values")

        # KNN imputation for numerical features
        imputer = KNNImputer(n_neighbors=5)
        X_train[self.numeric_features] = imputer.fit_transform(
            X_train[self.numeric_features]
        )
        X_test[self.numeric_features] = imputer.transform(X_test[self.numeric_features])

        # Mode imputation for categorical features
        for col in self.categorical_features:
            mode_value = X_train[col].mode()[0]
            X_train[col] = X_train[col].fillna(mode_value)
            X_test[col] = X_test[col].fillna(mode_value)

            # Convert to categorical type for XGBoost
            X_train[col] = X_train[col].astype("category")
            X_test[col] = X_test[col].astype("category")

        return X_train, X_test


class ModelTrainer:
    """Handles model training and hyperparameter optimization"""

    def __init__(self, config: Config):
        self.config = config
        self.task = Task.init(
            project_name="s4e11", task_name="xgboost_predict_mental_health"
        )

    def _optimize_hyperparameters(
        self, X_train: pd.DataFrame, y_train: pd.Series
    ) -> Dict:
        """Optimize hyperparameters using Optuna"""
        logging.info("Optimizing hyperparameters")

        def objective(trial):
            params = {
                "objective": "binary:logistic",
                "eval_metric": ["auc", "aucpr"],
                "learning_rate": trial.suggest_float(
                    "learning_rate", 1e-4, 0.1, log=True
                ),
                "max_depth": trial.suggest_int("max_depth", 4, 10),
                "subsample": trial.suggest_float("subsample", 0.1, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.05, 1.0),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 1000),
                "gamma": trial.suggest_float("gamma", 1e-8, 10.0, log=True),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
                "scale_pos_weight": sum(y_train == 0) / sum(y_train == 1),
                "tree_method": "hist",
                "random_state": self.config.random_seed,
                "enable_categorical": True,
            }

            try:
                cv_results = xgb.cv(
                    params,
                    xgb.DMatrix(X_train, label=y_train, enable_categorical=True),
                    num_boost_round=3000,
                    early_stopping_rounds=self.config.early_stopping_rounds,
                    nfold=self.config.cv_folds,
                    stratified=True,
                    shuffle=True,
                    seed=self.config.random_seed,
                    verbose_eval=False,
                )

                mean_auc = cv_results["test-auc-mean"].max()
                mean_aucpr = cv_results["test-aucpr-mean"].max()
                return 0.4 * mean_auc + 0.6 * mean_aucpr

            except Exception as e:
                raise optuna.TrialPruned()

        study = optuna.create_study(
            direction="maximize", pruner=optuna.pruners.MedianPruner(n_startup_trials=5)
        )

        # Load and enqueue previous best parameters if available
        try:
            with open(self.config.params_path, "r") as f:
                best_params = json.load(f)
                study.enqueue_trial(best_params)
        except FileNotFoundError:
            pass

        study.optimize(objective, n_trials=self.config.n_trials, n_jobs=4)

        best_params = study.best_params
        # Add number of rounds from early stopping
        cv_results = xgb.cv(
            best_params,
            xgb.DMatrix(X_train, label=y_train, enable_categorical=True),
            num_boost_round=3000,
            early_stopping_rounds=self.config.early_stopping_rounds,
            nfold=self.config.cv_folds,
            stratified=True,
        )
        best_params["n_estimators"] = len(cv_results)

        return best_params

    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> xgb.Booster:
        """Train the model with either default or optimized parameters"""
        logging.info("Training the model")

        if self.config.use_processed_data:
            with open(self.config.params_path, "r") as f:
                params = json.load(f)
        else:
            params = self._optimize_hyperparameters(X_train, y_train)
            with open(self.config.params_path, "w") as f:
                json.dump(params, f)

        dtrain = xgb.DMatrix(X_train, label=y_train, enable_categorical=True)
        logging.info(f"Training with parameters: {params}")
        model = xgb.train(params, dtrain, num_boost_round=params["n_estimators"])

        return model

    def validate(
        self, X_train: pd.DataFrame, y_train: pd.Series, params: Dict
    ) -> Tuple[float, xgb.Booster]:
        """Validate the model using cross-validation"""
        logging.info("Validating the model")

        dtrain = xgb.DMatrix(X_train, label=y_train, enable_categorical=True)
        cv_results = xgb.cv(
            params,
            dtrain,
            num_boost_round=params["n_estimators"],
            nfold=self.config.cv_folds,
            stratified=True,
            shuffle=True,
            seed=self.config.random_seed,
        )

        mean_auc = cv_results["test-auc-mean"].mean()

        with open("logs/s4_e11_xgboost_final_cv_score.log", "a") as f:
            f.write("\n")
            date_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"{date_time} - INFO - Mean AUC: {mean_auc}\n")
            f.write("\n")

        model = xgb.train(params, dtrain, num_boost_round=params["n_estimators"])
        return mean_auc, model


def main():
    config = Config()
    data_processor = DataProcessor(config)
    model_trainer = ModelTrainer(config)

    if config.use_processed_data:
        X_train, y_train, X_test, test_id = data_processor.load_processed_data()
    else:
        X_train, y_train, X_test, test_id = data_processor.load_raw_data()

    if config.run_validation:
        mean_auc, model = model_trainer.validate(
            X_train, y_train, json.load(open(config.params_path))
        )
        logging.info(f"Mean AUC: {mean_auc}")
    else:
        model = model_trainer.train(X_train, y_train)
        model.save_model(config.model_path)

    dtest = xgb.DMatrix(X_test, enable_categorical=True)
    y_pred = model.predict(dtest)
    y_pred_binary = (y_pred > 0.5).astype(int)

    Path("submissions/s4_e11").mkdir(parents=True, exist_ok=True)
    submission = pd.DataFrame({"id": test_id, "Depression": y_pred_binary})
    submission.to_csv("submissions/s4_e11/submission.csv", index=False)


if __name__ == "__main__":
    main()
