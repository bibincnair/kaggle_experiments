from datetime import datetime
import json
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import shap
import optuna
from catboost import CatBoostClassifier, Pool, cv, CatBoostError
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score
from sklearn.impute import KNNImputer
from clearml import Task
from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional, Any
from pathlib import Path
import logging

# Configure logging to save logs to a file in logs/ directory
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("logs/s4_e11_catboost_predict_mental_health.log"),
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
    model_path: str = "models/s4_e11_catboost_model.cbm"
    params_path: str = "models/s4_e11_catboost_best_params.json"
    data_path: str = "data/s4_e11/"


class DataProcessor:
    """Handles data loading and preprocessing"""

    def __init__(self, config: Config):
        self.config = config
        self.categorical_features = None

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

        # Identify categorical features
        self.categorical_features = X_train.select_dtypes(
            include=["object"]
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
        """Handle missing values using KNN imputation for numerical features"""
        logging.info("Handling missing values")
        numerical_features = X_train.select_dtypes(include=["float64", "int64"]).columns

        # KNN imputation for numerical features
        imputer = KNNImputer(n_neighbors=5)
        X_train[numerical_features] = imputer.fit_transform(X_train[numerical_features])
        X_test[numerical_features] = imputer.transform(X_test[numerical_features])

        # Mode imputation for categorical features
        for col in self.categorical_features:
            mode_value = X_train[col].mode()[0]
            X_train[col] = X_train[col].fillna(mode_value)
            X_test[col] = X_test[col].fillna(mode_value)

        return X_train, X_test


class ModelTrainer:
    """Handles model training and hyperparameter optimization"""

    def __init__(self, config: Config):
        self.config = config
        self.task = Task.init(
            project_name="s4e11", task_name="catboost_predict_mental_health"
        )

    def _optimize_hyperparameters(
        self, X_train: pd.DataFrame, y_train: pd.Series, categorical_features: List[str]
    ) -> Dict:
        """Optimize hyperparameters using Optuna"""
        logging.info("Optimizing hyperparameters")

        def objective(trial):
            params = {
                # Core parameters
                "iterations": trial.suggest_int("iterations", 100, 3000),
                "learning_rate": trial.suggest_float(
                    "learning_rate", 1e-4, 0.1, log=True
                ),
                "depth": trial.suggest_int("depth", 4, 10),  # As recommended in guide
                # Regularization
                "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-8, 10, log=True),
                "random_strength": trial.suggest_float(
                    "random_strength", 1e-8, 10, log=True
                ),
                # Tree growing policy
                "grow_policy": trial.suggest_categorical(
                    "grow_policy", ["SymmetricTree", "Depthwise", "Lossguide"]
                ),
                # Border count (as per guide)
                "border_count": trial.suggest_categorical("border_count", [64, 1024]),
                # Bootstrapping
                "bootstrap_type": trial.suggest_categorical(
                    "bootstrap_type", ["Bayesian", "Bernoulli", "MVS"]
                ),
                "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 1, 1000),
                "colsample_bylevel": trial.suggest_float(
                    "colsample_bylevel", 0.05, 1.0
                ),
                # Fixed parameters
                "loss_function": "Logloss",
                "eval_metric": "AUC",
                "verbose": False,
                "auto_class_weights": trial.suggest_categorical(
                    "auto_class_weights", ["None", "Balanced", "SqrtBalanced"]
                ),
                "random_seed": self.config.random_seed,
            }

            # Conditional parameters based on bootstrap_type
            if params["bootstrap_type"] == "Bayesian":
                params["bagging_temperature"] = trial.suggest_float(
                    "bagging_temperature", 0.0, 10.0
                )
            elif params["bootstrap_type"] == "Bernoulli":
                params["subsample"] = trial.suggest_float("subsample", 0.1, 1.0)

            # Conditional parameters for grow_policy
            if params["grow_policy"] == "Lossguide":
                params["max_leaves"] = trial.suggest_int("max_leaves", 8, 64)

            try:
                train_pool = Pool(
                    data=X_train, label=y_train, cat_features=categorical_features
                )

                cv_results = cv(
                    pool=train_pool,
                    params=params,
                    fold_count=self.config.cv_folds,
                    stratified=True,
                    shuffle=True,
                    early_stopping_rounds=100,
                    seed=self.config.random_seed,
                    logging_level="Verbose",
                )

                return np.mean(cv_results["test-AUC-mean"])
            except Exception as e:
                raise optuna.TrialPruned()

        study = optuna.create_study(
            direction="maximize", pruner=optuna.pruners.MedianPruner(n_startup_trials=5)
        )
        with open(Config().params_path, "r") as f:
            best_params = json.load(f)  
        study.enqueue_trial(best_params)
        study.optimize(objective, n_trials=self.config.n_trials, n_jobs=4)

        return study.best_params

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        categorical_features: Optional[List[str]] = None,
    ) -> CatBoostClassifier:
        """Train the model with either default or optimized parameters"""
        logging.info("Training the model")
        if self.config.use_processed_data:
            with open(self.config.params_path, "r") as f:
                params = json.load(f)
        else:
            params = self._optimize_hyperparameters(
                X_train, y_train, categorical_features
            )
            # Save best parameters
            with open(self.config.params_path, "w") as f:
                json.dump(params, f)

        model = CatBoostClassifier(**params)
        logging.info(f"Training with parameters: {params}")
        logging.info(f"Categorical features: {categorical_features}")
        model.fit(X_train, y_train, cat_features=categorical_features, verbose=True)

        return model

    def validate(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        categorical_features: List[str],
        params: Dict,
    ) -> Tuple[float, CatBoostClassifier]:
        """Load the best parameters and validate the model by training on the entire dataset and cv"""
        
        logging.info("Validating the model")
        # config does not have "loss_function": "Logloss" and "eval_metric": "AUC", add them
        params["loss_function"] = "Logloss"
        params["eval_metric"] = "AUC"
        
        model = CatBoostClassifier(**params)
        model.fit(X_train, y_train, cat_features=categorical_features, verbose=True)
        mean_auc = cv(
            Pool(data=X_train, label=y_train, cat_features=categorical_features),
            params=params,
            fold_count=self.config.cv_folds,
            stratified=True,
            shuffle=True,
            # early_stopping_rounds=100,
            seed=self.config.random_seed,
            logging_level="Silent",
        )["test-AUC-mean"].mean()
        # Append current mean auc value to logs/s4_e11_catboost_final_cv_score.log in format <date> <time> - INFO - Mean AUC: <mean_auc>
        with open("logs/s4_e11_catboost_final_cv_score.log", "a") as f:
            f.write("\n")
            date_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"{date_time} - INFO - Mean AUC: {mean_auc}\n")
            f.write("\n")
        return mean_auc, model


def main():
    config = Config()
    data_processor = DataProcessor(config)
    model_trainer = ModelTrainer(config)

    # Load data
    if config.use_processed_data:
        X_train, y_train, X_test, test_id = data_processor.load_processed_data()
        categorical_features = None  # Already processed
    else:
        X_train, y_train, X_test, test_id = data_processor.load_raw_data()
        categorical_features = data_processor.categorical_features

    if config.run_validation:
        # Validate model
        mean_auc, model = model_trainer.validate(
            X_train, y_train, categorical_features, json.load(open(config.params_path))
        )
        logging.info(f"Mean AUC: {mean_auc}")
    else:
        # Train model
        model = model_trainer.train(X_train, y_train, categorical_features)

    # Generate predictions
    y_pred = model.predict(X_test)

    # Create submission
    submission = pd.DataFrame({"id": test_id, "Depression": y_pred.astype(int)})
    submission.to_csv("submission.csv", index=False)


if __name__ == "__main__":
    main()
