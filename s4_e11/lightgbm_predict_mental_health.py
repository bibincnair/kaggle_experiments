import io
import json
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import shap
import optuna
import lightgbm as lgb
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import LabelEncoder


class MentalHealthClassifier:
    def __init__(self):
        self.model = None
        self.best_params = None
        self.imputer = None
        self.encoder = None
        self.shap_values = None

    def load_data(self):
        """Load data from data/s4_e11 folder, train.csv and test.csv"""
        train_data = pd.read_csv("data/s4_e11/train.csv")
        test_data = pd.read_csv("data/s4_e11/test.csv")
        X_train = train_data.drop("Depression", axis=1)
        y_train = train_data["Depression"]
        X_test = test_data
        return X_train, y_train, X_test

    def load_original_data(self):
        """Load original data from data/s4_e11 folder"""
        original_data = pd.read_csv("data/s4_e11/original.csv")
        X_original = original_data.drop("Depression", axis=1)
        y_original = original_data["Depression"]
        return X_original, y_original
    
    def load_processed_data(self):
        """Load processed data from data/s4_e11 folder"""
        X_train = pd.read_pickle("data/s4_e11/X_train.pkl")
        y_train = pd.read_pickle("data/s4_e11/y_train.pkl")
        X_test = pd.read_pickle("data/s4_e11/X_test.pkl")
        return X_train, y_train, X_test

    # Preprocessing transformers (same as in the original code)
    class CategoricalProcessor(BaseEstimator, TransformerMixin):
        def __init__(self):
            # Define mappings for ordinal encoding
            self.dietary_habits_mapping = {"Unhealthy": 1, "Moderate": 2, "Healthy": 3}
            self.sleep_duration_mapping = {
                "Less than 5 hours": 1,
                "5-6 hours": 2,
                "7-8 hours": 3,
                "More than 8 hours": 4,
            }
            # Columns to process
            self.one_hot_columns = ["City", "Profession", "Degree"]
            self.ordinal_columns = ["Dietary Habits", "Sleep Duration"]
            self.binary_columns = [
                "Gender",
                "Family History of Mental Illness",
                "Have you ever had suicidal thoughts ?",
            ]
            # To store modes and categories
            self.mode_values = {}
            self.one_hot_categories = {}

        def fit(self, X, y=None):
            # Compute mode for each column for imputation
            for col in (
                self.one_hot_columns + self.ordinal_columns + self.binary_columns
            ):
                self.mode_values[col] = X[col].mode()[0]
            # Collect categories for one-hot encoding
            for col in self.one_hot_columns:
                self.one_hot_categories[col] = (
                    X[col].fillna(self.mode_values[col]).unique().tolist()
                )
            return self

        def transform(self, X):
            X = X.copy()
            # Impute missing values with mode
            for col in (
                self.one_hot_columns + self.ordinal_columns + self.binary_columns
            ):
                X[col] = X[col].fillna(self.mode_values[col])
            # Ordinal encoding for 'Dietary Habits'
            X["Dietary Habits"] = X["Dietary Habits"].map(self.dietary_habits_mapping)
            # Ordinal encoding for 'Sleep Duration'
            X["Sleep Duration"] = X["Sleep Duration"].map(self.sleep_duration_mapping)
            # Binary encoding for binary columns
            for col in self.binary_columns:
                unique_values = sorted(X[col].unique())
                mapping = {value: idx for idx, value in enumerate(unique_values)}
                X[col] = X[col].map(mapping)
            # One-hot encoding for 'City', 'Profession', and 'Degree'
            dummies = pd.get_dummies(
                X[self.one_hot_columns], prefix=self.one_hot_columns
            )
            X = X.drop(columns=self.one_hot_columns)
            X = pd.concat([X, dummies], axis=1)
            return X

    class RareSampleRemover(BaseEstimator, TransformerMixin):
        def __init__(self, threshold=100):
            self.threshold = threshold
            self.frequent_categories_ = {}

        def fit(self, X):
            X = pd.DataFrame(X)
            for col in X.columns:
                counts = X[col].value_counts()
                self.frequent_categories_[col] = counts[
                    counts >= self.threshold
                ].index.tolist()
            return self

        def transform(self, X):
            X = pd.DataFrame(X)
            for col in X.columns:
                frequent = self.frequent_categories_[col]
                X[col] = X[col].where(X[col].isin(frequent), np.nan)
            return X

    class FeatureEngineerStudentProfession(BaseEstimator, TransformerMixin):
        """Feature engineering based on 'Working Professional or Student' feature"""

        def fit(self, X, y=None):
            return self

        def transform(self, X):
            X = X.copy()
            # For Students
            student_mask = X["Working Professional or Student"] == "Student"
            X.loc[student_mask, "Work Pressure"] = X.loc[
                student_mask, "Work Pressure"
            ].fillna(0)
            X.loc[student_mask, "Job Satisfaction"] = X.loc[
                student_mask, "Job Satisfaction"
            ].fillna(0)
            X.loc[student_mask, "Profession"] = X.loc[
                student_mask, "Profession"
            ].fillna("Student")
            # For Working Professionals
            professional_mask = (
                X["Working Professional or Student"] == "Working Professional"
            )
            X.loc[professional_mask, "Profession"] = X.loc[
                professional_mask, "Profession"
            ].fillna("Others")
            X.loc[professional_mask, "CGPA"] = X.loc[professional_mask, "CGPA"].fillna(
                0
            )
            X.loc[professional_mask, "Study Satisfaction"] = X.loc[
                professional_mask, "Study Satisfaction"
            ].fillna(0)
            X.loc[professional_mask, "Academic Pressure"] = X.loc[
                professional_mask, "Academic Pressure"
            ].fillna(0)
            # Drop 'Working Professional or Student' column
            X = X.drop("Working Professional or Student", axis=1)
            return X

    def impute_using_knn(self, X_train, X_test):
        """Impute NaN values using KNN imputer"""
        features = [
            "Study Satisfaction",
            "Academic Pressure",
            "Work/Study Hours",
            "CGPA",
        ]
        X_subset = X_train[features]
        imputer = KNNImputer(n_neighbors=5)
        X_imputed = imputer.fit_transform(X_subset)
        X_train[features] = X_imputed

        # Impute test data
        X_test_subset = X_test[features]
        X_test_imputed = imputer.transform(X_test_subset)
        X_test[features] = X_test_imputed
        self.imputer = imputer  # Save the imputer for later use
        return X_train, X_test

    def preprocess_categorical_columns(self, X_train, X_test):
        processor = self.CategoricalProcessor()
        processor.fit(X_train)
        X_train_processed = processor.transform(X_train)
        X_test_processed = processor.transform(X_test)
        return X_train_processed, X_test_processed

    def preprocess_numerical_columns(self, X_train, X_test):
        numerical_cols = X_train.select_dtypes(include=["number"]).columns
        imputer = SimpleImputer(strategy="median")
        X_train[numerical_cols] = imputer.fit_transform(X_train[numerical_cols])
        X_test[numerical_cols] = imputer.transform(X_test[numerical_cols])
        return X_train, X_test
    
    def save_processed_data(self, X_train, y_train, X_test, x_original, y_original, train_id, test_id):
        """Save processed data to data/s4_e11 folder"""
        X_train.to_pickle("data/s4_e11/processed_data/X_train.pkl")
        y_train.to_pickle("data/s4_e11/processed_data/y_train.pkl")
        X_test.to_pickle("data/s4_e11/processed_data/X_test.pkl")
                

    def preprocess_data(self, X_train, y_train, X_test, X_original, y_original):
        """Preprocess data"""
        # Save train and test IDs
        train_id = pd.DataFrame(X_train["id"])
        test_id = pd.DataFrame(X_test["id"])
        # Drop 'Name' and 'id' columns
        X_train = X_train.drop(["Name", "id"], axis=1)
        X_test = X_test.drop(["Name", "id"], axis=1)
        
        # If original data is provided, merge it with the training data after removing 'Name' and 'id' columns
        if X_original is not None:
            X_original = X_original.drop(["Name"], axis=1)
            # if y_original is categorical, encode it using LabelEncoder
            # print count of unique values in y_train and y_original
            if y_original.dtype == "object":
                encoder = LabelEncoder()
                # No as 0 and Yes as 1
                y_original = pd.DataFrame(encoder.fit_transform(y_original), columns=["Depression"])
            
            print(y_train.value_counts())
            print(y_original.value_counts())
            X_train = pd.concat([X_train, X_original], axis=0)
            y_train = pd.concat([y_train, y_original], axis=0)

        rare_sample_remover = self.RareSampleRemover()
        X_train = rare_sample_remover.fit_transform(X_train)
        X_test = rare_sample_remover.transform(X_test)

        feature_engineer = self.FeatureEngineerStudentProfession()
        X_train = feature_engineer.fit_transform(X_train)
        X_test = feature_engineer.transform(X_test)

        # Impute NaN using KNN imputer
        X_train, X_test = self.impute_using_knn(X_train, X_test)

        # Preprocess categorical columns
        X_train, X_test = self.preprocess_categorical_columns(X_train, X_test)

        # Preprocess numerical columns
        X_train, X_test = self.preprocess_numerical_columns(X_train, X_test)
        
        # Print count of instances of "No" in X_train and y_train
        print("Count of 'No' instances in X_train and y_train:")
        print(X_train[X_train == "No"].count())
        print(y_train[y_train == "No"].count())
        
        # pickle processed data to data/s4_e11 folder
        X_train.to_pickle("data/s4_e11/processed_data/X_train.pkl")
        y_train.to_pickle("data/s4_e11/processed_data/y_train.pkl")
        X_test.to_pickle("data/s4_e11/processed_data/X_test.pkl")
        # Save ids
        train_id.to_pickle("data/s4_e11/processed_data/train_id.pkl")
        test_id.to_pickle("data/s4_e11/processed_data/test_id.pkl")
        return X_train, y_train, X_test

    def train_model(self, X_train, y_train, params=None, fold_count=5):
        """Train LightGBM model with cross-validation"""
        if params is None:
            params = self.best_params if self.best_params else {}

        params.update({
            "objective": "binary",
            "metric": "auc",
            "verbose": -1,
            # "early_stopping_round": 50,
        })

        dataset = lgb.Dataset(X_train, label=y_train)
        cv_results = lgb.cv(
            params,
            dataset,
            stratified=True,
            nfold=fold_count,
            num_boost_round=1000,
            shuffle=True,
            seed=42,
            return_cvbooster=False,
        )

        mean_auc = cv_results["valid auc-mean"][-1]
        std_auc = cv_results["valid auc-stdv"][-1]
        print(f"CV AUC: {mean_auc:.4f} ± {std_auc:.4f}")
        
        # Check if early stopping is enabled, if so provide validation set and eval metric
        if "early_stopping_round" in params:
            # Train final model on full dataset
            self.model = lgb.LGBMClassifier(**params)
            X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.2, stratify=y_train, random_state=42)
            
            self.model.fit(
                X_tr, y_tr,
                eval_set=[(X_val, y_val)],
                eval_metric="auc",
            )
        else:
            self.model = lgb.LGBMClassifier(**params)
            self.model.fit(X_train, y_train)
        return self.model
    
    def evaluate_model(self, X_train, y_train):
        """Evaluate model performance with cross-validation and metrics"""
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        auc_scores = []

        for train_idx, val_idx in cv.split(X_train, y_train):
            X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
            y_pred = self.model.predict_proba(X_val)[:, 1]
            auc = roc_auc_score(y_val, y_pred)
            auc_scores.append(auc)

        metrics = {
            "auc_mean": np.mean(auc_scores),
            "auc_std": np.std(auc_scores),
        }
        return metrics

    def optimize_hyperparameters(self, X_train, y_train, n_trials=100):
        """Optimize hyperparameters using Optuna for LightGBM"""
        initial_params = {
            'learning_rate': 0.032832498290837335,
            'n_estimators': 1938,
            'num_leaves': 39,
            'max_depth': 14,
            'min_child_samples': 77,
            'subsample': 0.7891781484238849,
            'colsample_bytree': 0.7229807857262475,
            'reg_alpha': 3.755034962683313e-05,
            'reg_lambda': 8.62382802975515e-06,
            'min_split_gain': 4.598158157234433e-06,
            'early_stopping_round': 60
        }

        def objective(trial):
            param_grid = {
                "learning_rate": trial.suggest_float(
                    "learning_rate", 1e-3, 0.1, log=True
                ),
                "n_estimators": trial.suggest_int("n_estimators", 100, 5000),
                "num_leaves": trial.suggest_int("num_leaves", 20, 1500),
                "max_depth": trial.suggest_int("max_depth", 3, 15),
                "max_bin": trial.suggest_int("max_bin", 20, 500),
                "bagging_freq": trial.suggest_int("bagging_freq", 1, 10),
                "bagging_fraction": trial.suggest_float("bagging_fraction", 0.2, 1.0),
                "feature_fraction": trial.suggest_float("feature_fraction", 0.2, 1.0),
                "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
                "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 20, 15000),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
                "min_split_gain": trial.suggest_float(
                    "min_split_gain", 1e-8, 1.0, log=True
                ),
                "early_stopping_round": trial.suggest_int("early_stopping_round", 20, 100),
                "is_unbalance": True,
                # "num_boost_round": trial.suggest_int("num_boost_round", 100, 2000),
                "random_state": 42,
                "objective": "binary",
                "metric": "auc",
                # "verbose": -1,
            }
            #pruning_callback = optuna.integration.LightGBMPruningCallback(trial, "auc")
            cv_results = lgb.cv(
                param_grid,
                lgb.Dataset(X_train, label=y_train),
                stratified=True,
                nfold=7,
              #  callbacks=[pruning_callback],
            )
            # best_num_rounds = len(cv_results["valid auc-mean"])
            # mean_auc = cv_results["valid auc-mean"][-1]
            # std_auc = cv_results["valid auc-stdv"][-1]

            return cv_results["valid auc-mean"][-1]

        study = optuna.create_study(
            direction="maximize", pruner=optuna.pruners.MedianPruner()
        )
        if initial_params:
            study.enqueue_trial(initial_params)
        study.optimize(objective, n_trials=n_trials)

        print("\nBest trial:")
        print(f"  Value: {study.best_trial.value:.4f}")
        print(f"  Params: {study.best_trial.params}")

        self.best_params = study.best_trial.params
        return self.best_params

    def plot_feature_importance(self, X_train):
        """Plot SHAP-based feature importance"""
        explainer = shap.TreeExplainer(self.model)
        shap_values = explainer.shap_values(X_train)

        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, X_train, plot_type="bar", show=False)
        plt.tight_layout()
        plt.savefig("plots/s4_e11_lightgbm_feature_importance.png")
        plt.close()

        self.shap_values = shap_values
        return shap_values

    def plot_feature_interaction(self, X_train, feature1=None, feature2=None):
        """Plot SHAP interaction values between features"""
        if self.shap_values is None:
            explainer = shap.TreeExplainer(self.model)
            self.shap_values = explainer.shap_values(X_train)

        if feature1 is None:
            # Get top 2 important features
            feature_importance = np.abs(self.shap_values).mean(0)
            top_features = X_train.columns[np.argsort(-feature_importance)][:2]
            feature1, feature2 = top_features

        plt.figure(figsize=(10, 6))
        shap.dependence_plot(
            feature1, self.shap_values, X_train, interaction_index=feature2, show=False
        )
        plt.tight_layout()
        plt.savefig(f"plots/s4_e11_lightgbm_interaction_{feature1}_{feature2}.png")
        plt.close()

    def plot_prediction_explanation(self, X_train, sample_idx=0):
        """Plot SHAP force plot for prediction explanation"""
        explainer = shap.TreeExplainer(self.model)
        shap_values = explainer.shap_values(X_train.iloc[sample_idx : sample_idx + 1])

        plt.figure(figsize=(15, 3))
        shap.force_plot(
            explainer.expected_value[1],
            shap_values[1],
            X_train.iloc[sample_idx],
            matplotlib=True,
            show=False,
        )
        plt.tight_layout()
        plt.savefig(
            "plots/s4_e11_lightgbm_prediction_explanation.png",
            bbox_inches="tight",
            dpi=300,
        )
        plt.close()

    def create_submission_file(self, X_test, test_id, filename="submission.csv"):
        """Create submission file for test data"""
        # Predict on test data
        y_pred = self.model.predict(X_test).astype(int)
        submission = pd.DataFrame({"id": test_id, "Depression": y_pred})
        submission.to_csv(filename, index=False)

    # Additional methods like EDA can be added similarly

    def run_experiment(self):
        X_train, y_train, X_test = self.load_data()
        test_id = X_test["id"]
        
        # Load original data
        X_original, y_original = self.load_original_data()
        
        # Preprocess data
        X_train, y_train, X_test = self.preprocess_data(X_train, y_train, X_test, X_original, y_original)
        print("Finished preprocessing data.")
        return
        # Train model
        self.train_model(X_train, y_train)
        print("Finished training model.")
        # Evaluate model
        metrics = self.evaluate_model(X_train, y_train)
        print(f"Model evaluation metrics:\n{metrics}")
        # Create submission file
        self.create_submission_file(X_test, test_id, filename="submission_default.csv")
        print("Finished creating submission file.")
        # Optimize hyperparameters
        self.optimize_hyperparameters(X_train, y_train)
        print("Finished optimizing hyperparameters.")
        # Train model with best hyperparameters
        self.train_model(X_train, y_train, params=self.best_params)
        print("Finished training best model.")
        # Evaluate best model
        best_metrics = self.evaluate_model(X_train, y_train)
        print(f"Best model evaluation metrics:\n{best_metrics}")
        # Plot feature importance
        # self.plot_feature_importance(X_train)
        # # Plot feature interaction
        # self.plot_feature_interaction(X_train)
        # # Plot prediction explanation
        # self.plot_prediction_explanation(X_train)
        # Create submission file with best model
        self.create_submission_file(X_test, test_id, filename="submission_best.csv")
        # Save best model and parameters
        self.model.booster_.save_model("models/s4_e11_lightgbm_model.txt")
        with open("models/s4_e11_lightgbm_best_params.json", "w") as f:
            json.dump(self.best_params, f)


if __name__ == "__main__":
    classifier = MentalHealthClassifier()
    classifier.run_experiment()
