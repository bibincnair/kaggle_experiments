# Catboot based classification for mental health prediction
# 1. Use SparseGroupKFold for cross validation
# 2. Use Catboost for classification
# 3. Use Optuna for hyperparameter optimization
# 4. Use SHAP for feature importance
# 5. Use SHAP for feature interaction
# 6. Use SHAP for prediction explanation
# 7. Seaborn for EDA and visualization saved to plots folder with prefix s4_e11_catboost.

import io
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import shap
import optuna
from catboost import CatBoostClassifier, Pool, cv
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.experimental import enable_iterative_imputer


def load_data():
    """Load data from data/s4_e11 folder, train.csv and test.csv
    """
    train_data = pd.read_csv("data/s4_e11/train.csv")
    test_data = pd.read_csv("data/s4_e11/test.csv")
    X_train = train_data.drop("Depression", axis=1)
    y_train = train_data["Depression"]
    X_test = test_data
    return X_train, y_train, X_test

def load_original_data():
    """Load original data from data/s4_e11 folder
    """
    original_data = pd.read_csv("data/s4_e11/original.csv")
    X_original = original_data.drop("Depression", axis=1)
    y_original = original_data["Depression"]
    return X_original, y_original

class CategoricalProcessor(BaseEstimator, TransformerMixin):
    def __init__(self):
        # Define mappings for ordinal encoding
        self.dietary_habits_mapping = {'Unhealthy': 1, 'Moderate': 2, 'Healthy': 3}
        self.sleep_duration_mapping = {
            'Less than 5 hours': 1,
            '5-6 hours': 2,
            '7-8 hours': 3,
            'More than 8 hours': 4
        }
        # Columns to process
        self.one_hot_columns = ['City', 'Profession', 'Degree']
        self.ordinal_columns = ['Dietary Habits', 'Sleep Duration']
        self.binary_columns = [
            'Gender',
            'Family History of Mental Illness',
            'Have you ever had suicidal thoughts ?'
        ]
        # To store modes and categories
        self.mode_values = {}
        self.one_hot_categories = {}

    def fit(self, X, y=None):
        # Compute mode for each column for imputation
        for col in self.one_hot_columns + self.ordinal_columns + self.binary_columns:
            self.mode_values[col] = X[col].mode()[0]
        # Collect categories for one-hot encoding
        for col in self.one_hot_columns:
            self.one_hot_categories[col] = X[col].fillna(self.mode_values[col]).unique().tolist()
        return self

    def transform(self, X):
        X = X.copy()
        # Impute missing values with mode
        for col in self.one_hot_columns + self.ordinal_columns + self.binary_columns:
            X[col] = X[col].fillna(self.mode_values[col])
        # Ordinal encoding for 'Dietary Habits'
        X['Dietary Habits'] = X['Dietary Habits'].map(self.dietary_habits_mapping)
        # Ordinal encoding for 'Sleep Duration'
        X['Sleep Duration'] = X['Sleep Duration'].map(self.sleep_duration_mapping)
        # Binary encoding for binary columns
        for col in self.binary_columns:
            unique_values = sorted(X[col].unique())
            mapping = {value: idx for idx, value in enumerate(unique_values)}
            X[col] = X[col].map(mapping)
        # One-hot encoding for 'City', 'Profession', and 'Degree'
        dummies = pd.get_dummies(X[self.one_hot_columns], prefix=self.one_hot_columns)
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
            self.frequent_categories_[col] = counts[counts >= self.threshold].index.tolist()
        return self

    def transform(self, X):
        X = pd.DataFrame(X)
        for col in X.columns:
            frequent = self.frequent_categories_[col]
            X[col] = X[col].where(X[col].isin(frequent), np.nan)
        return X

class FeatureEngineerStudentProfession(BaseEstimator, TransformerMixin):
    """Based on "Working Professional or Student" feature, update NaN values:
    1. If feature is "Student", update NaN in "Work Pressure" to 0
    2. If feature is "Student", update NaN in "Job Satisfaction" to 0
    3. If feature is "Student", update NaN in "Profession" to "Student"
    4. If feature is "Working Professional", update NaN in "CGPA" to 0
    5. If feature is "Working Professional", update NaN in "Study Satisfaction" to 0
    6. If feature is "Working Professional", update NaN in "Academic Pressure" to 0
    """
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        # For Students
        student_mask = X["Working Professional or Student"] == "Student"
        X.loc[student_mask, "Work Pressure"] = X.loc[student_mask, "Work Pressure"].fillna(0)
        X.loc[student_mask, "Job Satisfaction"] = X.loc[student_mask, "Job Satisfaction"].fillna(0)
        X.loc[student_mask, "Profession"] = X.loc[student_mask, "Profession"].fillna("Student")
        # For Working Professionals
        professional_mask = X["Working Professional or Student"] == "Working Professional"
        X.loc[professional_mask, "Profession"] = X.loc[professional_mask, "Profession"].fillna("Others")
        X.loc[professional_mask, "CGPA"] = X.loc[professional_mask, "CGPA"].fillna(0)
        X.loc[professional_mask, "Study Satisfaction"] = X.loc[professional_mask, "Study Satisfaction"].fillna(0)
        X.loc[professional_mask, "Academic Pressure"] = X.loc[professional_mask, "Academic Pressure"].fillna(0)
        # drop "Working Professional or Student" column
        X = X.drop("Working Professional or Student", axis=1)
        return X
    
def print_work_student_profession_nan(X, y=None):
    """
    Print NaN coun in Work perssure, if "Student" in "Working Professional or Student"
    Print NaN coun in Job Satisfaction, if "Student" in "Working Professional or Student"
    Print NaN coun in Profession, if "Student" in "Working Professional or Student"
    Print NaN coun in CGPA, if "Working Professional" in "Working Professional or Student"
    Print NaN coun in Study Satisfaction, if "Working Professional" in "Working Professional or Student"
    Print NaN coun in Academic Pressure, if "Working Professional" in "Working Professional or Student"
    """
    print(f"Work Pressure NaN count: {X.loc[X['Working Professional or Student'] == 'Student']['Work Pressure'].isna().sum()}")
    print(f"Job Satisfaction NaN count: {X.loc[X['Working Professional or Student'] == 'Student']['Job Satisfaction'].isna().sum()}")
    print(f"Profession NaN count: {X.loc[X['Working Professional or Student'] == 'Student']['Profession'].isna().sum()}")
    print(f"CGPA NaN count: {X.loc[X['Working Professional or Student'] == 'Working Professional']['CGPA'].isna().sum()}")
    print(f"Study Satisfaction NaN count: {X.loc[X['Working Professional or Student'] == 'Working Professional']['Study Satisfaction'].isna().sum()}")
    print(f"Academic Pressure NaN count: {X.loc[X['Working Professional or Student'] == 'Working Professional']['Academic Pressure'].isna().sum()}")
    
    # Print NaN count in CGPA where y is 1
    print(f"CGPA NaN count where y is 1: {X.loc[y == 1]['CGPA'].isna().sum()}")
    
    
def impute_using_knn(X_train, X_test):
    """Impute NaN values using KNN imputer
    """
    features = ["Study Satisfaction", "Academic Pressure", "Work/Study Hours", "CGPA"]
    X_subset = X_train[features]
    imputer = KNNImputer(n_neighbors=5)
    X_imputed = imputer.fit_transform(X_subset)
    X_train[features] = X_imputed
    
    # Impute test data
    X_test_subset = X_test[features]
    X_test_imputed = imputer.transform(X_test_subset)
    X_test[features] = X_test_imputed
    return X_train, X_test, imputer

def preprocess_categorical_columns(X_train, X_test):
    processor = CategoricalProcessor()
    processor.fit(X_train)
    X_train_processed = processor.transform(X_train)
    X_test_processed = processor.transform(X_test)
    return X_train_processed, X_test_processed

def preprocess_numerical_columns(X_train, X_test):
    numerical_cols = X_train.select_dtypes(include=['number']).columns
    imputer = SimpleImputer(strategy='median')
    X_train[numerical_cols] = imputer.fit_transform(X_train[numerical_cols])
    X_test[numerical_cols] = imputer.transform(X_test[numerical_cols])
    return X_train, X_test

def preprocess_data(X_train, y_train, X_test):
    """Preprocess data
    """
    print(f"Nan count before transformation:\n{X_train.isna().sum()}")
    # Drop Name column
    X_train = X_train.drop("Name", axis=1)
    X_test = X_test.drop("Name", axis=1)
    # Drop ID column
    X_train = X_train.drop("id", axis=1)
    X_test = X_test.drop("id", axis=1)
    
    rare_sample_remover = RareSampleRemover()
    shape_before = X_train.shape
    X_train = rare_sample_remover.fit_transform(X_train)
    X_test = rare_sample_remover.transform(X_test)
    shape_after = X_train
    # print(f"################Describe rare sample remover#######################")
    # print(X_train.describe().to_string())
    # print(f"Shape before: {shape_before}, shape after: {shape_after}")
    feature_engineer_student_profession = FeatureEngineerStudentProfession()
    X_train = feature_engineer_student_profession.fit_transform(X_train)
    X_test = feature_engineer_student_profession.transform(X_test)
    # print(f"Shape before: {shape_before}, shape after: {shape_after}")
    # Print NaN count after transformation
    # print(f"NaN count after transformation:\n{X_train.isna().sum()}")
    # # describe(X_train)
    # print(f"################Describe feature eng#######################")
    # print(X_train.describe().to_string())
    # Impute NaN using KNN imputer
    X_train, X_test, imputer = impute_using_knn(X_train, X_test)
    # Preprocess categorical columns
    X_train, X_test = preprocess_categorical_columns(X_train, X_test)
    # Preprocess numerical columns
    X_train, X_test = preprocess_numerical_columns(X_train, X_test)
    # Fill NaN with m
    return X_train, y_train, X_test


def train_model(X_train, y_train, params=None, fold_count=5):
    """Train CatBoost model using CatBoostCV
    
    Args:
        X_train: Training features 
        y_train: Training targets
        params: Model parameters (optional)
        fold_count: Number of CV folds
        
    Returns:
        final_model: Trained model on full dataset
        cv_results: Cross validation results
    """
    if params is None:
        params = {
            'iterations': 1000,
            'learning_rate': 0.1,
            'depth': 6,
            'loss_function': 'Logloss',
            'eval_metric': 'AUC',
            'random_seed': 42,
            'early_stopping_rounds': 50
        }
    
    # Get categorical feature indices
    cat_features = [i for i, col in enumerate(X_train.columns) 
                   if X_train[col].dtype == 'object']
    
    # Create CatBoost pool with categorical features
    train_pool = Pool(X_train, y_train, cat_features=cat_features)
    
    # Perform stratified cross-validation
    cv_results = cv(
        pool=train_pool,
        params=params,
        fold_count=fold_count,
        stratified=True,
        shuffle=True,
        seed=42,
        verbose=100
    )
    
    # Log CV results
    mean_auc = np.mean(cv_results['test-AUC-mean'])
    std_auc = np.std(cv_results['test-AUC-mean'])
    print(f'CV AUC: {mean_auc:.4f} ± {std_auc:.4f}')
    
    # Train final model on full dataset
    final_model = CatBoostClassifier(**params)
    final_model.fit(
        X_train, y_train,
        cat_features=cat_features,
        verbose=100
    )
    
    return final_model, cv_results

def evaluate_model(X_train, y_train, model):
    """Evaluate model performance with cross-validation and metrics"""
    # Get categorical features
    cat_features = [i for i, col in enumerate(X_train.columns) 
                   if X_train[col].dtype == 'object']
    
    # Create train pool
    train_pool = Pool(X_train, y_train, cat_features=cat_features)
    
    # Perform CV evaluation
    cv_results = cv(
        pool=train_pool,
        params=model.get_params(),
        fold_count=5,
        stratified=True,
        shuffle=True,
        seed=42
    )
    
    # Calculate metrics
    metrics = {
        'auc_mean': np.mean(cv_results['test-AUC-mean']),
        'auc_std': np.std(cv_results['test-AUC-mean']),
        'logloss_mean': np.mean(cv_results['test-Logloss-mean']),
        'logloss_std': np.std(cv_results['test-Logloss-mean'])
    }
    
    return metrics, cv_results

def optimize_hyperparameters(X_train, y_train, n_trials=100):
    """Optimize hyperparameters using Optuna with proper categorical feature handling"""
    # Get categorical feature indices
    cat_features = [i for i, col in enumerate(X_train.columns) 
                   if X_train[col].dtype == 'object']
    
    def objective(trial):
        params = {
            'iterations': trial.suggest_int('iterations', 100, 1000),
            'learning_rate': trial.suggest_float('learning_rate', 1e-3, 1e-1, log=True),
            'depth': trial.suggest_int('depth', 4, 10),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-8, 1.0, log=True),
            'bootstrap_type': trial.suggest_categorical('bootstrap_type', ['Bayesian', 'Bernoulli', 'MVS']),
            'random_strength': trial.suggest_float('random_strength', 1e-8, 10.0),
            'bagging_temperature': trial.suggest_float('bagging_temperature', 0.01, 100.0, log=True),
            'od_type': 'Iter',
            'od_wait': trial.suggest_int('od_wait', 10, 50),
            'loss_function': 'Logloss',
            'eval_metric': 'AUC',
            'verbose': False
        }
        
        # Create pool with categorical features specified
        train_pool = Pool(
            data=X_train,
            label=y_train,
            cat_features=cat_features
        )
        
        cv_results = cv(
            pool=train_pool,
            params=params,
            fold_count=5,
            stratified=True,
            shuffle=True,
            seed=42
        )
        
        return np.mean(cv_results['test-AUC-mean'])
    
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials)
    
    return study.best_params, study.best_value

def plot_feature_importance(X_train, y_train, model):
    """Plot SHAP-based feature importance"""
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_train)
    
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_train, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig('plots/s4_e11_catboost_feature_importance.png')
    plt.close()
    
    return shap_values

def plot_feature_interaction(X_train, y_train, model, feature1=None, feature2=None):
    """Plot SHAP interaction values between features"""
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_train)
    
    if feature1 is None:
        # Get top 2 important features
        feature_importance = np.abs(shap_values).mean(0)
        top_features = X_train.columns[np.argsort(-feature_importance)][:2]
        feature1, feature2 = top_features
    
    plt.figure(figsize=(10, 6))
    shap.dependence_plot(
        feature1, shap_values, X_train,
        interaction_index=feature2,
        show=False
    )
    plt.tight_layout()
    plt.savefig(f'plots/s4_e11_catboost_interaction_{feature1}_{feature2}.png')
    plt.close()

def plot_prediction_explanation(X_train, y_train, model, sample_idx=0):
    """Plot SHAP force plot for prediction explanation"""
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_train.iloc[sample_idx:sample_idx+1])
    
    plt.figure(figsize=(15, 3))
    shap.force_plot(
        explainer.expected_value,
        shap_values[0],
        X_train.iloc[sample_idx],
        matplotlib=True,
        show=False
    )
    plt.tight_layout()
    plt.savefig('plots/s4_e11_catboost_prediction_explanation.png', 
                bbox_inches='tight', dpi=300)
    plt.close()

def create_submission_file(X_test, model):
    with open("submission.csv", "w") as f:
        f.write("id,Depression\n")
        for idx, row in X_test.iterrows():
            features = row.values
            prediction = model.predict(features)
            f.write(f"{idx},{prediction}\n")

def compare_train_test_categorical_features(X_train, X_test):
    """Compare categorical features in train and test data for difference in unique values

    Args:
        X_train (_type_): _description_
        X_test (_type_): _description_
    """
    cat_columns_train = X_train.select_dtypes(include=["object"]).columns
    with open("logs/s4_e11_catboost/s4_e11_catboost_categorical_features.log", "w") as f:
        f.write("\nTrain categorical features:\n")
        for col in cat_columns_train:
            A = X_train[col].astype(str).unique()
            B = X_test[col].astype(str).unique()
            C = np.setdiff1d(B, A)
            f.write(f"Col {col} has categories in train but not in test: {C}\n")
            if len(C) > 0:
                f.write(f" => {len(X_test.loc[X_test[col].isin(C)])} rows in test data\n")
            f.write("================================================================\n")


def eda(X_train, y_train):
    """Do exploratory data analysis using seaborn
       resulting plots are saved to plots/s4_e11_catboost folder with prefix s4_e11_catboost_
       logs are saved to logs/s4_e11_catboost folder with prefix s4_e11_catboost_
       """
    
    #1. Data overview
    with open("logs/s4_e11_catboost/s4_e11_catboost_data_overview.log", "w") as f:
        f.write(X_train.head().to_string())
        f.write("\n")
        f.write("Info:\n")
        buf = io.StringIO()
        X_train.info(buf=buf)
        f.write(buf.getvalue())
        f.write("\n")
        f.write("Describe:\n")
        f.write(X_train.describe().to_string())
        f.write("\n")
        f.write("Number of NaN per column:\n")
        f.write(X_train.isna().sum().to_string())
        f.write("\n")
        f.write("Value counts:\n")
        f.write(y_train.value_counts().to_string())
        f.write("\n")
        f.write("Describe y_train:\n")
        f.write(y_train.describe().to_string())
        f.write("\n")
        # write categorical and numerical columns
        f.write("Categorical columns:\n")
        cat_columns = X_train.select_dtypes(include=["object"]).columns
        f.write("Categorical columns:\n")
        f.write(", ".join(cat_columns))
        f.write("\n")
        f.write("Numerical columns:\n")
        num_columns = X_train.select_dtypes(exclude=["object"]).columns
        f.write(", ".join(num_columns))
        f.write("\n")
        # Print unique values for each column and count per unique value
        for col in X_train.columns:
            if col == "id":
                continue
            f.write(f"Unique values for {col}:\n")
            f.write(X_train[col].value_counts().to_string())
            f.write("\n")
        
    #2. Data distribution
    for col in X_train.columns:
        # if column names are too long, take first 10 characters
        plt.figure()
        sns.histplot(X_train[col])
        # Remove any special characters from column name and append name
        col_name = "".join(e for e in col if e.isalnum())
        plt.savefig(f"plots/s4_e11_catboost/s4_e11_catboost_{col_name}_histplot.png")
        plt.close()


def main():
    X_train, y_train, X_test = load_data()
    X_original, y_original = load_original_data()
    # eda(X_original, y_original)
    print_work_student_profession_nan(X_train, y=y_train)
    compare_train_test_categorical_features(X_train, X_test)
    X_train, y_train, X_test = preprocess_data(X_train, y_train, X_test)
    # Print NaN in train and test data
    print("NaN count in train data:")
    print(X_train.isna().sum())
    print("NaN count in test data:")
    print(y_train.isna().sum())
    # Print data types
    print(X_train.dtypes)
    # Print categorical columns
    cat_columns = X_train.select_dtypes(include=["object"]).columns
    print(f"Categorical columns: {cat_columns}")
    
    
    # eda(X_train, y_train)
    # Train model
    model, cv_results = train_model(X_train, y_train)
    # Evaluate model
    metrics, cv_results = evaluate_model(X_train, y_train, model)
    print(f"Model evaluation metrics:\n{metrics}")
    # Optimize hyperparameters
    best_params, best_value = optimize_hyperparameters(X_train, y_train)
    print(f"Best hyperparameters: {best_params}")
    print(f"Best AUC value: {best_value}")
    # Train model with best hyperparameters
    best_model, _ = train_model(X_train, y_train, params=best_params)
    # Evaluate best model
    best_metrics, _ = evaluate_model(X_train, y_train, best_model)
    print(f"Best model evaluation metrics:\n{best_metrics}")
    # Plot feature importance
    shap_values = plot_feature_importance(X_train, y_train, best_model)
    # Plot feature interaction
    plot_feature_interaction(X_train, y_train, best_model)
    # Plot prediction explanation
    plot_prediction_explanation(X_train, y_train, best_model)
    # Create submission file
    create_submission_file(X_test, best_model)
    # Save best model and best params to models folder
    best_model.save_model("models/s4_e11_catboost_model.cbm")
    import json
    with open("models/s4_e11_catboost_best_params.json", "w") as f:
        json.dump(best_params, f)


if __name__ == "__main__":
    main()
