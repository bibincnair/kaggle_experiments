import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    classification_report,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
    auc,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import optuna
import optuna.visualization
import xgboost as xgb
from functools import partial


class SleepDurationTransformer(BaseEstimator, TransformerMixin):
    """Transform sleep duration categories into numerical features."""

    def __init__(self):
        self.sleep_mapping = {
            "Less than 5 hours": 4.0,
            "7-8 hours": 7.5,
            "More than 8 hours": 9.0,
            "5-6 hours": 5.5,
            "3-4 hours": 3.5,
            "6-7 hours": 6.5,
            "4-5 hours": 4.5,
            "2-3 hours": 2.5,
            "4-6 hours": 5.0,
            "6-8 hours": 7.0,
            "1-6 hours": 3.5,
        }
        self.median_sleep = None

    def fit(self, X, y=None):
        # Handle different input types
        if isinstance(X, np.ndarray):
            X = X.reshape(-1)
        elif isinstance(X, pd.DataFrame):
            X = X.iloc[:, 0]

        sleep_hours = pd.Series(X).map(self.sleep_mapping)
        self.median_sleep = sleep_hours.median()
        return self

    def transform(self, X):
        # Handle different input types
        if isinstance(X, np.ndarray):
            X = X.reshape(-1)
        elif isinstance(X, pd.DataFrame):
            X = X.iloc[:, 0]

        sleep_hours = pd.Series(X).map(self.sleep_mapping).fillna(self.median_sleep)

        return np.vstack(
            [
                sleep_hours,
                sleep_hours.between(7, 9).astype(int),
                (sleep_hours < 6).astype(int),
                (sleep_hours > 9).astype(int),
            ]
        ).T


class StatusBasedImputer(BaseEstimator, TransformerMixin):
    """Impute values based on professional/student status."""

    def __init__(self):
        self.columns = [
            "Working Professional or Student",
            "Work Pressure",
            "Job Satisfaction",
            "Academic Pressure",
            "CGPA",
            "Study Satisfaction",
        ]

    def fit(self, X, y=None):
        return self

    def transform(self, X_orig):
        # Convert to DataFrame if necessary
        if isinstance(X_orig, np.ndarray):
            X_orig = pd.DataFrame(X_orig, columns=self.columns)

        X = X_orig.copy()
        mask_student = X["Working Professional or Student"] == "Student"
        mask_working = X["Working Professional or Student"] == "Working Professional"

        print(f"Mask count for students: {mask_student.sum()}")
        print(f"Mask count for working professionals: {mask_working.sum()}")

        # Set default values for work-related fields
        X.loc[mask_student, ["Work Pressure", "Job Satisfaction"]] = 0

        # Set default values for study-related fields
        X.loc[mask_working, ["Academic Pressure", "CGPA", "Study Satisfaction"]] = 0

        return X.drop("Working Professional or Student", axis=1).values


def remove_rare_samples(df, min_count=100):
    """Remove samples with rare categorical values."""
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns
    mask = pd.Series(True, index=df.index)

    for col in categorical_cols:
        counts = df[col].value_counts()
        valid_categories = counts[counts >= min_count].index
        mask &= df[col].isin(valid_categories)

    return df[mask].reset_index(drop=True), mask


def create_preprocessing_pipeline(X):
    """Create preprocessing pipeline with proper column handling."""
    # Get column groups
    numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

    # Remove special columns
    exclude_cols = {
        "Sleep Duration",
        "Depression",
        "id",
        "Name",
        "Working Professional or Student",
    }

    numeric_cols = [col for col in numeric_cols if col not in exclude_cols]
    categorical_cols = [col for col in categorical_cols if col not in exclude_cols]

    # Create status columns list
    status_cols = [
        "Working Professional or Student",
        "Work Pressure",
        "Job Satisfaction",
        "Academic Pressure",
        "CGPA",
        "Study Satisfaction",
    ]

    # Exclude status columns from other groups
    numeric_cols = [col for col in numeric_cols if col not in status_cols]
    categorical_cols = [col for col in categorical_cols if col not in status_cols]

    # Create transformers
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), numeric_cols),
            (
                "categorical",
                Pipeline(
                    [
                        (
                            "imputer",
                            SimpleImputer(strategy="constant", fill_value="Other"),
                        ),
                        (
                            "onehot",
                            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                        ),
                    ]
                ),
                categorical_cols,
            ),
            ("sleep", SleepDurationTransformer(), ["Sleep Duration"]),
            ("status", StatusBasedImputer(), status_cols),
        ],
        remainder="drop",
    )

    return preprocessor, numeric_cols, categorical_cols


def load_data(train_path, test_path):
    """Load and prepare the dataset."""
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)

    y_train = train_data["Depression"]
    X_train = train_data.drop(["Depression"], axis=1)
    # X_test = test_data.drop(['id', 'Name'], axis=1)
    # print unique values in the target column
    print("\nUnique values in the target column:")
    print(y_train.value_counts())
    return X_train, y_train, test_data


def validate_and_transform_data(X_train, y_train, X_test=None, min_count=100):
    """Validate and transform the data."""
    # Validate required columns
    required_columns = {
        "Gender",
        "Age",
        "City",
        "Working Professional or Student",
        "Sleep Duration",
    }
    missing_cols = required_columns - set(X_train.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Remove rare samples
    X_train_filtered, train_mask = remove_rare_samples(X_train, min_count)
    y_train_filtered = y_train[train_mask].reset_index(drop=True)

    # Create preprocessing pipeline
    preprocessor, numeric_cols, categorical_cols = create_preprocessing_pipeline(
        X_train_filtered
    )

    # Transform training data
    X_train_transformed = preprocessor.fit_transform(X_train_filtered)
    # Generate feature names
    feature_names = (
        numeric_cols
        + list(
            preprocessor.named_transformers_["categorical"]
            .named_steps["onehot"]
            .get_feature_names_out(categorical_cols)
        )
        + ["sleep_hours", "is_recommended_sleep", "is_short_sleep", "is_long_sleep"]
        + [
            "Work Pressure",
            "Job Satisfaction",
            "Academic Pressure",
            "CGPA",
            "Study Satisfaction",
        ]
    )

    # Convert to DataFrame
    X_train_transformed = pd.DataFrame(X_train_transformed, columns=feature_names)
    print(f"Transformed training data +++++++++++++++++++++++++")
    if X_test is not None:
        X_test_transformed = preprocessor.transform(X_test)
        X_test_transformed = pd.DataFrame(X_test_transformed, columns=feature_names)
        return X_train_transformed, y_train_filtered, X_test_transformed, preprocessor

    return X_train_transformed, y_train_filtered, preprocessor


def optimize_xgboost_with_optuna(X_train, y_train, n_trials=100, use_previous_best=True):
    """
    Optimize XGBoost hyperparameters using Optuna for imbalanced binary classification.
    
    Args:
        X_train: Training features
        y_train: Training labels
        n_trials: Number of optimization trials
        use_previous_best: Whether to use known best parameters as starting point
    """
    # Define known best parameters
    BEST_PARAMS = {
        'learning_rate': 0.053586264512840626,
        'n_estimators': 684,
        'max_depth': 5,
        'min_child_weight': 3.5819104187316104,
        'subsample': 0.7176744310587686,
        'colsample_bytree': 0.34278124480479555,
        'colsample_bylevel': 0.8519453146528222,
        'colsample_bynode': 0.9677853695964237,
        'reg_alpha': 5.204618698507569,
        'reg_lambda': 2.74786117067469
    }
    
    def suggest_params(trial, previous_best=None):
        """Define parameter search space with optimal bounds."""
        params = {}
        
        # Learning rate: Allow 50% variation around previous best
        if previous_best:
            lr_low = previous_best['learning_rate'] * 0.5
            lr_high = previous_best['learning_rate'] * 1.5
        else:
            lr_low, lr_high = 1e-3, 0.3
        params['learning_rate'] = trial.suggest_float('learning_rate', lr_low, lr_high, log=True)
        
        # Number of estimators: Allow 20% variation
        if previous_best:
            n_est_low = max(100, int(previous_best['n_estimators'] * 0.8))
            n_est_high = min(1000, int(previous_best['n_estimators'] * 1.2))
        else:
            n_est_low, n_est_high = 100, 1000
        params['n_estimators'] = trial.suggest_int('n_estimators', n_est_low, n_est_high)
        
        # Max depth: Plus/minus 1 from previous best
        if previous_best:
            depth_low = max(3, previous_best['max_depth'] - 1)
            depth_high = min(10, previous_best['max_depth'] + 1)
        else:
            depth_low, depth_high = 3, 10
        params['max_depth'] = trial.suggest_int('max_depth', depth_low, depth_high)
        
        # Min child weight: Allow 20% variation
        if previous_best:
            mcw_low = max(1, previous_best['min_child_weight'] * 0.8)
            mcw_high = previous_best['min_child_weight'] * 1.2
        else:
            mcw_low, mcw_high = 1, 10
        params['min_child_weight'] = trial.suggest_float('min_child_weight', mcw_low, mcw_high)
        
        # Sampling parameters: Handle 0-1 bounded parameters carefully
        for param_name in ['subsample', 'colsample_bytree', 'colsample_bylevel', 'colsample_bynode']:
            if previous_best:
                param_low = max(0.3, min(1.0, previous_best[param_name] * 0.8))
                param_high = min(1.0, previous_best[param_name] * 1.2)
            else:
                param_low, param_high = 0.3, 1.0
            params[param_name] = trial.suggest_float(param_name, param_low, param_high)
        
        # Regularization parameters: Allow 50% variation
        for param_name in ['reg_alpha', 'reg_lambda']:
            if previous_best:
                reg_low = previous_best[param_name] * 0.5
                reg_high = previous_best[param_name] * 1.5
            else:
                reg_low, reg_high = 1e-8, 10.0
            params[param_name] = trial.suggest_float(param_name, reg_low, reg_high, log=True)
        
        return params

    def objective(trial, X_train, y_train, previous_best=None):
        # Get suggested parameters
        params = suggest_params(trial, previous_best)
        
        # Add fixed parameters
        params.update({
            'objective': 'binary:logistic',
            'tree_method': 'hist',
            'random_state': 42,
            'eval_metric': ['aucpr', 'auc'],
            'scale_pos_weight': sum(y_train == 0) / sum(y_train == 1)
        })
        
        # Cross-validation setup
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = {metric: [] for metric in ['ap', 'auc', 'f1']}
        
        for train_idx, val_idx in skf.split(X_train, y_train):
            X_fold_train = X_train.iloc[train_idx]
            y_fold_train = y_train.iloc[train_idx]
            X_fold_val = X_train.iloc[val_idx]
            y_fold_val = y_train.iloc[val_idx]
            
            # Train model
            model = xgb.XGBClassifier(**params)
            model.fit(
                X_fold_train, y_fold_train,
                eval_set=[(X_fold_val, y_fold_val)],
                verbose=False
            )
            
            # Make predictions
            y_pred_proba = model.predict_proba(X_fold_val)[:, 1]
            y_pred = (y_pred_proba >= 0.5).astype(int)
            
            # Calculate metrics
            scores['ap'].append(average_precision_score(y_fold_val, y_pred_proba))
            scores['auc'].append(roc_auc_score(y_fold_val, y_pred_proba))
            scores['f1'].append(f1_score(y_fold_val, y_pred))
        
        # Calculate mean scores
        mean_scores = {k: np.mean(v) for k, v in scores.items()}
        
        # Store all metrics in trial
        for metric, value in mean_scores.items():
            trial.set_user_attr(f'{metric}_score', value)
        
        return mean_scores['ap']

    # Create study
    study = optuna.create_study(
        direction="maximize",
        study_name="xgboost_optimization",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=5,
            n_warmup_steps=5,
            interval_steps=1
        )
    )
    
    # Optimize
    objective_func = partial(
        objective, 
        X_train=X_train, 
        y_train=y_train,
        previous_best=BEST_PARAMS if use_previous_best else None
    )
    
    study.optimize(
        objective_func,
        n_trials=n_trials,
        show_progress_bar=True,
        callbacks=[lambda study, trial: print(f"Trial {trial.number}: AP={trial.value:.4f}")]
    )
    
    # Print results
    print("\nBest trial:")
    trial = study.best_trial
    for metric in ['ap', 'auc', 'f1']:
        print(f"  {metric.upper()} Score: {trial.user_attrs[f'{metric}_score']:.4f}")
    
    print("\nBest hyperparameters:")
    for key, value in trial.params.items():
        print(f"  {key}: {value}")
    
    # Train final model with best parameters
    best_params = study.best_params
    best_params.update({
        'objective': 'binary:logistic',
        'tree_method': 'hist',
        'eval_metric': ['aucpr', 'auc'],
        'random_state': 42,
        'scale_pos_weight': sum(y_train == 0) / sum(y_train == 1)
    })
    
    final_model = xgb.XGBClassifier(**best_params)
    final_model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train)],
        verbose=False
    )
    
    return final_model, study


def evaluate_imbalanced_model(model, X, y, threshold=0.5):
    """
    Evaluate model performance with metrics suitable for imbalanced classification.
    """
    
    # Get predictions
    y_pred_proba = model.predict_proba(X)[:, 1]
    y_pred = (y_pred_proba >= threshold).astype(int)
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y, y_pred))
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig("plots/confusion_matrix.png")
    
    # Plot ROC curve
    plt.figure(figsize=(8, 6))
    fpr, tpr, _ = roc_curve(y, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.savefig("plots/roc_curve.png")
    
    # Plot Precision-Recall curve
    plt.figure(figsize=(8, 6))
    precision, recall, _ = precision_recall_curve(y, y_pred_proba)
    pr_auc = auc(recall, precision)
    plt.plot(recall, precision, label=f'PR curve (AUC = {pr_auc:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.savefig("plots/pr_curve.png")
    
    # Feature importance plot
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    plt.figure(figsize=(12, 6))
    sns.barplot(data=feature_importance.head(20), x='importance', y='feature')
    plt.title('Top 20 Feature Importance')
    plt.savefig("plots/feature_importance.png")
    
    return {
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'feature_importance': feature_importance
    }

def create_submission_file(model, test_data, test_ids):
    """Create a submission file."""
    # Make predictions
    predictions = model.predict(test_data)
    # Create a submission file
    submission_df = pd.DataFrame({"id": test_ids, "Depression": predictions})
    submission_df.to_csv("submission.csv", index=False)
    print("Submission file created!")


def eda(data_orig):

    data = data_orig.copy()
    # Explore distribution of samples
    print("Dataset shape:", data.shape)
    print("Dataset info:")
    print(data.info())
    print("Missing values per column:")
    # Print everything to the console and prevent truncation
    pd.set_option("display.max_rows", None)
    print(data.isnull().sum())

    # Print unique values for categorical columns, also write to log file
    categorical_cols = data.select_dtypes(include=["object"]).columns
    with open("categorical_values.txt", "w") as f:
        for col in categorical_cols:
            # print(f"\nUnique values for {col}:")
            # print(data[col].value_counts())

            f.write(f"\nUnique values for {col}:\n")
            f.write(str(data[col].value_counts().to_string()))
            f.write("\n")

    # Drop id and name columns
    if "id" in data.columns:
        data = data.drop("id", axis=1)
    if "Name" in data.columns:
        data = data.drop("Name", axis=1)

    # Distribution plots for numerical variables
    numerical_cols = data.select_dtypes(include=["int64", "float64"]).columns
    num_cols = len(numerical_cols)
    fig, axes = plt.subplots(num_cols, 1, figsize=(10, 5 * num_cols))

    for i, col in enumerate(numerical_cols):
        sns.histplot(data[col], kde=True, ax=axes[i])
        axes[i].set_title(f"Distribution of {col}")

    plt.tight_layout()
    # plt.show()

    # Correlation matrix
    corr = data[numerical_cols].corr()
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr, annot=True, fmt=".2f")
    plt.title("Correlation Matrix")
    # plt.show()

    # Violin plots for numerical variables

    fig, axes = plt.subplots(
        len(numerical_cols), 1, figsize=(10, 5 * len(numerical_cols))
    )
    for i, col in enumerate(numerical_cols):
        sns.violinplot(x=data[col], ax=axes[i])
        axes[i].set_title(f"Violin plot of {col}")
    plt.tight_layout()
    plt.show()

    # Additional analysis: Count plots for categorical variables
    categorical_cols = data.select_dtypes(include=["object"]).columns
    fig, axes = plt.subplots(
        len(categorical_cols), 1, figsize=(10, 5 * len(categorical_cols))
    )
    for i, col in enumerate(categorical_cols):
        sns.countplot(y=data[col], order=data[col].value_counts().index, ax=axes[i])
        axes[i].set_title(f"Count plot of {col}")
    plt.tight_layout()
    plt.show()


def main():
    # Load data
    X_train, y_train, X_test = load_data(
        "data/s4_e11/train.csv", "data/s4_e11/test.csv"
    )

    # Save training and test ids
    train_ids = X_train["id"]
    test_ids = X_test["id"]

    eda(X_train)
    print("====================================")
    eda(X_test)
    return
    # Print initial shapes
    print(f"Initial training data shape: {X_train.shape}")
    print(f"Initial test data shape: {X_test.shape}")

    # Transform data
    X_train_transformed, y_train_filtered, X_test_transformed, preprocessor = (
        validate_and_transform_data(X_train, y_train, X_test)
    )

    print(f"Training data shape after preprocessing: {X_train_transformed.shape}")
    print(f"Class distribution:\n{y_train_filtered.value_counts(normalize=True)}")

    # Optimize hyperparameters using Optuna
    best_model, study = optimize_xgboost_with_optuna(
        X_train_transformed, 
        y_train_filtered,
        n_trials=100  # Adjust based on your computational budget
    )
    
        # Create submission
    create_submission_file(best_model, X_test_transformed, test_ids)
    
    # Evaluate the model
    results = evaluate_imbalanced_model(best_model, X_train_transformed, y_train_filtered)    
    # Save study results
    study_results = pd.DataFrame()
    study_results['value'] = [trial.value for trial in study.trials]
    study_results['params'] = [trial.params for trial in study.trials]
    study_results.to_csv('optuna_study_results.csv')

if __name__ == "__main__":
    main()
