import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

# Use xgboost for prediction
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    auc,
)
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer


# Load the dataset from data/s4_e11/train.csv and test.csv
def load_data():
    train_data = pd.read_csv("data/s4_e11/train.csv")
    test_data = pd.read_csv("data/s4_e11/test.csv")
    print(f"Columns in train data: {train_data.columns}")
    # Depression column is the target variable
    y_train = train_data["Depression"]
    # Drop the target variable from the training and test data
    X_train = train_data.drop("Depression", axis=1)
    return X_train, y_train, test_data

class RareSampleRemover(BaseEstimator, TransformerMixin):
    def __init__(self, min_count=100):
        self.min_count = min_count
        self.frequent_categories = {}
        self.categorical_cols = None
        self.retained_indices_ = None

    def fit(self, X, y=None):
        self.categorical_cols = X.select_dtypes(include=['object', 'category']).columns
        self.frequent_categories = {}
        for col in self.categorical_cols:
            counts = X[col].value_counts()
            self.frequent_categories[col] = counts[counts >= self.min_count].index.tolist()
        self.retained_indices_ = np.ones(len(X), dtype=bool)
        for col in self.categorical_cols:
            mask = X[col].isin(self.frequent_categories[col])
            self.retained_indices_ &= mask
        return self

    def transform(self, X):
        X_filtered = X.loc[self.retained_indices_].reset_index(drop=True)
        return X_filtered

class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.attribute_names].values


class SleepDurationTransformer(BaseEstimator, TransformerMixin):
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
        self.median_sleep = X["Sleep Duration"].map(self.sleep_mapping).median()
        return self

    def transform(self, X):
        X_copy = X.copy()
        sleep_hours = (
            X_copy["Sleep Duration"].map(self.sleep_mapping).fillna(self.median_sleep)
        )

        result = pd.DataFrame(
            {
                "sleep_hours": sleep_hours,
                "is_recommended_sleep": sleep_hours.between(7, 9).astype(int),
                "is_short_sleep": (sleep_hours < 6).astype(int),
                "is_long_sleep": (sleep_hours > 9).astype(int),
            }
        )
        return result


class StatusBasedImputer(BaseEstimator, TransformerMixin):

    def fit(self, X):
        return self

    def transform(self, X):
        X_copy = X.copy()

        # Masks for student and working
        mask_student = X_copy["Working Professional or Student"] == "Student"
        mask_working = (
            X_copy["Working Professional or Student"] == "Working Professional"
        )

        # Handle student specific columns
        X_copy.loc[mask_student, "Work Pressure"] = 0
        X_copy.loc[mask_student, "Job Satisfaction"] = 0

        # Handle working professional-specific columns
        X_copy.loc[mask_working, "Academic Pressure"] = 0
        X_copy.loc[mask_working, "CGPA"] = 0
        X_copy.loc[mask_working, "Study Satisfaction"] = 0

        return X_copy


def create_preprocessing_pipeline(X):
    numerical_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object", "category"]).columns.tolist()
    
    # Remove special columns
    if 'Sleep Duration' in categorical_features:
        categorical_features.remove('Sleep Duration')
    if 'Depression' in numerical_features:
        numerical_features.remove('Depression')
    if 'id' in numerical_features:
        numerical_features.remove('id')
    if 'Name' in categorical_features:
        categorical_features.remove('Name')
        
    # Create preprocessing steps
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='Other')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    # Create column transformer
    column_transformer = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features),
            ('sleep', SleepDurationTransformer(), ['Sleep Duration']),
            ('status_imputer', StatusBasedImputer(), 
             ['Working Professional or Student', 'Work Pressure', 'Job Satisfaction', 
              'Academic Pressure', 'CGPA', 'Study Satisfaction'])
        ])
    
        # Create the full pipeline with RareSampleRemover
    preprocessor = Pipeline(steps=[
        ('rare_sample_remover', RareSampleRemover(min_count=100)),
        ('column_transformer', column_transformer)
    ])
    return preprocessor

def get_feature_names(preprocessor, X):
    """Get feature names for transformed data"""
    # Access the column transformer through pipeline
    column_transformer = preprocessor.named_steps['column_transformer']
    
    # Get numerical and categorical features
    numerical_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object", "category"]).columns.tolist()
    
    # Remove special columns
    special_columns = ['Sleep Duration', 'Depression', 'id', 'Name']
    for col in special_columns:
        if col in numerical_features:
            numerical_features.remove(col)
        if col in categorical_features:
            categorical_features.remove(col)
    
    # Get transformed feature names for each transformer
    transformed_features = []
    
    # Add numerical features
    if numerical_features:
        transformed_features.extend(numerical_features)
    
    # Add categorical features (one-hot encoded)
    if categorical_features:
        cat_transformer = column_transformer.named_transformers_['cat']
        onehot_encoder = cat_transformer.named_steps['onehot']
        cat_feature_names = onehot_encoder.get_feature_names_out(categorical_features)
        transformed_features.extend(cat_feature_names)
    
    # Add sleep features
    sleep_features = ['sleep_hours', 'is_recommended_sleep', 'is_short_sleep', 'is_long_sleep']
    transformed_features.extend(sleep_features)
    
    return transformed_features

def validate_and_transform_data(X_train, y_train, X_test=None):
    """
    Validate and transform the data using sklearn pipeline
    """
    try:
        # Validate required columns
        required_columns = [
            'Gender', 'Age', 'City', 'Working Professional or Student',
            'Sleep Duration'
        ]
        missing_cols = set(required_columns) - set(X_train.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Create and fit preprocessing pipeline
        preprocessor = create_preprocessing_pipeline(X=X_train)
        
        # Transform training data
        X_train_transformed = preprocessor.fit_transform(X_train)
        print(f"shape of transformed data: {X_train_transformed.shape}")
        retained_indices = preprocessor.named_steps['rare_sample_remover'].retained_indices_
        # Align y_train accordingly
        y_train_transformed = y_train.loc[retained_indices].reset_index(drop=True)
        
        # Get feature names
        feature_names = get_feature_names(preprocessor, X_train)
        
        # Convert transformed data to DataFrame
        X_train_transformed = pd.DataFrame(X_train_transformed, columns=feature_names)
        
        # If X_test is provided, transform it using the same preprocessor
        if X_test is not None:
            X_test_transformed = preprocessor.transform(X_test)
            X_test_transformed = pd.DataFrame(X_test_transformed, columns=feature_names)
            return X_train_transformed, y_train_transformed, X_test_transformed, preprocessor
        else:
            return X_train_transformed, y_train_transformed, preprocessor
    except Exception as e:
        print(f"Error in data validation and transformation: {e}")
        raise
    
    
def eda(data):
    # Explore distribution of samples
    print("Dataset shape:", data.shape)
    print("Dataset info:")
    print(data.info())
    print("Missing values per column:")
    print(data.isnull().sum())

    # Print unique values for categorical columns, also write to log file
    categorical_cols = data.select_dtypes(include=["object"]).columns
    with open("categorical_values.txt", "w") as f:
        for col in categorical_cols:
            print(f"\nUnique values for {col}:")
            print(data[col].value_counts())

            f.write(f"\nUnique values for {col}:\n")
            f.write(str(data[col].value_counts().to_string()))
            f.write("\n")

    # Drop id and name columns
    data = data.drop(["id", "Name"], axis=1)

    # Distribution plots for numerical variables
    numerical_cols = data.select_dtypes(include=["int64", "float64"]).columns
    for col in numerical_cols:
        plt.figure(figsize=(8, 4))
        sns.histplot(data[col], kde=True)
        plt.title(f"Distribution of {col}")
        # plt.show()

    # Correlation matrix
    corr = data[numerical_cols].corr()
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr, annot=True, fmt=".2f")
    plt.title("Correlation Matrix")
    # plt.show()

    # Violin plots for numerical variables
    for col in numerical_cols:
        plt.figure(figsize=(8, 4))
        sns.violinplot(x=data[col])
        plt.title(f"Violin plot of {col}")
        # plt.show()

    # Additional analysis: Count plots for categorical variables
    categorical_cols = data.select_dtypes(include=["object"]).columns
    for col in categorical_cols:
        plt.figure(figsize=(8, 4))
        sns.countplot(y=data[col], order=data[col].value_counts().index)
        plt.title(f"Count plot of {col}")
        # plt.show()
    plt.show()


def main():
    train_x, train_y, test_x = load_data()
    # eda(train_x)
    train_x, train_y, test_x, preprocessor = validate_and_transform_data(
        train_x, train_y, test_x
    )


if __name__ == "__main__":
    main()
