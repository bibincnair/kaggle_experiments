import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("TkAgg")


# Load the dataset
def load_data():
    data = pd.read_csv("data/train.csv")
    # print column names
    print(data.columns)
    # drop id column
    data = data.drop("id", axis=1)
    return data


def create_submission_file(model, test_data, test_ids):
    # Make predictions
    predictions = model.predict(test_data)
    # Create a submission file
    submission_df = pd.DataFrame({"id": test_ids, "loan_status": predictions})
    submission_df.to_csv("submission.csv", index=False)
    print("Submission file created!")


def preprocess(data):
    # Convert categorical column "person_home_ownership" and "cb_person_default_on_file"
    # Use one-hot encoding
    data = pd.get_dummies(
        data, columns=["person_home_ownership", "cb_person_default_on_file"]
    )
    return data


def eda(data):
    print(data.info())
    print(data.describe())
    print(data.isnull().sum())

    # Check the distribution of the target variable
    print(data["loan_status"].value_counts())

    # Handle categorical columns
    categorical_cols = ["person_home_ownership", "cb_person_default_on_file"]

    for col in categorical_cols:
        print(f"\nDistribution of {col}:")
        print(data[col].value_counts())
        print(f"\nDistribution of {col} by loan_status:")
        print(data.groupby("loan_status")[col].value_counts().unstack())

        # Visualize categorical columns
        plt.figure(figsize=(10, 6))
        sns.countplot(x=col, hue="loan_status", data=data)
        plt.title(f"Distribution of {col} by Loan Status")
        plt.show()

    # Correlation heatmap for numerical columns
    numeric_cols = data.select_dtypes(include=["int64", "float64"]).columns
    corr = data[numeric_cols].corr()
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Heatmap of Numerical Features")
    plt.show()


def main():
    data = load_data()
    eda(data)
    # Drop the missing values
    data = data.dropna()


if __name__ == "__main__":
    main()
