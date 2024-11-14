import pytorch_tabular.models as models
from pytorch_tabular import TabularModel
from pytorch_tabular.config import (
    DataConfig,
    OptimizerConfig,
    TrainerConfig,
    ExperimentConfig,
)
from sklearn.model_selection import train_test_split

import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


class NNMentalHealthClassifier:
    def __init__(self):
        self.data_config = DataConfig(
            target=["Depression"],
            continuous_cols=[
                "Age",
                "Academic Pressure",
                "Work Pressure",
                "CGPA",
                "Study Satisfaction",
                "Job Satisfaction",
                "Work/Study Hours",
                "Financial Stress",
            ],
            categorical_cols=[
                "Gender",
                "City",
                "Working Professional or Student",
                "Profession",
                "Sleep Duration",
                "Dietary Habits",
                "Degree",
                "Have you ever had suicidal thoughts ?",
                "Family History of Mental Illness",
            ],
        )
        self.trainer_config = TrainerConfig(
            auto_lr_find=True, batch_size=1024, max_epochs=100
        )
        self.optimizer_config = OptimizerConfig()
        self.model_config = models.TabNetModelConfig(
            task="classification",
        )

    def create_model(self, categorical_cols, continuous_cols):
        self.data_config = DataConfig(
            target=["Depression"],
            continuous_cols=continuous_cols,
            categorical_cols=categorical_cols,
        )
        self.model = TabularModel(
            data_config=self.data_config,
            model_config=self.model_config,
            optimizer_config=self.optimizer_config,
            trainer_config=self.trainer_config,
            verbose=True,
        )

    def load_data(self):
        """Load data from data/s4_e11 folder, train.csv and test.csv"""
        train_data = pd.read_csv("data/s4_e11/train.csv")
        test_data = pd.read_csv("data/s4_e11/test.csv")
        X_train = train_data
        # Define target variable and set column name to "Depression", defined as Dataframe
        y_train = pd.DataFrame(train_data["Depression"])
        X_test = test_data
        # Print column names
        print(X_train.columns)
        # Print categorical columns
        print(X_train.select_dtypes(include=["object"]).columns)
        # Print continuous columns
        print(X_train.select_dtypes(include=["number"]).columns)
        # Print target column
        print(y_train.columns)
        return X_train, y_train, X_test

    def load_original_data(self):
        """Load original data from data/s4_e11 folder"""
        original_data = pd.read_csv("data/s4_e11/original.csv")
        X_original = original_data.drop("Depression", axis=1)
        y_original = original_data["Depression"]
        return X_original, y_original

    def load_preprocess_data(self):
        """Loads preprocessed data from the processed_data folder

        Raises:
            FileNotFoundError: _description_

        Returns:
            pd.Dataframe: data frame: X_train, X_test, y_train
        """
        try:
            # Load pickled X_train, X_test, y_train from data/s4_e11/processed_data folder
            X_train = pd.read_pickle(f"data/s4_e11/processed_data/X_train.pkl")
            X_test = pd.read_pickle(f"data/s4_e11/processed_data/X_test.pkl")
            y_train = pd.read_pickle(f"data/s4_e11/processed_data/y_train.pkl")
            train_id = pd.read_pickle(f"data/s4_e11/processed_data/train_id.pkl")
            test_id = pd.read_pickle(f"data/s4_e11/processed_data/test_id.pkl")
        except:
            raise FileNotFoundError("Please run the data preprocessing script first")

        return X_train, X_test, y_train, train_id, test_id

    def train_model(self, train_data):
        """Trains the model

        Args:
            X_train (pd.DataFrame): Training data
            y_train (pd.DataFrame): Training target
        """
        train, val = train_test_split(
            train_data,
            test_size=0.2,
            random_state=42,
            shuffle=True,
            stratify=train_data["Depression"],
        )
        self.model.fit(train=train, validation=val)

    def predict(self, X_test):
        """Predicts the target variable

        Args:
            X_test (pd.DataFrame): Test data

        Returns:
            np.array: Predicted target
        """
        return self.model.predict(X_test)


def main():
    nn = NNMentalHealthClassifier()
    X_train, X_test, y_train, train_id, test_id = nn.load_preprocess_data()
    # Print column names
    print(X_train.columns)
    print(X_test.columns)
    # Convert y_train to dataframe and name the column "Depression"
    y_train = pd.DataFrame(y_train, columns=["Depression"])
    train_data = pd.concat([X_train, y_train], axis=1)
    print(y_train.columns)
    nn.create_model(
        list(X_train.select_dtypes(include=["object"]).columns),
        list(X_train.select_dtypes(include=["number"]).columns),
    )
    nn.train_model(train_data)
    y_pred = nn.predict(X_test)
    print(y_pred)


if __name__ == "__main__":
    main()
