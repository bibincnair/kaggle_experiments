import pandas as pd
import polars as pl
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import StandardScaler
import os


class CMIDataset:
    def __init__(self, data_path):
        self.data_path = data_path
        self.season_dtype = pl.Enum(["Spring", "Summer", "Fall", "Winter"])
        self.cat_columns = [
            "Basic_Demos-Enroll_Season",
            "CGAS-Season",
            "Physical-Season",
            "Fitness_Endurance-Season",
            "FGC-Season",
            "BIA-Season",
            "PAQ_A-Season",
            "PAQ_C-Season",
            "SDS-Season",
            "PreInt_EduHx-Season",
        ]

    def load_data(self):
        train = pl.read_csv(f"{self.data_path}/train.csv")
        test = pl.read_csv(f"{self.data_path}/test.csv")
        sample = pd.read_csv(f"{self.data_path}/sample_submission.csv")

        train = train.with_columns(pl.col("^.*Season$").cast(self.season_dtype))
        test = test.with_columns(pl.col("^.*Season$").cast(self.season_dtype))

        return train, test, sample

    def load_time_series(self, dirname):
        def process_file(filename):
            df = pd.read_parquet(os.path.join(dirname, filename, "part-0.parquet"))
            df.drop("step", axis=1, inplace=True)
            return df.describe().values.reshape(-1), filename.split("=")[1]

        ids = os.listdir(dirname)
        with ThreadPoolExecutor() as executor:
            results = list(
                tqdm(
                    executor.map(lambda fname: process_file(fname), ids), total=len(ids)
                )
            )

        stats, indexes = zip(*results)
        df = pd.DataFrame(stats, columns=[f"stat_{i}" for i in range(len(stats[0]))])
        df["id"] = indexes
        return df

    def feature_engineering(self, df):
        df = self._create_basic_features(df)
        df = self._create_advanced_features(df)
        return df

    def _create_basic_features(self, df):
        df["BMI_Age"] = df["Physical-BMI"] * df["Basic_Demos-Age"]
        df["Internet_Hours_Age"] = (
            df["PreInt_EduHx-computerinternet_hoursday"] * df["Basic_Demos-Age"]
        )
        df["BMI_Internet_Hours"] = (
            df["Physical-BMI"] * df["PreInt_EduHx-computerinternet_hoursday"]
        )
        return df

    def _create_advanced_features(self, df):
        # BIA-related features
        df["BFP_BMI"] = df["BIA-BIA_Fat"] / df["BIA-BIA_BMI"]
        df["FFMI_BFP"] = df["BIA-BIA_FFMI"] / df["BIA-BIA_Fat"]
        df["FMI_BFP"] = df["BIA-BIA_FMI"] / df["BIA-BIA_Fat"]
        df["LST_TBW"] = df["BIA-BIA_LST"] / df["BIA-BIA_TBW"]
        df["BFP_BMR"] = df["BIA-BIA_Fat"] * df["BIA-BIA_BMR"]
        df["BFP_DEE"] = df["BIA-BIA_Fat"] * df["BIA-BIA_DEE"]

        # Physical metrics features
        df["BMR_Weight"] = df["BIA-BIA_BMR"] / df["Physical-Weight"]
        df["DEE_Weight"] = df["BIA-BIA_DEE"] / df["Physical-Weight"]
        df["SMM_Height"] = df["BIA-BIA_SMM"] / df["Physical-Height"]
        df["Muscle_to_Fat"] = df["BIA-BIA_SMM"] / df["BIA-BIA_FMI"]
        df["Hydration_Status"] = df["BIA-BIA_TBW"] / df["Physical-Weight"]
        df["ICW_TBW"] = df["BIA-BIA_ICW"] / df["BIA-BIA_TBW"]
        return df

    def preprocess_categorical(self, df):
        for c in self.cat_columns:
            df[c] = df[c].fillna("Missing")
            df[c] = df[c].astype("category")
        return df

    def encode_categorical(self, train_df, test_df):
        for col in self.cat_columns:
            train_mapping = {val: idx for idx, val in enumerate(train_df[col].unique())}
            test_mapping = {val: idx for idx, val in enumerate(test_df[col].unique())}

            train_df[col] = train_df[col].replace(train_mapping).astype(int)
            test_df[col] = test_df[col].replace(test_mapping).astype(int)

        return train_df, test_df
