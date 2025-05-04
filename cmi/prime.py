import numpy as np
import pandas as pd
import os
import re
from sklearn.base import clone, BaseEstimator, RegressorMixin
from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from scipy.optimize import minimize
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import polars as pl
import polars.selectors as cs
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, FormatStrFormatter, PercentFormatter
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from keras.models import Model
from keras.layers import Input, Dense
from keras.optimizers import Adam
import torch
import torch.nn as nn
import torch.optim as optim
from colorama import Fore, Style
from IPython.display import clear_output
import warnings
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.ensemble import (
    VotingRegressor,
    RandomForestRegressor,
    GradientBoostingRegressor,
)
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.pipeline import Pipeline
from pytorch_tabnet.tab_model import TabNetRegressor
from pytorch_tabnet.callbacks import Callback

warnings.filterwarnings("ignore")
pd.options.display.max_columns = None

# Constants
SEED = 42
n_splits = 5
target_labels = ["None", "Mild", "Moderate", "Severe"]
season_dtype = pl.Enum(["Spring", "Summer", "Fall", "Winter"])


def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


class AutoEncoder(nn.Module):
    def __init__(self, input_dim, encoding_dim):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, encoding_dim * 3),
            nn.ReLU(),
            nn.Linear(encoding_dim * 3, encoding_dim * 2),
            nn.ReLU(),
            nn.Linear(encoding_dim * 2, encoding_dim),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, input_dim * 2),
            nn.ReLU(),
            nn.Linear(input_dim * 2, input_dim * 3),
            nn.ReLU(),
            nn.Linear(input_dim * 3, input_dim),
            nn.Sigmoid(),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


class TabNetWrapper(BaseEstimator, RegressorMixin):
    def __init__(self, **kwargs):
        self.model = TabNetRegressor(**kwargs)
        self.kwargs = kwargs
        self.imputer = SimpleImputer(strategy="median")
        self.best_model_path = "best_tabnet_model.pt"

    def fit(self, X, y):
        X_imputed = self.imputer.fit_transform(X)

        if hasattr(y, "values"):
            y = y.values

        X_train, X_valid, y_train, y_valid = train_test_split(
            X_imputed, y, test_size=0.2, random_state=42
        )

        history = self.model.fit(
            X_train=X_train,
            y_train=y_train.reshape(-1, 1),
            eval_set=[(X_valid, y_valid.reshape(-1, 1))],
            eval_name=["valid"],
            eval_metric=["mse"],
            max_epochs=200,
            patience=20,
            batch_size=1024,
            virtual_batch_size=128,
            num_workers=0,
            drop_last=False,
            callbacks=[
                TabNetPretrainedModelCheckpoint(
                    filepath=self.best_model_path,
                    monitor="valid_mse",
                    mode="min",
                    save_best_only=True,
                    verbose=True,
                )
            ],
        )

        if os.path.exists(self.best_model_path):
            self.model.load_model(self.best_model_path)
            os.remove(self.best_model_path)

        return self

    def predict(self, X):
        X_imputed = self.imputer.transform(X)
        return self.model.predict(X_imputed).flatten()


class TabNetPretrainedModelCheckpoint(Callback):
    def __init__(
        self, filepath, monitor="val_loss", mode="min", save_best_only=True, verbose=1
    ):
        super().__init__()
        self.filepath = filepath
        self.monitor = monitor
        self.mode = mode
        self.save_best_only = save_best_only
        self.verbose = verbose
        self.best = float("inf") if mode == "min" else -float("inf")

    def on_train_begin(self, logs=None):
        self.model = self.trainer

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        current = logs.get(self.monitor)
        if current is None:
            return

        if (self.mode == "min" and current < self.best) or (
            self.mode == "max" and current > self.best
        ):
            if self.verbose:
                print(
                    f"\nEpoch {epoch}: {self.monitor} improved from {self.best:.4f} to {current:.4f}"
                )
            self.best = current
            if self.save_best_only:
                self.model.save_model(self.filepath)


def process_file(filename, dirname):
    df = pd.read_parquet(os.path.join(dirname, filename, "part-0.parquet"))
    df.drop("step", axis=1, inplace=True)
    return df.describe().values.reshape(-1), filename.split("=")[1]


def load_time_series(dirname) -> pd.DataFrame:
    ids = os.listdir(dirname)

    with ThreadPoolExecutor() as executor:
        results = list(
            tqdm(
                executor.map(lambda fname: process_file(fname, dirname), ids),
                total=len(ids),
            )
        )

    stats, indexes = zip(*results)

    df = pd.DataFrame(stats, columns=[f"stat_{i}" for i in range(len(stats[0]))])
    df["id"] = indexes
    return df


def perform_autoencoder(df, encoding_dim=50, epochs=50, batch_size=32):
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df)

    data_tensor = torch.FloatTensor(df_scaled)

    input_dim = data_tensor.shape[1]
    autoencoder = AutoEncoder(input_dim, encoding_dim)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(autoencoder.parameters())

    for epoch in range(epochs):
        for i in range(0, len(data_tensor), batch_size):
            batch = data_tensor[i : i + batch_size]
            optimizer.zero_grad()
            reconstructed = autoencoder(batch)
            loss = criterion(reconstructed, batch)
            loss.backward()
            optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}]")

    with torch.no_grad():
        encoded_data = autoencoder.encoder(data_tensor).numpy()

    df_encoded = pd.DataFrame(
        encoded_data, columns=[f"Enc_{i + 1}" for i in range(encoded_data.shape[1])]
    )

    return df_encoded


def feature_engineering(df):
    season_cols = [col for col in df.columns if "Season" in col]
    df = df.drop(season_cols, axis=1)
    df["BMI_Age"] = df["Physical-BMI"] * df["Basic_Demos-Age"]
    df["Internet_Hours_Age"] = (
        df["PreInt_EduHx-computerinternet_hoursday"] * df["Basic_Demos-Age"]
    )
    df["BMI_Internet_Hours"] = (
        df["Physical-BMI"] * df["PreInt_EduHx-computerinternet_hoursday"]
    )
    df["BFP_BMI"] = df["BIA-BIA_Fat"] / df["BIA-BIA_BMI"]
    df["FFMI_BFP"] = df["BIA-BIA_FFMI"] / df["BIA-BIA_Fat"]
    df["FMI_BFP"] = df["BIA-BIA_FMI"] / df["BIA-BIA_Fat"]
    df["LST_TBW"] = df["BIA-BIA_LST"] / df["BIA-BIA_TBW"]
    df["BFP_BMR"] = df["BIA-BIA_Fat"] * df["BIA-BIA_BMR"]
    df["BFP_DEE"] = df["BIA-BIA_Fat"] * df["BIA-BIA_DEE"]
    df["BMR_Weight"] = df["BIA-BIA_BMR"] / df["Physical-Weight"]
    df["DEE_Weight"] = df["BIA-BIA_DEE"] / df["Physical-Weight"]
    df["SMM_Height"] = df["BIA-BIA_SMM"] / df["Physical-Height"]
    df["Muscle_to_Fat"] = df["BIA-BIA_SMM"] / df["BIA-BIA_FMI"]
    df["Hydration_Status"] = df["BIA-BIA_TBW"] / df["Physical-Weight"]
    df["ICW_TBW"] = df["BIA-BIA_ICW"] / df["BIA-BIA_TBW"]

    return df


def quadratic_weighted_kappa(y_true, y_pred):
    return cohen_kappa_score(y_true, y_pred, weights="quadratic")


def threshold_Rounder(oof_non_rounded, thresholds):
    return np.where(
        oof_non_rounded < thresholds[0],
        0,
        np.where(
            oof_non_rounded < thresholds[1],
            1,
            np.where(oof_non_rounded < thresholds[2], 2, 3),
        ),
    )


def evaluate_predictions(thresholds, y_true, oof_non_rounded):
    rounded_p = threshold_Rounder(oof_non_rounded, thresholds)
    return -quadratic_weighted_kappa(y_true, rounded_p)


def TrainML(model_class, test_data):
    X = train.drop(["sii"], axis=1)
    y = train["sii"]

    SKF = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)

    train_S = []
    test_S = []

    oof_non_rounded = np.zeros(len(y), dtype=float)
    oof_rounded = np.zeros(len(y), dtype=int)
    test_preds = np.zeros((len(test_data), n_splits))

    for fold, (train_idx, test_idx) in enumerate(
        tqdm(SKF.split(X, y), desc="Training Folds", total=n_splits)
    ):
        X_train, X_val = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[test_idx]

        model = clone(model_class)
        model.fit(X_train, y_train)

        y_train_pred = model.predict(X_train)
        y_val_pred = model.predict(X_val)

        oof_non_rounded[test_idx] = y_val_pred
        y_val_pred_rounded = y_val_pred.round(0).astype(int)
        oof_rounded[test_idx] = y_val_pred_rounded

        train_kappa = quadratic_weighted_kappa(
            y_train, y_train_pred.round(0).astype(int)
        )
        val_kappa = quadratic_weighted_kappa(y_val, y_val_pred_rounded)

        train_S.append(train_kappa)
        test_S.append(val_kappa)

        test_preds[:, fold] = model.predict(test_data)

        print(
            f"Fold {fold+1} - Train QWK: {train_kappa:.4f}, Validation QWK: {val_kappa:.4f}"
        )
        clear_output(wait=True)

    print(f"Mean Train QWK --> {np.mean(train_S):.4f}")
    print(f"Mean Validation QWK ---> {np.mean(test_S):.4f}")

    KappaOPtimizer = minimize(
        evaluate_predictions,
        x0=[0.5, 1.5, 2.5],
        args=(y, oof_non_rounded),
        method="Nelder-Mead",
    )
    assert KappaOPtimizer.success, "Optimization did not converge."

    oof_tuned = threshold_Rounder(oof_non_rounded, KappaOPtimizer.x)
    tKappa = quadratic_weighted_kappa(y, oof_tuned)

    print(
        f"----> || Optimized QWK SCORE :: {Fore.CYAN}{Style.BRIGHT} {tKappa:.3f}{Style.RESET_ALL}"
    )

    tpm = test_preds.mean(axis=1)
    tpTuned = threshold_Rounder(tpm, KappaOPtimizer.x)

    submission = pd.DataFrame({"id": sample["id"], "sii": tpTuned})

    return submission


if __name__ == "__main__":
    seed_everything(2024)

    # Model parameters
    LightGBM_Params = {
        "learning_rate": 0.046,
        "max_depth": 12,
        "num_leaves": 478,
        "min_data_in_leaf": 13,
        "feature_fraction": 0.893,
        "bagging_fraction": 0.784,
        "bagging_freq": 4,
        "lambda_l1": 10,
        "lambda_l2": 0.01,
        "device": "cpu",
    }

    XGB_Params = {
        "learning_rate": 0.05,
        "max_depth": 6,
        "n_estimators": 200,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 1,
        "reg_lambda": 5,
        "random_state": SEED,
        "tree_method": "gpu_hist",
    }

    CatBoost_Params = {
        "learning_rate": 0.05,
        "depth": 6,
        "iterations": 200,
        "random_state": SEED,
        "verbose": 0,
    }

TabNet_Params = {
    "n_d": 64,
    "n_a": 64,
    "n_steps": 5,
    "gamma": 1.5,
    "n_independent": 2,
    "n_shared": 2,
    "lambda_sparse": 1e-4,
    "optimizer_fn": torch.optim.Adam,
    "optimizer_params": dict(lr=2e-2, weight_decay=1e-5),
    "mask_type": "entmax",
    "scheduler_params": dict(mode="min", patience=10, min_lr=1e-5, factor=0.5),
    "scheduler_fn": torch.optim.lr_scheduler.ReduceLROnPlateau,
    "verbose": 1,
    "device_name": "cuda" if torch.cuda.is_available() else "cpu",
}

# Load data
train = pd.read_csv(
    "/kaggle/input/child-mind-institute-problematic-internet-use/train.csv"
)
test = pd.read_csv(
    "/kaggle/input/child-mind-institute-problematic-internet-use/test.csv"
)
sample = pd.read_csv(
    "/kaggle/input/child-mind-institute-problematic-internet-use/sample_submission.csv"
)

# Load time series data
train_ts = load_time_series(
    "/kaggle/input/child-mind-institute-problematic-internet-use/series_train.parquet"
)
test_ts = load_time_series(
    "/kaggle/input/child-mind-institute-problematic-internet-use/series_test.parquet"
)

# Process time series features
df_train = train_ts.drop("id", axis=1)
df_test = test_ts.drop("id", axis=1)

train_ts_encoded = perform_autoencoder(
    df_train, encoding_dim=60, epochs=100, batch_size=32
)
test_ts_encoded = perform_autoencoder(
    df_test, encoding_dim=60, epochs=100, batch_size=32
)

time_series_cols = train_ts_encoded.columns.tolist()
train_ts_encoded["id"] = train_ts["id"]
test_ts_encoded["id"] = test_ts["id"]

# Merge data
train = pd.merge(train, train_ts_encoded, how="left", on="id")
test = pd.merge(test, test_ts_encoded, how="left", on="id")

# Handle missing values
imputer = KNNImputer(n_neighbors=5)
numeric_cols = train.select_dtypes(include=["float64", "int64"]).columns
imputed_data = imputer.fit_transform(train[numeric_cols])
train_imputed = pd.DataFrame(imputed_data, columns=numeric_cols)
train_imputed["sii"] = train_imputed["sii"].round().astype(int)

for col in train.columns:
    if col not in numeric_cols:
        train_imputed[col] = train[col]

train = train_imputed

# Feature engineering
train = feature_engineering(train)
train = train.dropna(thresh=10, axis=0)
test = feature_engineering(test)

train = train.drop("id", axis=1)
test = test.drop("id", axis=1)

# Define feature columns
featuresCols = [
    "Basic_Demos-Age",
    "Basic_Demos-Sex",
    "CGAS-CGAS_Score",
    "Physical-BMI",
    "Physical-Height",
    "Physical-Weight",
    "Physical-Waist_Circumference",
    "Physical-Diastolic_BP",
    "Physical-HeartRate",
    "Physical-Systolic_BP",
    "Fitness_Endurance-Max_Stage",
    "Fitness_Endurance-Time_Mins",
    "Fitness_Endurance-Time_Sec",
    "FGC-FGC_CU",
    "FGC-FGC_CU_Zone",
    "FGC-FGC_GSND",
    "FGC-FGC_GSND_Zone",
    "FGC-FGC_GSD",
    "FGC-FGC_GSD_Zone",
    "FGC-FGC_PU",
    "FGC-FGC_PU_Zone",
    "FGC-FGC_SRL",
    "FGC-FGC_SRL_Zone",
    "FGC-FGC_SRR",
    "FGC-FGC_SRR_Zone",
    "FGC-FGC_TL",
    "FGC-FGC_TL_Zone",
    "BIA-BIA_Activity_Level_num",
    "BIA-BIA_BMC",
    "BIA-BIA_BMI",
    "BIA-BIA_BMR",
    "BIA-BIA_DEE",
    "BIA-BIA_ECW",
    "BIA-BIA_FFM",
    "BIA-BIA_FFMI",
    "BIA-BIA_FMI",
    "BIA-BIA_Fat",
    "BIA-BIA_Frame_num",
    "BIA-BIA_ICW",
    "BIA-BIA_LDM",
    "BIA-BIA_LST",
    "BIA-BIA_SMM",
    "BIA-BIA_TBW",
    "PAQ_A-PAQ_A_Total",
    "PAQ_C-PAQ_C_Total",
    "SDS-SDS_Total_Raw",
    "SDS-SDS_Total_T",
    "PreInt_EduHx-computerinternet_hoursday",
    "sii",
    "BMI_Age",
    "Internet_Hours_Age",
    "BMI_Internet_Hours",
    "BFP_BMI",
    "FFMI_BFP",
    "FMI_BFP",
    "LST_TBW",
    "BFP_BMR",
    "BFP_DEE",
    "BMR_Weight",
    "DEE_Weight",
    "SMM_Height",
    "Muscle_to_Fat",
    "Hydration_Status",
    "ICW_TBW",
]

featuresCols += time_series_cols

# Prepare final datasets
train = train[featuresCols]
train = train.dropna(subset="sii")
test = test[featuresCols[:-1]]  # Exclude 'sii' from test features

# Create and train models
Light = LGBMRegressor(
    **LightGBM_Params, random_state=SEED, verbose=-1, n_estimators=300
)
XGB_Model = XGBRegressor(**XGB_Params)
CatBoost_Model = CatBoostRegressor(**CatBoost_Params)
TabNet_Model = TabNetWrapper(**TabNet_Params)

# Create ensemble model
voting_model = VotingRegressor(
    estimators=[
        ("lightgbm", Light),
        ("xgboost", XGB_Model),
        ("catboost", CatBoost_Model),
        ("tabnet", TabNet_Model),
    ],
    weights=[4.0, 4.0, 5.0, 4.0],
)

# Train and get predictions
submission1 = TrainML(voting_model, test)

# Alternative ensemble with pipelines
ensemble_pipeline = VotingRegressor(
    estimators=[
        (
            "lgb",
            Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("regressor", LGBMRegressor(random_state=SEED)),
                ]
            ),
        ),
        (
            "xgb",
            Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("regressor", XGBRegressor(random_state=SEED)),
                ]
            ),
        ),
        (
            "cat",
            Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("regressor", CatBoostRegressor(random_state=SEED, silent=True)),
                ]
            ),
        ),
        (
            "rf",
            Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("regressor", RandomForestRegressor(random_state=SEED)),
                ]
            ),
        ),
        (
            "gb",
            Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("regressor", GradientBoostingRegressor(random_state=SEED)),
                ]
            ),
        ),
    ]
)

submission2 = TrainML(ensemble_pipeline, test)

# Combine predictions using majority voting
submissions = [submission1, submission2]
for sub in submissions:
    sub.sort_values(by="id", inplace=True)
    sub.reset_index(drop=True, inplace=True)

combined = pd.DataFrame(
    {
        "id": submissions[0]["id"],
        "sii_1": submissions[0]["sii"],
        "sii_2": submissions[1]["sii"],
    }
)

# Get majority vote
combined["final_sii"] = combined[["sii_1", "sii_2"]].mode(axis=1)[0]

# Create final submission
final_submission = combined[["id", "final_sii"]].rename(columns={"final_sii": "sii"})
final_submission.to_csv("submission.csv", index=False)
