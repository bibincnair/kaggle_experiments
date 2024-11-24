import os
import numpy as np
from sklearn.model_selection import StratifiedKFold
from dataset import CMIDataset
from models import create_ensemble_model
from metrics import quadratic_weighted_kappa, optimize_thresholds, threshold_rounder


def train_and_predict(train_data, test_data, n_splits=5):
    X = train_data.drop(["sii"], axis=1)
    y = train_data["sii"]

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=2024)

    oof_predictions = np.zeros(len(y))
    test_predictions = np.zeros((len(test_data), n_splits))

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = create_ensemble_model(cat_features=None)
        model.fit(X_train, y_train)

        oof_predictions[val_idx] = model.predict(X_val)
        test_predictions[:, fold] = model.predict(test_data)

    optimal_thresholds = optimize_thresholds(y, oof_predictions)
    test_pred_mean = test_predictions.mean(axis=1)
    final_predictions = threshold_rounder(test_pred_mean, optimal_thresholds)

    return final_predictions


def main():
    # Initialize dataset
    dataset = CMIDataset("/kaggle/input/child-mind-institute-problematic-internet-use")
    train, test, sample = dataset.load_data()

    # Load time series data
    train_ts = dataset.load_time_series("series_train.parquet")
    test_ts = dataset.load_time_series("series_test.parquet")

    # Merge time series features
    train = pd.merge(train, train_ts, how="left", on="id")
    test = pd.merge(test, test_ts, how="left", on="id")

    # Feature engineering
    train = dataset.feature_engineering(train)
    test = dataset.feature_engineering(test)

    # Preprocess categorical features
    train = dataset.preprocess_categorical(train)
    test = dataset.preprocess_categorical(test)

    # Encode categorical features
    train, test = dataset.encode_categorical(train, test)

    # Train and predict
    predictions = train_and_predict(train, test)

    # Create submission
    submission = pd.DataFrame({"id": sample["id"], "sii": predictions})

    submission.to_csv("submission.csv", index=False)


if __name__ == "__main__":
    main()
