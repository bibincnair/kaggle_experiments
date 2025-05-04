from datetime import datetime, timedelta
from typing import List, Tuple
import json
import polars as pl
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error
from sklearn.impute import KNNImputer
from clearml import Task
from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional, Any
from pathlib import Path
import logging
import shap
import optuna

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("logs/s5_e4_podcast_listening.log"),
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
    model_path: str = "models/s5_e4_catboost_model.cbm"
    params_path: str = "models/s5_e4_catboost_best_params.json"
    data_path: str = "data/s5_e4/"


class EDA:
    def __init__(self, data: pl.DataFrame):
        self.data = data

    def plot_listening_time_distribution(self) -> plt.Axes:
        """Plots the distribution of listening time.

        Returns:
            plt.Axes: The matplotlib axes object containing the plot.
        """
        plt.figure(figsize=(10, 6))
        ax = sns.histplot(self.data["Listening_Time_minutes"], bins=30, kde=True)
        ax.set_title("Distribution of Podcast Listening Time")
        ax.set_xlabel("Listening Time (minutes)")
        ax.set_ylabel("Frequency")
        return ax

    def plot_listening_time_by_podcast(self) -> plt.Axes:
        """Plots boxplots of listening time grouped by podcast name.

        Returns:
            plt.Axes: The matplotlib axes object containing the plot.
        """
        plt.figure(figsize=(12, 6))
        ax = sns.boxplot(
            x="Podcast_Name", y="Listening_Time_minutes", data=self.data.to_pandas()
        )
        ax.set_title("Podcast Listening Time by Podcast")
        ax.set_xlabel("Podcast Name")
        ax.set_ylabel("Listening Time (minutes)")
        plt.xticks(rotation=90)
        return ax

    def print_features(self):
        """Find categorical and numerical features in the dataset."""
        categorical_features = self.data.select(pl.col(pl.Utf8)).columns
        numerical_features = self.data.select(pl.col(pl.Float64)).columns
        print("Categorical Features:", categorical_features)
        print("Numerical Features:", numerical_features)
        return categorical_features, numerical_features

    def print_statistics(self):
        """Prints descriptive statistics for key columns."""
        print("Listening Time Statistics:")
        print(self.data["Listening_Time_minutes"].describe())
        print("\nPodcast Name Statistics:")
        print(self.data["Podcast_Name"].value_counts())
        print("\nEpisode Title Statistics:")
        # Consider limiting output for potentially very long lists of titles
        print(
            self.data["Episode_Title"].value_counts().head(20)
        )  # Print top 20 episode titles
        print("\nListening Time by Podcast Name (Summary Stats):")
        # Calculate specific stats within agg()
        print(
            self.data.group_by("Podcast_Name").agg(
                pl.col("Listening_Time_minutes").count().alias("count"),
                pl.col("Listening_Time_minutes").mean().alias("mean"),
                pl.col("Listening_Time_minutes").std().alias("std"),
                pl.col("Listening_Time_minutes").min().alias("min"),
                pl.col("Listening_Time_minutes").median().alias("median"),
                pl.col("Listening_Time_minutes").max().alias("max"),
            )
        )
        print(
            "\nListening Time by Episode Title (Summary Stats - Top 20 Episodes by Count):"
        )
        # Get value counts first to find top episodes
        top_episodes = (
            self.data["Episode_Title"].value_counts().head(20)["Episode_Title"]
        )
        # Filter data for top episodes and then calculate stats
        print(
            self.data.filter(pl.col("Episode_Title").is_in(top_episodes))
            .group_by("Episode_Title")
            .agg(
                pl.col("Listening_Time_minutes").count().alias("count"),
                pl.col("Listening_Time_minutes").mean().alias("mean"),
                pl.col("Listening_Time_minutes").std().alias("std"),
                pl.col("Listening_Time_minutes").min().alias("min"),
                pl.col("Listening_Time_minutes").median().alias("median"),
                pl.col("Listening_Time_minutes").max().alias("max"),
            )
            .sort("count", descending=True)  # Sort by count for clarity
        )


class DataPreprocessor:
    """
    A class for preprocessing data using Polars, including feature engineering,
    imputation, label encoding, and train-validation splitting.

    Args:
        target_column (str): The name of the target variable column.
        random_seed (int): The random seed for reproducibility in splitting. Defaults to 42.
    """

    def __init__(self, target_column: str, random_seed: int = 42):
        if not target_column:
            raise ValueError("target_column must be specified.")
        self.target_column = target_column
        self.random_seed = random_seed

        # State fitted on training data
        self.imputation_values_: Dict[str, float] = {}
        self.numerical_features_for_imputation_: List[str] = []
        # We don't need to store categorical features explicitly beforehand,
        # as label encoding in transform will apply to all string columns found at that time.

        logging.info(
            f"DataPreprocessor initialized for target '{self.target_column}' with seed {self.random_seed}"
        )

    def _engineer_features(self, data: pl.DataFrame) -> pl.DataFrame:
        """
        Applies feature engineering steps.
        This method should be customized based on specific project needs.

        Args:
            data (pl.DataFrame): The input dataframe.

        Returns:
            pl.DataFrame: The dataframe with engineered features.
        """
        engineered_data = data.clone()  # Work on a copy
        logging.info("Applying feature engineering...")

        # --- Example Feature Engineering ---
        # Combine 'Publication_Day' and 'Episode_Sentiment' if they exist
        day_col = "Publication_Day"
        sentiment_col = "Episode_Sentiment"
        new_col_name = "Day_Sentiment"

        if (
            day_col in engineered_data.columns
            and sentiment_col in engineered_data.columns
        ):
            try:
                engineered_data = engineered_data.with_columns(
                    (
                        pl.col(day_col).cast(pl.String)  # Use pl.String
                        + "_"
                        + pl.col(sentiment_col).cast(pl.String)  # Use pl.String
                    ).alias(new_col_name)
                )
                logging.info(f"Successfully engineered feature: '{new_col_name}'")
            except Exception as e:
                logging.warning(
                    f"Could not engineer '{new_col_name}' feature: {e}. Skipping."
                )
        else:
            logging.info(
                f"Skipping '{new_col_name}' engineering: Columns '{day_col}' or '{sentiment_col}' not found."
            )

        # Add more feature engineering steps here as needed...
        # Example: Extract hour from a datetime column 'Timestamp'
        # timestamp_col = "Timestamp"
        # if timestamp_col in engineered_data.columns:
        #     try:
        #         engineered_data = engineered_data.with_columns(
        #             pl.col(timestamp_col).dt.hour().alias("Hour_Of_Day")
        #         )
        #         logging.info("Successfully engineered feature: 'Hour_Of_Day'")
        #     except Exception as e:
        #          logging.warning(f"Could not engineer 'Hour_Of_Day': {e}. Skipping.")

        return engineered_data

    def fit(self, data: pl.DataFrame):
        """
        Fits the preprocessor on the training data.
        Identifies numerical features and calculates imputation values (median).

        Args:
            data (pl.DataFrame): The training dataframe.
        """
        logging.info(f"Fitting preprocessor on data with shape {data.shape}...")

        # --- Reset state from previous fits ---
        self.imputation_values_ = {}
        self.numerical_features_for_imputation_ = []

        # --- Prepare data for fitting ---
        # Work on a copy to avoid modifying the original DataFrame passed to fit
        data_to_fit = data.clone()

        # Drop rows with missing target *before* fitting imputation values
        initial_height = data_to_fit.height
        data_to_fit = data_to_fit.drop_nulls(subset=[self.target_column])
        rows_dropped = initial_height - data_to_fit.height
        if rows_dropped > 0:
            logging.warning(
                f"Dropped {rows_dropped} rows with missing target "
                f"('{self.target_column}') before fitting."
            )
        if data_to_fit.is_empty():
            raise ValueError(
                "No data remaining after dropping rows with missing target. Cannot fit."
            )

        # --- Identify Numerical Features for Imputation ---
        # Select all numeric columns except the target column
        self.numerical_features_for_imputation_ = data_to_fit.select(
            pl.col(pl.NUMERIC_DTYPES).exclude(self.target_column)
        ).columns
        logging.info(
            f"Identified numerical features for imputation: {self.numerical_features_for_imputation_}"
        )

        # --- Calculate Imputation Values (Median) ---
        if not self.numerical_features_for_imputation_:
            logging.info("No numerical features found for imputation.")
        else:
            median_exprs = [
                pl.median(col).alias(col)
                for col in self.numerical_features_for_imputation_
            ]
            # Calculate medians safely
            try:
                medians_df = data_to_fit.select(median_exprs)
                medians_dict = medians_df.to_dicts()[0]  # Get medians as a dictionary
            except Exception as e:
                logging.error(
                    f"Error calculating medians: {e}. Imputation values might be incorrect.",
                    exc_info=True,
                )
                medians_dict = {}  # Fallback to empty dict

            for col in self.numerical_features_for_imputation_:
                median_val = medians_dict.get(col)
                if median_val is not None:
                    self.imputation_values_[col] = median_val
                else:
                    # Handle cases where median might be null (e.g., all-null column or error during calculation)
                    self.imputation_values_[col] = 0.0  # Default imputation value
                    logging.warning(
                        f"Median for numerical column '{col}' could not be calculated or is null. "
                        f"Using 0.0 for imputation."
                    )
            logging.info(f"Calculated imputation medians: {self.imputation_values_}")

        # No explicit fitting needed for Polars categorical casting (Label Encoding)
        # or standard scaling (scaler would be fit here if used)
        logging.info("Preprocessor fitting complete.")

    def transform(
        self, data: pl.DataFrame, drop_original_categoricals: bool = False
    ) -> pl.DataFrame:
        """
        Transforms the data using the fitted preprocessor steps:
        1. Applies feature engineering.
        2. Imputes missing numerical values using fitted medians.
        3. Applies label encoding to all string/categorical columns.
        4. (Optional) Applies scaling to numerical features.

        Args:
            data (pl.DataFrame): The dataframe to transform (train, validation, or test).
            drop_original_categoricals (bool): If True, drops the original string columns
                                               after label encoding. Defaults to False.

        Returns:
            pl.DataFrame: The transformed dataframe.
        """
        if not self.imputation_values_ and self.numerical_features_for_imputation_:
            logging.warning(
                "Transform called before fit, or fit found no numerical features. Imputation might not work as expected."
            )

        logging.info(f"Transforming data with shape {data.shape}...")
        transformed_data = data.clone()  # Work on a copy

        # --- 1. Apply Feature Engineering ---
        # Feature engineering might add or modify columns
        transformed_data = self._engineer_features(transformed_data)
        logging.info(f"Shape after feature engineering: {transformed_data.shape}")

        # --- 2. Apply Imputation to Numerical Features ---
        if self.imputation_values_:
            impute_exprs = [
                pl.col(col)
                .fill_null(
                    self.imputation_values_.get(
                        col, 0.0
                    )  # Use fitted median or default 0.0
                )
                .alias(col)  # Ensure the original column name is kept
                for col in self.numerical_features_for_imputation_
                if col in transformed_data.columns  # Apply only if column exists
            ]
            if impute_exprs:
                transformed_data = transformed_data.with_columns(impute_exprs)
                logging.info(
                    f"Applied median imputation to numerical columns: "
                    f"{[col for col in self.numerical_features_for_imputation_ if col in transformed_data.columns]}"
                )
        else:
            logging.info(
                "Skipping numerical imputation (no values calculated during fit or no numerical columns found)."
            )

        # --- 3. Apply Label Encoding to String/Categorical Features ---
        # Identify all string or categorical columns *currently* in the dataframe, excluding the target
        # This automatically includes original categoricals and any engineered string features.
        string_like_cols_to_encode = transformed_data.select(
            pl.col(pl.String, pl.Categorical).exclude(self.target_column)
        ).columns

        if string_like_cols_to_encode:
            encode_exprs = [
                # Cast to Categorical, then get the underlying physical representation (integer)
                pl.col(col).cast(pl.Categorical).to_physical().alias(f"{col}_LE")
                for col in string_like_cols_to_encode
            ]
            transformed_data = transformed_data.with_columns(encode_exprs)
            logging.info(
                f"Applied Label Encoding (Categorical -> Physical) to: {string_like_cols_to_encode}"
            )

            if drop_original_categoricals:
                # Drop original string columns only if requested
                cols_to_drop = [
                    col
                    for col in string_like_cols_to_encode
                    if col in transformed_data.columns
                ]
                if cols_to_drop:
                    transformed_data = transformed_data.drop(
                        cols_to_drop
                    )  # Use 'columns' argument
                    logging.info(
                        f"Dropped original categorical columns: {cols_to_drop}"
                    )
        else:
            logging.info("No string/categorical columns found for label encoding.")

        # --- 4. Add Scaling Here (Optional) ---
        # Example using a hypothetical StandardScaler-like logic for Polars
        # Note: Scaler should be fitted in the `fit` method on training data only.
        # numerical_cols_for_scaling = self.numerical_features_for_imputation_ # Or define separately
        # if hasattr(self, 'scaler_means_') and hasattr(self, 'scaler_stds_'): # Check if scaler was fitted
        #    scale_exprs = []
        #    for col in numerical_cols_for_scaling:
        #        if col in transformed_data.columns and col in self.scaler_means_:
        #             mean = self.scaler_means_[col]
        #             std = self.scaler_stds_[col]
        #             if std != 0: # Avoid division by zero
        #                  scale_exprs.append(((pl.col(col) - mean) / std).alias(f"{col}_scaled"))
        #             else:
        #                  scale_exprs.append(pl.lit(0.0).alias(f"{col}_scaled")) # Handle zero std dev
        #                  logging.warning(f"Std deviation for '{col}' is zero. Scaled feature set to 0.")
        #    if scale_exprs:
        #        transformed_data = transformed_data.with_columns(scale_exprs)
        #        logging.info(f"Applied scaling to: {[expr.meta.output_name() for expr in scale_exprs]}")
        #        # Optionally drop original numerical columns used for scaling
        #        # transformed_data = transformed_data.drop(numerical_cols_for_scaling)

        logging.info(
            f"Transformation complete. Final data shape: {transformed_data.shape}"
        )
        return transformed_data

    def fit_transform_split(
        self,
        data: pl.DataFrame,
        test_size: float = 0.2,
        drop_original_categoricals_in_transform: bool = False,
    ) -> Tuple[pl.DataFrame, pl.DataFrame]:
        """
        Splits the data into training and validation sets, fits the preprocessor
        on the training set, and transforms both sets.

        Args:
            data (pl.DataFrame): The full dataset to split and process.
            test_size (float): The proportion of the dataset to include in the validation split.
                               Defaults to 0.2.
            drop_original_categoricals_in_transform (bool): Passed to the transform method.
                                                            If True, drops original string columns
                                                            after encoding. Defaults to False.


        Returns:
            Tuple[pl.DataFrame, pl.DataFrame]: A tuple containing the transformed
                                               training dataframe and the transformed
                                               validation dataframe.
        """
        logging.info(f"Performing fit, transform, and split (test_size={test_size})...")

        if not 0 < test_size < 1:
            raise ValueError("test_size must be between 0 and 1.")

        # --- Drop rows with missing target *before* splitting ---
        data_cleaned = data.drop_nulls(subset=[self.target_column])
        rows_dropped = data.height - data_cleaned.height
        if rows_dropped > 0:
            logging.warning(
                f"Dropped {rows_dropped} rows with missing target variable "
                f"('{self.target_column}') before splitting."
            )
        if data_cleaned.is_empty():
            raise ValueError(
                "No data remaining after dropping rows with missing target. Cannot split."
            )

        # --- Robust Split using anti-join ---
        # Add a temporary index for reliable splitting using with_row_index
        temp_index_col = "__temp_idx__"
        data_with_idx = data_cleaned.with_row_index(
            name=temp_index_col
        )  # Use with_row_index

        # Sample indices for the training set using 'fraction' argument
        train_indices = data_with_idx.select(temp_index_col).sample(
            fraction=(1.0 - test_size),
            shuffle=True,
            seed=self.random_seed,  # Use fraction instead of frac
        )

        # Create training data by joining on the sampled indices
        train_data = data_with_idx.join(
            train_indices, on=temp_index_col, how="inner"
        ).drop(temp_index_col)

        # Create validation data using an anti-join to get the remaining rows
        val_data = data_with_idx.join(
            train_indices, on=temp_index_col, how="anti"
        ).drop(temp_index_col)
        # ------------------------------------

        logging.info(
            f"Split complete: Train shape {train_data.shape}, Validation shape {val_data.shape}"
        )

        # --- Fit the preprocessor ONLY on the training data ---
        self.fit(train_data)

        # --- Transform both training and validation data ---
        logging.info("Transforming training data...")
        train_transformed = self.transform(
            train_data,
            drop_original_categoricals=drop_original_categoricals_in_transform,
        )

        logging.info("Transforming validation data...")
        val_transformed = self.transform(
            val_data, drop_original_categoricals=drop_original_categoricals_in_transform
        )

        return train_transformed, val_transformed
    
class CatBoostModel:
    """Wraps a CatBoost Regressor model."""

    def __init__(self, params: Dict[str, Any]):
        """Initializes the CatBoost Regressor model.

        Args:
            params (Dict[str, Any]): Parameters for the CatBoostRegressor.
                                     It's recommended to include 'random_seed',
                                     'loss_function' (e.g., 'RMSE'), 'eval_metric' (e.g., 'RMSE'),
                                     'early_stopping_rounds'.
        """
        self.params = params
        # Ensure essential params are present or set defaults
        self.params.setdefault('loss_function', 'RMSE')
        self.params.setdefault('eval_metric', 'RMSE')
        self.params.setdefault('random_seed', 42) # Ensure reproducibility
        self.model = CatBoostRegressor(**self.params)
        self.categorical_features_indices_: Optional[List[int]] = None

    def _prepare_data(self, data: pl.DataFrame, target_column: Optional[str] = None) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        """Converts Polars DataFrame to Pandas and separates features/target."""
        if target_column and target_column in data.columns:
            X = data.drop(target_column).to_pandas()
            y = data[target_column].to_pandas()
            return X, y
        else:
            # If no target column (e.g., for prediction on new data)
            X = data.to_pandas()
            return X, None

    def _identify_categorical_features(self, X: pd.DataFrame):
        """Identifies categorical features based on '_LE' suffix or object/category dtype."""
        # Prioritize columns ending with _LE as created by our preprocessor
        le_cols = [col for col in X.columns if col.endswith('_LE')]
        if le_cols:
             self.categorical_features_indices_ = [X.columns.get_loc(col) for col in le_cols]
             logging.info(f"Identified Label Encoded features for CatBoost: {le_cols}")
        else:
             # Fallback: Identify object or category columns if no _LE suffix found
             cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
             if cat_cols:
                 self.categorical_features_indices_ = [X.columns.get_loc(col) for col in cat_cols]
                 logging.info(f"Identified object/category features for CatBoost: {cat_cols}")
             else:
                 self.categorical_features_indices_ = [] # No categorical features found
                 logging.info("No specific categorical features identified for CatBoost.")


    def fit(self,
            train_data: pl.DataFrame,
            target_column: str,
            val_data: Optional[pl.DataFrame] = None,
            verbose: bool = True):
        """Fits the model on training data, optionally using validation data for early stopping.

        Args:
            train_data (pl.DataFrame): The training data (features + target).
            target_column (str): The name of the target variable column.
            val_data (Optional[pl.DataFrame]): The validation data (features + target). Defaults to None.
            verbose (bool): Whether to print CatBoost training progress. Defaults to True.
        """
        logging.info("Preparing data for fitting...")
        X_train, y_train = self._prepare_data(train_data, target_column)

        if X_train is None or y_train is None:
             raise ValueError("Training data preparation failed.")

        # Identify categorical features from the training features
        self._identify_categorical_features(X_train)

        eval_set = None
        if val_data is not None:
            logging.info("Preparing validation data...")
            X_val, y_val = self._prepare_data(val_data, target_column)
            if X_val is None or y_val is None:
                 logging.warning("Validation data preparation failed. Proceeding without eval_set.")
            else:
                 # Ensure validation columns match training columns (order and presence)
                 X_val = X_val[X_train.columns] # Reorder/select columns to match training
                 eval_set = (X_val, y_val)
                 logging.info("Validation set prepared.")

        logging.info(f"Starting model training with {X_train.shape[1]} features...")
        self.model.fit(
            X_train, y_train,
            cat_features=self.categorical_features_indices_,
            eval_set=eval_set,
            verbose=verbose,
            # early_stopping_rounds parameter is taken from self.params if provided
        )
        logging.info("Model training complete.")

    def predict(self, data: pl.DataFrame) -> List[float]:
        """Generates predictions for the input data.

        Args:
            data (pl.DataFrame): Data containing features (target column optional, will be ignored).

        Returns:
            List[float]: The predicted values.
        """
        logging.info("Generating predictions...")
        # Prepare data (target column doesn't matter here)
        X, _ = self._prepare_data(data)

        # Ensure prediction columns match training columns used during fit
        if hasattr(self.model, 'feature_names_') and self.model.feature_names_:
             try:
                 X = X[self.model.feature_names_] # Reorder/select columns
             except KeyError as e:
                  raise ValueError(f"Prediction data missing columns required by the model: {e}")

        predictions = self.model.predict(X)
        logging.info(f"Generated {len(predictions)} predictions.")
        return predictions.tolist() # Return as list

    def evaluate(self, data: pl.DataFrame, target_column: str) -> float:
         """Evaluates the model on the given data using the primary eval metric (e.g., RMSE)."""
         logging.info("Evaluating model...")
         X, y_true = self._prepare_data(data, target_column)
         if X is None or y_true is None:
              raise ValueError("Evaluation data preparation failed.")

         # Ensure evaluation columns match training columns
         if hasattr(self.model, 'feature_names_') and self.model.feature_names_:
             try:
                 X = X[self.model.feature_names_]
             except KeyError as e:
                  raise ValueError(f"Evaluation data missing columns required by the model: {e}")

         y_pred = self.model.predict(X)
         # Use a common regression metric, adapt if needed
         score = mean_squared_error(y_true, y_pred, squared=False) # Get RMSE
         logging.info(f"Evaluation complete. RMSE: {score:.4f}")
         return score


    def save_model(self, model_path: str):
        """Saves the trained model to a file."""
        logging.info(f"Saving model to {model_path}...")
        path = Path(model_path)
        path.parent.mkdir(parents=True, exist_ok=True) # Ensure directory exists
        self.model.save_model(str(path))
        logging.info("Model saved successfully.")

    def load_model(self, model_path: str):
        """Loads a model from a file."""
        logging.info(f"Loading model from {model_path}...")
        if not Path(model_path).exists():
             raise FileNotFoundError(f"Model file not found at {model_path}")
        self.model.load_model(model_path)
        # It's good practice to re-identify categorical features if possible,
        # although CatBoost often stores this info internally.
        logging.info("Model loaded successfully.")

if __name__ == "__main__":
    # Initialize the task
    # task = Task.init(project_name="s5_e4", task_name="Podcast Listening Time Analysis")
    # task.set_base_task("s5_e4_podcast_listening_time")
    # task.connect(Config())

    # Load the data
    data_path = Path(Config.data_path)
    data = pl.read_csv(f"{data_path}/train.csv")
    # Print data summary
    print(data.describe())
    # print column names
    print(data.columns)

    # Perform EDA
    eda = EDA(data)
    eda.print_statistics()
    eda.print_features()
    eda.plot_listening_time_distribution()
    eda.plot_listening_time_by_podcast()

    preprocessor = DataPreprocessor(
        target_column="Listening_Time_minutes", random_seed=123
    )
    try:
        train_df, val_df = preprocessor.fit_transform_split(data, test_size=0.2, drop_original_categoricals_in_transform=True)
        logging.info(f"Preprocessing and split finished.")
        logging.info(f"Train data shape after preprocessing: {train_df.shape}")
        logging.info(f"Validation data shape after preprocessing: {val_df.shape}")
        logging.info(f"Train columns: {train_df.columns}")
        logging.info(f"Validation columns: {val_df.columns}")
    except ValueError as e:
        logging.error(f"Error during preprocessing: {e}")

    except Exception as e:
        logging.error(f"An error occurred: {e}")

    plt.show()
    
    config = Config() 
    catboost_params = {
        'iterations': 1000,
        'learning_rate': 0.05,
        'depth': 6,
        'l2_leaf_reg': 3,
        'loss_function': 'RMSE',
        'eval_metric': 'RMSE',
        'random_seed': config.random_seed,
        'early_stopping_rounds': 50 # Stop if eval metric doesn't improve for 50 rounds
    }

    model = CatBoostModel(params=catboost_params)

    try:
        # Fit the model using training and validation data
        model.fit(
            train_data=train_df,
            target_column=preprocessor.target_column,
            val_data=val_df, # Pass validation data here
            verbose=100 # Print progress every 100 iterations
        )

        # Evaluate on validation set
        val_rmse = model.evaluate(val_df, preprocessor.target_column)
        logging.info(f"Final Validation RMSE: {val_rmse:.4f}")

        # Save the trained model
        model.save_model(config.model_path)

    except Exception as e:
        logging.error(f"An error occurred during model training or evaluation: {e}", exc_info=True)
    
    # Use the model to predict on test data and save results
    # Submission File
    # For each id in the test set, you must predict the Listening_Time_minutes of the podcast. The file should contain a header and have the following format:
    # id,Listening_Time_minutes
    # 750000,45.437
    # 750001,45.437
    # 750002,45.437
    # etc.
    test_data_raw = pl.read_csv(f"{data_path}/test.csv")
    logging.info(f"Raw test data loaded. Shape: {test_data_raw.shape}")

    test_data_transformed = preprocessor.transform(
        test_data_raw, drop_original_categoricals=True
    )
    logging.info(
        f"Test data shape after preprocessing: {test_data_transformed.shape}"
    )

    predictions = model.predict(test_data_transformed)

    submission_df = pl.DataFrame({
        "id": test_data_raw["id"],
        "Listening_Time_minutes": predictions
    })

    submission_path = Path(f"{data_path}/submission.csv")
    submission_df.write_csv(submission_path)
    logging.info(f"Submission file created successfully at: {submission_path}")
    print(f"\nSubmission file preview:\n{submission_df.head()}")
    

