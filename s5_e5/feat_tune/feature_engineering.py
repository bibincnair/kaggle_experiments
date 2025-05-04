import featuretools as ft
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.decomposition import PCA


def manual_features(
    df: pd.DataFrame,
    add_bmi=True,
    use_bmi_only=True,
    add_log_duration=True,
    add_temp_elevation=True,
) -> pd.DataFrame:
    df = df.copy()
    if add_temp_elevation:
        df["Temp_Elevation"] = df["Body_Temp"] - 37.0
    if add_bmi:
        df["Height_m"] = df["Height"] / 100
        df["BMI"] = df["Weight"] / (df["Height_m"] ** 2 + 1e-6)
        df.drop(columns=["Height_m"], inplace=True)
        if use_bmi_only:
            df.drop(columns=["Height", "Weight"], errors="ignore", inplace=True)
    if add_log_duration:
        df["Log_Duration"] = np.log1p(df["Duration"])
    return df


def run_featuretools_dfs(
    df: pd.DataFrame,
    entity_id: str = "exercises",
    index_col: str = "id",
    primitives: list = None,
    n_jobs: int = -1,
) -> pd.DataFrame:
    """
    Single‐table DFS with only pairwise multiply/divide + absolute,
    faster and less explosive than all four primitives.
    """
    if primitives is None:
        primitives = ["multiply_numeric", "divide_numeric", "absolute"]
    df = df.copy()
    # ensure we can make a unique index
    if not df[index_col].is_unique:
        df = df.reset_index(drop=True).rename(columns={"index": index_col})
    es = ft.EntitySet(id="es")
    numeric_cols = df.select_dtypes(include="number").columns.drop(index_col).tolist()

    var_types = {c: ft.variable_types.Numeric for c in numeric_cols}
    es = es.add_dataframe(
        dataframe_name=entity_id,
        dataframe=df,
        index=index_col,
        variable_types=var_types,
        make_index=False,
    )
    fm, _ = ft.dfs(
        entityset=es,
        target_dataframe_name=entity_id,
        agg_primitives=[],
        trans_primitives=primitives,
        max_depth=1,
        verbose=False,
        n_jobs=n_jobs,
    )
    return fm


def reduce_dimensionality(
    train: pd.DataFrame,
    test: pd.DataFrame,
    variance_threshold=0.95,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, PCA]:
    """
    Fit PCA on train, keep components covering `variance_threshold` of variance,
    transform both train/test.
    """
    cols = train.columns.drop(["id", "Calories"], errors="ignore")
    pca = PCA(n_components=variance_threshold, random_state=random_state)
    X_tr = pca.fit_transform(train[cols])
    X_te = pca.transform(test[cols])
    df_tr = pd.DataFrame(
        X_tr, columns=[f"PC{i+1}" for i in range(X_tr.shape[1])], index=train.index
    )
    df_te = pd.DataFrame(X_te, columns=df_tr.columns, index=test.index)
    # reattach id/target
    if "Calories" in train:
        df_tr["Calories"] = train["Calories"].values
    if "id" in train:
        df_tr["id"] = train["id"].values
        df_te["id"] = test["id"].values
    return df_tr, df_te, pca
