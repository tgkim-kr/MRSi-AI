from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def data_split(
    data_input: pd.DataFrame,
    sampling: str = "over",
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Split data into train / test / validation sets with optional oversampling
    of the positive class to fixed sample sizes (mirrors original script).

    Parameters
    ----------
    data_input : pd.DataFrame
        Input dataframe containing a 'label' column.
    sampling : {"over", "none"}, optional
        If 'over', oversample positive class to fixed sizes.
        If 'none', keep natural class distribution.

    Returns
    -------
    x_train, y_train, x_test, y_test, x_val, y_val
    """
    df = data_input.copy()

    cat_cols = [
        "label",
        "as1_area",
        "as1_edua",
        "v1_fdm_rel",
        "v1_house",
        "as1_sex",
        "pa",
        "urinfqlnc",
        "wafqlnc",
        "smoke",
        "drink",
        "pa_diff",
        "urinfqlnc_diff",
        "wafqlnc_diff",
        "smoke_diff",
        "drink_diff",
    ]
    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].astype("category")

    # Positive class
    data_1 = df[df["label"] == 1].reset_index(drop=True)
    train_1 = data_1.sample(frac=0.8, random_state=123)
    train_re_1 = data_1.drop(train_1.index)
    test_1 = train_re_1.sample(frac=0.5, random_state=123)
    validation_1 = train_re_1.drop(test_1.index)

    if sampling == "over":
        # These numbers are taken from the original code
        train_1 = train_1.sample(n=36679, random_state=123, replace=True)
        test_1 = test_1.sample(n=4584, random_state=123, replace=True)
        validation_1 = validation_1.sample(n=4584, random_state=123, replace=True)

    # Negative class
    data_0 = df[df["label"] == 0].reset_index(drop=True)
    train_0 = data_0.sample(frac=0.8, random_state=123)
    train_re_0 = data_0.drop(train_0.index)
    test_0 = train_re_0.sample(frac=0.5, random_state=123)
    validation_0 = train_re_0.drop(test_0.index)

    # Combine and shuffle
    train = (
        pd.concat([train_0, train_1])
        .sample(frac=1.0, random_state=123)
        .reset_index(drop=True)
    )
    test = (
        pd.concat([test_0, test_1])
        .sample(frac=1.0, random_state=123)
        .reset_index(drop=True)
    )
    validation = (
        pd.concat([validation_0, validation_1])
        .sample(frac=1.0, random_state=123)
        .reset_index(drop=True)
    )

    x_train = train.drop(columns=["label"])
    y_train = train["label"]
    x_test = test.drop(columns=["label"])
    y_test = test["label"]
    x_val = validation.drop(columns=["label"])
    y_val = validation["label"]

    return x_train, y_train, x_test, y_test, x_val, y_val


def feature_selection(data: pd.DataFrame, feature_: str) -> pd.DataFrame:
    """
    Select feature subsets based on a high-level feature set name.

    Parameters
    ----------
    data : pd.DataFrame
        Input dataframe.
    feature_ : {"all","bio","physical","life","Non_invasive"}
        Name of the feature subset to keep.

    Returns
    -------
    pd.DataFrame
        Dataframe with selected features.
    """
    df = data.copy()

    bio_ = []
    for i in [
        "TChl",
        "HDL",
        "TG",
        "ALT",
        "AST",
        "Bun",
        "Creatinine",
        "CRP",
        "WBC",
        "RBC",
        "Plat",
        "Hct",
        "Hb",
    ]:
        bio_.append(i.lower())
        bio_.append(i.lower() + "_diff")

    physical_ = []
    for i in ["WEIGHT", "BMI", "WAIST_AVG", "SBP", "DBP"]:
        physical_.append(i.lower())
        physical_.append(i.lower() + "_diff")

    life_ = []
    for i in [
        "SMOKE",
        "PACKYR_Current",
        "DRINK",
        "INCOME",
        "MEDCST",
        "HTFDCST",
        "URINFQLNC",
        "WAFQLNC",
        "PA",
        "TOTALC",
    ]:
        life_.append(i.lower())
        life_.append(i.lower() + "_diff")

    Non_invasive_ = []
    for i in [
        "SMOKE",
        "PACKYR_Current",
        "DRINK",
        "INCOME",
        "MEDCST",
        "HTFDCST",
        "URINFQLNC",
        "WAFQLNC",
        "PA",
        "TOTALC",
        "WEIGHT",
        "BMI",
        "WAIST_AVG",
        "SBP",
        "DBP",
    ]:
        Non_invasive_.append(i.lower())
        Non_invasive_.append(i.lower() + "_diff")

    if feature_ == "all":
        return df
    elif feature_ == "bio":
        df = df.drop(columns=physical_, errors="ignore").drop(
            columns=life_, errors="ignore"
        )
    elif feature_ == "physical":
        df = df.drop(columns=bio_, errors="ignore").drop(
            columns=life_, errors="ignore"
        )
    elif feature_ == "life":
        df = df.drop(columns=bio_, errors="ignore").drop(
            columns=physical_, errors="ignore"
        )
    elif feature_ == "Non_invasive":
        df = df.drop(columns=bio_, errors="ignore")
    else:
        raise ValueError(f"Unknown feature set: {feature_}")

    return df


def apply_pca(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """
    Apply PCA to a subset of features and keep enough components to explain
    at least 90% of variance, capped at 10 components.

    Parameters
    ----------
    df : pd.DataFrame
        Feature matrix to be transformed.
    prefix : str
        Prefix for generated PCA column names.

    Returns
    -------
    pd.DataFrame
        Dataframe of PCA components.
    """
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df)

    pca = PCA()
    pca.fit(df_scaled)
    explained_variance = np.cumsum(pca.explained_variance_ratio_)
    num_components = min(10, int(np.argmax(explained_variance >= 0.9) + 1))

    pca = PCA(n_components=num_components)
    principal_components = pca.fit_transform(df_scaled)

    pca_columns = [f"{prefix}_pca_{i + 1}" for i in range(num_components)]
    df_pca = pd.DataFrame(principal_components, columns=pca_columns, index=df.index)

    return df_pca


def pca_df(data: pd.DataFrame, pca_: str, features_: str) -> pd.DataFrame:
    """
    Apply PCA to designated feature groups and concatenate/replace them.

    Parameters
    ----------
    data : pd.DataFrame
        Input dataframe.
    pca_ : {"none","plus","only"}
        PCA mode.
    features_ : {"all","bio","physical","life","Non_invasive"}
        Feature set name.

    Returns
    -------
    pd.DataFrame
        Dataframe with PCA-transformed features.
    """
    df = data.copy()
    df_pca = pd.DataFrame(index=df.index)

    bio_ = []
    for i in [
        "TChl",
        "HDL",
        "TG",
        "ALT",
        "AST",
        "Bun",
        "Creatinine",
        "CRP",
        "WBC",
        "RBC",
        "Plat",
        "Hct",
        "Hb",
    ]:
        bio_.append(i.lower())
        bio_.append(i.lower() + "_diff")

    physical_ = []
    for i in ["WEIGHT", "BMI", "WAIST_AVG", "SBP", "DBP"]:
        physical_.append(i.lower())
        physical_.append(i.lower() + "_diff")

    life_ = []
    for i in [
        "SMOKE",
        "PACKYR_Current",
        "DRINK",
        "INCOME",
        "MEDCST",
        "HTFDCST",
        "URINFQLNC",
        "WAFQLNC",
        "PA",
        "TOTALC",
    ]:
        life_.append(i.lower())
        life_.append(i.lower() + "_diff")

    Non_invasive_ = []
    for i in [
        "SMOKE",
        "PACKYR_Current",
        "DRINK",
        "INCOME",
        "MEDCST",
        "HTFDCST",
        "URINFQLNC",
        "WAFQLNC",
        "PA",
        "TOTALC",
        "WEIGHT",
        "BMI",
        "WAIST_AVG",
        "SBP",
        "DBP",
    ]:
        Non_invasive_.append(i.lower())
        Non_invasive_.append(i.lower() + "_diff")

    all_features_ = bio_ + physical_ + life_

    if pca_ == "none":
        return df

    if pca_ in ("plus", "only"):
        if features_ == "bio":
            df_pca = apply_pca(df[bio_], "bio")
            if pca_ == "only":
                df = df.drop(columns=bio_, errors="ignore")
        if features_ == "physical":
            df_pca = apply_pca(df[physical_], "physical")
            if pca_ == "only":
                df = df.drop(columns=physical_, errors="ignore")
        if features_ == "life":
            df_pca = apply_pca(df[life_], "life")
            if pca_ == "only":
                df = df.drop(columns=life_, errors="ignore")
        if features_ == "Non_invasive":
            df_pca = apply_pca(df[Non_invasive_], "Non_invasive")
            if pca_ == "only":
                df = df.drop(columns=Non_invasive_, errors="ignore")
        if features_ == "all":
            df_bio = apply_pca(df[bio_], "bio")
            df_physical = apply_pca(df[physical_], "physical")
            df_life = apply_pca(df[life_], "life")
            df_pca = pd.concat([df_bio, df_physical, df_life], axis=1)
            if pca_ == "only":
                df = df.drop(columns=all_features_, errors="ignore")

        df = pd.concat([df, df_pca], axis=1)
        return df

    raise ValueError(f"Unknown PCA mode: {pca_}")
