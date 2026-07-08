from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


BIO_FEATURES: tuple[str, ...] = tuple(
    item.lower()
    for name in [
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
    ]
    for item in (name, f"{name}_diff")
)

PHYSICAL_FEATURES: tuple[str, ...] = tuple(
    item.lower()
    for name in ["WEIGHT", "BMI", "WAIST_AVG", "SBP", "DBP"]
    for item in (name, f"{name}_diff")
)

LIFESTYLE_FEATURES: tuple[str, ...] = tuple(
    item.lower()
    for name in [
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
    ]
    for item in (name, f"{name}_diff")
)

NON_INVASIVE_FEATURES: tuple[str, ...] = PHYSICAL_FEATURES + LIFESTYLE_FEATURES

ALL_MODELED_FEATURES: tuple[str, ...] = (
    BIO_FEATURES + PHYSICAL_FEATURES + LIFESTYLE_FEATURES
)

FEATURE_GROUPS: dict[str, tuple[str, ...]] = {
    "bio": BIO_FEATURES,
    "physical": PHYSICAL_FEATURES,
    "life": LIFESTYLE_FEATURES,
    "Non_invasive": NON_INVASIVE_FEATURES,
    "all": ALL_MODELED_FEATURES,
}


def _existing_columns(data: pd.DataFrame, columns: tuple[str, ...] | list[str]) -> list[str]:
    """Return columns that are present in the dataframe, preserving order."""
    return [col for col in columns if col in data.columns]


def feature_selection(data: pd.DataFrame, feature_: str) -> pd.DataFrame:
    """
    Select one feature set for model training.

    Parameters
    ----------
    data:
        Input dataframe containing engineered features, `rid`, and `label`.

    feature_:
        Feature-set name. Supported values are:
        - "all"
        - "bio"
        - "physical"
        - "life"
        - "Non_invasive"

    Returns
    -------
    pandas.DataFrame
        Dataframe with irrelevant feature groups removed. Columns not included
        in the predefined feature groups, such as `rid` and `label`, are
        preserved.
    """
    if feature_ not in FEATURE_GROUPS:
        raise ValueError(
            f"Unsupported feature set: {feature_}. "
            f"Supported values are {list(FEATURE_GROUPS)}."
        )

    data = data.copy()

    if feature_ == "all":
        return data

    if feature_ == "bio":
        drop_cols = PHYSICAL_FEATURES + LIFESTYLE_FEATURES
    elif feature_ == "physical":
        drop_cols = BIO_FEATURES + LIFESTYLE_FEATURES
    elif feature_ == "life":
        drop_cols = BIO_FEATURES + PHYSICAL_FEATURES
    elif feature_ == "Non_invasive":
        drop_cols = BIO_FEATURES
    else:
        drop_cols = ()

    return data.drop(columns=_existing_columns(data, drop_cols))


def apply_pca(df: pd.DataFrame, prefix: str, max_components: int = 10) -> pd.DataFrame:
    """
    Fit PCA to a feature block and return principal-component columns.

    The number of components is the smaller of:
    - `max_components`
    - the minimum number of components explaining at least 90% variance
    - the number of available input columns/samples

    Parameters
    ----------
    df:
        Numeric dataframe for one feature block.

    prefix:
        Prefix used for PCA column names.

    max_components:
        Maximum number of PCA components.

    Returns
    -------
    pandas.DataFrame
        PCA-transformed dataframe with the same index as `df`.
    """
    if df.empty:
        return pd.DataFrame(index=df.index)

    df_numeric = df.apply(pd.to_numeric, errors="coerce")

    if df_numeric.isna().any().any():
        raise ValueError(
            f"PCA input for '{prefix}' contains missing or non-numeric values. "
            "Run imputation before PCA or set pca mode to 'none'."
        )

    n_samples, n_features = df_numeric.shape
    max_allowed_components = min(max_components, n_samples, n_features)

    if max_allowed_components < 1:
        return pd.DataFrame(index=df.index)

    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df_numeric)

    pca_full = PCA(n_components=max_allowed_components)
    pca_full.fit(df_scaled)

    explained_variance = np.cumsum(pca_full.explained_variance_ratio_)
    n_components_90 = int(np.argmax(explained_variance >= 0.9) + 1)
    n_components = min(max_allowed_components, n_components_90)

    pca = PCA(n_components=n_components)
    principal_components = pca.fit_transform(df_scaled)

    pca_columns = [f"{prefix}_pca_{i + 1}" for i in range(n_components)]

    return pd.DataFrame(principal_components, columns=pca_columns, index=df.index)


def pca_df(data: pd.DataFrame, pca_: str, features_: str) -> pd.DataFrame:
    """
    Add or replace feature groups with PCA-derived components.

    Parameters
    ----------
    data:
        Input dataframe after feature-set selection.

    pca_:
        PCA mode. Supported values are:
        - "none": return data unchanged
        - "plus": append PCA columns while keeping original columns
        - "only": replace the selected feature groups with PCA columns

    features_:
        Feature-set name used to determine which feature blocks are transformed.

    Returns
    -------
    pandas.DataFrame
        Dataframe with PCA columns added or substituted.
    """
    if pca_ not in {"none", "plus", "only"}:
        raise ValueError("pca_ must be one of 'none', 'plus', or 'only'.")

    if features_ not in FEATURE_GROUPS:
        raise ValueError(
            f"Unsupported feature set: {features_}. "
            f"Supported values are {list(FEATURE_GROUPS)}."
        )

    data = data.copy()

    if pca_ == "none":
        return data

    pca_blocks: list[pd.DataFrame] = []
    drop_cols: list[str] = []

    if features_ == "all":
        groups_to_transform = {
            "bio": BIO_FEATURES,
            "physical": PHYSICAL_FEATURES,
            "life": LIFESTYLE_FEATURES,
        }
    else:
        groups_to_transform = {features_: FEATURE_GROUPS[features_]}

    for prefix, columns in groups_to_transform.items():
        cols = _existing_columns(data, columns)

        if not cols:
            continue

        pca_blocks.append(apply_pca(data[cols], prefix=prefix))

        if pca_ == "only":
            drop_cols.extend(cols)

    if pca_ == "only" and drop_cols:
        data = data.drop(columns=drop_cols)

    if pca_blocks:
        df_pca = pd.concat(pca_blocks, axis=1)
        data = pd.concat([data, df_pca], axis=1)

    return data


def data_split(data_input, sampling):
    """
    Split the input dataframe into train, test, and validation sets using
    participant-level splitting based on the `rid` column.

    This function prevents participant-level data leakage by ensuring that
    the same `rid` does not appear in more than one split.

    Split ratio:
        - Train: 80%
        - Test: 10%
        - Validation: 10%

    Parameters
    ----------
    data_input : pandas.DataFrame
        Input dataframe. It must contain the following columns:
        - rid   : participant identifier
        - label : binary target variable encoded as 0/1

    sampling : str
        Sampling strategy.
        - "over": oversample positive-class rows within each split
        - Any other value: no resampling

    Returns
    -------
    x_train, y_train, x_test, y_test, x_validation, y_validation
        Feature matrices and target vectors for train, test, and validation.
        The `rid`, `label`, and internal temporary label column are removed
        from the feature matrices.
    """

    random_state = 123
    label_tmp_col = "__label_num_for_split__"

    categorical_cols = [
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

    oversampling_targets = {
        "train": 36679,
        "test": 4584,
        "validation": 4584,
    }

    data_input = data_input.copy()

    if "rid" not in data_input.columns:
        raise ValueError("The input dataframe must contain a 'rid' column.")

    if "label" not in data_input.columns:
        raise ValueError("The input dataframe must contain a 'label' column.")

    if data_input["rid"].isna().any():
        raise ValueError(
            "The 'rid' column contains missing values. "
            "Missing participant IDs must be handled before splitting."
        )

    if label_tmp_col in data_input.columns:
        raise ValueError(
            f"Temporary column name conflict: '{label_tmp_col}' already exists."
        )

    def clean_nominal_category_columns(df, categorical_cols):
        """
        Convert nominal categorical columns to pandas category dtype.

        Numeric-looking categories such as 1.0 and 2.0 are converted to
        string categories "1" and "2" so that nominal variables are not
        treated as continuous numeric variables.
        """
        df = df.copy()

        for col in categorical_cols:
            if col not in df.columns:
                continue

            s = df[col]
            non_null = s.dropna()

            if pd.api.types.is_numeric_dtype(s):
                if len(non_null) == 0:
                    df[col] = s.astype("string")
                elif ((non_null % 1) == 0).all():
                    df[col] = s.astype("Int64").astype("string")
                else:
                    df[col] = s.astype("string")
            else:
                temp = pd.to_numeric(s, errors="coerce")
                temp_non_null = temp.dropna()

                all_non_null_values_are_numeric = temp.notna().sum() == non_null.shape[0]
                all_numeric_values_are_integer_like = (
                    len(temp_non_null) > 0
                    and ((temp_non_null % 1) == 0).all()
                )

                if all_non_null_values_are_numeric and all_numeric_values_are_integer_like:
                    df[col] = temp.astype("Int64").astype("string")
                else:
                    df[col] = s.astype("string")

            df[col] = df[col].astype("category")

        return df

    def safe_train_test_split(df, test_size, random_state):
        """
        Split RID-level data with stratification when possible.

        Stratification is based on the RID-level label-composition group:
        only_0, only_1, or mixed. If stratification is not possible because
        some strata have too few RIDs, the function falls back to a regular
        RID-level split without stratification.
        """
        if df.empty:
            raise ValueError("Cannot split an empty RID summary dataframe.")

        n_samples = df.shape[0]

        if n_samples < 2:
            raise ValueError(
                f"At least 2 unique RIDs are required for splitting. "
                f"Current number of unique RIDs: {n_samples}"
            )

        strata_counts = df["strata"].value_counts()
        n_classes = strata_counts.shape[0]

        if isinstance(test_size, float):
            n_test = int(np.ceil(n_samples * test_size))
        else:
            n_test = int(test_size)

        n_train = n_samples - n_test

        if n_train <= 0 or n_test <= 0:
            raise ValueError(
                f"Invalid split size. n_samples={n_samples}, "
                f"n_train={n_train}, n_test={n_test}, test_size={test_size}"
            )

        can_stratify = (
            n_classes >= 2
            and strata_counts.min() >= 2
            and n_train >= n_classes
            and n_test >= n_classes
        )

        stratify = df["strata"] if can_stratify else None

        train_df, test_df = train_test_split(
            df,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify,
        )

        return train_df.reset_index(drop=True), test_df.reset_index(drop=True)

    def oversample_positive(df, n_positive, random_state):
        """
        Oversample positive-class rows within a single split.

        All negative rows are preserved. Positive rows are sampled with
        replacement until the requested positive-class sample size is reached.
        """
        if n_positive <= 0:
            raise ValueError(
                f"n_positive must be a positive integer. Current value: {n_positive}"
            )

        df_0 = df[df[label_tmp_col] == 0]
        df_1 = df[df[label_tmp_col] == 1]

        if len(df_1) == 0:
            raise ValueError(
                "Positive-class oversampling was requested, but this split "
                "contains no rows with label 1."
            )

        df_1_over = df_1.sample(
            n=n_positive,
            random_state=random_state,
            replace=True,
        )

        return pd.concat([df_0, df_1_over], axis=0)

    # ------------------------------------------------------------------
    # Validate and normalize the binary target label.
    # ------------------------------------------------------------------
    label_numeric = pd.to_numeric(
        data_input["label"].astype("string"),
        errors="raise",
    )

    if label_numeric.isna().any():
        raise ValueError("The 'label' column contains missing values.")

    if not ((label_numeric % 1) == 0).all():
        invalid_values = sorted(label_numeric[~((label_numeric % 1) == 0)].unique())
        raise ValueError(
            "The 'label' column must contain integer-like binary values. "
            f"Invalid values: {invalid_values}"
        )

    data_input[label_tmp_col] = label_numeric.astype(int)

    valid_labels = set(data_input[label_tmp_col].unique())

    if not valid_labels.issubset({0, 1}):
        raise ValueError(
            "The 'label' column must be binary and encoded as 0/1. "
            f"Current label values: {sorted(valid_labels)}"
        )

    data_input["label"] = data_input[label_tmp_col].astype(int)

    # ------------------------------------------------------------------
    # Normalize nominal categorical variables.
    # ------------------------------------------------------------------
    data_input = clean_nominal_category_columns(
        data_input,
        categorical_cols,
    )

    # ------------------------------------------------------------------
    # Build RID-level label-composition strata.
    #
    # only_0 : all rows for the RID have label 0
    # only_1 : all rows for the RID have label 1
    # mixed  : the RID has both label 0 and label 1 rows
    # ------------------------------------------------------------------
    rid_summary = (
        data_input
        .groupby("rid")[label_tmp_col]
        .agg(total="count", positive="sum")
        .reset_index()
    )

    rid_summary["negative"] = rid_summary["total"] - rid_summary["positive"]

    rid_summary["strata"] = np.select(
        [
            rid_summary["positive"] == 0,
            rid_summary["negative"] == 0,
        ],
        [
            "only_0",
            "only_1",
        ],
        default="mixed",
    )

    # ------------------------------------------------------------------
    # First split: train 80%, remaining 20%.
    # Second split: remaining 20% -> test 10%, validation 10%.
    # ------------------------------------------------------------------
    train_rid_df, remain_rid_df = safe_train_test_split(
        rid_summary,
        test_size=0.2,
        random_state=random_state,
    )

    test_rid_df, validation_rid_df = safe_train_test_split(
        remain_rid_df,
        test_size=0.5,
        random_state=random_state,
    )

    train_rids = set(train_rid_df["rid"])
    test_rids = set(test_rid_df["rid"])
    validation_rids = set(validation_rid_df["rid"])

    # ------------------------------------------------------------------
    # Check RID leakage explicitly.
    # Do not use assert here because assertions can be disabled with
    # Python optimization flags.
    # ------------------------------------------------------------------
    if not train_rids.isdisjoint(test_rids):
        raise RuntimeError("RID leakage detected between train and test splits.")

    if not train_rids.isdisjoint(validation_rids):
        raise RuntimeError("RID leakage detected between train and validation splits.")

    if not test_rids.isdisjoint(validation_rids):
        raise RuntimeError("RID leakage detected between test and validation splits.")

    # ------------------------------------------------------------------
    # Extract rows by RID.
    # ------------------------------------------------------------------
    train = data_input[data_input["rid"].isin(train_rids)].copy()
    test = data_input[data_input["rid"].isin(test_rids)].copy()
    validation = data_input[data_input["rid"].isin(validation_rids)].copy()

    # ------------------------------------------------------------------
    # Optional oversampling.
    # The oversampling is performed within each split only, so it does not
    # introduce RID leakage across train, test, and validation sets.
    # ------------------------------------------------------------------
    if sampling == "over":
        train = oversample_positive(
            train,
            n_positive=oversampling_targets["train"],
            random_state=random_state,
        )

        test = oversample_positive(
            test,
            n_positive=oversampling_targets["test"],
            random_state=random_state,
        )

        validation = oversample_positive(
            validation,
            n_positive=oversampling_targets["validation"],
            random_state=random_state,
        )

    # ------------------------------------------------------------------
    # Shuffle each split and reset indices.
    # ------------------------------------------------------------------
    train = train.sample(frac=1, random_state=random_state).reset_index(drop=True)
    test = test.sample(frac=1, random_state=random_state).reset_index(drop=True)
    validation = validation.sample(frac=1, random_state=random_state).reset_index(drop=True)

    # ------------------------------------------------------------------
    # Separate X and y.
    # Remove RID and temporary label columns from feature matrices.
    # ------------------------------------------------------------------
    drop_cols = ["label", "rid", label_tmp_col]

    x_train = train.drop(columns=drop_cols)
    y_train = train["label"].astype(int)

    x_test = test.drop(columns=drop_cols)
    y_test = test["label"].astype(int)

    x_validation = validation.drop(columns=drop_cols)
    y_validation = validation["label"].astype(int)

    return x_train, y_train, x_test, y_test, x_validation, y_validation
