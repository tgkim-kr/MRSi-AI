import numpy as np
import pandas as pd
from sas7bdat import SAS7BDAT
from pathlib import Path
from typing import List, Sequence

SENTINEL = 99999

BASE_COLS = ["as1_area", "as1_edua", "v1_fdm_rel", "v1_house", "as1_sex"]

BASE_VARS = [
    "SBP", "DBP", "BMI", "WEIGHT", "WAIST_AVG", "PA", "SMOKE", "PACKYR_Current",
    "DRINK", "TOTALC", "HTFDCST", "MEDCST", "URINFQLNC", "WAFQLNC", "INCOME",
    "ALT", "AST", "Bun", "Creatinine", "Hb", "Hct", "HDL", "CRP",
    "Ins0", "Ins120", "Ins60", "Plat", "RBC", "TChl", "TG", "WBC",
]

GLUCOSE_VARS = ["Glu0", "Glu120", "Hba1c"]

# Variables treated as categorical when computing visit-to-visit transitions
CATEGORICAL_2_LEVEL = ["pa", "urinfqlnc", "wafqlnc"]
CATEGORICAL_3_LEVEL = ["smoke", "drink"]


def _is_extreme_missing(value) -> bool:
    """Return True for NaN or sentinel-like missing values."""
    if pd.isna(value):
        return True
    try:
        return (value > 90000) or (value < -90000)
    except TypeError:
        return False


def _ordered_diff_names() -> List[str]:
    """Return unique longitudinal variable names in a deterministic order."""
    diff_names: List[str] = []
    for visit in range(1, 11):
        for var in BASE_VARS + GLUCOSE_VARS:
            raw = f"V{visit}_{var}"
            name = "_".join(raw.split("_")[1:]).lower()
            if name not in diff_names:
                diff_names.append(name)
    return diff_names


def sas_trans(path: str) -> pd.DataFrame:
    """
    Load a SAS7BDAT file and rescale income-related columns.

    Parameters
    ----------
    path : str
        Path to the .sas7bdat file.

    Returns
    -------
    pd.DataFrame
        Loaded dataframe with rescaled income columns.
    """
    with SAS7BDAT(path) as f:
        data = f.to_data_frame()

    # For V1_INCOME ~ V10_INCOME:
    # - 99999 is treated as sentinel (missing)
    # - other values are divided by 10,000.
    for i in range(1, 11):
        col = f"V{i}_INCOME"
        if col in data.columns:
            mask = data[col] != SENTINEL
            data.loc[mask, col] = data.loc[mask, col] / 10000

    return data


def feature_selection_processing(data: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize column names, select relevant features, and apply deterministic
    visit-wise forward filling.

    Notes
    -----
    This function only performs subject-level deterministic preprocessing.
    It does not estimate imputation parameters from the whole dataset. Mean,
    mode, scaling, and PCA fitting should be performed later after
    participant-level train/validation/test splitting to avoid data leakage.
    """
    data = data.copy()
    data.columns = [c.lower() for c in data.columns]

    age_cols: List[str] = [c for c in data.columns if "_age" in c]

    prefixes = [f"V{i}_" for i in range(1, 11)]
    diff_cols_raw = (
        [p + v for p in prefixes for v in BASE_VARS]
        + [f"V{i}_{v}" for i in range(1, 11) for v in GLUCOSE_VARS]
    )
    diff_cols = [c.lower() for c in diff_cols_raw]

    stat_cols = ["dmcase", "dmstatus_sub"]
    id_cols = ["rid"]

    selected_cols: List[str] = []
    selected_cols.extend(id_cols)
    selected_cols.extend(stat_cols)
    selected_cols.extend(BASE_COLS)
    selected_cols.extend(age_cols)
    selected_cols.extend(diff_cols)

    missing_cols = [c for c in selected_cols if c not in data.columns]
    if missing_cols:
        preview = ", ".join(missing_cols[:20])
        suffix = " ..." if len(missing_cols) > 20 else ""
        raise KeyError(
            f"The input data is missing required columns: {preview}{suffix}"
        )

    df = data[selected_cols].copy()

    # Default dmstatus_sub to 0 if missing.
    df["dmstatus_sub"] = df["dmstatus_sub"].fillna(0)

    # Use a sentinel during deterministic forward filling.
    for col in df.columns:
        df[col] = df[col].fillna(SENTINEL)

    # Forward-fill visit-based variables (v2_*–v10_*) using the previous visit.
    # This is deterministic within each participant and does not use cohort-level
    # summary statistics.
    var_names = _ordered_diff_names()
    for var in var_names:
        for visit in range(2, 11):
            curr_col = f"v{visit}_{var}"
            prev_col = f"v{visit - 1}_{var}"
            if curr_col in df.columns and prev_col in df.columns:
                df[curr_col] = df[curr_col].mask(df[curr_col] == SENTINEL, df[prev_col])

    # Forward-fill age columns (as2_age–as10_age) using the previous visit.
    for visit in range(2, 11):
        curr_col = f"as{visit}_age"
        prev_col = f"as{visit - 1}_age"
        if curr_col in df.columns and prev_col in df.columns:
            df[curr_col] = df[curr_col].mask(df[curr_col] == SENTINEL, df[prev_col])

    return df


def get_diff_code(v1: int, v2: int, levels: Sequence[int]) -> int:
    """
    Encode a (v1, v2) pair into an integer transition code.

    This helper is kept for compatibility with the original script.
    """
    if v1 not in levels or v2 not in levels:
        return SENTINEL
    return levels.index(v1) * len(levels) + levels.index(v2) + 1


def _encode_transition(v1: float, v2: float, levels: Sequence[int]) -> int:
    """Encode categorical visit-to-visit transitions."""
    if _is_extreme_missing(v1) or _is_extreme_missing(v2):
        return SENTINEL

    try:
        v1_int = int(v1)
        v2_int = int(v2)
    except (TypeError, ValueError):
        return SENTINEL

    return get_diff_code(v1_int, v2_int, levels)


def _compute_diff(row: pd.Series, var_name: str, visit: int):
    """
    Compute the visit-to-visit difference or categorical transition.

    Missing values are returned as SENTINEL instead of being converted into
    artificial numeric differences, e.g. 99999 - 99999 = 0.
    """
    curr_col = f"v{visit}_{var_name}"
    next_col = f"v{visit + 1}_{var_name}"

    v_curr = row[curr_col]
    v_next = row[next_col]

    if var_name in CATEGORICAL_2_LEVEL:
        return _encode_transition(v_curr, v_next, levels=[0, 1])

    if var_name in CATEGORICAL_3_LEVEL:
        return _encode_transition(v_curr, v_next, levels=[0, 1, 2])

    if _is_extreme_missing(v_curr) or _is_extreme_missing(v_next):
        return SENTINEL

    return v_next - v_curr


def _get_intervals(dmcase: float, mode: str) -> List[tuple[int, int]]:
    """
    Compute (visit_index, label) pairs for a given dmcase value.

    The logic follows the original analysis script and determines how many
    intervals each participant contributes and with which label.
    """
    intervals: List[tuple[int, int]] = []

    if mode == "diag":
        if dmcase == 1:
            intervals = [(1, 1)]
        elif 1 < dmcase < 10:
            intervals = [(j, 0 if j < dmcase else 1) for j in range(1, int(dmcase) + 1)]
        elif dmcase == 10:
            intervals = [(j, 0) for j in range(1, 10)]
        elif dmcase > 10:
            intervals = [(j, 0) for j in range(1, int(20 - dmcase))]

    elif mode == "drug":
        if dmcase == 2:
            intervals = [(1, 1)]
        elif 2 < dmcase < 10:
            intervals = [(j, 0 if j < dmcase - 1 else 1) for j in range(1, int(dmcase))]
        elif dmcase == 10:
            intervals = [(j, 0) for j in range(1, 10)]
        elif dmcase > 10:
            intervals = [(j, 0) for j in range(1, int(20 - dmcase))]

    else:
        raise ValueError("mode must be either 'diag' or 'drug'.")

    return intervals


def _build_longitudinal_subset(df_sub: pd.DataFrame, mode: str) -> pd.DataFrame:
    """
    Construct a longitudinal dataframe for one endpoint definition.

    Each output row corresponds to one visit interval for one participant.
    The participant identifier `rid` is intentionally preserved so that
    downstream code can split train/validation/test sets at the participant
    level and prevent leakage across repeated observations.
    """
    diff_names = _ordered_diff_names()

    rows: List[dict] = []

    for _, row in df_sub.iterrows():
        dmcase = row["dmcase"]
        intervals = _get_intervals(dmcase, mode)

        for visit, label in intervals:
            out = {
                "rid": row["rid"],
                "data_type": mode,
                "visit": visit,
                "label": label,
                "age": row.get(f"as{visit}_age", SENTINEL),
            }

            for base_col in BASE_COLS:
                out[base_col] = row[base_col]

            for var in diff_names:
                out[var] = row[f"v{visit}_{var}"]
                out[f"{var}_diff"] = _compute_diff(row, var, visit)

            rows.append(out)

    return pd.DataFrame(rows)


def _apply_basic_filters(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply deterministic exclusion criteria used in the original script.

    This filter does not estimate cohort-level statistics; it only removes
    rows lacking required baseline variables.
    """
    if df.empty:
        return df

    mask = pd.Series(True, index=df.index)

    for col in ["as1_edua", "v1_house"]:
        if col in df.columns:
            mask &= ~df[col].apply(_is_extreme_missing)

    return df.loc[mask].reset_index(drop=True)


def _replace_sentinel_with_nan(
    df: pd.DataFrame,
    protected_cols: Sequence[str] = ("rid", "label", "visit", "data_type"),
) -> pd.DataFrame:
    """
    Replace sentinel-like missing values with NaN.

    Imputation is deliberately not performed here. Train-set-only imputation
    should be performed in the modeling pipeline after participant-level
    splitting.
    """
    df = df.copy()

    for col in df.columns:
        if col in protected_cols:
            continue

        converted = pd.to_numeric(df[col], errors="coerce")
        df[col] = converted.where(
            ~((converted > 90000) | (converted < -90000)),
            np.nan,
        )

    return df


def build_longitudinal_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build the longitudinal modeling dataset without cohort-level imputation.

    Returns
    -------
    pd.DataFrame
        Longitudinal dataset containing:
        - rid: participant identifier retained for participant-level splitting
        - data_type: endpoint definition ("diag" or "drug")
        - visit: interval start visit
        - label: target label
        - current visit features
        - visit-to-visit difference features

    Important
    ---------
    This function does not perform mean/mode imputation, scaling, or PCA.
    Those steps should be fit on the training set only after participant-level
    splitting in the modeling code.
    """
    # Build longitudinal datasets for diag / drug endpoint definitions.
    df_diag = df[df["dmstatus_sub"] != 1.1].reset_index(drop=True)
    df_drug = df[df["dmstatus_sub"] != 1.2].reset_index(drop=True)

    df_diag_all = _build_longitudinal_subset(df_diag, mode="diag")
    df_drug_all = _build_longitudinal_subset(df_drug, mode="drug")

    df_all = pd.concat([df_drug_all, df_diag_all], ignore_index=True)

    df_all = _apply_basic_filters(df_all)
    df_all = _replace_sentinel_with_nan(df_all)

    return df_all


def label_split_imputation(df: pd.DataFrame) -> pd.DataFrame:
    """
    Backward-compatible wrapper for the previous function name.

    The old implementation generated longitudinal rows, imputed the full
    dataset, and dropped `rid`. That could make participant-level splitting
    and leakage control difficult. The revised behavior intentionally keeps
    `rid` and defers mean/mode imputation to the modeling pipeline.
    """
    return build_longitudinal_dataset(df)


def run_full_preprocessing(
    sas_data_path: str,
    glu_data_path: str,
    output_csv_path: str,
) -> pd.DataFrame:
    """
    Run the full preprocessing pipeline.

    Steps
    -----
    1. Load SAS data and rescale income columns (`sas_trans`).
    2. Load glucose Excel file and replace '.' with sentinel missing values.
    3. Merge cohort and glucose data by RID.
    4. Run feature selection and deterministic visit-wise forward filling.
    5. Cast non-ID columns to float when possible.
    6. Build the longitudinal dataset while preserving `rid`.
    7. Save the final dataset as CSV.

    Notes
    -----
    The output CSV is not fully imputed by design. Downstream model training
    should perform participant-level train/validation/test splitting first,
    then fit imputation, scaling, and PCA using the training set only.
    """
    data = sas_trans(sas_data_path)

    data_glu = pd.read_excel(glu_data_path).replace(".", SENTINEL)

    # Convert all columns except the first (RID) to float if possible.
    for col in list(data_glu.columns)[1:]:
        try:
            data_glu[col] = data_glu[col].astype("float")
        except Exception:
            continue

    data = pd.merge(data, data_glu, how="left", left_on="RID", right_on="RID")

    df = feature_selection_processing(data)

    for col in list(df.columns)[1:]:
        try:
            df[col] = df[col].astype("float")
        except Exception:
            continue

    df_result = build_longitudinal_dataset(df)

    output_path = Path(output_csv_path)
    if output_path.parent != Path("."):
        output_path.parent.mkdir(parents=True, exist_ok=True)

    df_result.to_csv(output_path, index=False)

    return df_result
