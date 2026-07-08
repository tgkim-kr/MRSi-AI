from __future__ import annotations

from pathlib import Path
from typing import Iterable, Literal

import numpy as np
import pandas as pd
from tqdm import tqdm


MISSING_SENTINEL = 99999
N_VISITS = 10

BASE_COLUMNS = (
    "as1_area",
    "as1_edua",
    "v1_fdm_rel",
    "v1_house",
    "as1_sex",
)

STATUS_COLUMNS = (
    "dmcase",
    "dmstatus_sub",
)

# Baseline and follow-up measurements used to construct interval-level features.
# For each feature, values are expected as v1_<feature>, ..., v10_<feature>.
LONGITUDINAL_FEATURES = (
    "sbp",
    "dbp",
    "bmi",
    "weight",
    "waist_avg",
    "pa",
    "smoke",
    "packyr_current",
    "drink",
    "totalc",
    "htfdcst",
    "medcst",
    "urinfqlnc",
    "wafqlnc",
    "income",
    "alt",
    "ast",
    "bun",
    "creatinine",
    "hb",
    "hct",
    "hdl",
    "crp",
    "ins0",
    "ins120",
    "ins60",
    "plat",
    "rbc",
    "tchl",
    "tg",
    "wbc",
    "glu0",
    "glu120",
    "hba1c",
)

# Binary nominal variables. Visit-to-visit changes are encoded as:
# 0->0: 1, 0->1: 2, 1->0: 3, 1->1: 4.
BINARY_TRANSITION_FEATURES = (
    "pa",
    "urinfqlnc",
    "wafqlnc",
)

# Three-level nominal variables. Visit-to-visit changes are encoded as:
# previous_value * 3 + next_value + 1 for values in {0, 1, 2}.
TERNARY_TRANSITION_FEATURES = (
    "smoke",
    "drink",
)

DRUG_BASELINE_DROP_COLUMNS = (
    "as1_edua",
    "v1_house",
)


PathLike = str | Path
InputData = pd.DataFrame | PathLike


def load_cohort_data(cohort_path: PathLike) -> pd.DataFrame:
    """Load the cohort data from CSV or SAS7BDAT.

    Parameters
    ----------
    cohort_path:
        Path to a cohort file. Supported extensions are `.csv` and `.sas7bdat`.

    Returns
    -------
    pandas.DataFrame
        Loaded cohort dataframe.
    """
    cohort_path = Path(cohort_path)
    suffix = cohort_path.suffix.lower()

    if suffix == ".csv":
        return pd.read_csv(cohort_path)

    if suffix == ".sas7bdat":
        try:
            from sas7bdat import SAS7BDAT
        except ImportError as exc:
            raise ImportError(
                "Reading SAS7BDAT files requires the `sas7bdat` package. "
                "Install it with `pip install sas7bdat` or provide a CSV file."
            ) from exc

        with SAS7BDAT(str(cohort_path)) as reader:
            return reader.to_data_frame()

    raise ValueError(
        f"Unsupported cohort file format: {cohort_path.suffix}. "
        "Supported formats are .csv and .sas7bdat."
    )


def _load_input_data(data: InputData, loader) -> pd.DataFrame:
    """Load a dataframe from a path or return a copy of an existing dataframe."""
    if isinstance(data, pd.DataFrame):
        return data.copy()
    return loader(data)


def _load_glucose_data(glucose_path: PathLike) -> pd.DataFrame:
    """Load glucose/insulin data from an Excel file."""
    return pd.read_excel(glucose_path)


def _lowercase_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of `df` with lower-case column names."""
    df = df.copy()
    df.columns = [str(col).lower() for col in df.columns]
    return df


def _required_columns(n_visits: int = N_VISITS) -> list[str]:
    """Return the expected raw input columns after lower-casing."""
    required = ["rid", *STATUS_COLUMNS, *BASE_COLUMNS]
    required.extend([f"as{visit}_age" for visit in range(1, n_visits + 1)])
    required.extend(
        f"v{visit}_{feature}"
        for visit in range(1, n_visits + 1)
        for feature in LONGITUDINAL_FEATURES
    )
    return required


def _validate_columns(df: pd.DataFrame, required_columns: Iterable[str]) -> None:
    """Raise a clear error if required columns are missing."""
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        preview = ", ".join(missing_columns[:30])
        suffix = "" if len(missing_columns) <= 30 else f", ... (+{len(missing_columns) - 30} more)"
        raise ValueError(
            "Input data are missing required columns: "
            f"{preview}{suffix}"
        )


def _prepare_glucose_data(glucose_data: pd.DataFrame) -> pd.DataFrame:
    """Clean glucose/insulin data before merging with the cohort dataframe."""
    glucose_data = glucose_data.copy()
    glucose_data = glucose_data.replace(".", MISSING_SENTINEL)

    for col in glucose_data.columns[1:]:
        glucose_data[col] = glucose_data[col].astype(float)

    return glucose_data


def _merge_cohort_and_glucose(
    cohort_data: pd.DataFrame,
    glucose_data: pd.DataFrame,
    id_col_original: str = "RID",
) -> pd.DataFrame:
    """Merge cohort-level data with glucose/insulin measurements by RID."""
    if id_col_original not in cohort_data.columns:
        raise ValueError(f"Cohort data must contain the '{id_col_original}' column.")

    if id_col_original not in glucose_data.columns:
        raise ValueError(f"Glucose data must contain the '{id_col_original}' column.")

    return pd.merge(
        cohort_data,
        glucose_data,
        how="left",
        left_on=id_col_original,
        right_on=id_col_original,
    )


def _scale_income_columns(
    df: pd.DataFrame,
    n_visits: int = N_VISITS,
    missing_sentinel: int | float = MISSING_SENTINEL,
) -> pd.DataFrame:
    """Optionally scale raw income columns by 10,000.

    This reproduces the one-time SAS-to-CSV conversion step used in the original
    notebook. Keep this disabled if income columns were already converted.
    """
    df = df.copy()

    for visit in range(1, n_visits + 1):
        col = f"V{visit}_INCOME"
        if col not in df.columns:
            continue

        df[col] = pd.to_numeric(df[col], errors="coerce")
        valid_mask = df[col].notna() & (df[col] != missing_sentinel)
        df.loc[valid_mask, col] = df.loc[valid_mask, col] / 10000

    return df


def _carry_forward_missing_values(
    df: pd.DataFrame,
    n_visits: int = N_VISITS,
    missing_sentinel: int | float = MISSING_SENTINEL,
) -> pd.DataFrame:
    """Carry forward visit-level missing values encoded with `missing_sentinel`.

    For v2-v10 measurements and as2_age-as10_age, missing values are replaced by
    the previous visit's value. This follows the original preprocessing logic.
    """
    df = df.copy()

    for feature in LONGITUDINAL_FEATURES:
        for visit in range(2, n_visits + 1):
            current_col = f"v{visit}_{feature}"
            previous_col = f"v{visit - 1}_{feature}"
            df[current_col] = df[current_col].mask(
                df[current_col] == missing_sentinel,
                df[previous_col],
            )

    for visit in range(2, n_visits + 1):
        current_col = f"as{visit}_age"
        previous_col = f"as{visit - 1}_age"
        df[current_col] = df[current_col].mask(
            df[current_col] == missing_sentinel,
            df[previous_col],
        )

    return df


def _encode_nominal_transition(
    previous_value,
    next_value,
    n_levels: int,
    missing_sentinel: int | float = MISSING_SENTINEL,
):
    """Encode a nominal visit-to-visit transition as a compact integer code."""
    try:
        previous_int = int(previous_value)
        next_int = int(next_value)
    except (TypeError, ValueError):
        return missing_sentinel

    valid_values = set(range(n_levels))
    if previous_int not in valid_values or next_int not in valid_values:
        return missing_sentinel

    return previous_int * n_levels + next_int + 1


def _intervals_for_diag(dmcase) -> list[tuple[int, int]]:
    """Return (visit, label) intervals for the diagnosis-based outcome."""
    try:
        dmcase = int(dmcase)
    except (TypeError, ValueError):
        return []

    if dmcase == 1:
        return [(1, 1)]

    if 1 < dmcase < 10:
        return [(visit, int(visit == dmcase)) for visit in range(1, dmcase + 1)]

    if dmcase == 10:
        return [(visit, 0) for visit in range(1, 10)]

    if dmcase > 10:
        return [(visit, 0) for visit in range(1, int(20 - dmcase))]

    return []


def _intervals_for_drug(dmcase) -> list[tuple[int, int]]:
    """Return (visit, label) intervals for the medication-based outcome."""
    try:
        dmcase = int(dmcase)
    except (TypeError, ValueError):
        return []

    if dmcase == 2:
        return [(1, 1)]

    if 2 < dmcase < 10:
        return [(visit, int(visit == dmcase - 1)) for visit in range(1, dmcase)]

    if dmcase == 10:
        return [(visit, 0) for visit in range(1, 10)]

    if dmcase > 10:
        return [(visit, 0) for visit in range(1, int(20 - dmcase))]

    return []


def _build_interval_examples(
    source_df: pd.DataFrame,
    outcome_type: Literal["diag", "drug"],
    n_visits: int = N_VISITS,
    missing_sentinel: int | float = MISSING_SENTINEL,
    show_progress: bool = True,
) -> pd.DataFrame:
    """Construct interval-level rows from participant-level longitudinal data."""
    if outcome_type == "diag":
        interval_fn = _intervals_for_diag
    elif outcome_type == "drug":
        interval_fn = _intervals_for_drug
    else:
        raise ValueError("outcome_type must be either 'diag' or 'drug'.")

    rows = []
    iterator = source_df.iterrows()
    if show_progress:
        iterator = tqdm(iterator, total=len(source_df), desc=f"Building {outcome_type} rows")

    for _, participant in iterator:
        intervals = interval_fn(participant["dmcase"])

        for visit, label in intervals:
            next_visit = visit + 1
            if next_visit > n_visits:
                raise ValueError(
                    f"Invalid interval for RID={participant['rid']}: "
                    f"visit {visit} requires visit {next_visit}, but n_visits={n_visits}."
                )

            row = {
                "rid": participant["rid"],
                "label": int(label),
            }

            for base_col in BASE_COLUMNS:
                row[base_col] = participant[base_col]

            row["age"] = participant[f"as{visit}_age"]

            for feature in LONGITUDINAL_FEATURES:
                current_col = f"v{visit}_{feature}"
                next_col = f"v{next_visit}_{feature}"
                row[feature] = participant[current_col]

                if feature in BINARY_TRANSITION_FEATURES:
                    row[f"{feature}_diff"] = _encode_nominal_transition(
                        participant[current_col],
                        participant[next_col],
                        n_levels=2,
                        missing_sentinel=missing_sentinel,
                    )
                elif feature in TERNARY_TRANSITION_FEATURES:
                    row[f"{feature}_diff"] = _encode_nominal_transition(
                        participant[current_col],
                        participant[next_col],
                        n_levels=3,
                        missing_sentinel=missing_sentinel,
                    )
                else:
                    row[f"{feature}_diff"] = participant[next_col] - participant[current_col]

            rows.append(row)

    columns = ["rid", "label", *BASE_COLUMNS, "age"]
    columns.extend(LONGITUDINAL_FEATURES)
    columns.extend(f"{feature}_diff" for feature in LONGITUDINAL_FEATURES)

    return pd.DataFrame(rows, columns=columns)


def _replace_sentinel_values_with_nan(
    df: pd.DataFrame,
    missing_threshold: int | float = 90000,
) -> pd.DataFrame:
    """Replace extreme sentinel values with NaN."""
    df = df.copy()

    for col in df.columns:
        if col == "rid":
            continue

        numeric_values = pd.to_numeric(df[col], errors="coerce")
        sentinel_mask = (numeric_values > missing_threshold) | (numeric_values < -missing_threshold)
        df.loc[sentinel_mask, col] = np.nan

    return df


def _fill_with_mode(series: pd.Series) -> pd.Series:
    """Fill missing values in a series using the first mode when available."""
    mode_values = series.mode(dropna=True)
    if mode_values.empty:
        return series
    return series.fillna(mode_values.iloc[0])


def _impute_interval_examples(df: pd.DataFrame) -> pd.DataFrame:
    """Impute missing values in the interval-level dataframe.

    Categorical baseline variables and nominal transition variables are imputed
    with the mode. Continuous variables and continuous visit-to-visit changes are
    imputed with the mean.
    """
    df = _replace_sentinel_values_with_nan(df)

    categorical_cols = set(BASE_COLUMNS)
    categorical_cols.update(BINARY_TRANSITION_FEATURES)
    categorical_cols.update(TERNARY_TRANSITION_FEATURES)
    categorical_cols.update(f"{feature}_diff" for feature in BINARY_TRANSITION_FEATURES)
    categorical_cols.update(f"{feature}_diff" for feature in TERNARY_TRANSITION_FEATURES)

    for col in categorical_cols:
        if col in df.columns:
            df[col] = _fill_with_mode(df[col])

    numeric_cols = [
        col
        for col in df.columns
        if col not in {"rid", "label"} and col not in categorical_cols
    ]

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        mean_value = df[col].mean(skipna=True)
        if not pd.isna(mean_value):
            df[col] = df[col].fillna(mean_value)

    return df


def preprocess_asas_data(
    cohort_data: InputData,
    glucose_data: InputData,
    output_csv: PathLike | None = None,
    *,
    scale_income: bool = False,
    impute_missing: bool = True,
    drop_drug_missing_baseline: bool = True,
    show_progress: bool = True,
) -> pd.DataFrame:
    """Run the full ASAS preprocessing pipeline.

    Parameters
    ----------
    cohort_data:
        Cohort-level longitudinal data as a dataframe or a path to `.csv` or
        `.sas7bdat` file.

    glucose_data:
        Glucose/insulin measurement data as a dataframe or a path to an Excel
        file. The first column must be `RID`, matching the cohort data.

    output_csv:
        Optional path for saving the preprocessed dataframe.

    scale_income:
        Whether to divide V1_INCOME-V10_INCOME by 10,000 before lower-casing
        column names. Enable only when the raw income variables have not already
        been scaled.

    impute_missing:
        Whether to replace sentinel-coded missing values and impute missing
        entries in the final interval-level dataframe. This reproduces the
        original preprocessing notebook behavior. If leakage-free imputation is
        required, set this to False and perform imputation after train/test/
        validation splitting.

    drop_drug_missing_baseline:
        Whether to drop medication-outcome rows with missing `as1_edua` or
        `v1_house`, matching the original notebook behavior.

    show_progress:
        Whether to show tqdm progress bars when constructing interval-level rows.

    Returns
    -------
    pandas.DataFrame
        Preprocessed interval-level dataframe with a `data_type` column where
        values are `diag` or `drug`.
    """
    cohort_df = _load_input_data(cohort_data, load_cohort_data)
    glucose_df = _load_input_data(glucose_data, _load_glucose_data)
    glucose_df = _prepare_glucose_data(glucose_df)

    if scale_income:
        cohort_df = _scale_income_columns(cohort_df)

    merged_df = _merge_cohort_and_glucose(cohort_df, glucose_df)
    merged_df = _lowercase_columns(merged_df)

    required_cols = _required_columns()
    _validate_columns(merged_df, required_cols)

    df = merged_df[required_cols].copy()
    df["dmcase"] = pd.to_numeric(df["dmcase"], errors="coerce")
    df["dmstatus_sub"] = pd.to_numeric(df["dmstatus_sub"], errors="coerce").fillna(0)
    df = df.fillna(MISSING_SENTINEL)

    df = _carry_forward_missing_values(df)

    diag_source = df[df["dmstatus_sub"] != 1.1].reset_index(drop=True)
    drug_source = df[df["dmstatus_sub"] != 1.2].reset_index(drop=True)

    diag_examples = _build_interval_examples(
        diag_source,
        outcome_type="diag",
        show_progress=show_progress,
    )

    drug_examples = _build_interval_examples(
        drug_source,
        outcome_type="drug",
        show_progress=show_progress,
    )

    if drop_drug_missing_baseline:
        for col in DRUG_BASELINE_DROP_COLUMNS:
            drug_examples = drug_examples[drug_examples[col] != MISSING_SENTINEL]
        drug_examples = drug_examples.reset_index(drop=True)

    if impute_missing:
        diag_examples = _impute_interval_examples(diag_examples)
        drug_examples = _impute_interval_examples(drug_examples)

    diag_examples["data_type"] = "diag"
    drug_examples["data_type"] = "drug"

    preprocessed_df = pd.concat(
        [drug_examples, diag_examples],
        axis=0,
        ignore_index=True,
    )

    if output_csv is not None:
        output_csv = Path(output_csv)
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        preprocessed_df.to_csv(output_csv, index=False)

    return preprocessed_df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run ASAS preprocessing and save an interval-level CSV file."
    )
    parser.add_argument("--cohort-data", required=True, help="Path to cohort CSV or SAS7BDAT file.")
    parser.add_argument("--glucose-data", required=True, help="Path to glucose/insulin Excel file.")
    parser.add_argument("--output-csv", required=True, help="Path to save the preprocessed CSV file.")
    parser.add_argument(
        "--scale-income",
        action="store_true",
        help="Scale V1_INCOME-V10_INCOME by 10,000 before preprocessing.",
    )
    parser.add_argument(
        "--no-impute",
        action="store_true",
        help="Do not impute missing values in the final preprocessed dataframe.",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable tqdm progress bars.",
    )

    args = parser.parse_args()

    preprocess_asas_data(
        cohort_data=args.cohort_data,
        glucose_data=args.glucose_data,
        output_csv=args.output_csv,
        scale_income=args.scale_income,
        impute_missing=not args.no_impute,
        show_progress=not args.no_progress,
    )
