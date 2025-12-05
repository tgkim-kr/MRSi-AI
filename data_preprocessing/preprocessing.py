#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
from sas7bdat import SAS7BDAT
from typing import List


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
            mask = data[col] != 99999
            data.loc[mask, col] = data.loc[mask, col] / 10000

    return data


def feature_selection_processing(data: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize column names, select relevant features, and perform simple
    forward-fill imputation across visits.

    The logic is kept close to the original script:
    - Column names are converted to lowercase.
    - A set of base, age, and longitudinal variables are selected.
    - Missing values are first filled with 99999.
    - For visit-based variables (v2_*–v10_*), sentinel values (99999)
      are replaced by the value from the previous visit (v1_*–v9_*).
    - Age columns (as2_age–as10_age) are also forward-filled in the
      same way.

    Parameters
    ----------
    data : pd.DataFrame
        Raw merged dataframe (SAS + glucose).

    Returns
    -------
    pd.DataFrame
        Feature-selected dataframe with simple forward-fill.
    """
    data = data.copy()
    data.columns = [c.lower() for c in data.columns]

    # Collect age-related columns (e.g., as1_age, as2_age, ...)
    age_cols: List[str] = [c for c in data.columns if "_age" in c]

    base_cols = ["as1_area", "as1_edua", "v1_fdm_rel", "v1_house", "as1_sex"]

    prefixes = [f"V{i}_" for i in range(1, 11)]
    base_vars = [
        "SBP", "DBP", "BMI", "WEIGHT", "WAIST_AVG", "PA", "SMOKE", "PACKYR_Current",
        "DRINK", "TOTALC", "HTFDCST", "MEDCST", "URINFQLNC", "WAFQLNC", "INCOME",
        "ALT", "AST", "Bun", "Creatinine", "Hb", "Hct", "HDL", "CRP",
        "Ins0", "Ins120", "Ins60", "Plat", "RBC", "TChl", "TG", "WBC",
    ]
    glucose_vars = ["Glu0", "Glu120", "Hba1c"]

    diff_cols_raw = (
        [p + v for p in prefixes for v in base_vars]
        + [f"V{i}_{v}" for i in range(1, 11) for v in glucose_vars]
    )
    diff_cols = [c.lower() for c in diff_cols_raw]

    # Collect base names (e.g., sbp, dbp, ...) from v*_sbp style columns.
    diff_ori = list(
        set("_".join(c.split("_")[1:]).lower() for c in diff_cols_raw)
    )

    stat_cols = ["dmcase", "dmstatus_sub"]
    id_cols = ["rid"]

    selected_cols: List[str] = []
    selected_cols.extend(id_cols)
    selected_cols.extend(stat_cols)
    selected_cols.extend(base_cols)
    selected_cols.extend(age_cols)
    selected_cols.extend(diff_cols)

    df = data[selected_cols].copy()

    # Default dmstatus_sub to 0 if missing
    df["dmstatus_sub"] = df["dmstatus_sub"].fillna(0)

    # Fill all remaining missing values with sentinel 99999
    for col in df.columns:
        df[col] = df[col].fillna(99999)

    # Collect the tail names of v*_ variables (e.g., sbp, dbp, ...)
    var_names = set()
    for c in df.columns:
        if "_" in c:
            try:
                var_names.add("_".join(c.split("_")[1:]))
            except Exception:
                continue

    # Remove names that are not true longitudinal variables
    for drop_name in ["", "sub", "area", "edua", "fdm_rel", "house", "sex", "age"]:
        var_names.discard(drop_name)

    # Forward-fill visit-based variables (v2_*–v10_*) using the previous visit
    for idx in range(len(df)):
        for var in var_names:
            for visit in range(2, 11):
                curr_col = f"v{visit}_{var}"
                prev_col = f"v{visit-1}_{var}"
                if curr_col in df.columns and prev_col in df.columns:
                    if df.loc[idx, curr_col] == 99999:
                        df.loc[idx, curr_col] = df.loc[idx, prev_col]

    # Forward-fill age columns (as2_age–as10_age) using the previous visit
    for idx in range(len(df)):
        for visit in range(2, 11):
            curr_col = f"as{visit}_age"
            prev_col = f"as{visit-1}_age"
            if curr_col in df.columns and prev_col in df.columns:
                if df.loc[idx, curr_col] == 99999:
                    df.loc[idx, curr_col] = df.loc[idx, prev_col]

    return df


def get_diff_code(v1: int, v2: int, levels: List[int]) -> int:
    """
    Encode a (v1, v2) pair into an integer code based on a list of levels.

    This helper is kept for compatibility with the original script.

    Parameters
    ----------
    v1 : int
        Previous level.
    v2 : int
        Next level.
    levels : list of int
        Ordered list of possible levels.

    Returns
    -------
    int
        Encoded difference code or 99999 if either value is not in levels.
    """
    if v1 not in levels or v2 not in levels:
        return 99999
    return levels.index(v1) * len(levels) + levels.index(v2) + 1


def label_split_imputation(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a longitudinal dataset with labels and differences, then apply
    simple imputation.

    Steps:
    ------
    1. Split the cohort into two subsets based on `dmstatus_sub`
       (diag / drug).
    2. For each row, generate multiple time points (visits) with:
       - label (0/1) determined by `dmcase` and mode (diag / drug),
       - baseline features (base_cols),
       - age at each visit,
       - longitudinal variables and their differences between visits.
    3. For the combined longitudinal dataset, perform imputation:
       - Filter out rows with sentinel 99999 in some baseline columns.
       - Convert extreme sentinel values to NaN.
       - Impute continuous features with mean.
       - Impute categorical features with mode.
       - Drop `rid` after imputation.
    4. Add a `data_type` column ("diag" / "drug") and concatenate.

    Parameters
    ----------
    df : pd.DataFrame
        Output of `feature_selection_processing`.

    Returns
    -------
    pd.DataFrame
        Longitudinal, imputed dataset containing both diag and drug subsets.
    """
    base_cols = ["as1_area", "as1_edua", "v1_fdm_rel", "v1_house", "as1_sex"]

    prefixes = [f"V{i}_" for i in range(1, 11)]
    base_vars = [
        "SBP", "DBP", "BMI", "WEIGHT", "WAIST_AVG", "PA", "SMOKE", "PACKYR_Current",
        "DRINK", "TOTALC", "HTFDCST", "MEDCST", "URINFQLNC", "WAFQLNC", "INCOME",
        "ALT", "AST", "Bun", "Creatinine", "Hb", "Hct", "HDL", "CRP",
        "Ins0", "Ins120", "Ins60", "Plat", "RBC", "TChl", "TG", "WBC",
    ]
    glucose_vars = ["Glu0", "Glu120", "Hba1c"]

    diff_names: List[str] = []
    for p in prefixes:
        for v in base_vars + glucose_vars:
            raw = p + v
            name = "_".join(raw.split("_")[1:]).lower()
            if name not in diff_names:
                diff_names.append(name)
    diff_ori = diff_names

    # Variables treated as categorical when computing differences
    chr_2 = ["pa", "urinfqlnc", "wafqlnc"]  # 2-level categorical
    chr_3 = ["smoke", "drink"]             # 3-level categorical

    def encode_chr2(v1: float, v2: float) -> int:
        """
        Encode a (v1, v2) pair for a 2-level categorical variable.

        Returns
        -------
        int
            Encoded state transition or 99999 for invalid values.
        """
        if v1 == 0:
            if v2 == 0:
                return 1
            elif v2 == 1:
                return 2
            else:
                return 99999
        elif v1 == 1:
            if v2 == 0:
                return 3
            elif v2 == 1:
                return 4
            else:
                return 99999
        else:
            return 99999

    def encode_chr3(v1: float, v2: float) -> int:
        """
        Encode a (v1, v2) pair for a 3-level categorical variable.

        Returns
        -------
        int
            Encoded state transition or 99999 for invalid values.
        """
        if v1 == 0:
            if v2 == 0:
                return 1
            elif v2 == 1:
                return 2
            elif v2 == 2:
                return 3
            else:
                return 99999
        elif v1 == 1:
            if v2 == 0:
                return 4
            elif v2 == 1:
                return 5
            elif v2 == 2:
                return 6
            else:
                return 99999
        elif v1 == 2:
            if v2 == 0:
                return 7
            elif v2 == 1:
                return 8
            elif v2 == 2:
                return 9
            else:
                return 99999
        else:
            return 99999

    def compute_diff(row: pd.Series, var_name: str, j: int):
        """
        Compute the difference (or encoded transition) between visit j and j+1
        for a given variable.
        """
        v_curr = row[f"v{j}_{var_name}"]
        v_next = row[f"v{j+1}_{var_name}"]

        if var_name in chr_2:
            return encode_chr2(v_curr, v_next)
        elif var_name in chr_3:
            return encode_chr3(v_curr, v_next)
        else:
            return v_next - v_curr

    def get_intervals(dmcase: float, mode: str):
        """
        Compute (visit_index, label) pairs for a given `dmcase` value.

        The logic directly follows the original script, and determines how
        many time points each subject should contribute, and with which label.
        """
        intervals = []
        if mode == "diag":
            if dmcase == 1:
                intervals = [(1, 1)]
            elif 1 < dmcase < 10:
                intervals = [
                    (j, 0 if j < dmcase else 1) for j in range(1, int(dmcase) + 1)
                ]
            elif dmcase == 10:
                intervals = [(j, 0) for j in range(1, 10)]
            elif dmcase > 10:
                intervals = [(j, 0) for j in range(1, int(20 - dmcase))]
        else:  # mode == "drug"
            if dmcase == 2:
                intervals = [(1, 1)]
            elif 2 < dmcase < 10:
                intervals = [
                    (j, 0 if j < dmcase - 1 else 1) for j in range(1, int(dmcase))
                ]
            elif dmcase == 10:
                intervals = [(j, 0) for j in range(1, 10)]
            elif dmcase > 10:
                intervals = [(j, 0) for j in range(1, int(20 - dmcase))]
        return intervals

    def build_longitudinal(df_sub: pd.DataFrame, mode: str) -> pd.DataFrame:
        """
        Construct a longitudinal dataframe for a given subset and mode.

        For each subject:
        - Determine the list of (visit, label) intervals using `get_intervals`.
        - For each interval, append:
          - rid
          - label
          - baseline features (base_cols)
          - age at that visit
          - current longitudinal features
          - difference between visit j and j+1 for each longitudinal feature.
        """
        rid_list: List[int] = []
        label_list: List[int] = []
        age_list: List[float] = []

        base_dict = {b: [] for b in base_cols}
        val_dict = {k: [] for k in diff_ori}
        diff_dict = {k: [] for k in diff_ori}

        for _, row in df_sub.iterrows():
            dmcase = row["dmcase"]
            intervals = get_intervals(dmcase, mode)

            for j, lbl in intervals:
                rid_list.append(row["rid"])
                label_list.append(lbl)

                for b in base_cols:
                    base_dict[b].append(row[b])

                age_list.append(row[f"as{j}_age"])

                for v in diff_ori:
                    val_dict[v].append(row[f"v{j}_{v}"])
                    diff_dict[v].append(compute_diff(row, v, j))

        data_dict = {
            "rid": rid_list,
            "label": label_list,
            "age": age_list,
        }
        for b in base_cols:
            data_dict[b] = base_dict[b]
        for v in diff_ori:
            data_dict[v] = val_dict[v]
        for v in diff_ori:
            data_dict[f"{v}_diff"] = diff_dict[v]

        return pd.DataFrame(data_dict)

    def impute_dataset(df_all: pd.DataFrame) -> pd.DataFrame:
        """
        Apply simple filtering and imputation to the longitudinal dataset.

        Steps
        -----
        1. Filter subjects with sentinel values in some key baseline features.
        2. Replace extreme sentinel values (> 90000 or < -90000) with NaN.
        3. Impute:
           - continuous columns: mean
           - categorical columns (chr_2, chr_3 and their *_diff): mode
        4. Drop `rid` at the end (not used as a feature).
        """
        df_all = df_all[
            (df_all["as1_edua"] != 99999) & (df_all["v1_house"] != 99999)
        ].reset_index(drop=True)

        df_impu = df_all.copy()

        # Convert sentinel values to NaN
        for col in df_impu.columns:
            if col == "rid":
                continue
            df_impu[col] = df_impu[col].where(
                ~((df_impu[col] > 90000) | (df_impu[col] < -90000)),
                np.nan,
            )

        cont_cols = df_impu.columns.tolist()
        for c in ["rid", "label"] + base_cols:
            if c in cont_cols:
                cont_cols.remove(c)

        # Remove categorical variables from continuous list
        for c in chr_2:
            for cc in [c, f"{c}_diff"]:
                if cc in cont_cols:
                    cont_cols.remove(cc)
        for c in chr_3:
            for cc in [c, f"{c}_diff"]:
                if cc in cont_cols:
                    cont_cols.remove(cc)

        # Continuous variables: mean imputation
        for col in cont_cols:
            df_impu[col] = df_impu[col].fillna(df_impu[col].mean())

        # 2-level categorical variables: mode imputation
        for c in chr_2:
            if c in df_impu:
                df_impu[c] = df_impu[c].fillna(df_impu[c].mode()[0])
            diff_col = f"{c}_diff"
            if diff_col in df_impu:
                df_impu[diff_col] = df_impu[diff_col].fillna(
                    df_impu[diff_col].mode()[0]
                )

        # 3-level categorical variables: mode imputation
        for c in chr_3:
            if c in df_impu:
                df_impu[c] = df_impu[c].fillna(df_impu[c].mode()[0])
            diff_col = f"{c}_diff"
            if diff_col in df_impu:
                df_impu[diff_col] = df_impu[diff_col].fillna(
                    df_impu[diff_col].mode()[0]
                )

        # Drop rid as it is not used as an input feature
        if "rid" in df_impu:
            df_impu = df_impu.drop("rid", axis=1)

        return df_impu

    # Build longitudinal datasets for diag / drug
    df_diag = df[df["dmstatus_sub"] != 1.1].reset_index(drop=True)
    df_drug = df[df["dmstatus_sub"] != 1.2].reset_index(drop=True)

    df_diag_all = build_longitudinal(df_diag, mode="diag")
    df_drug_all = build_longitudinal(df_drug, mode="drug")

    df_diag_all_impu = impute_dataset(df_diag_all)
    df_drug_all_impu = impute_dataset(df_drug_all)

    df_diag_all_impu["data_type"] = "diag"
    df_drug_all_impu["data_type"] = "drug"

    df_all_impu = pd.concat([df_drug_all_impu, df_diag_all_impu]).reset_index(
        drop=True
    )

    return df_all_impu


def run_full_preprocessing(
    sas_data_path: str,
    glu_data_path: str,
    output_csv_path: str,
) -> pd.DataFrame:
    """
    Run the full preprocessing pipeline from the original script.

    Steps
    -----
    1. Load SAS data and rescale income columns (`sas_trans`).
    2. Load glucose Excel file and replace '.' with '99999'.
    3. Merge cohort and glucose data by RID.
    4. Run feature selection and basic forward-fill
       (`feature_selection_processing`).
    5. Cast all non-ID columns to float when possible.
    6. Build the longitudinal, imputed dataset
       (`label_split_imputation`).
    7. Save the final dataset as CSV.

    Parameters
    ----------
    sas_data_path : str
        Path to the .sas7bdat file (cohort data).
    glu_data_path : str
        Path to the glucose .xlsx file.
    output_csv_path : str
        Destination path for the final CSV file.

    Returns
    -------
    pd.DataFrame
        Final preprocessed dataframe.
    """
    data = sas_trans(sas_data_path)

    data_glu = (
        pd.read_excel(glu_data_path)
        .replace(".", "99999")
    )

    # Convert all columns except the first (RID) to float if possible
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

    df_result = label_split_imputation(df)
    df_result.to_csv(output_csv_path, index=False)

    return df_result

