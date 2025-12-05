import json
from typing import List, Dict, Any, Optional

import pandas as pd
from tqdm import tqdm

from .feature_engineering import data_split, feature_selection, pca_df
from .models import (
    train_lgbm_model,
    train_xgb_model,
    train_lr_model,
    train_rf_model,
    train_ann_model,
    train_rnn_model,
)
from .utils import convert_floats_and_ints


MODEL_FUNCS = {
    "LR": train_lr_model,
    "RF": train_rf_model,
    "XGB": train_xgb_model,
    "LGBM": train_lgbm_model,
    "ANN": train_ann_model,
    "RNN": train_rnn_model,
}


def _drop_insulin_glucose_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop columns containing 'ins', 'glu', or 'hba1c' in their names,
    matching the original script.
    """
    cols_to_drop = [
        c
        for c in df.columns
        if ("ins" in c.lower()) or ("glu" in c.lower()) or ("hba1c" in c.lower())
    ]
    return df.drop(columns=cols_to_drop)


def run_experiment(
    csv_path: str,
    output_json_path: str,
    result_root: str = "result",
    feature_sets: Optional[List[str]] = None,
    pca_modes: Optional[List[str]] = None,
    models_to_run: Optional[List[str]] = None,
    sampling: str = "over",
) -> Dict[str, Any]:
    """
    Run the full training experiment across multiple feature sets,
    PCA modes, and models.

    Parameters
    ----------
    csv_path : str
        Path to preprocessed CSV file.
    output_json_path : str
        Path to JSON summary output.
    result_root : str, optional
        Base directory for all model artifacts.
    feature_sets : list of str, optional
        List of feature set names. Default: ["all","bio","physical","life","Non_invasive"].
    pca_modes : list of str, optional
        List of PCA modes. Default: ["plus","none","only"].
    models_to_run : list of str, optional
        List of model names among {"LR","RF","XGB","LGBM","ANN","RNN"}.
        Default: all of them.
    sampling : {"over","none"}, optional
        Sampling strategy for data_split.

    Returns
    -------
    dict
        Nested dictionary:
        results[feature_set][pca_mode][model_name] = {...}
    """
    if feature_sets is None:
        feature_sets = ["all", "bio", "physical", "life", "Non_invasive"]
    if pca_modes is None:
        pca_modes = ["plus", "none", "only"]
    if models_to_run is None:
        models_to_run = ["XGB", "LR", "RF", "LGBM", "ANN", "RNN"]

    # Load data
    data = pd.read_csv(csv_path)

    # Filter to 'diag' rows if column exists
    if "data_type" in data.columns:
        data = (
            data[data["data_type"] == "diag"]
            .reset_index(drop=True)
            .drop(columns=["data_type"])
        )

    data = _drop_insulin_glucose_columns(data)

    results: Dict[str, Dict[str, Dict[str, Any]]] = {}

    for feature_set in feature_sets:
        results[feature_set] = {}
        for pca_mode in pca_modes:
            results[feature_set][pca_mode] = {}

            df_fs = feature_selection(data, feature_set)
            df_fs_pca = pca_df(df_fs, pca_mode, feature_set)

            x_train, y_train, x_test, y_test, x_val, y_val = data_split(
                df_fs_pca, sampling=sampling
            )

            for model_name in tqdm(models_to_run, desc=f"{feature_set}-{pca_mode}"):
                if model_name not in MODEL_FUNCS:
                    raise ValueError(f"Unknown model: {model_name}")

                train_fn = MODEL_FUNCS[model_name]
                auc_test, auc_val, info = train_fn(
                    x_train,
                    y_train,
                    x_test,
                    y_test,
                    x_val,
                    y_val,
                    features_tag=feature_set,
                    pca_tag=pca_mode,
                    output_root=result_root,
                )

                results[feature_set][pca_mode][model_name] = {
                    "auc_test": auc_test,
                    "auc_validation": auc_val,
                    "performance": info["performance"],
                    "importance": info["importance"],
                    "params": info["params"],
                }

    results_converted = convert_floats_and_ints(results)

    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(results_converted, f, indent=4, ensure_ascii=False)

    return results_converted
