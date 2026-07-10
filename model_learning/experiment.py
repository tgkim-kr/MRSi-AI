from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm

from .feature_engineering import data_split, feature_selection, pca_df
from .models import MODEL_REGISTRY
from .utils import ensure_dir, find_best_cutoff, save_json, save_roc_curve


DEFAULT_CPU_MODELS: tuple[str, ...] = ("LR", "RF", "XGB", "LGBM")
DEFAULT_GPU_MODELS: tuple[str, ...] = ("ANN", "RNN")
DEFAULT_PCA_MODES: tuple[str, ...] = ("plus", "none", "only")
DEFAULT_FEATURE_SETS: tuple[str, ...] = (
    "all",
    "bio",
    "physical",
    "life",
    "Non_invasive",
)


def load_training_data(
    input_csv: str | Path,
    *,
    data_type: str | None = "diag",
    data_type_col: str = "data_type",
    drop_contains: tuple[str, ...] = ("ins", "glu", "hba1c"),
) -> pd.DataFrame:
    """
    Load the preprocessed data used for model training.

    Parameters
    ----------
    input_csv:
        Path to the preprocessed CSV file.

    data_type:
        If provided, keep rows where `data_type_col == data_type`.

    data_type_col:
        Column containing analysis data type labels such as "diag" or "drug".

    drop_contains:
        Drop any column whose name contains one of these substrings.

    Returns
    -------
    pandas.DataFrame
        Training dataframe after optional row filtering and column removal.
    """
    data = pd.read_csv(input_csv)

    if data_type is not None:
        if data_type_col not in data.columns:
            raise ValueError(
                f"data_type='{data_type}' was requested, but column "
                f"'{data_type_col}' is not present in the input data."
            )

        data = (
            data[data[data_type_col] == data_type]
            .reset_index(drop=True)
            .drop(columns=[data_type_col])
        )

    drop_cols = [
        col
        for col in data.columns
        if any(pattern.lower() in col.lower() for pattern in drop_contains)
    ]

    if drop_cols:
        data = data.drop(columns=drop_cols)

    return data.copy()


def prepare_combo_data(
    features_name: str,
    pca_name: str,
    data_ori_: pd.DataFrame,
    sampling_method: str,
):
    """
    Prepare train/validation/test matrices for one feature/PCA combination.
    """
    data_tmp = feature_selection(data_ori_.copy(), features_name)
    data_tmp = pca_df(data_tmp, pca_name, features_name)

    x_train, y_train, x_validation, y_validation, x_test, y_test = data_split(
        data_tmp,
        sampling_method,
    )

    return x_train, y_train, x_validation, y_validation, x_test, y_test


def _call_model(
    model_name: str,
    model_func,
    x_train,
    y_train,
    x_validation,
    y_validation,
    x_test,
    y_test,
    *,
    save_dir: Path,
    random_state: int,
    make_plots: bool,
    make_shap: bool,
    save_xgb_trees: bool,
    run_label: str,
):
    """
    Call one model function with model-specific keyword arguments.
    """
    kwargs = {
        "save_dir": save_dir,
        "random_state": random_state,
        "make_plots": make_plots,
    }

    if model_name in {"LGBM", "XGB"}:
        kwargs["make_shap"] = make_shap

    if model_name == "XGB":
        kwargs["save_trees"] = save_xgb_trees
        kwargs["run_label"] = run_label

    return model_func(
        x_train,
        y_train,
        x_validation,
        y_validation,
        x_test,
        y_test,
        **kwargs,
    )


def run_models_for_combo(
    features_name: str,
    pca_name: str,
    data_ori_: pd.DataFrame,
    model_names: Iterable[str],
    sampling_method: str,
    *,
    output_dir: str | Path = "result",
    random_state: int = 123,
    make_plots: bool = True,
    make_shap: bool = True,
    save_xgb_trees: bool = False,
):
    """
    Run a list of models for one feature/PCA combination.

    This function is process-safe because it does not depend on global
    variables such as `features` or `pca`.
    """
    output_dir = Path(output_dir)

    combo_result = {}
    pred_test_dict = {}
    pred_validation_dict = {}

    x_train, y_train, x_validation, y_validation, x_test, y_test = prepare_combo_data(
        features_name,
        pca_name,
        data_ori_,
        sampling_method,
    )

    for model_name in model_names:
        if model_name not in MODEL_REGISTRY:
            raise ValueError(
                f"Unsupported model name: {model_name}. "
                f"Supported models are {list(MODEL_REGISTRY)}."
            )

        model_func = MODEL_REGISTRY[model_name]
        model_save_dir = ensure_dir(output_dir / features_name / pca_name / model_name)

        auc_validation, auc_test, params, y_pred_prob_validation, y_pred_prob_test, performance, importance = _call_model(
            model_name,
            model_func,
            x_train,
            y_train,
            x_validation,
            y_validation,
            x_test,
            y_test,
            save_dir=model_save_dir,
            random_state=random_state,
            make_plots=make_plots,
            make_shap=make_shap,
            save_xgb_trees=save_xgb_trees,
            run_label=f"{features_name}__{pca_name}__{model_name}",
        )

        y_pred_prob_test = np.asarray(y_pred_prob_test).reshape(-1)
        y_test_array = np.asarray(y_test).reshape(-1)

        if len(y_pred_prob_test) != len(y_test_array):
            raise ValueError(
                f"[{features_name}-{pca_name}-{model_name}] prediction length "
                f"does not match y_test length. "
                f"prediction: {y_pred_prob_test.shape}, "
                f"y_test: {len(y_test_array)}"
            )

        y_pred_prob_validation = np.asarray(y_pred_prob_validation).reshape(-1)
        y_validation_array = np.asarray(y_validation).reshape(-1)

        if len(y_pred_prob_validation) != len(y_validation_array):
            raise ValueError(
                f"[{features_name}-{pca_name}-{model_name}] validation prediction length "
                f"does not match y_validation length. "
                f"prediction: {y_pred_prob_validation.shape}, "
                f"y_validation: {len(y_validation_array)}"
            )

        pred_test_dict[model_name] = y_pred_prob_test
        pred_validation_dict[model_name] = y_pred_prob_validation

        combo_result[model_name] = {
            "auc_validation": auc_validation,
            "auc_test": auc_test,
            "performance": performance,
            "importance": importance,
            "params": params,
        }

    return {
        "features": features_name,
        "pca": pca_name,
        "result": combo_result,
        "pred_test_dict": pred_test_dict,
        "pred_validation_dict": pred_validation_dict,
        "y_validation": np.asarray(y_validation).reshape(-1),
        "y_test": np.asarray(y_test).reshape(-1),
    }


def add_ensemble_result(
    features_name: str,
    pca_name: str,
    combo_result: dict,
    pred_test_dict: dict,
    pred_validation_dict: dict,
    y_validation,
    y_test,
    model_names: Iterable[str],
    *,
    output_dir: str | Path = "result",
    make_plots: bool = True,
) -> dict:
    """
    Add an ensemble result based on the average of model predictions.
    """
    output_dir = Path(output_dir)
    save_dir = ensure_dir(output_dir / features_name / pca_name / "Ensemble")

    model_names = list(model_names)
    missing_test_models = [
        model_name for model_name in model_names if model_name not in pred_test_dict
    ]
    missing_validation_models = [
        model_name for model_name in model_names if model_name not in pred_validation_dict
    ]

    if missing_test_models or missing_validation_models:
        raise ValueError(
            f"[{features_name}-{pca_name}] Ensemble predictions are missing. "
            f"test missing: {missing_test_models}; "
            f"validation missing: {missing_validation_models}"
        )

    y_validation = np.asarray(y_validation).reshape(-1)
    y_test = np.asarray(y_test).reshape(-1)

    ensemble_pred_validation_list = []
    ensemble_pred_test_list = []

    for model_name in model_names:
        y_pred_prob_validation = np.asarray(pred_validation_dict[model_name]).reshape(-1)
        y_pred_prob_test = np.asarray(pred_test_dict[model_name]).reshape(-1)

        if len(y_pred_prob_validation) != len(y_validation):
            raise ValueError(
                f"[{features_name}-{pca_name}-{model_name}] Ensemble validation prediction "
                f"length mismatch. prediction: {y_pred_prob_validation.shape}, "
                f"y_validation: {len(y_validation)}"
            )

        if len(y_pred_prob_test) != len(y_test):
            raise ValueError(
                f"[{features_name}-{pca_name}-{model_name}] Ensemble test prediction "
                f"length mismatch. prediction: {y_pred_prob_test.shape}, "
                f"y_test: {len(y_test)}"
            )

        ensemble_pred_validation_list.append(y_pred_prob_validation)
        ensemble_pred_test_list.append(y_pred_prob_test)

    y_pred_prob_validation = np.vstack(ensemble_pred_validation_list).mean(axis=0)
    y_pred_prob_test = np.vstack(ensemble_pred_test_list).mean(axis=0)

    performance = find_best_cutoff(
        y_validation_true=y_validation,
        y_validation_pred_prob=y_pred_prob_validation,
        y_test_true=y_test,
        y_test_pred_prob=y_pred_prob_test,
    )
    roc_auc = save_roc_curve(
        y_test,
        y_pred_prob_test,
        save_dir / "ROC_curve.jpg",
    ) if make_plots else float("nan")

    from sklearn.metrics import roc_auc_score

    auc_validation = float(roc_auc_score(y_validation, y_pred_prob_validation))

    if not make_plots:
        roc_auc = float(roc_auc_score(y_test, y_pred_prob_test))

    combo_result["Ensemble"] = {
        "auc_validation": auc_validation,
        "auc_test": roc_auc,
        "performance": performance,
        "importance": {},
        "params": {
            "method": "average of predicted probabilities",
            "models": model_names,
        },
    }

    return combo_result


def _make_tasks(
    feature_sets: Iterable[str],
    pca_modes: Iterable[str],
) -> list[tuple[str, str]]:
    """
    Make feature/PCA combination tasks.
    """
    return [
        (features_name, pca_name)
        for features_name in feature_sets
        for pca_name in pca_modes
    ]


def run_experiment(
    data: pd.DataFrame | None = None,
    *,
    input_csv: str | Path | None = None,
    output_dir: str | Path = "result",
    result_json: str | Path | None = None,
    data_type: str | None = "diag",
    drop_contains: tuple[str, ...] = ("ins", "glu", "hba1c"),
    feature_sets: Iterable[str] = DEFAULT_FEATURE_SETS,
    pca_modes: Iterable[str] = DEFAULT_PCA_MODES,
    cpu_models: Iterable[str] = DEFAULT_CPU_MODELS,
    gpu_models: Iterable[str] = DEFAULT_GPU_MODELS,
    sampling_method: str = "no",
    n_jobs: int = 3,
    random_state: int = 123,
    make_plots: bool = True,
    make_shap: bool = True,
    save_xgb_trees: bool = False,
) -> dict:
    """
    Run the full model-learning experiment.

    Parameters
    ----------
    data:
        Pre-loaded dataframe. If None, `input_csv` must be provided.

    input_csv:
        Path to a preprocessed CSV file.

    output_dir:
        Directory where result plots, trained models, and intermediate outputs
        are saved.

    result_json:
        Optional JSON output path. If None, uses `output_dir / "result.json"`.

    data_type:
        Data type to retain from the preprocessed CSV, e.g. "diag".
        Ignored when `data` is provided directly.

    drop_contains:
        Column-name substrings to remove before model training.

    feature_sets:
        Feature-set names to evaluate.

    pca_modes:
        PCA modes to evaluate.

    cpu_models:
        Models that may be executed in parallel workers.

    gpu_models:
        Models executed sequentially in the main process. This is useful for
        TensorFlow/Keras models that should not be sent to joblib workers.

    sampling_method:
        Sampling method passed to `data_split`.

    n_jobs:
        Number of joblib workers for CPU models.

    random_state:
        Random seed passed to model training functions.

    make_plots:
        Whether to save ROC and feature-importance plots.

    make_shap:
        Whether to save SHAP plots for tree boosting models.

    save_xgb_trees:
        Whether to save all XGBoost trees. This can be slow and should usually
        be enabled only for final selected models.

    Returns
    -------
    dict
        Nested result dictionary: result[feature_set][pca_mode][model_name].
    """
    output_dir = Path(output_dir)
    ensure_dir(output_dir)

    if data is None:
        if input_csv is None:
            raise ValueError("Either `data` or `input_csv` must be provided.")

        data_ori = load_training_data(
            input_csv,
            data_type=data_type,
            drop_contains=drop_contains,
        )
    else:
        data_ori = data.copy()

    feature_sets = list(feature_sets)
    pca_modes = list(pca_modes)
    cpu_models = list(cpu_models)
    gpu_models = list(gpu_models)
    all_models = cpu_models + gpu_models

    tasks = _make_tasks(feature_sets, pca_modes)

    cpu_results = Parallel(
        n_jobs=n_jobs,
        backend="loky",
        verbose=10,
    )(
        delayed(run_models_for_combo)(
            features_name,
            pca_name,
            data_ori,
            cpu_models,
            sampling_method,
            output_dir=output_dir,
            random_state=random_state,
            make_plots=make_plots,
            make_shap=make_shap,
            save_xgb_trees=save_xgb_trees,
        )
        for features_name, pca_name in tasks
    )

    result_pack = {}

    for item in cpu_results:
        key = (item["features"], item["pca"])

        result_pack[key] = {
            "result": item["result"],
            "pred_test_dict": item["pred_test_dict"],
            "pred_validation_dict": item["pred_validation_dict"],
            "y_validation": item["y_validation"],
            "y_test": item["y_test"],
        }

    for features_name, pca_name in tqdm(tasks, desc="GPU models"):
        if not gpu_models:
            continue

        gpu_item = run_models_for_combo(
            features_name,
            pca_name,
            data_ori,
            gpu_models,
            sampling_method,
            output_dir=output_dir,
            random_state=random_state,
            make_plots=make_plots,
            make_shap=make_shap,
            save_xgb_trees=save_xgb_trees,
        )

        key = (features_name, pca_name)

        y_validation_cpu = np.asarray(result_pack[key]["y_validation"]).reshape(-1)
        y_validation_gpu = np.asarray(gpu_item["y_validation"]).reshape(-1)
        y_test_cpu = np.asarray(result_pack[key]["y_test"]).reshape(-1)
        y_test_gpu = np.asarray(gpu_item["y_test"]).reshape(-1)

        if len(y_validation_cpu) != len(y_validation_gpu):
            raise ValueError(
                f"[{features_name}-{pca_name}] CPU/GPU y_validation lengths differ. "
                f"CPU: {len(y_validation_cpu)}, GPU: {len(y_validation_gpu)}"
            )

        if not np.array_equal(y_validation_cpu, y_validation_gpu):
            raise ValueError(
                f"[{features_name}-{pca_name}] CPU/GPU y_validation values differ. "
                "The data split is not deterministic across runs."
            )

        if len(y_test_cpu) != len(y_test_gpu):
            raise ValueError(
                f"[{features_name}-{pca_name}] CPU/GPU y_test lengths differ. "
                f"CPU: {len(y_test_cpu)}, GPU: {len(y_test_gpu)}"
            )

        if not np.array_equal(y_test_cpu, y_test_gpu):
            raise ValueError(
                f"[{features_name}-{pca_name}] CPU/GPU y_test values differ. "
                "The data split is not deterministic across runs."
            )

        result_pack[key]["result"].update(gpu_item["result"])
        result_pack[key]["pred_test_dict"].update(gpu_item["pred_test_dict"])
        result_pack[key]["pred_validation_dict"].update(gpu_item["pred_validation_dict"])

    for features_name, pca_name in tasks:
        key = (features_name, pca_name)

        result_pack[key]["result"] = add_ensemble_result(
            features_name=features_name,
            pca_name=pca_name,
            combo_result=result_pack[key]["result"],
            pred_test_dict=result_pack[key]["pred_test_dict"],
            pred_validation_dict=result_pack[key]["pred_validation_dict"],
            y_validation=result_pack[key]["y_validation"],
            y_test=result_pack[key]["y_test"],
            model_names=all_models,
            output_dir=output_dir,
            make_plots=make_plots,
        )

    result = {}

    for features_name in feature_sets:
        result[features_name] = {}

        for pca_name in pca_modes:
            key = (features_name, pca_name)
            result[features_name][pca_name] = result_pack[key]["result"]

    if result_json is None:
        result_json = output_dir / "result.json"

    save_json(result, result_json)

    return result


def summarize_auc_results(result: dict) -> pd.DataFrame:
    """
    Convert the nested experiment result dictionary into an AUC summary table.
    """
    rows = []

    for feature_set, pca_dict in result.items():
        for pca_mode, model_dict in pca_dict.items():
            for model_name, model_result in model_dict.items():
                rows.append({
                    "feature_set": feature_set,
                    "pca_mode": pca_mode,
                    "model": model_name,
                    "auc_test": model_result.get("auc_test"),
                })

    return (
        pd.DataFrame(rows)
        .sort_values("auc_test", ascending=False)
        .reset_index(drop=True)
    )
