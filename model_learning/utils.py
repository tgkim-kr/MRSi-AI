from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.metrics import roc_auc_score, roc_curve


def convert_floats_and_ints(obj: Any) -> Any:
    """
    Convert NumPy scalar values to native Python values for JSON serialization.
    """
    if isinstance(obj, dict):
        return {k: convert_floats_and_ints(v) for k, v in obj.items()}

    if isinstance(obj, list):
        return [convert_floats_and_ints(v) for v in obj]

    if isinstance(obj, tuple):
        return tuple(convert_floats_and_ints(v) for v in obj)

    if isinstance(obj, np.floating):
        return float(obj)

    if isinstance(obj, np.integer):
        return int(obj)

    if isinstance(obj, np.ndarray):
        return obj.tolist()

    return obj


def save_json(obj: Any, output_path: str | os.PathLike[str]) -> None:
    """
    Save an object as UTF-8 JSON after converting NumPy scalar types.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(
            convert_floats_and_ints(obj),
            f,
            indent=4,
            ensure_ascii=False,
        )


def safe_filename(x: Any) -> str:
    """
    Return a filesystem-safe string for model/run identifiers.
    """
    return re.sub(r"[^a-zA-Z0-9가-힣_.-]+", "_", str(x))


def ensure_dir(path: str | os.PathLike[str]) -> Path:
    """
    Create a directory if needed and return it as a pathlib.Path object.
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def find_best_cutoff(
    y_validation_true,
    y_validation_pred_prob,
    y_test_true,
    y_test_pred_prob,
    cutoff_grid: np.ndarray | None = None,
) -> dict[str, float]:
    """
    Find the best probability cutoff on the validation set,
    then evaluate precision, recall, F1, and accuracy on the test set.

    Parameters
    ----------
    y_validation_true:
        Ground-truth binary labels for the validation set.
        Used only to select the best cutoff.

    y_validation_pred_prob:
        Predicted positive-class probabilities for the validation set.
        Used only to select the best cutoff.

    y_test_true:
        Ground-truth binary labels for the test set.
        Used to calculate final performance metrics.

    y_test_pred_prob:
        Predicted positive-class probabilities for the test set.
        Used to calculate final performance metrics using the
        validation-derived cutoff.

    cutoff_grid:
        Candidate cutoffs. If None, uses 0.01 to 0.99.

    Returns
    -------
    dict
        Best cutoff selected from the validation set and test-set precision,
        recall, F1, and accuracy.
    """
    if cutoff_grid is None:
        cutoff_grid = np.arange(0.01, 1.0, 0.01)

    y_validation_true = np.asarray(y_validation_true).reshape(-1)
    y_validation_pred_prob = np.asarray(y_validation_pred_prob).reshape(-1)

    y_test_true = np.asarray(y_test_true).reshape(-1)
    y_test_pred_prob = np.asarray(y_test_pred_prob).reshape(-1)

    if len(y_validation_true) != len(y_validation_pred_prob):
        raise ValueError(
            "y_validation_true and y_validation_pred_prob lengths differ: "
            f"{len(y_validation_true)} vs {len(y_validation_pred_prob)}"
        )

    if len(y_test_true) != len(y_test_pred_prob):
        raise ValueError(
            "y_test_true and y_test_pred_prob lengths differ: "
            f"{len(y_test_true)} vs {len(y_test_pred_prob)}"
        )

    # ------------------------------------------------------------
    # 1. Find best cutoff using validation set
    # ------------------------------------------------------------
    best_cutoff = 0.5
    best_validation_f1 = -1.0

    for cutoff in cutoff_grid:
        y_validation_pred = (y_validation_pred_prob >= cutoff).astype(int)
        validation_f1 = f1_score(
            y_validation_true,
            y_validation_pred,
            zero_division=0,
        )

        if validation_f1 > best_validation_f1:
            best_validation_f1 = validation_f1
            best_cutoff = float(cutoff)

    # ------------------------------------------------------------
    # 2. Evaluate test performance using validation-derived cutoff
    # ------------------------------------------------------------
    y_test_pred = (y_test_pred_prob >= best_cutoff).astype(int)

    precision = precision_score(y_test_true, y_test_pred, zero_division=0)
    recall = recall_score(y_test_true, y_test_pred, zero_division=0)
    f1 = f1_score(y_test_true, y_test_pred, zero_division=0)
    accuracy = accuracy_score(y_test_true, y_test_pred)

    return {
        "Cutoff": float(best_cutoff),
        "Precision": float(precision),
        "Recall": float(recall),
        "F1 Score": float(f1),
        "Accuracy": float(accuracy),
    }


def save_roc_curve(
    y_true,
    y_pred_prob,
    output_path: str | os.PathLike[str],
    title: str = "Receiver Operating Characteristic (ROC)",
    dpi: int = 500,
) -> float:
    """
    Save a ROC curve and return test AUC.
    """
    y_true = np.asarray(y_true).reshape(-1)
    y_pred_prob = np.asarray(y_pred_prob).reshape(-1)

    if len(y_true) != len(y_pred_prob):
        raise ValueError(
            "y_true and y_pred_prob lengths differ: "
            f"{len(y_true)} vs {len(y_pred_prob)}"
        )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fpr, tpr, _ = roc_curve(y_true, y_pred_prob)
    roc_auc = roc_auc_score(y_true, y_pred_prob)

    plt.figure()
    plt.plot(fpr, tpr, "r--", label=f"Test AUC = {roc_auc:.4f}")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.savefig(output_path, dpi=dpi)
    plt.close()

    return float(roc_auc)


def validate_prediction_length(
    y_true,
    y_pred_prob,
    context: str,
) -> None:
    """
    Validate that prediction length matches target length.
    """
    y_true = np.asarray(y_true).reshape(-1)
    y_pred_prob = np.asarray(y_pred_prob).reshape(-1)

    if len(y_true) != len(y_pred_prob):
        raise ValueError(
            f"[{context}] prediction length does not match y_true length. "
            f"prediction: {y_pred_prob.shape}, y_true: {len(y_true)}"
        )
