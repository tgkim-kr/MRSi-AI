from typing import Any, Dict
import numpy as np
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
)


def convert_floats_and_ints(obj: Any) -> Any:
    """
    Recursively convert NumPy scalar types (float32, float64, int32, int64)
    into native Python float/int so that the object becomes JSON-serializable.
    """
    if isinstance(obj, dict):
        return {k: convert_floats_and_ints(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [convert_floats_and_ints(v) for v in obj]
    if isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    return obj


def find_best_cutoff(y_true, y_pred_prob) -> Dict[str, float]:
    """
    Search over probability thresholds and find the one that maximizes F1 score.

    Parameters
    ----------
    y_true : array-like
        True binary labels.
    y_pred_prob : array-like
        Predicted probabilities for the positive class.

    Returns
    -------
    dict
        Dictionary containing Cutoff, Precision, Recall, F1 Score, Accuracy.
    """
    best_cutoff = 0.0
    best_f1 = 0.0
    best_metrics: Dict[str, float] = {}

    for cutoff in np.arange(0.01, 1.0, 0.01):
        y_pred = (y_pred_prob >= cutoff).astype(int)

        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        accuracy = accuracy_score(y_true, y_pred)

        if f1 > best_f1:
            best_f1 = f1
            best_cutoff = float(cutoff)
            best_metrics = {
                "Cutoff": best_cutoff,
                "Precision": float(precision),
                "Recall": float(recall),
                "F1 Score": float(f1),
                "Accuracy": float(accuracy),
            }

    return best_metrics
