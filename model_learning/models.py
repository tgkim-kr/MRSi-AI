import os
import pickle
from typing import Dict, Any, Tuple, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
import lightgbm as lgb
import xgboost as xgb
from xgboost import to_graphviz
from scipy import stats
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, SimpleRNN
from tensorflow.keras.optimizers import Adam

from .utils import find_best_cutoff


# -------------------------------------------------------------------
# Helper: ROC curve plotter
# -------------------------------------------------------------------
def _plot_and_save_roc(
    y_true,
    y_prob,
    save_path: str,
    title: str = "Receiver Operating Characteristic (ROC)",
) -> float:
    """
    Plot ROC curve and save it, returning the AUC.
    """
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc_val = float(roc_auc_score(y_true, y_prob))

    plt.figure()
    plt.plot(fpr, tpr, "r--", label=f"Validation AUC = {auc_val:.4f}")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.savefig(save_path, dpi=500)
    plt.close()

    return auc_val


# -------------------------------------------------------------------
# LightGBM
# -------------------------------------------------------------------
def _extract_lgbm_importance(final_model, x_train: pd.DataFrame) -> Dict[str, float]:
    """
    Extract feature importance from a trained LightGBM model.
    """
    feature_importance = final_model.feature_importance()
    features = x_train.columns
    importance_dict = {
        feature: float(importance)
        for feature, importance in zip(features, feature_importance)
    }
    return importance_dict


def train_lgbm_model(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    x_val: pd.DataFrame,
    y_val: pd.Series,
    features_tag: str,
    pca_tag: str,
    output_root: str = "result",
) -> Tuple[float, float, Dict[str, Any]]:
    """
    Train a LightGBM model with a grid search similar to the original script
    and save model artifacts (ROC, feature importance, SHAP, tree image).
    """
    y_train = y_train.astype(int)
    y_test = y_test.astype(int)
    y_val = y_val.astype(int)

    base_dir = os.path.join(output_root, features_tag, pca_tag, "LGBM")
    os.makedirs(base_dir, exist_ok=True)

    train_data = lgb.Dataset(x_train, label=y_train)
    test_data = lgb.Dataset(x_test, label=y_test)

    learning_rate_list = [0.01, 0.1]
    max_depth_list = [5, 30]
    num_leaves_list = [10, 100]
    feature_fraction_list = [0.5, 0.9]
    bagging_fraction_list = [0.5, 0.9]
    num_iterations_list = [50, 200]
    lambda_l1_list = [0.0, 0.1]
    lambda_l2_list = [0.0, 0.1]

    param_list = pd.DataFrame(
        columns=[
            "learning_rate",
            "max_depth",
            "num_leaves",
            "feature_fraction",
            "bagging_fraction",
            "num_iterations",
            "lambda_l1",
            "lambda_l2",
            "AUC",
        ]
    )

    best_auc = 0.0
    best_params = None

    for learning_rate in learning_rate_list:
        for max_depth in max_depth_list:
            for num_leaves in num_leaves_list:
                for feature_fraction in feature_fraction_list:
                    for bagging_fraction in bagging_fraction_list:
                        for num_iterations in num_iterations_list:
                            for lambda_l1 in lambda_l1_list:
                                for lambda_l2 in lambda_l2_list:
                                    params = {
                                        "learning_rate": learning_rate,
                                        "max_depth": max_depth,
                                        "boosting": "gbdt",
                                        "objective": "binary",
                                        "metric": "auc",
                                        "num_leaves": num_leaves,
                                        "feature_fraction": feature_fraction,
                                        "bagging_fraction": bagging_fraction,
                                        "lambda_l1": lambda_l1,
                                        "lambda_l2": lambda_l2,
                                    }

                                    model = lgb.train(
                                        params,
                                        train_data,
                                        num_iterations,
                                        valid_sets=[test_data],
                                        verbose_eval=False,
                                        early_stopping_rounds=10,
                                    )

                                    y_pred_prob = model.predict(x_test)
                                    auc = roc_auc_score(y_test, y_pred_prob)

                                    param_list = pd.concat(
                                        [
                                            param_list,
                                            pd.DataFrame(
                                                [
                                                    {
                                                        "learning_rate": learning_rate,
                                                        "max_depth": max_depth,
                                                        "num_leaves": num_leaves,
                                                        "feature_fraction": feature_fraction,
                                                        "bagging_fraction": bagging_fraction,
                                                        "num_iterations": num_iterations,
                                                        "lambda_l1": lambda_l1,
                                                        "lambda_l2": lambda_l2,
                                                        "AUC": auc,
                                                    }
                                                ]
                                            ),
                                        ],
                                        ignore_index=True,
                                    )

                                    if auc > best_auc:
                                        best_auc = float(auc)
                                        best_params = params

    param_list = param_list.sort_values(by=["AUC"], ascending=False).reset_index(
        drop=True
    )
    best_num_iterations = int(param_list.loc[0, "num_iterations"])

    final_model = lgb.train(
        best_params,
        train_data,
        best_num_iterations,
        valid_sets=[test_data],
        verbose_eval=False,
        early_stopping_rounds=10,
    )

    importance = _extract_lgbm_importance(final_model, x_train)

    y_pred_prob_val = final_model.predict(x_val)
    performance = find_best_cutoff(y_val, y_pred_prob_val)

    # ROC curve (validation)
    roc_auc_val = _plot_and_save_roc(
        y_val,
        y_pred_prob_val,
        save_path=os.path.join(base_dir, "ROC_curve.jpg"),
        title="ROC (LightGBM)",
    )

    # Feature importance barplot
    feature_importance_df = pd.DataFrame(
        {"Feature": x_train.columns, "Importance": final_model.feature_importance()}
    ).sort_values(by="Importance", ascending=False)

    plt.figure(figsize=(10, 18))
    sns.barplot(x="Importance", y="Feature", data=feature_importance_df)
    plt.title("Feature Importance (LightGBM)")
    plt.xlabel("Importance Score")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(os.path.join(base_dir, "feature_importance.jpg"), dpi=500)
    plt.close()

    # SHAP
    explainer = shap.TreeExplainer(final_model)
    shap_values = explainer(x_val)

    fig = plt.figure(figsize=(10, 6))
    shap.plots.beeswarm(shap_values, show=False, max_display=20)
    fig.savefig(
        os.path.join(base_dir, "shap_summary_plot.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)

    # Single tree plot
    fig, ax = plt.subplots(figsize=(30, 10))
    lgb.plot_tree(final_model, tree_index=0, ax=ax)
    plt.savefig(os.path.join(base_dir, "lgbm_tree.png"), dpi=500, bbox_inches="tight")
    plt.close()

    info = {
        "params": best_params,
        "performance": performance,
        "importance": importance,
        "y_pred_prob_val": y_pred_prob_val,
    }

    return best_auc, roc_auc_val, info


# -------------------------------------------------------------------
# XGBoost
# -------------------------------------------------------------------
def train_xgb_model(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    x_val: pd.DataFrame,
    y_val: pd.Series,
    features_tag: str,
    pca_tag: str,
    output_root: str = "result",
) -> Tuple[float, float, Dict[str, Any]]:
    """
    Train an XGBoost model with a small hyperparameter search and
    save model, feature importance, SHAP summary, and tree visualizations.
    """
    y_train = y_train.astype(int)
    y_test = y_test.astype(int)
    y_val = y_val.astype(int)

    base_dir = os.path.join(output_root, features_tag, pca_tag, "XGB")
    os.makedirs(base_dir, exist_ok=True)

    learning_rate_list = [0.1]
    max_depth_list = [5]
    n_estimators_list = [100]
    colsample_bytree_list = [0.5]
    subsample_list = [0.5]
    reg_lambda_list = [0.0]
    reg_alpha_list = [0.0]

    param_list = pd.DataFrame(
        columns=[
            "learning_rate",
            "max_depth",
            "n_estimators",
            "colsample_bytree",
            "subsample",
            "reg_lambda",
            "reg_alpha",
            "AUC",
        ]
    )
    best_auc = 0.0
    best_params = None

    for learning_rate in learning_rate_list:
        for max_depth in max_depth_list:
            for n_estimators in n_estimators_list:
                for colsample_bytree in colsample_bytree_list:
                    for subsample in subsample_list:
                        for reg_lambda in reg_lambda_list:
                            for reg_alpha in reg_alpha_list:
                                params = {
                                    "learning_rate": learning_rate,
                                    "max_depth": max_depth,
                                    "n_estimators": n_estimators,
                                    "colsample_bytree": colsample_bytree,
                                    "subsample": subsample,
                                    "reg_lambda": reg_lambda,
                                    "reg_alpha": reg_alpha,
                                    "objective": "binary:logistic",
                                    "eval_metric": "auc",
                                    "use_label_encoder": False,
                                    "enable_categorical": True,
                                    "tree_method": "hist",
                                }

                                model = xgb.XGBClassifier(**params)
                                model.fit(
                                    x_train,
                                    y_train,
                                    eval_set=[(x_test, y_test)],
                                    verbose=False,
                                )

                                y_pred_prob = model.predict_proba(x_test)[:, 1]
                                auc = roc_auc_score(y_test, y_pred_prob)

                                param_list = pd.concat(
                                    [
                                        param_list,
                                        pd.DataFrame(
                                            [
                                                {
                                                    "learning_rate": learning_rate,
                                                    "max_depth": max_depth,
                                                    "n_estimators": n_estimators,
                                                    "colsample_bytree": colsample_bytree,
                                                    "subsample": subsample,
                                                    "reg_lambda": reg_lambda,
                                                    "reg_alpha": reg_alpha,
                                                    "AUC": auc,
                                                }
                                            ]
                                        ),
                                    ],
                                    ignore_index=True,
                                )

                                if auc > best_auc:
                                    best_auc = float(auc)
                                    best_params = params

    param_list = param_list.sort_values(by=["AUC"], ascending=False).reset_index(
        drop=True
    )

    final_model = xgb.XGBClassifier(**best_params)
    final_model.fit(
        x_train,
        y_train,
        eval_set=[(x_test, y_test)],
        verbose=False,
    )

    # Save model
    with open(os.path.join(base_dir, "xgb_model.pkl"), "wb") as f:
        pickle.dump(final_model, f)

    # Validation predictions
    y_pred_prob_val = final_model.predict_proba(x_val)[:, 1]
    performance = find_best_cutoff(y_val, y_pred_prob_val)

    # ROC curve (validation)
    roc_auc_val = _plot_and_save_roc(
        y_val,
        y_pred_prob_val,
        save_path=os.path.join(base_dir, "ROC_curve.jpg"),
        title="ROC (XGBoost)",
    )

    # Feature importance plot
    plt.figure(figsize=(50, 30))
    xgb.plot_importance(final_model)
    plt.title("Feature Importance (XGBoost)")
    plt.tight_layout()
    plt.savefig(os.path.join(base_dir, "feature_importance.jpg"), dpi=500)
    plt.close()

    importance_scores = final_model.feature_importances_
    importance = {
        feature: float(score)
        for feature, score in zip(x_train.columns, importance_scores)
    }

    # SHAP summary plot
    explainer = shap.TreeExplainer(
        final_model, feature_perturbation="tree_path_dependent"
    )
    shap_values = explainer.shap_values(x_val)

    fig = plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, x_val, show=False, max_display=20)
    fig.savefig(
        os.path.join(base_dir, "shap_summary_plot.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)

    # Tree visualizations
    booster = final_model.get_booster()
    num_trees = booster.num_boosted_rounds()

    tree_dir = os.path.join(base_dir, "trees")
    os.makedirs(tree_dir, exist_ok=True)

    for i in range(num_trees):
        dot = to_graphviz(booster, num_trees=i, rankdir="LR")
        dot.render(
            filename=os.path.join(tree_dir, f"tree_{i}"),
            format="png",
            cleanup=True,
        )

    trees = booster.get_dump(with_stats=True)
    tree_txt_dir = os.path.join(base_dir, "trees_txt")
    os.makedirs(tree_txt_dir, exist_ok=True)

    for i, tree in enumerate(trees):
        with open(os.path.join(tree_txt_dir, f"tree_{i}.txt"), "w") as f:
            f.write(tree)

    info = {
        "params": best_params,
        "performance": performance,
        "importance": importance,
        "y_pred_prob_val": y_pred_prob_val,
    }

    return best_auc, roc_auc_val, info


# -------------------------------------------------------------------
# Logistic Regression
# -------------------------------------------------------------------
def train_lr_model(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    x_val: pd.DataFrame,
    y_val: pd.Series,
    features_tag: str,
    pca_tag: str,
    output_root: str = "result",
) -> Tuple[float, float, Dict[str, Any]]:
    """
    Train a Logistic Regression model with one-hot encoded features and
    a small grid search over C and penalty.
    """
    base_dir = os.path.join(output_root, features_tag, pca_tag, "LR")
    os.makedirs(base_dir, exist_ok=True)

    # One-hot encoding
    x_train_enc = pd.get_dummies(x_train)
    x_test_enc = pd.get_dummies(x_test)
    x_val_enc = pd.get_dummies(x_val)

    # Align columns
    x_train_enc, x_test_enc = x_train_enc.align(
        x_test_enc, join="left", axis=1, fill_value=0
    )
    x_train_enc, x_val_enc = x_train_enc.align(
        x_val_enc, join="left", axis=1, fill_value=0
    )

    y_train = y_train.astype(int)
    y_test = y_test.astype(int)
    y_val = y_val.astype(int)

    C_list = [0.001, 0.01, 0.1, 1.0, 10.0]
    penalty_list = ["l1", "l2"]

    param_list = pd.DataFrame({"C": [], "penalty": [], "AUC": []})

    best_auc = 0.0
    best_params = None

    for C in C_list:
        for penalty in penalty_list:
            try:
                solver = "liblinear" if penalty == "l1" else "lbfgs"
                model = LogisticRegression(
                    C=C, penalty=penalty, solver=solver, max_iter=500
                )
                model.fit(x_train_enc, y_train)

                y_pred_prob_test = model.predict_proba(x_test_enc)[:, 1]
                auc = roc_auc_score(y_test, y_pred_prob_test)

                param_list = pd.concat(
                    [param_list, pd.DataFrame([{"C": C, "penalty": penalty, "AUC": auc}])],
                    ignore_index=True,
                )

                if auc > best_auc:
                    best_auc = float(auc)
                    best_params = {"C": C, "penalty": penalty}

            except Exception as e:
                print(f"Error with C={C}, penalty={penalty}: {e}")
                continue

    final_model = LogisticRegression(
        C=best_params["C"],
        penalty=best_params["penalty"],
        solver="liblinear" if best_params["penalty"] == "l1" else "lbfgs",
        max_iter=500,
    )
    final_model.fit(x_train_enc, y_train)

    y_pred_prob_val = final_model.predict_proba(x_val_enc)[:, 1]
    performance = find_best_cutoff(y_val, y_pred_prob_val)
    roc_auc_val = roc_auc_score(y_val, y_pred_prob_val)

    # ROC curve
    _ = _plot_and_save_roc(
        y_val,
        y_pred_prob_val,
        save_path=os.path.join(base_dir, "ROC_curve.jpg"),
        title="ROC (Logistic Regression)",
    )

    # Coefficients and p-values (approximation, as in original code)
    x_train_numeric = x_train_enc
    coef = final_model.coef_[0]
    feature_names = x_train_numeric.columns
    n = len(y_train)

    # Simple standard error approximation (same logic as original)
    std_x = np.std(x_train_numeric, 0)
    var_x = np.var(x_train_numeric, 0)
    mean_x = np.mean(x_train_numeric, 0)

    standard_errors = std_x * np.sqrt((1.0 / n) + (mean_x**2 / (var_x + 1e-8)))
    z_scores = coef / (standard_errors + 1e-8)
    p_values = stats.norm.sf(np.abs(z_scores)) * 2.0

    importance = {
        feature: [float(weight), float(p_val)]
        for feature, weight, p_val in zip(feature_names, coef, p_values)
    }

    # Save model and encoded feature names
    with open(os.path.join(base_dir, "lr_model.pkl"), "wb") as f:
        pickle.dump(
            {
                "model": final_model,
                "feature_names": list(feature_names),
            },
            f,
        )

    info = {
        "params": best_params,
        "performance": performance,
        "importance": importance,
        "y_pred_prob_val": y_pred_prob_val,
    }

    return best_auc, float(roc_auc_val), info


# -------------------------------------------------------------------
# Random Forest
# -------------------------------------------------------------------
def train_rf_model(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    x_val: pd.DataFrame,
    y_val: pd.Series,
    features_tag: str,
    pca_tag: str,
    output_root: str = "result",
) -> Tuple[float, float, Dict[str, Any]]:
    """
    Train a RandomForestClassifier with grid search similar to the original code
    and save ROC and feature-importance plots.
    """
    base_dir = os.path.join(output_root, features_tag, pca_tag, "RF")
    os.makedirs(base_dir, exist_ok=True)

    y_train = y_train.astype(int)
    y_test = y_test.astype(int)
    y_val = y_val.astype(int)

    n_estimators_list = [50, 100, 200]
    max_depth_list = [5, 10, 30, None]
    min_samples_split_list = [2, 5, 10]
    min_samples_leaf_list = [1, 2, 5]

    param_list = pd.DataFrame(
        {
            "n_estimators": [],
            "max_depth": [],
            "min_samples_split": [],
            "min_samples_leaf": [],
            "AUC": [],
        }
    )

    best_auc = 0.0
    best_params = None

    for n_estimators in n_estimators_list:
        for max_depth in max_depth_list:
            for min_samples_split in min_samples_split_list:
                for min_samples_leaf in min_samples_leaf_list:
                    model = RandomForestClassifier(
                        n_estimators=n_estimators,
                        max_depth=max_depth,
                        min_samples_split=min_samples_split,
                        min_samples_leaf=min_samples_leaf,
                        random_state=42,
                        n_jobs=-1,
                    )

                    model.fit(x_train, y_train)

                    y_pred_prob_test = model.predict_proba(x_test)[:, 1]
                    auc = roc_auc_score(y_test, y_pred_prob_test)

                    param_list = pd.concat(
                        [
                            param_list,
                            pd.DataFrame(
                                [
                                    {
                                        "n_estimators": n_estimators,
                                        "max_depth": max_depth,
                                        "min_samples_split": min_samples_split,
                                        "min_samples_leaf": min_samples_leaf,
                                        "AUC": auc,
                                    }
                                ]
                            ),
                        ],
                        ignore_index=True,
                    )

                    if auc > best_auc:
                        best_auc = float(auc)
                        best_params = {
                            "n_estimators": n_estimators,
                            "max_depth": max_depth,
                            "min_samples_split": min_samples_split,
                            "min_samples_leaf": min_samples_leaf,
                        }

    param_list = param_list.sort_values(by=["AUC"], axis=0, ascending=False)
    param_list.reset_index(drop=True, inplace=True)

    final_model = RandomForestClassifier(
        n_estimators=best_params["n_estimators"],
        max_depth=best_params["max_depth"],
        min_samples_split=best_params["min_samples_split"],
        min_samples_leaf=best_params["min_samples_leaf"],
        random_state=42,
        n_jobs=-1,
    )
    final_model.fit(x_train, y_train)

    y_pred_prob_val = final_model.predict_proba(x_val)[:, 1]
    performance = find_best_cutoff(y_val, y_pred_prob_val)
    roc_auc_val = roc_auc_score(y_val, y_pred_prob_val)

    # ROC curve
    _ = _plot_and_save_roc(
        y_val,
        y_pred_prob_val,
        save_path=os.path.join(base_dir, "ROC_curve.jpg"),
        title="ROC (Random Forest)",
    )

    # Feature importance barplot
    importances = final_model.feature_importances_
    feature_names = x_train.columns

    plt.figure(figsize=(10, 6))
    sns.barplot(x=importances, y=feature_names)
    plt.title("Feature Importance (Random Forest)")
    plt.xlabel("Importance Score")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(os.path.join(base_dir, "feature_importance.jpg"), dpi=500)
    plt.close()

    importance_dict = {
        feature: float(importance)
        for feature, importance in zip(feature_names, importances)
    }

    # Save model
    with open(os.path.join(base_dir, "rf_model.pkl"), "wb") as f:
        pickle.dump(final_model, f)

    info = {
        "params": best_params,
        "performance": performance,
        "importance": importance_dict,
        "y_pred_prob_val": y_pred_prob_val,
    }

    return best_auc, float(roc_auc_val), info


# -------------------------------------------------------------------
# ANN
# -------------------------------------------------------------------
def train_ann_model(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    x_val: pd.DataFrame,
    y_val: pd.Series,
    features_tag: str,
    pca_tag: str,
    output_root: str = "result",
) -> Tuple[float, float, Dict[str, Any]]:
    """
    Train a feed-forward neural network (ANN) with a small grid search,
    similar to the original script.
    """
    base_dir = os.path.join(output_root, features_tag, pca_tag, "ANN")
    os.makedirs(base_dir, exist_ok=True)

    x_train_arr = np.asarray(x_train, dtype=np.float32)
    x_test_arr = np.asarray(x_test, dtype=np.float32)
    x_val_arr = np.asarray(x_val, dtype=np.float32)

    y_train_arr = np.asarray(y_train, dtype=np.float32)
    y_test_arr = np.asarray(y_test, dtype=np.float32)
    y_val_arr = np.asarray(y_val, dtype=np.float32)

    hidden_units_list = [64, 256]
    hidden_layers_list = [1, 3]
    learning_rate_list = [0.001, 0.01]
    batch_size_list = [32]

    param_list = pd.DataFrame(
        {
            "hidden_units": [],
            "hidden_layers": [],
            "learning_rate": [],
            "batch_size": [],
            "AUC": [],
        }
    )

    best_auc = 0.0
    best_params = None

    for hidden_units in hidden_units_list:
        for hidden_layers in hidden_layers_list:
            for learning_rate in learning_rate_list:
                for batch_size in batch_size_list:
                    model = Sequential()
                    model.add(
                        Dense(
                            hidden_units,
                            activation="relu",
                            input_shape=(x_train_arr.shape[1],),
                        )
                    )

                    for _ in range(hidden_layers - 1):
                        model.add(Dense(hidden_units, activation="relu"))

                    model.add(Dropout(0.2))
                    model.add(Dense(1, activation="sigmoid"))

                    model.compile(
                        optimizer=Adam(learning_rate=learning_rate),
                        loss="binary_crossentropy",
                        metrics=["AUC"],
                    )

                    model.fit(
                        x_train_arr,
                        y_train_arr,
                        validation_data=(x_test_arr, y_test_arr),
                        epochs=50,
                        batch_size=batch_size,
                        verbose=0,
                    )

                    y_pred_prob_test = model.predict(x_test_arr).flatten()
                    auc = roc_auc_score(y_test_arr, y_pred_prob_test)

                    param_list = pd.concat(
                        [
                            param_list,
                            pd.DataFrame(
                                [
                                    {
                                        "hidden_units": hidden_units,
                                        "hidden_layers": hidden_layers,
                                        "learning_rate": learning_rate,
                                        "batch_size": batch_size,
                                        "AUC": auc,
                                    }
                                ]
                            ),
                        ],
                        ignore_index=True,
                    )

                    if auc > best_auc:
                        best_auc = float(auc)
                        best_params = {
                            "hidden_units": hidden_units,
                            "hidden_layers": hidden_layers,
                            "learning_rate": learning_rate,
                            "batch_size": batch_size,
                        }

    param_list = param_list.sort_values(by=["AUC"], ascending=False).reset_index(
        drop=True
    )

    final_model = Sequential()
    final_model.add(
        Dense(
            best_params["hidden_units"],
            activation="relu",
            input_shape=(x_train_arr.shape[1],),
        )
    )

    for _ in range(best_params["hidden_layers"] - 1):
        final_model.add(Dense(best_params["hidden_units"], activation="relu"))

    final_model.add(Dropout(0.2))
    final_model.add(Dense(1, activation="sigmoid"))

    final_model.compile(
        optimizer=Adam(learning_rate=best_params["learning_rate"]),
        loss="binary_crossentropy",
        metrics=["AUC"],
    )

    final_model.fit(
        x_train_arr,
        y_train_arr,
        validation_data=(x_test_arr, y_test_arr),
        epochs=50,
        batch_size=best_params["batch_size"],
        verbose=0,
    )

    y_pred_prob_val = final_model.predict(x_val_arr).flatten()
    performance = find_best_cutoff(y_val_arr, y_pred_prob_val)
    roc_auc_val = roc_auc_score(y_val_arr, y_pred_prob_val)

    # ROC curve
    _ = _plot_and_save_roc(
        y_val_arr,
        y_pred_prob_val,
        save_path=os.path.join(base_dir, "ROC_curve.jpg"),
        title="ROC (ANN)",
    )

    # Save model
    final_model.save(os.path.join(base_dir, "ann_model.h5"))

    info = {
        "params": best_params,
        "performance": performance,
        "importance": {},  # non-trivial for ANN
        "y_pred_prob_val": y_pred_prob_val,
    }

    return best_auc, float(roc_auc_val), info


# -------------------------------------------------------------------
# RNN
# -------------------------------------------------------------------
def train_rnn_model(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    x_val: pd.DataFrame,
    y_val: pd.Series,
    features_tag: str,
    pca_tag: str,
    output_root: str = "result",
) -> Tuple[float, float, Dict[str, Any]]:
    """
    Train a SimpleRNN-based model on single-timestep sequences (shape: (N, 1, num_features)),
    mirroring the original RNN implementation.
    """
    base_dir = os.path.join(output_root, features_tag, pca_tag, "RNN")
    os.makedirs(base_dir, exist_ok=True)

    num_features = x_train.shape[1]

    x_train_arr = np.array(x_train, dtype=np.float32).reshape(
        (x_train.shape[0], 1, num_features)
    )
    x_test_arr = np.array(x_test, dtype=np.float32).reshape(
        (x_test.shape[0], 1, num_features)
    )
    x_val_arr = np.array(x_val, dtype=np.float32).reshape(
        (x_val.shape[0], 1, num_features)
    )

    y_train_arr = np.array(y_train, dtype=np.float32)
    y_test_arr = np.array(y_test, dtype=np.float32)
    y_val_arr = np.array(y_val, dtype=np.float32)

    hidden_units_list = [32, 128]
    hidden_layers_list = [1, 3]
    learning_rate_list = [0.001, 0.01]
    batch_size_list = [32]

    best_auc = 0.0
    best_params = None

    for hidden_units in hidden_units_list:
        for hidden_layers in hidden_layers_list:
            for learning_rate in learning_rate_list:
                for batch_size in batch_size_list:
                    model = Sequential()
                    model.add(
                        SimpleRNN(
                            hidden_units,
                            activation="tanh",
                            input_shape=(1, num_features),
                            return_sequences=(hidden_layers > 1),
                        )
                    )

                    for i in range(hidden_layers - 1):
                        is_last_layer = i == hidden_layers - 2
                        model.add(
                            SimpleRNN(
                                hidden_units,
                                activation="tanh",
                                return_sequences=not is_last_layer,
                            )
                        )

                    model.add(Dropout(0.2))
                    model.add(Dense(16, activation="relu"))
                    model.add(Dense(1, activation="sigmoid"))

                    model.compile(
                        optimizer=Adam(learning_rate=learning_rate),
                        loss="binary_crossentropy",
                        metrics=["AUC"],
                    )

                    model.fit(
                        x_train_arr,
                        y_train_arr,
                        validation_data=(x_test_arr, y_test_arr),
                        epochs=10,
                        batch_size=batch_size,
                        verbose=0,
                    )

                    y_pred_prob_test = model.predict(x_test_arr).flatten()
                    auc = roc_auc_score(y_test_arr, y_pred_prob_test)

                    if auc > best_auc:
                        best_auc = float(auc)
                        best_params = {
                            "hidden_units": hidden_units,
                            "hidden_layers": hidden_layers,
                            "learning_rate": learning_rate,
                            "batch_size": batch_size,
                        }

    final_model = Sequential()
    final_model.add(
        SimpleRNN(
            best_params["hidden_units"],
            activation="tanh",
            input_shape=(1, num_features),
            return_sequences=(best_params["hidden_layers"] > 1),
        )
    )

    for i in range(best_params["hidden_layers"] - 1):
        is_last_layer = i == best_params["hidden_layers"] - 2
        final_model.add(
            SimpleRNN(
                best_params["hidden_units"],
                activation="tanh",
                return_sequences=not is_last_layer,
            )
        )

    final_model.add(Dropout(0.2))
    final_model.add(Dense(16, activation="relu"))
    final_model.add(Dense(1, activation="sigmoid"))

    final_model.compile(
        optimizer=Adam(learning_rate=best_params["learning_rate"]),
        loss="binary_crossentropy",
        metrics=["AUC"],
    )

    final_model.fit(
        x_train_arr,
        y_train_arr,
        validation_data=(x_test_arr, y_test_arr),
        epochs=10,
        batch_size=best_params["batch_size"],
        verbose=0,
    )

    y_pred_prob_val = final_model.predict(x_val_arr).flatten()
    performance = find_best_cutoff(y_val_arr, y_pred_prob_val)
    roc_auc_val = roc_auc_score(y_val_arr, y_pred_prob_val)

    # ROC curve
    _ = _plot_and_save_roc(
        y_val_arr,
        y_pred_prob_val,
        save_path=os.path.join(base_dir, "ROC_curve.jpg"),
        title="ROC (RNN)",
    )

    # Save model
    final_model.save(os.path.join(base_dir, "rnn_model.h5"))

    info = {
        "params": best_params,
        "performance": performance,
        "importance": {},  # not defined for RNN here
        "y_pred_prob_val": y_pred_prob_val,
    }

    return best_auc, float(roc_auc_val), info
