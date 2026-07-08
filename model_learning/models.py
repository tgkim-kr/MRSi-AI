from __future__ import annotations

import os
import pickle
import uuid
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import seaborn as sns

from .utils import ensure_dir, find_best_cutoff, safe_filename, save_roc_curve


def _require_lightgbm():
    try:
        import lightgbm as lgb
    except ImportError as exc:
        raise ImportError(
            "LightGBM is required for the LGBM model. "
            "Install it with `pip install lightgbm` or remove LGBM from the model list."
        ) from exc

    return lgb


def _require_xgboost():
    try:
        import xgboost as xgb
    except ImportError as exc:
        raise ImportError(
            "XGBoost is required for the XGB model. "
            "Install it with `pip install xgboost` or remove XGB from the model list."
        ) from exc

    return xgb


def _require_shap():
    try:
        import shap
    except ImportError as exc:
        raise ImportError(
            "SHAP is required to generate SHAP summary plots. "
            "Install it with `pip install shap` or call the model with `make_shap=False`."
        ) from exc

    return shap


def _require_tensorflow():
    try:
        from tensorflow.keras.layers import Dense, Dropout, Input, SimpleRNN
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.optimizers import Adam
        import tensorflow as tf
    except ImportError as exc:
        raise ImportError(
            "TensorFlow is required for ANN/RNN models. "
            "Install it with `pip install tensorflow` or remove ANN/RNN from the model list."
        ) from exc

    return tf, Sequential, Input, Dense, Dropout, SimpleRNN, Adam


def _one_hot_align(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    x_validation: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    One-hot encode categorical columns using a combined column space.
    """
    combined = pd.concat(
        [
            x_train.copy().assign(__split__="train"),
            x_test.copy().assign(__split__="test"),
            x_validation.copy().assign(__split__="validation"),
        ],
        axis=0,
    )

    cat_cols = combined.select_dtypes(
        include=["category", "object", "string"]
    ).columns.tolist()
    cat_cols = [col for col in cat_cols if col != "__split__"]

    combined = pd.get_dummies(
        combined,
        columns=cat_cols,
        dummy_na=True,
    )

    x_train_encoded = (
        combined[combined["__split__"] == "train"]
        .drop(columns=["__split__"])
        .astype("float32")
    )
    x_test_encoded = (
        combined[combined["__split__"] == "test"]
        .drop(columns=["__split__"])
        .astype("float32")
    )
    x_validation_encoded = (
        combined[combined["__split__"] == "validation"]
        .drop(columns=["__split__"])
        .astype("float32")
    )

    return x_train_encoded, x_test_encoded, x_validation_encoded


def _align_categorical_columns_for_boosting(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    x_validation: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Align category levels across train/test/validation for categorical boosting.

    XGBoost and LightGBM can use pandas category dtype, but category columns
    should have consistent categories across splits.
    """
    combined = pd.concat(
        [
            x_train.copy().assign(__split__="train"),
            x_test.copy().assign(__split__="test"),
            x_validation.copy().assign(__split__="validation"),
        ],
        axis=0,
    )

    cat_cols = combined.select_dtypes(
        include=["category", "object", "string"]
    ).columns.tolist()
    cat_cols = [col for col in cat_cols if col != "__split__"]

    for col in cat_cols:
        combined[col] = combined[col].astype("object").astype("category")

    x_train_aligned = combined[combined["__split__"] == "train"].drop(columns=["__split__"])
    x_test_aligned = combined[combined["__split__"] == "test"].drop(columns=["__split__"])
    x_validation_aligned = combined[combined["__split__"] == "validation"].drop(columns=["__split__"])

    return x_train_aligned, x_test_aligned, x_validation_aligned


def _extract_lgbm_feature_importance(final_model: Any, x_train: pd.DataFrame) -> dict[str, float]:
    feature_importance = final_model.feature_importance()
    return {
        feature: float(importance)
        for feature, importance in zip(x_train.columns, feature_importance)
    }


def LGBM(
    x_train,
    y_train,
    x_test,
    y_test,
    x_validation,
    y_validation,
    *,
    save_dir,
    random_state=123,
    make_plots=True,
    make_shap=True,
):
    """
    Train and evaluate a LightGBM classifier using a small manual grid search.
    """
    lgb = _require_lightgbm()
    save_dir = ensure_dir(save_dir)

    x_train, x_test, x_validation = _align_categorical_columns_for_boosting(
        x_train,
        x_test,
        x_validation,
    )

    train_data = lgb.Dataset(x_train, label=y_train)
    test_data = lgb.Dataset(x_test, label=y_test, reference=train_data)

    learning_rate_list = [0.01, 0.1]
    max_depth_list = [5, 30]
    num_leaves_list = [10, 100]
    feature_fraction_list = [0.5, 0.9]
    bagging_fraction_list = [0.5, 0.9]
    num_iterations_list = [50, 200]
    lambda_l1_list = [0, 0.1]
    lambda_l2_list = [0, 0.1]

    param_list = []
    best_auc = -np.inf
    best_params = None
    best_num_iterations = None

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
                                        "verbosity": -1,
                                        "random_state": random_state,
                                    }

                                    model = lgb.train(
                                        params,
                                        train_data,
                                        num_boost_round=num_iterations,
                                        valid_sets=[test_data],
                                        valid_names=["test"],
                                        callbacks=[
                                            lgb.early_stopping(
                                                stopping_rounds=10,
                                                verbose=False,
                                            ),
                                            lgb.log_evaluation(period=0),
                                        ],
                                    )

                                    y_pred_prob_test = model.predict(
                                        x_test,
                                        num_iteration=model.best_iteration,
                                    )
                                    auc_test = roc_auc_score(y_test, y_pred_prob_test)

                                    param_list.append({
                                        "learning_rate": learning_rate,
                                        "max_depth": max_depth,
                                        "num_leaves": num_leaves,
                                        "feature_fraction": feature_fraction,
                                        "bagging_fraction": bagging_fraction,
                                        "num_iterations": num_iterations,
                                        "lambda_l1": lambda_l1,
                                        "lambda_l2": lambda_l2,
                                        "AUC": float(auc_test),
                                    })

                                    if auc_test > best_auc:
                                        best_auc = float(auc_test)
                                        best_params = params.copy()
                                        best_num_iterations = num_iterations

    if best_params is None or best_num_iterations is None:
        raise RuntimeError("LGBM grid search failed to fit any model.")

    pd.DataFrame(param_list).sort_values("AUC", ascending=False).to_csv(
        save_dir / "parameter_search.csv",
        index=False,
    )

    final_model = lgb.train(
        best_params,
        train_data,
        num_boost_round=best_num_iterations,
        valid_sets=[test_data],
        valid_names=["test"],
        callbacks=[
            lgb.early_stopping(stopping_rounds=10, verbose=False),
            lgb.log_evaluation(period=0),
        ],
    )

    with (save_dir / "lgbm_model.pkl").open("wb") as f:
        pickle.dump(final_model, f)

    y_pred_prob = final_model.predict(
        x_validation,
        num_iteration=final_model.best_iteration,
    )

    performance = find_best_cutoff(y_validation, y_pred_prob)
    roc_auc = roc_auc_score(y_validation, y_pred_prob)
    importance = _extract_lgbm_feature_importance(final_model, x_train)

    if make_plots:
        save_roc_curve(y_validation, y_pred_prob, save_dir / "ROC_curve.jpg")

        feature_importance = pd.DataFrame({
            "Feature": x_train.columns,
            "Importance": final_model.feature_importance(),
        }).sort_values("Importance", ascending=False)

        plt.figure(figsize=(10, 18))
        sns.barplot(x="Importance", y="Feature", data=feature_importance)
        plt.title("Feature Importance (LightGBM)")
        plt.xlabel("Importance Score")
        plt.ylabel("Feature")
        plt.savefig(save_dir / "feature_importance.jpg", dpi=500, bbox_inches="tight")
        plt.close()

        try:
            fig, ax = plt.subplots(figsize=(30, 10))
            lgb.plot_tree(final_model, tree_index=0, ax=ax)
            plt.savefig(save_dir / "lgbm_tree.png", dpi=500, bbox_inches="tight")
            plt.close(fig)
        except Exception as exc:
            print(f"Warning: failed to save LightGBM tree plot: {exc}")

    if make_shap:
        shap = _require_shap()
        try:
            explainer = shap.TreeExplainer(final_model)
            shap_values = explainer(x_validation)

            fig = plt.figure(figsize=(10, 6))
            shap.plots.beeswarm(shap_values, show=False, max_display=20)
            fig.savefig(
                save_dir / "shap_summary_plot.png",
                dpi=300,
                bbox_inches="tight",
            )
            plt.close(fig)
        except Exception as exc:
            print(f"Warning: failed to save LightGBM SHAP plot: {exc}")

    best_params_return = best_params.copy()
    best_params_return["num_iterations"] = best_num_iterations
    best_params_return["best_iteration"] = final_model.best_iteration

    return best_auc, float(roc_auc), best_params_return, y_pred_prob, performance, importance


def XGB(
    x_train,
    y_train,
    x_test,
    y_test,
    x_validation,
    y_validation,
    *,
    save_dir,
    random_state=123,
    save_trees=False,
    make_plots=True,
    make_shap=True,
    run_label=None,
):
    """
    Train and evaluate an XGBoost classifier using a small manual grid search.
    """
    xgb = _require_xgboost()
    save_dir = ensure_dir(save_dir)

    x_train, x_test, x_validation = _align_categorical_columns_for_boosting(
        x_train,
        x_test,
        x_validation,
    )

    if run_label is None:
        run_label = f"pid{os.getpid()}__{uuid.uuid4().hex[:8]}"

    run_id = safe_filename(run_label)

    learning_rate_list = [0.01, 0.1]
    max_depth_list = [5, 30]
    n_estimators_list = [10, 100]
    colsample_bytree_list = [0.5, 1.0]
    subsample_list = [0.5, 1.0]
    reg_lambda_list = [0, 0.1]
    reg_alpha_list = [0, 0.1]

    param_list = []
    best_auc = -np.inf
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
                                    "enable_categorical": True,
                                    "tree_method": "hist",
                                    "random_state": random_state,
                                    "n_jobs": 1,
                                }

                                model = xgb.XGBClassifier(**params)

                                model.fit(
                                    x_train,
                                    y_train,
                                    eval_set=[(x_test, y_test)],
                                    verbose=False,
                                )

                                y_pred_prob_test = model.predict_proba(x_test)[:, 1]
                                auc_test = roc_auc_score(y_test, y_pred_prob_test)

                                param_list.append({
                                    "learning_rate": learning_rate,
                                    "max_depth": max_depth,
                                    "n_estimators": n_estimators,
                                    "colsample_bytree": colsample_bytree,
                                    "subsample": subsample,
                                    "reg_lambda": reg_lambda,
                                    "reg_alpha": reg_alpha,
                                    "AUC": float(auc_test),
                                })

                                if auc_test > best_auc:
                                    best_auc = float(auc_test)
                                    best_params = params.copy()

    if best_params is None:
        raise RuntimeError("XGB grid search failed to fit any model.")

    pd.DataFrame(param_list).sort_values("AUC", ascending=False).to_csv(
        save_dir / "parameter_search.csv",
        index=False,
    )

    final_model = xgb.XGBClassifier(**best_params)
    final_model.fit(
        x_train,
        y_train,
        eval_set=[(x_test, y_test)],
        verbose=False,
    )

    with (save_dir / "xgb_model.pkl").open("wb") as f:
        pickle.dump(final_model, f)

    y_pred_prob = final_model.predict_proba(x_validation)[:, 1]
    performance = find_best_cutoff(y_validation, y_pred_prob)
    roc_auc = roc_auc_score(y_validation, y_pred_prob)

    importance_scores = final_model.feature_importances_
    importance = {
        feature: float(score)
        for feature, score in zip(x_train.columns, importance_scores)
    }

    if make_plots:
        save_roc_curve(y_validation, y_pred_prob, save_dir / "ROC_curve.jpg")

        plt.figure(figsize=(50, 30))
        xgb.plot_importance(final_model)
        plt.title("Feature Importance (XGBoost)")
        plt.savefig(save_dir / "feature_importance.jpg", dpi=500, bbox_inches="tight")
        plt.close()

    if make_shap:
        try:
            x_shap = x_validation.copy()

            if len(x_shap) > 1000:
                x_shap = x_shap.sample(n=1000, random_state=random_state)

            dvalid = xgb.DMatrix(x_shap, enable_categorical=True)
            booster = final_model.get_booster()

            shap_contribs = booster.predict(dvalid, pred_contribs=True)
            shap_values = shap_contribs[:, :-1]

            shap = _require_shap()
            shap.summary_plot(shap_values, x_shap, show=False, max_display=20)

            plt.savefig(
                save_dir / "shap_summary_plot.png",
                dpi=300,
                bbox_inches="tight",
            )
            plt.close()
        except Exception as exc:
            print(f"Warning: failed to save XGBoost SHAP plot: {exc}")

    if save_trees:
        booster = final_model.get_booster()
        num_trees = booster.num_boosted_rounds()

        tree_dir = ensure_dir(save_dir / "trees")
        tree_txt_dir = ensure_dir(save_dir / "trees_txt")

        for i in range(num_trees):
            dot = xgb.to_graphviz(
                booster,
                num_trees=i,
                rankdir="LR",
            )

            tree_filename = f"tree_{run_id}_{i}"

            dot.render(
                filename=tree_filename,
                directory=tree_dir,
                format="png",
                cleanup=True,
            )

        trees = booster.get_dump(with_stats=True)

        for i, tree in enumerate(trees):
            tree_txt_path = tree_txt_dir / f"tree_{run_id}_{i}.txt"
            with tree_txt_path.open("w", encoding="utf-8") as f:
                f.write(tree)

    return best_auc, float(roc_auc), best_params, y_pred_prob, performance, importance


def LR(
    x_train,
    y_train,
    x_test,
    y_test,
    x_validation,
    y_validation,
    *,
    save_dir,
    random_state=123,
    make_plots=True,
):
    """
    Train and evaluate logistic regression using a small manual grid search.
    """
    save_dir = ensure_dir(save_dir)

    x_train, x_test, x_validation = _one_hot_align(x_train, x_test, x_validation)

    C_list = [0.001, 0.01, 0.1, 1.0, 10.0]
    penalty_list = ["l1", "l2"]

    param_list = []
    best_auc = -np.inf
    best_params = None

    for C in C_list:
        for penalty in penalty_list:
            try:
                solver = "liblinear" if penalty == "l1" else "lbfgs"
                model = LogisticRegression(
                    C=C,
                    penalty=penalty,
                    solver=solver,
                    max_iter=500,
                    random_state=random_state,
                )
                model.fit(x_train, y_train)

                y_pred_prob_test = model.predict_proba(x_test)[:, 1]
                auc_test = roc_auc_score(y_test, y_pred_prob_test)

                param_list.append({
                    "C": C,
                    "penalty": penalty,
                    "AUC": float(auc_test),
                })

                if auc_test > best_auc:
                    best_auc = float(auc_test)
                    best_params = {"C": C, "penalty": penalty}

            except Exception as exc:
                print(f"Warning: LR failed for C={C}, penalty={penalty}: {exc}")
                continue

    if best_params is None:
        raise RuntimeError("LR grid search failed to fit any model.")

    pd.DataFrame(param_list).sort_values("AUC", ascending=False).to_csv(
        save_dir / "parameter_search.csv",
        index=False,
    )

    final_model = LogisticRegression(
        C=best_params["C"],
        penalty=best_params["penalty"],
        solver="liblinear" if best_params["penalty"] == "l1" else "lbfgs",
        max_iter=500,
        random_state=random_state,
    )
    final_model.fit(x_train, y_train)

    with (save_dir / "lr_model.pkl").open("wb") as f:
        pickle.dump(final_model, f)

    y_pred_prob = final_model.predict_proba(x_validation)[:, 1]
    performance = find_best_cutoff(y_validation, y_pred_prob)
    roc_auc = roc_auc_score(y_validation, y_pred_prob)

    if make_plots:
        save_roc_curve(y_validation, y_pred_prob, save_dir / "ROC_curve.jpg")

    coef = final_model.coef_[0]
    feature_names = x_train.columns
    n = len(y_train)

    variance = np.var(x_train, axis=0)
    variance = variance.replace(0, np.nan)

    standard_errors = np.std(x_train, axis=0) * np.sqrt(
        (1 / n) + (np.mean(x_train, axis=0) ** 2 / variance)
    )
    z_scores = coef / standard_errors
    p_values = stats.norm.sf(np.abs(z_scores)) * 2

    importance = {
        feature: [float(weight), None if pd.isna(p_val) else float(p_val)]
        for feature, weight, p_val in zip(feature_names, coef, p_values)
    }

    return best_auc, float(roc_auc), best_params, y_pred_prob, performance, importance


def RF(
    x_train,
    y_train,
    x_test,
    y_test,
    x_validation,
    y_validation,
    *,
    save_dir,
    random_state=123,
    n_jobs=3,
    make_plots=True,
):
    """
    Train and evaluate a random forest classifier using a small manual grid search.
    """
    save_dir = ensure_dir(save_dir)

    x_train, x_test, x_validation = _one_hot_align(x_train, x_test, x_validation)

    n_estimators_list = [50, 100, 200]
    max_depth_list = [5, 10, 30, None]
    min_samples_split_list = [2, 5, 10]
    min_samples_leaf_list = [1, 2, 5]

    param_list = []
    best_auc = -np.inf
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
                        random_state=random_state,
                        n_jobs=n_jobs,
                    )

                    model.fit(x_train, y_train)

                    y_pred_prob_test = model.predict_proba(x_test)[:, 1]
                    auc_test = roc_auc_score(y_test, y_pred_prob_test)

                    param_list.append({
                        "n_estimators": n_estimators,
                        "max_depth": max_depth,
                        "min_samples_split": min_samples_split,
                        "min_samples_leaf": min_samples_leaf,
                        "AUC": float(auc_test),
                    })

                    if auc_test > best_auc:
                        best_auc = float(auc_test)
                        best_params = {
                            "n_estimators": n_estimators,
                            "max_depth": max_depth,
                            "min_samples_split": min_samples_split,
                            "min_samples_leaf": min_samples_leaf,
                        }

    if best_params is None:
        raise RuntimeError("RF grid search failed to fit any model.")

    pd.DataFrame(param_list).sort_values("AUC", ascending=False).to_csv(
        save_dir / "parameter_search.csv",
        index=False,
    )

    final_model = RandomForestClassifier(
        n_estimators=best_params["n_estimators"],
        max_depth=best_params["max_depth"],
        min_samples_split=best_params["min_samples_split"],
        min_samples_leaf=best_params["min_samples_leaf"],
        random_state=random_state,
        n_jobs=n_jobs,
    )

    final_model.fit(x_train, y_train)

    with (save_dir / "rf_model.pkl").open("wb") as f:
        pickle.dump(final_model, f)

    y_pred_prob = final_model.predict_proba(x_validation)[:, 1]
    performance = find_best_cutoff(y_validation, y_pred_prob)
    roc_auc = roc_auc_score(y_validation, y_pred_prob)

    importances = final_model.feature_importances_
    feature_names = x_train.columns
    importance = {
        feature: float(score)
        for feature, score in zip(feature_names, importances)
    }

    if make_plots:
        save_roc_curve(y_validation, y_pred_prob, save_dir / "ROC_curve.jpg")

        plt.figure(figsize=(10, 6))
        sns.barplot(x=importances, y=feature_names)
        plt.title("Feature Importance (Random Forest)")
        plt.xlabel("Importance Score")
        plt.ylabel("Feature")
        plt.savefig(save_dir / "feature_importance.jpg", dpi=500, bbox_inches="tight")
        plt.close()

    return best_auc, float(roc_auc), best_params, y_pred_prob, performance, importance


def ANN(
    x_train,
    y_train,
    x_test,
    y_test,
    x_validation,
    y_validation,
    *,
    save_dir,
    random_state=123,
    make_plots=True,
):
    """
    Train and evaluate a feed-forward neural network classifier.
    """
    tf, Sequential, Input, Dense, Dropout, _, Adam = _require_tensorflow()
    save_dir = ensure_dir(save_dir)

    tf.random.set_seed(random_state)
    np.random.seed(random_state)

    x_train, x_test, x_validation = _one_hot_align(x_train, x_test, x_validation)

    y_train = np.asarray(y_train).astype("float32")
    y_test = np.asarray(y_test).astype("float32")
    y_validation = np.asarray(y_validation).astype("float32")

    hidden_units_list = [64, 256]
    hidden_layers_list = [1, 3]
    learning_rate_list = [0.001, 0.01]
    batch_size_list = [32]

    param_list = []
    best_auc = -np.inf
    best_params = None

    for hidden_units in hidden_units_list:
        for hidden_layers in hidden_layers_list:
            for learning_rate in learning_rate_list:
                for batch_size in batch_size_list:
                    model = Sequential()
                    model.add(Input(shape=(x_train.shape[1],)))
                    model.add(Dense(hidden_units, activation="relu"))

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
                        x_train,
                        y_train,
                        validation_data=(x_test, y_test),
                        epochs=50,
                        batch_size=batch_size,
                        verbose=0,
                    )

                    y_pred_prob_test = model.predict(x_test, verbose=0).flatten()
                    auc_test = roc_auc_score(y_test, y_pred_prob_test)

                    param_list.append({
                        "hidden_units": hidden_units,
                        "hidden_layers": hidden_layers,
                        "learning_rate": learning_rate,
                        "batch_size": batch_size,
                        "AUC": float(auc_test),
                    })

                    if auc_test > best_auc:
                        best_auc = float(auc_test)
                        best_params = {
                            "hidden_units": hidden_units,
                            "hidden_layers": hidden_layers,
                            "learning_rate": learning_rate,
                            "batch_size": batch_size,
                        }

    if best_params is None:
        raise RuntimeError("ANN grid search failed to fit any model.")

    pd.DataFrame(param_list).sort_values("AUC", ascending=False).to_csv(
        save_dir / "parameter_search.csv",
        index=False,
    )

    final_model = Sequential()
    final_model.add(Input(shape=(x_train.shape[1],)))
    final_model.add(Dense(best_params["hidden_units"], activation="relu"))

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
        x_train,
        y_train,
        validation_data=(x_test, y_test),
        epochs=50,
        batch_size=best_params["batch_size"],
        verbose=0,
    )

    final_model.save(save_dir / "ann_model.keras")

    y_pred_prob = final_model.predict(x_validation, verbose=0).flatten()
    performance = find_best_cutoff(y_validation, y_pred_prob)
    roc_auc = roc_auc_score(y_validation, y_pred_prob)

    if make_plots:
        save_roc_curve(y_validation, y_pred_prob, save_dir / "ROC_curve.jpg")

    importance = {}

    return best_auc, float(roc_auc), best_params, y_pred_prob, performance, importance


def RNN(
    x_train,
    y_train,
    x_test,
    y_test,
    x_validation,
    y_validation,
    *,
    save_dir,
    random_state=123,
    make_plots=True,
):
    """
    Train and evaluate a simple RNN classifier.

    For tabular data, this treats the feature vector as a single time step.
    Categorical columns are one-hot encoded before reshaping.
    """
    tf, Sequential, _, Dense, Dropout, SimpleRNN, Adam = _require_tensorflow()
    save_dir = ensure_dir(save_dir)

    tf.random.set_seed(random_state)
    np.random.seed(random_state)

    x_train, x_test, x_validation = _one_hot_align(x_train, x_test, x_validation)

    num_features = x_train.shape[1]

    x_train = np.asarray(x_train, dtype=np.float32).reshape((x_train.shape[0], 1, num_features))
    x_test = np.asarray(x_test, dtype=np.float32).reshape((x_test.shape[0], 1, num_features))
    x_validation = np.asarray(x_validation, dtype=np.float32).reshape((x_validation.shape[0], 1, num_features))

    y_train = np.asarray(y_train, dtype=np.float32)
    y_test = np.asarray(y_test, dtype=np.float32)
    y_validation = np.asarray(y_validation, dtype=np.float32)

    hidden_units_list = [32, 128]
    hidden_layers_list = [1, 3]
    learning_rate_list = [0.001, 0.01]
    batch_size_list = [32]

    param_list = []
    best_auc = -np.inf
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
                        x_train,
                        y_train,
                        validation_data=(x_test, y_test),
                        epochs=10,
                        batch_size=batch_size,
                        verbose=0,
                    )

                    y_pred_prob_test = model.predict(x_test, verbose=0).flatten()
                    auc_test = roc_auc_score(y_test, y_pred_prob_test)

                    param_list.append({
                        "hidden_units": hidden_units,
                        "hidden_layers": hidden_layers,
                        "learning_rate": learning_rate,
                        "batch_size": batch_size,
                        "AUC": float(auc_test),
                    })

                    if auc_test > best_auc:
                        best_auc = float(auc_test)
                        best_params = {
                            "hidden_units": hidden_units,
                            "hidden_layers": hidden_layers,
                            "learning_rate": learning_rate,
                            "batch_size": batch_size,
                        }

    if best_params is None:
        raise RuntimeError("RNN grid search failed to fit any model.")

    pd.DataFrame(param_list).sort_values("AUC", ascending=False).to_csv(
        save_dir / "parameter_search.csv",
        index=False,
    )

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
        x_train,
        y_train,
        validation_data=(x_test, y_test),
        epochs=10,
        batch_size=best_params["batch_size"],
        verbose=0,
    )

    final_model.save(save_dir / "rnn_model.keras")

    y_pred_prob = final_model.predict(x_validation, verbose=0).flatten()
    performance = find_best_cutoff(y_validation, y_pred_prob)
    roc_auc = roc_auc_score(y_validation, y_pred_prob)

    if make_plots:
        save_roc_curve(y_validation, y_pred_prob, save_dir / "ROC_curve.jpg")

    importance = {}

    return best_auc, float(roc_auc), best_params, y_pred_prob, performance, importance


MODEL_REGISTRY = {
    "LR": LR,
    "RF": RF,
    "XGB": XGB,
    "LGBM": LGBM,
    "ANN": ANN,
    "RNN": RNN,
}
