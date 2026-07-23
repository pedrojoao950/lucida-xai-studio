from __future__ import annotations

import hashlib
import json
import platform
import time
import uuid
from typing import Literal

import lightgbm
import numpy as np
import pandas as pd
import shap
import sklearn
import xgboost
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler
from lightgbm import LGBMClassifier
from lime.lime_tabular import LimeTabularExplainer
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    auc,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    log_loss,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, TimeSeriesSplit
from sklearn.inspection import partial_dependence
from sklearn.isotonic import IsotonicRegression
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

SEED = 42
app = FastAPI(title="LÚCIDA Science API", version="7.4")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:5173",
        "https://explainable-ai-studio.pedrojoao950.chatgpt.site",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _preprocessor(
    frame: pd.DataFrame,
    encoding: str,
    scaling: str,
    numeric_imputer: str,
    categorical_imputer: str,
) -> ColumnTransformer:
    numeric = frame.select_dtypes(include=np.number).columns.tolist()
    categorical = [column for column in frame.columns if column not in numeric]
    scaler = (
        StandardScaler()
        if scaling == "standard"
        else MinMaxScaler()
        if scaling == "minmax"
        else "passthrough"
    )
    number_pipe = SkPipeline(
        [("imputer", SimpleImputer(strategy=numeric_imputer)), ("scaler", scaler)]
    )
    category_imputer = (
        SimpleImputer(strategy="constant", fill_value="Ausente")
        if categorical_imputer == "constant"
        else SimpleImputer(strategy="most_frequent")
    )
    encoder = (
        OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        if encoding == "ordinal"
        else OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    )
    category_pipe = SkPipeline([("imputer", category_imputer), ("encoder", encoder)])
    return ColumnTransformer(
        [("numeric", number_pipe, numeric), ("categorical", category_pipe, categorical)],
        remainder="drop",
    )


def _models(class_weight: bool, positive_weight: float):
    weight = "balanced" if class_weight else None
    return [
        ("Regressão Logística", LogisticRegression(max_iter=2000, class_weight=weight, random_state=SEED)),
        ("Random Forest", RandomForestClassifier(n_estimators=250, class_weight=weight, random_state=SEED, n_jobs=-1)),
        ("Gradient Boosting", GradientBoostingClassifier(random_state=SEED)),
        ("XGBoost", XGBClassifier(n_estimators=250, max_depth=5, learning_rate=.05, subsample=.9, colsample_bytree=.9, scale_pos_weight=positive_weight if class_weight else 1, random_state=SEED, n_jobs=-1)),
        ("LightGBM", LGBMClassifier(n_estimators=250, learning_rate=.05, class_weight=weight, random_state=SEED, verbosity=-1, n_jobs=-1)),
    ]


def _summary(values: list[float]) -> dict[str, float]:
    return {"mean": float(np.mean(values)), "std": float(np.std(values))}


def _points(first: np.ndarray, second: np.ndarray, limit: int = 120):
    positions = np.unique(np.linspace(0, len(first) - 1, min(limit, len(first))).astype(int))
    return [{"x": float(first[i]), "y": float(second[i])} for i in positions]


def _apply_calibrator(method: str, train_probability, train_y, probability):
    if method == "none":
        return np.asarray(probability), None
    if method == "isotonic":
        fitted = IsotonicRegression(out_of_bounds="clip").fit(train_probability, train_y)
        return np.asarray(fitted.predict(probability)), fitted
    fitted = LogisticRegression(random_state=SEED).fit(np.asarray(train_probability).reshape(-1, 1), train_y)
    return fitted.predict_proba(np.asarray(probability).reshape(-1, 1))[:, 1], fitted


def _threshold_point(y_true, probability, candidate: float):
    prediction = (np.asarray(probability) >= candidate).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, prediction, labels=[0, 1]).ravel()
    return {
        "threshold": float(candidate),
        "recall": float(recall_score(y_true, prediction, zero_division=0)),
        "precision": float(precision_score(y_true, prediction, zero_division=0)),
        "f1": float(f1_score(y_true, prediction, zero_division=0)),
        "specificity": float(tn / max(1, tn + fp)),
        "false_positive_rate": float(fp / max(1, fp + tn)),
        "false_negative_rate": float(fn / max(1, fn + tp)),
        "predicted_positive": int(tp + fp),
    }


def _bootstrap_intervals(y_true, probability, threshold: float, samples: int = 400):
    truth = np.asarray(y_true)
    probability = np.asarray(probability)
    rng = np.random.default_rng(SEED)
    values = {metric: [] for metric in ("recall", "precision", "f1", "auc_roc", "auc_pr")}
    for _ in range(samples):
        index = rng.integers(0, len(truth), len(truth))
        sampled_y, sampled_probability = truth[index], probability[index]
        if len(np.unique(sampled_y)) < 2:
            continue
        sampled_prediction = (sampled_probability >= threshold).astype(int)
        precision_curve, recall_curve, _ = precision_recall_curve(sampled_y, sampled_probability)
        values["recall"].append(recall_score(sampled_y, sampled_prediction, zero_division=0))
        values["precision"].append(precision_score(sampled_y, sampled_prediction, zero_division=0))
        values["f1"].append(f1_score(sampled_y, sampled_prediction, zero_division=0))
        values["auc_roc"].append(roc_auc_score(sampled_y, sampled_probability))
        values["auc_pr"].append(auc(recall_curve, precision_curve))
    return {
        metric: {"lower": float(np.quantile(scores, .025)), "upper": float(np.quantile(scores, .975))}
        for metric, scores in values.items() if scores
    }


def _display_feature(name: str):
    return name.replace("numeric__", "").replace("categorical__", "").replace("_", " ")


@app.get("/")
def root():
    return {"service": "LÚCIDA Science API", "version": "7.4", "status": "online"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/v1/train")
async def train(
    file: UploadFile = File(...),
    target: str = Form(...),
    validation: Literal["stratified", "temporal"] = Form("stratified"),
    folds: int = Form(5),
    date_column: str = Form(""),
    encoding: Literal["onehot", "ordinal"] = Form("onehot"),
    scaling: Literal["standard", "minmax", "none"] = Form("standard"),
    numeric_imputer: Literal["median", "mean"] = Form("median"),
    categorical_imputer: Literal["most_frequent", "constant"] = Form("most_frequent"),
    imbalance: Literal["none", "class_weight", "smote", "undersampling"] = Form("class_weight"),
    excluded_features: str = Form("[]"),
    threshold: float = Form(.5),
    calibration: Literal["auto", "none", "isotonic", "platt"] = Form("auto"),
    protected_feature: str = Form(""),
    xai_model: str = Form(""),
):
    if folds not in (5, 10):
        raise HTTPException(422, "folds deve ser 5 ou 10")
    if not .05 <= threshold <= .95:
        raise HTTPException(422, "threshold deve estar entre 0.05 e 0.95")
    raw = await file.read()
    try:
        frame = pd.read_csv(pd.io.common.BytesIO(raw), sep=None, engine="python")
        excluded = json.loads(excluded_features)
    except Exception as error:
        raise HTTPException(422, f"CSV ou configuração inválida: {error}") from error
    if target not in frame:
        raise HTTPException(422, "Coluna-alvo não encontrada")
    if not isinstance(excluded, list):
        raise HTTPException(422, "excluded_features deve ser uma lista JSON")
    frame = frame.dropna(subset=[target]).drop_duplicates().reset_index(drop=True)
    if validation == "temporal":
        if not date_column or date_column not in frame:
            raise HTTPException(422, "Selecione uma coluna temporal válida")
        frame[date_column] = pd.to_datetime(frame[date_column], errors="coerce")
        frame = frame.dropna(subset=[date_column]).sort_values(date_column).reset_index(drop=True)
    excluded = [str(column) for column in excluded if column in frame and column != target]
    feature_columns = [column for column in frame.columns if column not in {target, *excluded}]
    if date_column in feature_columns:
        feature_columns.remove(date_column)
    if not feature_columns:
        raise HTTPException(422, "Nenhuma variável explicativa disponível")
    X = frame[feature_columns]
    labels = frame[target].astype(str)
    classes = labels.value_counts().index.tolist()
    if len(classes) != 2:
        raise HTTPException(422, "A fase 7.2 suporta classificação binária")
    positive_class = str(classes[-1])
    y = (labels == positive_class).astype(int)
    cut = int(len(frame) * .8)
    if cut < folds or len(frame) - cut < 2:
        raise HTTPException(422, "Dataset insuficiente para validação e holdout")
    X_dev, X_test, y_dev, y_test = X.iloc[:cut], X.iloc[cut:], y.iloc[:cut], y.iloc[cut:]
    negative, positive = np.bincount(y_dev, minlength=2)
    positive_weight = float(negative / max(1, positive))
    splitter = (
        TimeSeriesSplit(n_splits=folds)
        if validation == "temporal"
        else StratifiedKFold(n_splits=folds, shuffle=True, random_state=SEED)
    )
    split_iterator = splitter.split(X_dev, y_dev)
    splits = list(split_iterator)
    results = []
    fitted_artifacts = {}
    for algorithm, estimator in _models(imbalance == "class_weight", positive_weight):
        started = time.perf_counter()
        fold_metrics = []
        oof_probability = np.full(len(X_dev), np.nan)
        for fold_index, (train_index, valid_index) in enumerate(splits, 1):
            steps = [("preprocess", _preprocessor(X_dev, encoding, scaling, numeric_imputer, categorical_imputer))]
            if imbalance == "smote":
                steps.append(("balance", SMOTE(random_state=SEED)))
            elif imbalance == "undersampling":
                steps.append(("balance", RandomUnderSampler(random_state=SEED)))
            steps.append(("model", estimator))
            pipeline = Pipeline(steps)
            fit_options = {}
            if imbalance == "class_weight" and algorithm == "Gradient Boosting":
                fit_options["model__sample_weight"] = compute_sample_weight("balanced", y_dev.iloc[train_index])
            pipeline.fit(X_dev.iloc[train_index], y_dev.iloc[train_index], **fit_options)
            probability = pipeline.predict_proba(X_dev.iloc[valid_index])[:, 1]
            oof_probability[valid_index] = probability
            prediction = (probability >= threshold).astype(int)
            fold_metrics.append({
                "fold": fold_index,
                "recall": float(recall_score(y_dev.iloc[valid_index], prediction, zero_division=0)),
                "f1": float(f1_score(y_dev.iloc[valid_index], prediction, zero_division=0)),
                "auc_roc": float(roc_auc_score(y_dev.iloc[valid_index], probability)),
                "auc_pr": float(auc(*reversed(precision_recall_curve(y_dev.iloc[valid_index], probability)[:2]))),
            })
        final_steps = [("preprocess", _preprocessor(X_dev, encoding, scaling, numeric_imputer, categorical_imputer))]
        if imbalance == "smote":
            final_steps.append(("balance", SMOTE(random_state=SEED)))
        elif imbalance == "undersampling":
            final_steps.append(("balance", RandomUnderSampler(random_state=SEED)))
        final_steps.append(("model", estimator))
        final_fit_options = {}
        if imbalance == "class_weight" and algorithm == "Gradient Boosting":
            final_fit_options["model__sample_weight"] = compute_sample_weight("balanced", y_dev)
        final_pipeline = Pipeline(final_steps).fit(X_dev, y_dev, **final_fit_options)
        raw_probability = final_pipeline.predict_proba(X_test)[:, 1]
        calibration_mask = np.isfinite(oof_probability)
        development_probability = oof_probability[calibration_mask]
        development_y = y_dev.iloc[np.where(calibration_mask)[0]].to_numpy()
        calibration_splits = StratifiedKFold(n_splits=min(5, int(np.bincount(development_y).min())), shuffle=True, random_state=SEED)
        calibrated_oof = {"none": development_probability.copy()}
        selection_scores = {"none": float(brier_score_loss(development_y, development_probability))}
        for method in ("isotonic", "platt"):
            candidate_probability = np.full(len(development_y), np.nan)
            for calibration_train, calibration_valid in calibration_splits.split(development_probability, development_y):
                candidate_probability[calibration_valid], _ = _apply_calibrator(
                    method,
                    development_probability[calibration_train],
                    development_y[calibration_train],
                    development_probability[calibration_valid],
                )
            calibrated_oof[method] = candidate_probability
            selection_scores[method] = float(brier_score_loss(development_y, candidate_probability))
        selected_calibration = min(selection_scores, key=selection_scores.get) if calibration == "auto" else calibration
        probability, _ = _apply_calibrator(
            selected_calibration, development_probability, development_y, raw_probability
        )
        development_decision_probability = calibrated_oof[selected_calibration]
        development_threshold_curve = [
            _threshold_point(development_y, development_decision_probability, candidate)
            for candidate in np.linspace(.05, .95, 19)
        ]
        best_threshold = max(development_threshold_curve, key=lambda point: point["f1"])
        prediction = (probability >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_test, prediction, labels=[0, 1]).ravel()
        fpr_curve, tpr_curve, _ = roc_curve(y_test, probability)
        precision_curve, recall_curve, _ = precision_recall_curve(y_test, probability)
        fraction_positive, mean_predicted = calibration_curve(y_test, probability, n_bins=8, strategy="quantile")
        fairness = None
        if protected_feature and protected_feature in X_test:
            fairness_groups = []
            for group in sorted(X_test[protected_feature].astype(str).unique()):
                group_mask = X_test[protected_feature].astype(str).to_numpy() == group
                group_y, group_prediction = y_test.to_numpy()[group_mask], prediction[group_mask]
                if not len(group_y):
                    continue
                gtn, gfp, gfn, gtp = confusion_matrix(group_y, group_prediction, labels=[0, 1]).ravel()
                fairness_groups.append({
                    "group": group,
                    "rows": int(group_mask.sum()),
                    "positive_rate": float(group_prediction.mean()),
                    "recall": float(recall_score(group_y, group_prediction, zero_division=0)),
                    "precision": float(precision_score(group_y, group_prediction, zero_division=0)),
                    "false_positive_rate": float(gfp / max(1, gfp + gtn)),
                    "false_negative_rate": float(gfn / max(1, gfn + gtp)),
                })
            if fairness_groups:
                fairness = {
                    "protected_feature": protected_feature,
                    "groups": fairness_groups,
                    "demographic_parity_gap": float(max(g["positive_rate"] for g in fairness_groups) - min(g["positive_rate"] for g in fairness_groups)),
                    "equal_opportunity_gap": float(max(g["recall"] for g in fairness_groups) - min(g["recall"] for g in fairness_groups)),
                    "fpr_gap": float(max(g["false_positive_rate"] for g in fairness_groups) - min(g["false_positive_rate"] for g in fairness_groups)),
                }
        holdout = {
            "rows": len(y_test),
            "recall": float(recall_score(y_test, prediction, zero_division=0)),
            "precision": float(precision_score(y_test, prediction, zero_division=0)),
            "f1": float(f1_score(y_test, prediction, zero_division=0)),
            "auc_roc": float(roc_auc_score(y_test, probability)),
            "auc_pr": float(auc(recall_curve, precision_curve)),
            "specificity": float(tn / max(1, tn + fp)),
            "false_positive_rate": float(fp / max(1, fp + tn)),
            "false_negative_rate": float(fn / max(1, fn + tp)),
            "predicted_positive": int(tp + fp),
            "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
            "roc_curve": _points(fpr_curve, tpr_curve),
            "pr_curve": _points(recall_curve, precision_curve),
            "confidence_intervals_95": _bootstrap_intervals(y_test, probability, threshold),
        }
        calibration_result = {
            "method_requested": calibration,
            "method_selected": selected_calibration,
            "selection_source": "cross_fitted_out_of_fold_development",
            "selection_brier": selection_scores,
            "accepted": selected_calibration != "none",
            "brier_before": float(brier_score_loss(y_test, raw_probability)),
            "brier_after": float(brier_score_loss(y_test, probability)),
            "log_loss_before": float(log_loss(y_test, np.clip(raw_probability, 1e-7, 1 - 1e-7))),
            "log_loss_after": float(log_loss(y_test, np.clip(probability, 1e-7, 1 - 1e-7))),
            "curve": [{"predicted": float(x), "observed": float(y)} for x, y in zip(mean_predicted, fraction_positive)],
        }
        recommended_holdout = _threshold_point(y_test, probability, best_threshold["threshold"])
        results.append({
            "algorithm": algorithm,
            "seconds": float(time.perf_counter() - started),
            **{metric: _summary([fold[metric] for fold in fold_metrics]) for metric in ("recall", "f1", "auc_roc", "auc_pr")},
            "fold_metrics": fold_metrics,
            "holdout": holdout,
            "calibration": calibration_result,
            "threshold_analysis": {
                "active_threshold": threshold,
                "recommended_threshold": best_threshold["threshold"],
                "criterion": "maximum_cross_fitted_oof_f1",
                "selection_source": "development_out_of_fold",
                "holdout_used_for_selection": False,
                "curve": development_threshold_curve,
                "holdout_at_recommended_threshold": recommended_holdout,
            },
            "fairness": fairness,
        })
        fitted_artifacts[algorithm] = {
            "pipeline": final_pipeline,
            "X_dev": X_dev,
            "X_test": X_test,
        }
    champion = max(results, key=lambda result: result["auc_pr"]["mean"] + result["f1"]["mean"])["algorithm"]
    explanation_model = xai_model if xai_model in fitted_artifacts else champion
    explanation_artifact = fitted_artifacts[explanation_model]
    explanation_pipeline = explanation_artifact["pipeline"]
    preprocess = explanation_pipeline.named_steps["preprocess"]
    model = explanation_pipeline.named_steps["model"]
    transformed_dev = preprocess.transform(explanation_artifact["X_dev"])
    transformed_test = preprocess.transform(explanation_artifact["X_test"])
    feature_names = [str(name) for name in preprocess.get_feature_names_out()]
    background = transformed_dev[: min(100, len(transformed_dev))]
    explained = transformed_test[: min(40, len(transformed_test))]
    try:
        if explanation_model == "Regressão Logística":
            shap_values = shap.LinearExplainer(model, background)(explained).values
        else:
            shap_values = shap.TreeExplainer(model)(explained).values
        if shap_values.ndim == 3:
            shap_values = shap_values[:, :, -1]
        global_importance = np.mean(np.abs(shap_values), axis=0)
        shap_result = [
            {"feature": feature_names[index], "display_feature": _display_feature(feature_names[index]), "mean_abs_shap": float(global_importance[index])}
            for index in np.argsort(global_importance)[::-1][:12]
        ]
    except Exception as error:
        shap_result = [{"status": "unavailable", "reason": str(error)[:180]}]
    try:
        lime_explainer = LimeTabularExplainer(
            np.asarray(background),
            feature_names=feature_names,
            class_names=["negative", positive_class],
            mode="classification",
            random_state=SEED,
        )
        lime_explanation = lime_explainer.explain_instance(
            np.asarray(explained[0]),
            model.predict_proba,
            num_features=min(10, len(feature_names)),
        )
        lime_result = [{"condition": condition, "display_condition": _display_feature(condition), "weight": float(weight), "scale": "transformed"} for condition, weight in lime_explanation.as_list()]
    except Exception as error:
        lime_result = [{"status": "unavailable", "reason": str(error)[:180]}]
    pdp_result = []
    for feature in explanation_artifact["X_dev"].select_dtypes(include=np.number).columns[:5]:
        try:
            dependence = partial_dependence(explanation_pipeline, explanation_artifact["X_dev"], [feature], grid_resolution=12)
            grid = dependence["grid_values"][0]
            average = dependence["average"][0]
            pdp_result.append({"feature": str(feature), "points": [{"value": float(x), "prediction": float(y)} for x, y in zip(grid, average)]})
        except Exception:
            continue
    return {
        "schema_version": "7.4",
        "experiment_id": str(uuid.uuid4()),
        "dataset_sha256": hashlib.sha256(raw).hexdigest(),
        "random_seed": SEED,
        "runtime": {
            "python": platform.python_version(),
            "scikit_learn": sklearn.__version__,
            "xgboost": xgboost.__version__,
            "lightgbm": lightgbm.__version__,
        },
        "positive_class": positive_class,
        "threshold": threshold,
        "validation": validation,
        "development_rows": len(X_dev),
        "test_rows": len(X_test),
        "pipeline": {
            "numeric_imputer": numeric_imputer,
            "categorical_imputer": categorical_imputer,
            "encoding": encoding,
            "scaling": scaling,
            "imbalance": imbalance,
            "excluded_features": excluded,
            "features_used": feature_columns,
            "date_column": date_column or None,
        },
        "results": results,
        "champion": champion,
        "scientific_protocol": {
            "holdout_locked": True,
            "calibration_selection": "cross_fitted_out_of_fold_development",
            "threshold_selection": "cross_fitted_out_of_fold_development",
            "holdout_used_for_selection": False,
            "confidence_intervals": "bootstrap_95_percent_400_resamples",
        },
        "explainability": {
            "source": "science-api",
            "status": "scientific",
            "model": explanation_model,
            "shap_global": shap_result,
            "lime_local": {
                "holdout_row": 0,
                "features": lime_result,
                "raw_instance": {
                    str(key): (None if pd.isna(value) else value.item() if hasattr(value, "item") else value)
                    for key, value in explanation_artifact["X_test"].iloc[0].to_dict().items()
                },
                "note": "Condições LIME usam a escala transformada; valores originais são fornecidos em raw_instance.",
            },
            "partial_dependence": pdp_result,
        },
    }
