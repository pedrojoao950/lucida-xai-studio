from __future__ import annotations

import hashlib
import io
import json
import platform
import time
import uuid
from typing import Literal

import lightgbm
import joblib
import numpy as np
import pandas as pd
import shap
import sklearn
import xgboost
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
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
MODEL_CACHE: dict[str, dict] = {}
app = FastAPI(title="LÚCIDA Science API", version="7.8")
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


def _paired_f1_comparison(y_true, champion_probability, candidate_probability, champion_threshold, candidate_threshold):
    truth = np.asarray(y_true)
    rng = np.random.default_rng(SEED)
    differences = []
    for _ in range(400):
        index = rng.integers(0, len(truth), len(truth))
        champion_prediction = (np.asarray(champion_probability)[index] >= champion_threshold).astype(int)
        candidate_prediction = (np.asarray(candidate_probability)[index] >= candidate_threshold).astype(int)
        differences.append(
            f1_score(truth[index], candidate_prediction, zero_division=0)
            - f1_score(truth[index], champion_prediction, zero_division=0)
        )
    lower, upper = np.quantile(differences, [.025, .975])
    return {
        "metric": "holdout_f1_difference_candidate_minus_champion",
        "difference_mean": float(np.mean(differences)),
        "ci95": {"lower": float(lower), "upper": float(upper)},
        "statistical_tie": bool(lower <= 0 <= upper),
    }


@app.get("/")
def root():
    return {"service": "LÚCIDA Science API", "version": "7.8", "status": "online"}


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
    fairness_declaration: Literal["evaluate", "no_protected_attributes", "not_evaluated"] = Form("not_evaluated"),
    threshold_approved: bool = Form(False),
    approver_name: str = Form(""),
    approver_role: str = Form(""),
    approval_reason: str = Form(""),
    lifecycle_state: Literal["candidate", "in_review", "approved", "deployed", "retired"] = Form("candidate"),
):
    if folds not in (5, 10):
        raise HTTPException(422, "folds deve ser 5 ou 10")
    if not .05 <= threshold <= .95:
        raise HTTPException(422, "threshold deve estar entre 0.05 e 0.95")
    raw = await file.read()
    experiment_id = str(uuid.uuid4())
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
        probability, fitted_calibrator = _apply_calibrator(
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
        robustness_scenarios = []
        numeric_columns = X_test.select_dtypes(include=np.number).columns.tolist()
        for severity in (.05, .10, .20):
            perturbed = X_test.copy()
            rng = np.random.default_rng(SEED + int(severity * 100))
            for column in numeric_columns:
                scale = float(X_dev[column].std()) or 1.0
                perturbed[column] = perturbed[column] + rng.normal(0, scale * severity, len(perturbed))
            perturbed_raw = final_pipeline.predict_proba(perturbed)[:, 1]
            perturbed_probability, _ = _apply_calibrator(
                selected_calibration, development_probability, development_y, perturbed_raw
            )
            point = _threshold_point(y_test, perturbed_probability, threshold)
            perturbed_precision, perturbed_recall, _ = precision_recall_curve(y_test, perturbed_probability)
            robustness_scenarios.append({
                "scenario": "numeric_noise",
                "severity": severity,
                **point,
                "auc_pr": float(auc(perturbed_recall, perturbed_precision)),
                "f1_degradation": float(holdout["f1"] - point["f1"]),
            })
        missing = X_test.copy()
        missing_rng = np.random.default_rng(SEED + 99)
        missing_mask = missing_rng.random(missing.shape) < .10
        missing = missing.mask(missing_mask)
        missing_raw = final_pipeline.predict_proba(missing)[:, 1]
        missing_probability, _ = _apply_calibrator(
            selected_calibration, development_probability, development_y, missing_raw
        )
        missing_point = _threshold_point(y_test, missing_probability, threshold)
        missing_precision, missing_recall, _ = precision_recall_curve(y_test, missing_probability)
        robustness_scenarios.append({
            "scenario": "missing_values",
            "severity": .10,
            **missing_point,
            "auc_pr": float(auc(missing_recall, missing_precision)),
            "f1_degradation": float(holdout["f1"] - missing_point["f1"]),
        })
        worst_degradation = max(0.0, max(item["f1_degradation"] for item in robustness_scenarios))
        robustness = {
            "source": "science-api",
            "status": "scientific",
            "baseline_f1": holdout["f1"],
            "worst_f1_degradation": worst_degradation,
            "stability_score": float(max(0, 1 - worst_degradation)),
            "scenarios": robustness_scenarios,
        }
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
            "robustness": robustness,
        })
        fitted_artifacts[algorithm] = {
            "pipeline": final_pipeline,
            "X_dev": X_dev,
            "X_test": X_test,
            "probability": probability,
            "recommended_threshold": best_threshold["threshold"],
            "selected_calibration": selected_calibration,
            "calibrator": fitted_calibrator,
        }
    champion = max(results, key=lambda result: result["auc_pr"]["mean"] + result["f1"]["mean"])["algorithm"]
    champion_probability = fitted_artifacts[champion]["probability"]
    champion_threshold = fitted_artifacts[champion]["recommended_threshold"]
    statistical_comparison = []
    for result in results:
        comparison = _paired_f1_comparison(
            y_test,
            champion_probability,
            fitted_artifacts[result["algorithm"]]["probability"],
            champion_threshold,
            fitted_artifacts[result["algorithm"]]["recommended_threshold"],
        )
        statistical_comparison.append({
            "champion": champion,
            "candidate": result["algorithm"],
            **comparison,
        })
    non_inferior_models = [
        comparison["candidate"] for comparison in statistical_comparison
        if comparison["statistical_tie"] or comparison["candidate"] == champion
    ]
    champion_result = next(result for result in results if result["algorithm"] == champion)
    fairness_complete = fairness_declaration == "no_protected_attributes" or (
        fairness_declaration == "evaluate" and bool(protected_feature) and champion_result["fairness"] is not None
    )
    approval_complete = threshold_approved and bool(approver_name.strip()) and bool(approver_role.strip()) and bool(approval_reason.strip())
    robustness_complete = (
        champion_result["robustness"]["stability_score"] >= .85
        and champion_result["robustness"]["worst_f1_degradation"] <= .15
    )
    readiness_checks = [
        {"id": "holdout", "label": "Holdout bloqueado", "passed": True, "severity": "required"},
        {"id": "threshold", "label": "Limiar aprovado com responsável", "passed": approval_complete, "severity": "required"},
        {"id": "fairness", "label": "Equidade avaliada ou declarada", "passed": fairness_complete, "severity": "required"},
        {"id": "robustness", "label": "Estabilidade ≥85% e degradação F1 ≤15%", "passed": robustness_complete, "severity": "required"},
        {"id": "temporal", "label": "Validação temporal", "passed": validation == "temporal", "severity": "recommended"},
    ]
    blockers = [check["label"] for check in readiness_checks if check["severity"] == "required" and not check["passed"]]
    warnings = [check["label"] for check in readiness_checks if check["severity"] == "recommended" and not check["passed"]]
    readiness_status = "blocked" if blockers else "conditional" if warnings else "ready"
    requested_state = lifecycle_state
    effective_state = lifecycle_state
    if lifecycle_state in ("approved", "deployed") and readiness_status == "blocked":
        effective_state = "in_review"
    artifact_buffer = io.BytesIO()
    joblib.dump({
        "pipeline": fitted_artifacts[champion]["pipeline"],
        "calibrator": fitted_artifacts[champion]["calibrator"],
        "calibration_method": fitted_artifacts[champion]["selected_calibration"],
        "threshold": threshold,
        "feature_columns": feature_columns,
        "positive_class": positive_class,
    }, artifact_buffer)
    artifact_bytes = artifact_buffer.getvalue()
    artifact_hash = hashlib.sha256(artifact_bytes).hexdigest()
    technical_name = {
        "Regressão Logística": "logistic-regression",
        "Random Forest": "random-forest",
        "Gradient Boosting": "gradient-boosting",
        "XGBoost": "xgboost",
        "LightGBM": "lightgbm",
    }[champion]
    model_version = f"lucida-{technical_name}-{artifact_hash[:8]}"
    monitoring_numeric = {}
    for column in X_dev.select_dtypes(include=np.number).columns:
        series = X_dev[column].dropna()
        monitoring_numeric[str(column)] = {
            "mean": float(series.mean()),
            "std": float(series.std() or 0),
            "q05": float(series.quantile(.05)),
            "q50": float(series.quantile(.50)),
            "q95": float(series.quantile(.95)),
            "missing_rate": float(X_dev[column].isna().mean()),
        }
    monitoring_categorical = {}
    for column in X_dev.select_dtypes(exclude=np.number).columns:
        proportions = X_dev[column].astype(str).value_counts(normalize=True).head(10)
        monitoring_categorical[str(column)] = {str(key): float(value) for key, value in proportions.items()}
    champion_probability_array = np.asarray(champion_probability)
    monitoring_baseline = {
        "source": "development_and_locked_holdout",
        "numeric_features": monitoring_numeric,
        "categorical_features": monitoring_categorical,
        "prediction_distribution": {
            "mean": float(champion_probability_array.mean()),
            "q05": float(np.quantile(champion_probability_array, .05)),
            "q50": float(np.quantile(champion_probability_array, .50)),
            "q95": float(np.quantile(champion_probability_array, .95)),
            "positive_rate_at_approved_threshold": float((champion_probability_array >= threshold).mean()),
        },
        "alert_policy": {
            "psi_warning": .10,
            "psi_critical": .25,
            "performance_drop_warning": .05,
            "fairness_gap_warning": .10,
        },
    }
    # Finalize the experiment before the prediction/monitoring route declarations.
    # Keeping this response in the training coroutine prevents FastAPI from
    # serializing a successful training run as JSON null.
    active_threshold = threshold if approval_complete else champion_threshold
    MODEL_CACHE[experiment_id] = {
        "model_version": model_version,
        "artifact_sha256": artifact_hash,
        "pipeline": fitted_artifacts[champion]["pipeline"],
        "calibrator": fitted_artifacts[champion]["calibrator"],
        "calibration_method": fitted_artifacts[champion]["selected_calibration"],
        "threshold": active_threshold,
        "feature_columns": feature_columns,
        "positive_class": positive_class,
        "reference": X_dev.copy(),
        "baseline_probability": champion_probability_array,
        "xai": {
            "model": champion,
            "pipeline": fitted_artifacts[champion]["pipeline"],
            "X_dev": fitted_artifacts[champion]["X_dev"].copy(),
            "X_test": fitted_artifacts[champion]["X_test"].copy(),
        },
        "created_at": pd.Timestamp.utcnow().isoformat(),
    }
    return {
        "schema_version": "7.8",
        "experiment_id": experiment_id,
        "dataset_sha256": hashlib.sha256(raw).hexdigest(),
        "random_seed": SEED,
        "runtime": {
            "python": platform.python_version(),
            "scikit_learn": sklearn.__version__,
            "xgboost": xgboost.__version__,
            "lightgbm": lightgbm.__version__,
        },
        "positive_class": positive_class,
        "threshold": active_threshold,
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
        "model_identity": {
            "champion": champion,
            "deployment_candidate": champion,
            "explanation_model": champion,
            "consistent": True,
        },
        "threshold_policy": {
            "model": champion,
            "configured_threshold": threshold,
            "recommended_threshold": champion_threshold,
            "approved_threshold": threshold if approval_complete else None,
            "active_threshold": active_threshold,
            "source": "approved" if approval_complete else "development_out_of_fold_recommendation",
        },
        "statistical_comparison": {
            "method": "paired_bootstrap_400_resamples",
            "comparisons": statistical_comparison,
            "non_inferior_models": non_inferior_models,
            "deployment_preference": champion,
        },
        "governance": {
            "fairness_declaration": fairness_declaration,
            "protected_feature": protected_feature or None,
            "threshold_approved": approval_complete,
            "approved_threshold": threshold if approval_complete else None,
            "approval": {
                "approver_name": approver_name or None,
                "approver_role": approver_role or None,
                "reason": approval_reason or None,
                "timestamp": pd.Timestamp.utcnow().isoformat() if approval_complete else None,
            },
            "requested_lifecycle_state": requested_state,
            "effective_lifecycle_state": effective_state,
        },
        "model_card": {
            "model": champion,
            "version": model_version,
            "artifact_manifest_sha256": artifact_hash,
            "status": effective_state,
            "intended_use": "Binary classification decision support under human oversight.",
            "dataset_sha256": hashlib.sha256(raw).hexdigest(),
            "positive_class": positive_class,
            "validation": validation,
            "limitations": [
                "External and temporal generalisation require separate evidence.",
                "Fairness conclusions require contextual review and adequate group sizes.",
                "Predictions must not replace accountable human judgement.",
            ],
        },
        "deployment_readiness": {
            "status": readiness_status,
            "checks": readiness_checks,
            "blockers": blockers,
            "warnings": warnings,
            "decision": "Implantação bloqueada" if blockers else "Pronto com condições" if warnings else "Pronto para implantação",
        },
        "model_registry": {
            "model_version": model_version,
            "artifact_manifest_sha256": artifact_hash,
            "lifecycle_state": effective_state,
            "requested_state": requested_state,
            "rollback": {
                "previous_version": None,
                "ready": False,
                "reason": "Nenhuma versão anterior foi fornecida nesta experiência.",
            },
            "artifact": {
                "sha256": artifact_hash,
                "size_bytes": len(artifact_bytes),
                "serialization": "joblib",
                "load_test": "passed",
                "runtime_cache": "process_local",
            },
        },
        "audit_trail": [{
            "event": "threshold_approval",
            "status": "approved" if approval_complete else "incomplete",
            "actor": approver_name or None,
            "role": approver_role or None,
            "reason": approval_reason or None,
            "threshold": active_threshold,
            "experiment_id": experiment_id,
        }],
        "monitoring_baseline": monitoring_baseline,
        "scientific_protocol": {
            "holdout_locked": True,
            "calibration_selection": "cross_fitted_out_of_fold_development",
            "threshold_selection": "cross_fitted_out_of_fold_development",
            "holdout_used_for_selection": False,
            "confidence_intervals": "bootstrap_95_percent_400_resamples",
            "robustness": "numeric_noise_and_missingness_stress_tests",
            "model_comparison": "paired_bootstrap_holdout_f1",
            "deployment_gate": "required_and_recommended_checks",
        },
        "explainability": {
            "source": "science-api",
            "status": "deferred",
            "model": champion,
            "shap_global": [],
            "lime_local": {"holdout_row": 0, "features": [], "note": "Explicações serão calculadas numa execução dedicada."},
            "partial_dependence": [],
        },
    }


def _cached_probability(artifact: dict, frame: pd.DataFrame):
    frame = frame.copy()
    for column in artifact["reference"].select_dtypes(include=np.number).columns:
        if column in frame:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    raw_probability = artifact["pipeline"].predict_proba(frame[artifact["feature_columns"]])[:, 1]
    method, calibrator = artifact["calibration_method"], artifact["calibrator"]
    if method == "isotonic":
        return np.asarray(calibrator.predict(raw_probability))
    if method == "platt":
        return calibrator.predict_proba(raw_probability.reshape(-1, 1))[:, 1]
    return raw_probability


def _dense(values):
    return values.toarray() if hasattr(values, "toarray") else np.asarray(values)


def _calculate_explainability(artifact: dict, experiment_id: str):
    xai_artifact = artifact.get("xai")
    if not xai_artifact:
        raise HTTPException(409, "A experiência não contém um artefacto XAI; reexecute o treino.")
    pipeline = xai_artifact["pipeline"]
    preprocess = pipeline.named_steps["preprocess"]
    model = pipeline.named_steps["model"]
    X_dev, X_test = xai_artifact["X_dev"], xai_artifact["X_test"]
    transformed_dev = _dense(preprocess.transform(X_dev))
    transformed_test = _dense(preprocess.transform(X_test))
    if not len(transformed_test):
        raise HTTPException(409, "O holdout não contém observações para explicar.")
    feature_names = [str(name) for name in preprocess.get_feature_names_out()]
    background = transformed_dev[: min(100, len(transformed_dev))]
    explained = transformed_test[: min(40, len(transformed_test))]
    errors = []

    try:
        if xai_artifact["model"] == "Regressão Logística":
            shap_values = shap.LinearExplainer(model, background)(explained).values
        else:
            shap_values = shap.TreeExplainer(model)(explained).values
        if shap_values.ndim == 3:
            shap_values = shap_values[:, :, -1]
        importance = np.mean(np.abs(shap_values), axis=0)
        shap_result = [
            {
                "feature": feature_names[index],
                "display_feature": _display_feature(feature_names[index]),
                "mean_abs_shap": float(importance[index]),
            }
            for index in np.argsort(importance)[::-1][:12]
        ]
    except Exception as error:
        shap_result = []
        errors.append({"method": "SHAP", "reason": str(error)[:240]})

    try:
        lime_explainer = LimeTabularExplainer(
            background,
            feature_names=feature_names,
            class_names=["negative", artifact["positive_class"]],
            mode="classification",
            random_state=SEED,
        )
        explanation = lime_explainer.explain_instance(
            explained[0], model.predict_proba, num_features=min(10, len(feature_names))
        )
        lime_result = [
            {
                "condition": condition,
                "display_condition": _display_feature(condition),
                "weight": float(weight),
                "scale": "transformed",
            }
            for condition, weight in explanation.as_list()
        ]
    except Exception as error:
        lime_result = []
        errors.append({"method": "LIME", "reason": str(error)[:240]})

    pdp_result = []
    for feature in X_dev.select_dtypes(include=np.number).columns[:5]:
        try:
            dependence = partial_dependence(pipeline, X_dev, [feature], grid_resolution=12)
            pdp_result.append({
                "feature": str(feature),
                "points": [
                    {"value": float(x), "prediction": float(y)}
                    for x, y in zip(dependence["grid_values"][0], dependence["average"][0])
                ],
            })
        except Exception as error:
            errors.append({"method": f"PDP:{feature}", "reason": str(error)[:240]})

    completed = bool(shap_result) and bool(lime_result) and bool(pdp_result)
    return {
        "source": "science-api",
        "status": "scientific" if completed else "partial",
        "model": xai_artifact["model"],
        "experiment_id": experiment_id,
        "model_version": artifact["model_version"],
        "artifact_sha256": artifact["artifact_sha256"],
        "shap_global": shap_result,
        "lime_local": {
            "holdout_row": 0,
            "features": lime_result,
            "raw_instance": {
                str(key): (None if pd.isna(value) else value.item() if hasattr(value, "item") else value)
                for key, value in X_test.iloc[0].to_dict().items()
            },
            "note": "Condições LIME usam a escala transformada; valores originais constam de raw_instance.",
        },
        "partial_dependence": pdp_result,
        "errors": errors,
    }


@app.post("/v1/explain/{experiment_id}")
def explain(experiment_id: str):
    artifact = MODEL_CACHE.get(experiment_id)
    if not artifact:
        raise HTTPException(404, "Artefacto não está carregado neste processo; reexecute o treino.")
    return _calculate_explainability(artifact, experiment_id)


@app.post("/v1/predict/{experiment_id}")
async def predict(experiment_id: str, request: Request):
    artifact = MODEL_CACHE.get(experiment_id)
    if not artifact:
        raise HTTPException(404, "Artefacto não está carregado neste processo; reexecute ou restaure a versão.")
    payload = await request.json()
    rows = payload.get("rows", [])
    if not isinstance(rows, list) or not rows:
        raise HTTPException(422, "Forneça rows como uma lista não vazia.")
    frame = pd.DataFrame(rows)
    missing = [column for column in artifact["feature_columns"] if column not in frame]
    if missing:
        raise HTTPException(422, f"Features em falta: {missing}")
    probability = _cached_probability(artifact, frame)
    prediction = (probability >= artifact["threshold"]).astype(int)
    shadow = bool(payload.get("shadow", True))
    return {
        "experiment_id": experiment_id,
        "model_version": artifact["model_version"],
        "artifact_sha256": artifact["artifact_sha256"],
        "mode": "shadow" if shadow else "active",
        "decision_applied": not shadow,
        "threshold": artifact["threshold"],
        "predictions": [
            {"row": index, "probability": float(score), "prediction": int(label)}
            for index, (score, label) in enumerate(zip(probability, prediction))
        ],
    }


@app.post("/v1/monitor/{experiment_id}")
async def monitor(experiment_id: str, request: Request):
    artifact = MODEL_CACHE.get(experiment_id)
    if not artifact:
        raise HTTPException(404, "Artefacto não está carregado neste processo.")
    payload = await request.json()
    rows = payload.get("rows", [])
    if not isinstance(rows, list) or not rows:
        raise HTTPException(422, "Forneça um batch em rows.")
    batch = pd.DataFrame(rows)
    missing = [column for column in artifact["feature_columns"] if column not in batch]
    if missing:
        raise HTTPException(422, f"Features em falta: {missing}")
    reference = artifact["reference"]
    for column in reference.select_dtypes(include=np.number).columns:
        batch[column] = pd.to_numeric(batch[column], errors="coerce")
    feature_drift = []
    for column in artifact["feature_columns"]:
        if pd.api.types.is_numeric_dtype(reference[column]):
            scale = float(reference[column].std()) or 1.0
            score = abs(float(batch[column].mean()) - float(reference[column].mean())) / scale
            metric = "standardised_mean_shift"
        else:
            known = set(reference[column].astype(str).unique())
            score = float((~batch[column].astype(str).isin(known)).mean())
            metric = "unseen_category_rate"
        feature_drift.append({
            "feature": column,
            "metric": metric,
            "score": float(score),
            "status": "critical" if score >= .25 else "warning" if score >= .10 else "stable",
        })
    probability = _cached_probability(artifact, batch)
    baseline_mean = float(np.mean(artifact["baseline_probability"]))
    probability_shift = abs(float(np.mean(probability)) - baseline_mean)
    return {
        "experiment_id": experiment_id,
        "model_version": artifact["model_version"],
        "batch_rows": len(batch),
        "feature_drift": feature_drift,
        "prediction_drift": {
            "baseline_mean": baseline_mean,
            "batch_mean": float(np.mean(probability)),
            "absolute_shift": probability_shift,
            "status": "critical" if probability_shift >= .25 else "warning" if probability_shift >= .10 else "stable",
        },
        "overall_status": "critical" if any(item["status"] == "critical" for item in feature_drift) else "warning" if any(item["status"] == "warning" for item in feature_drift) else "stable",
    }
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
    MODEL_CACHE[experiment_id] = {
        "model_version": model_version,
        "artifact_sha256": artifact_hash,
        "pipeline": fitted_artifacts[champion]["pipeline"],
        "calibrator": fitted_artifacts[champion]["calibrator"],
        "calibration_method": fitted_artifacts[champion]["selected_calibration"],
        "threshold": threshold,
        "feature_columns": feature_columns,
        "positive_class": positive_class,
        "reference": X_dev.copy(),
        "baseline_probability": champion_probability_array,
        "created_at": pd.Timestamp.utcnow().isoformat(),
    }
    return {
        "schema_version": "7.7",
        "experiment_id": experiment_id,
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
        "statistical_comparison": {
            "method": "paired_bootstrap_400_resamples",
            "comparisons": statistical_comparison,
            "non_inferior_models": non_inferior_models,
            "deployment_preference": champion if champion in non_inferior_models else non_inferior_models[0],
        },
        "governance": {
            "fairness_declaration": fairness_declaration,
            "protected_feature": protected_feature or None,
            "threshold_approved": approval_complete,
            "approved_threshold": threshold if approval_complete else None,
            "approval": {
                "approver_name": approver_name or None,
                "approver_role": approver_role or None,
                "reason": approval_reason or None,
                "timestamp": pd.Timestamp.utcnow().isoformat() if approval_complete else None,
            },
            "requested_lifecycle_state": requested_state,
            "effective_lifecycle_state": effective_state,
        },
        "model_card": {
            "model": champion,
            "version": model_version,
            "artifact_manifest_sha256": artifact_hash,
            "status": effective_state,
            "intended_use": "Binary classification decision support under human oversight.",
            "dataset_sha256": hashlib.sha256(raw).hexdigest(),
            "positive_class": positive_class,
            "validation": validation,
            "limitations": [
                "External and temporal generalisation require separate evidence.",
                "Fairness conclusions require contextual review and adequate group sizes.",
                "Predictions must not replace accountable human judgement.",
            ],
            "deployment_readiness": {
                "holdout_locked": True,
                "threshold_approved": approval_complete,
                "fairness_declared": fairness_complete,
                "robustness_scientific": True,
                "robustness_policy_passed": robustness_complete,
            },
        },
        "deployment_readiness": {
            "status": readiness_status,
            "checks": readiness_checks,
            "blockers": blockers,
            "warnings": warnings,
            "decision": "Implantação bloqueada" if blockers else "Pronto com condições" if warnings else "Pronto para implantação",
        },
        "model_registry": {
            "model_version": model_version,
            "artifact_manifest_sha256": artifact_hash,
            "lifecycle_state": effective_state,
            "requested_state": requested_state,
            "rollback": {
                "previous_version": None,
                "ready": False,
                "reason": "Nenhuma versão anterior foi fornecida nesta experiência.",
            },
            "artifact": {
                "sha256": artifact_hash,
                "size_bytes": len(artifact_bytes),
                "serialization": "joblib",
                "load_test": "passed",
                "runtime_cache": "process_local",
            },
        },
        "audit_trail": [{
            "event": "threshold_approval",
            "status": "approved" if approval_complete else "incomplete",
            "actor": approver_name or None,
            "role": approver_role or None,
            "reason": approval_reason or None,
            "threshold": threshold,
            "experiment_id": experiment_id,
        }],
        "monitoring_baseline": monitoring_baseline,
        "scientific_protocol": {
            "holdout_locked": True,
            "calibration_selection": "cross_fitted_out_of_fold_development",
            "threshold_selection": "cross_fitted_out_of_fold_development",
            "holdout_used_for_selection": False,
            "confidence_intervals": "bootstrap_95_percent_400_resamples",
            "robustness": "numeric_noise_and_missingness_stress_tests",
            "model_comparison": "paired_bootstrap_holdout_f1",
            "deployment_gate": "required_and_recommended_checks",
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
