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
import sklearn
import xgboost
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    auc,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, TimeSeriesSplit
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

SEED = 42
app = FastAPI(title="LÚCIDA Science API", version="7.2")
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


@app.get("/")
def root():
    return {"service": "LÚCIDA Science API", "version": "7.2", "status": "online"}


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
    for algorithm, estimator in _models(imbalance == "class_weight", positive_weight):
        started = time.perf_counter()
        fold_metrics = []
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
        probability = final_pipeline.predict_proba(X_test)[:, 1]
        prediction = (probability >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_test, prediction, labels=[0, 1]).ravel()
        fpr_curve, tpr_curve, _ = roc_curve(y_test, probability)
        precision_curve, recall_curve, _ = precision_recall_curve(y_test, probability)
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
        }
        results.append({
            "algorithm": algorithm,
            "seconds": float(time.perf_counter() - started),
            **{metric: _summary([fold[metric] for fold in fold_metrics]) for metric in ("recall", "f1", "auc_roc", "auc_pr")},
            "fold_metrics": fold_metrics,
            "holdout": holdout,
        })
    champion = max(results, key=lambda result: result["auc_pr"]["mean"] + result["f1"]["mean"])["algorithm"]
    return {
        "schema_version": "7.2",
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
    }
