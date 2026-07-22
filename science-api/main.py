"""Scientific training service for LÚCIDA Explainable AI Studio."""
from __future__ import annotations

import io
import time
from typing import Literal

import numpy as np
import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from lightgbm import LGBMClassifier
from pydantic import BaseModel
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from xgboost import XGBClassifier

app = FastAPI(title="LÚCIDA Science API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
    allow_origins=[
    "http://localhost:3000",
    "http://localhost:5173",
    "https://explainable-ai-studio.pedrojoao950.chatgpt.site",
],
)


class Health(BaseModel):
    status: str
    engine: str
    algorithms: list[str]


def _models(class_weight: str | None) -> dict[str, object]:
    return {
        "Regressão Logística": LogisticRegression(max_iter=2000, class_weight=class_weight, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=250, class_weight=class_weight, n_jobs=-1, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=150, random_state=42),
        "XGBoost": XGBClassifier(n_estimators=200, learning_rate=.05, max_depth=5, eval_metric="logloss", n_jobs=-1, random_state=42),
        "LightGBM": LGBMClassifier(n_estimators=200, learning_rate=.05, class_weight=class_weight, verbosity=-1, n_jobs=-1, random_state=42),
    }


def _preprocessor(frame: pd.DataFrame, encoding: str, scaling: str) -> ColumnTransformer:
    numeric = frame.select_dtypes(include=np.number).columns.tolist()
    categorical = frame.columns.difference(numeric).tolist()
    numeric_steps: list[tuple[str, object]] = [("imputer", SimpleImputer(strategy="median"))]
    if scaling == "standard":
        numeric_steps.append(("scaler", StandardScaler()))
    encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1) if encoding == "ordinal" else OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    return ColumnTransformer([
        ("numeric", Pipeline(numeric_steps), numeric),
        ("categorical", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("encoder", encoder)]), categorical),
    ], remainder="drop")


def _binary_target(series: pd.Series) -> tuple[np.ndarray, list[str]]:
    labels = series.astype(str).fillna("Ausente")
    classes = sorted(labels.unique().tolist())
    if len(classes) != 2:
        raise HTTPException(422, "A Fase 6 suporta inicialmente classificação binária.")
    return (labels == classes[1]).astype(int).to_numpy(), classes


@app.get("/health", response_model=Health)
def health() -> Health:
    return Health(status="ready", engine="python-scientific", algorithms=list(_models(None)))


@app.post("/v1/train")
async def train(
    file: UploadFile = File(...),
    target: str = Form(...),
    validation: Literal["stratified", "temporal"] = Form("stratified"),
    folds: int = Form(5),
    date_column: str | None = Form(None),
    encoding: Literal["onehot", "ordinal"] = Form("onehot"),
    scaling: Literal["standard", "none"] = Form("standard"),
    imbalance: Literal["class_weight", "none"] = Form("class_weight"),
):
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(415, "Envie um ficheiro CSV.")
    raw = await file.read()
    if len(raw) > 25 * 1024 * 1024:
        raise HTTPException(413, "O CSV excede 25 MB.")
    try:
        frame = pd.read_csv(io.BytesIO(raw), sep=None, engine="python")
    except Exception as exc:
        raise HTTPException(422, f"CSV inválido: {exc}") from exc
    if target not in frame:
        raise HTTPException(422, "Coluna-alvo não encontrada.")
    frame = frame.dropna(subset=[target]).drop_duplicates().reset_index(drop=True)
    if len(frame) < max(30, folds * 4):
        raise HTTPException(422, "Observações insuficientes para validação robusta.")
    if validation == "temporal":
        if not date_column or date_column not in frame:
            raise HTTPException(422, "Selecione uma coluna temporal válida.")
        frame[date_column] = pd.to_datetime(frame[date_column], errors="coerce")
        frame = frame.dropna(subset=[date_column]).sort_values(date_column)
    y, classes = _binary_target(frame[target])
    drop = [target] + ([date_column] if date_column else [])
    X = frame.drop(columns=drop, errors="ignore")
    splitter = TimeSeriesSplit(n_splits=folds) if validation == "temporal" else StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
    class_weight = "balanced" if imbalance == "class_weight" else None
    results = []
    for name, estimator in _models(class_weight).items():
        started = time.perf_counter()
        fold_metrics = []
        for train_index, test_index in splitter.split(X, y):
            pipeline = Pipeline([("prepare", _preprocessor(X.iloc[train_index], encoding, scaling)), ("model", estimator)])
            pipeline.fit(X.iloc[train_index], y[train_index])
            probability = pipeline.predict_proba(X.iloc[test_index])[:, 1]
            prediction = (probability >= .5).astype(int)
            fold_metrics.append({
                "recall": recall_score(y[test_index], prediction, zero_division=0),
                "f1": f1_score(y[test_index], prediction, zero_division=0),
                "auc_roc": roc_auc_score(y[test_index], probability),
                "auc_pr": average_precision_score(y[test_index], probability),
            })
        results.append({"algorithm": name, "seconds": round(time.perf_counter()-started, 3), **{key: {"mean": round(float(np.mean([fold[key] for fold in fold_metrics])), 5), "std": round(float(np.std([fold[key] for fold in fold_metrics])), 5)} for key in fold_metrics[0]}})
    champion = max(results, key=lambda result: result["auc_pr"]["mean"] + result["f1"]["mean"])
    return {"mode": "scientific", "rows": len(frame), "features": X.shape[1], "classes": classes, "validation": validation, "folds": folds, "results": results, "champion": champion["algorithm"]}
