# 06-cicd/app.py
from typing import List, Optional

import pandas as pd
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from joblib import load
import os

MODEL_PATH = "xgb_attrition_pipeline.joblib"

app = FastAPI(title="Attrition Prediction API")


class AttritionRow(BaseModel):
    Age: float
    JobLevel: int
    MonthlyIncome: float
    DistanceFromHome: float
    Education: int
    JobSatisfaction: int
    WorkLifeBalance: int
    PerformanceRating: int
    YearsAtCompany: float
    Gender: str
    OverTime: str
    MaritalStatus: str
    Source: str


def load_model():
    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(f"Model file not found at {MODEL_PATH}")
    return load(MODEL_PATH)


model = load_model()


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/predict-row")
async def predict_row(row: AttritionRow):
    df = pd.DataFrame([row.dict()])
    try:
        proba = model.predict_proba(df)[0, 1]
        pred = int(proba >= 0.5)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    return {
        "attrition_score": float(proba),
        "prediction": pred,
        "threshold": 0.5,
    }


@app.post("/predict-csv")
async def predict_csv(file: UploadFile = File(...)):
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are supported")

    try:
        contents = await file.read()
        df = pd.read_csv(pd.io.common.BytesIO(contents))
        proba = model.predict_proba(df)[:, 1]
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    return {
        "n_rows": int(len(df)),
        "attrition_scores": [float(p) for p in proba],
    }
