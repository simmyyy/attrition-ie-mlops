from __future__ import annotations

from contextlib import asynccontextmanager
from enum import Enum

import os
import pandas as pd
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel, Field
from joblib import load


MODEL_PATH = "xgb_attrition_pipeline.joblib"


class SourceEnum(str, Enum):
    company = "Company"
    industry = "Industry"


class AttritionRow(BaseModel):
    Age: float = Field(..., ge=18, le=80, description="Age in years (18-80)")
    JobLevel: int = Field(..., ge=1, le=5, description="Job level (1-5)")
    MonthlyIncome: float = Field(..., gt=0, description="Monthly income > 0")
    DistanceFromHome: float = Field(..., ge=0, description="Distance from home in km")
    Education: int = Field(..., ge=1, le=5, description="Education level (1-5)")
    JobSatisfaction: int = Field(..., ge=1, le=4, description="Job satisfaction (1-4)")
    WorkLifeBalance: int = Field(..., ge=1, le=4, description="Work-life balance (1-4)")
    PerformanceRating: int = Field(
        ..., ge=1, le=4, description="Performance rating (1-4)"
    )
    YearsAtCompany: float = Field(..., ge=0, description="Years at company (>= 0)")

    Gender: str = Field(..., description="e.g. 'Male', 'Female'")
    OverTime: str = Field(..., description="'Yes' or 'No'")
    MaritalStatus: str = Field(..., description="e.g. 'Single', 'Married'")
    Source: SourceEnum = Field(..., description="'Company' or 'Industry'")

    class Config:
        json_schema_extra = {
            "example": {
                "Age": 35,
                "JobLevel": 2,
                "MonthlyIncome": 5000,
                "DistanceFromHome": 10,
                "Education": 3,
                "JobSatisfaction": 3,
                "WorkLifeBalance": 3,
                "PerformanceRating": 3,
                "YearsAtCompany": 5,
                "Gender": "Male",
                "OverTime": "Yes",
                "MaritalStatus": "Single",
                "Source": "Company",
            }
        }


class PredictionResponse(BaseModel):
    attrition_score: float = Field(..., ge=0.0, le=1.0)
    prediction: int = Field(..., ge=0, le=1)
    threshold: float
    model_version: str | None = None


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


@asynccontextmanager
async def lifespan(app: FastAPI):
    model = None
    if os.path.exists(MODEL_PATH):
        try:
            model = load(MODEL_PATH)
            print(f"[startup] Model loaded from {MODEL_PATH}")
        except Exception as e:
            print(f"[startup] Failed to load model: {e}")
    else:
        print(f"[startup] Model file not found at {MODEL_PATH}")

    app.state.model = model
    yield


app = FastAPI(
    title="Attrition Prediction API",
    description="Predict employee attrition probability.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/", response_model=dict)
async def root():
    return {"message": "Welcome to the Attrition Prediction API"}


@app.get("/health", response_model=HealthResponse)
async def health():
    loaded = app.state.model is not None
    return HealthResponse(status="ok" if loaded else "degraded", model_loaded=loaded)


@app.post("/predict-row", response_model=PredictionResponse)
async def predict_row(row: AttritionRow):
    if app.state.model is None:
        raise HTTPException(status_code=500, detail="Model not loaded. Check /health.")

    df = pd.DataFrame([row.dict()])
    try:
        proba = app.state.model.predict_proba(df)[0, 1]
        pred = int(proba >= 0.5)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {e}")

    return PredictionResponse(
        attrition_score=float(proba),
        prediction=pred,
        threshold=0.5,
        model_version="attrition-xgb",
    )


@app.post("/predict-csv", response_model=dict)
async def predict_csv(file: UploadFile = File(...)):
    if app.state.model is None:
        raise HTTPException(status_code=500, detail="Model not loaded. Check /health.")

    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are supported")

    try:
        contents = await file.read()
        df = pd.read_csv(pd.io.common.BytesIO(contents))

        expected_cols = [
            "Age",
            "JobLevel",
            "MonthlyIncome",
            "DistanceFromHome",
            "Education",
            "JobSatisfaction",
            "WorkLifeBalance",
            "PerformanceRating",
            "YearsAtCompany",
            "Gender",
            "OverTime",
            "MaritalStatus",
            "Source",
        ]
        missing = [c for c in expected_cols if c not in df.columns]
        if missing:
            raise HTTPException(
                status_code=400,
                detail=f"Missing columns in CSV: {missing}",
            )

        proba = app.state.model.predict_proba(df)[:, 1]
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {e}")

    return {
        "n_rows": int(len(df)),
        "attrition_scores": [float(p) for p in proba],
    }
