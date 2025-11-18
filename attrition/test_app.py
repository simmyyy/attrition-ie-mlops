import os
from io import StringIO

import pandas as pd
from fastapi.testclient import TestClient

from app import app, MODEL_PATH


def test_model_file_exists():
    assert os.path.exists(MODEL_PATH), f"Model file not found at {MODEL_PATH}"


def test_health():
    with TestClient(app) as client:
        res = client.get("/health")
        assert res.status_code == 200

        data = res.json()
        assert "status" in data
        assert "model_loaded" in data

        assert data["model_loaded"] is True
        assert data["status"] == "ok"


def test_predict_row():
    payload = {
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

    with TestClient(app) as client:
        res = client.post("/predict-row", json=payload)
        assert res.status_code == 200

        data = res.json()
        # Response model: attrition_score, prediction, threshold, model_version
        assert "attrition_score" in data
        assert "prediction" in data
        assert "threshold" in data

        assert isinstance(data["attrition_score"], (float, int))
        assert 0.0 <= data["attrition_score"] <= 1.0
        assert data["prediction"] in (0, 1)


def test_predict_csv():
    csv_str = (
        "Age,JobLevel,MonthlyIncome,DistanceFromHome,Education,"
        "JobSatisfaction,WorkLifeBalance,PerformanceRating,YearsAtCompany,"
        "Gender,OverTime,MaritalStatus,Source\n"
        "35,2,5000,10,3,3,3,3,5,Male,Yes,Single,Company\n"
        "40,3,7000,5,4,4,2,3,10,Female,No,Married,Industry\n"
    )
    df = pd.read_csv(StringIO(csv_str))
    files = {"file": ("test.csv", df.to_csv(index=False), "text/csv")}

    with TestClient(app) as client:
        res = client.post("/predict-csv", files=files)
        assert res.status_code == 200

        data = res.json()
        assert "n_rows" in data
        assert "attrition_scores" in data

        assert data["n_rows"] == 2
        assert len(data["attrition_scores"]) == 2

        for s in data["attrition_scores"]:
            val = float(s)
            assert 0.0 <= val <= 1.0
