# 06-cicd/test_app.py
from fastapi.testclient import TestClient
import os
import pandas as pd
from io import StringIO

from app import app, MODEL_PATH

client = TestClient(app)


def test_model_file_exists():
    assert os.path.exists(MODEL_PATH)


def test_health():
    res = client.get("/health")
    assert res.status_code == 200
    assert res.json()["status"] == "ok"


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
    res = client.post("/predict-row", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert "attrition_score" in data
    assert 0.0 <= data["attrition_score"] <= 1.0


def test_predict_csv():
    csv_str = """Age,JobLevel,MonthlyIncome,DistanceFromHome,Education,JobSatisfaction,WorkLifeBalance,PerformanceRating,YearsAtCompany,Gender,OverTime,MaritalStatus,Source
35,2,5000,10,3,3,3,3,5,Male,Yes,Single,Company
40,3,7000,5,4,4,2,3,10,Female,No,Married,Industry
"""
    df = pd.read_csv(StringIO(csv_str))
    files = {"file": ("test.csv", df.to_csv(index=False), "text/csv")}
    res = client.post("/predict-csv", files=files)
    assert res.status_code == 200
    data = res.json()
    assert data["n_rows"] == 2
    assert len(data["attrition_scores"]) == 2
