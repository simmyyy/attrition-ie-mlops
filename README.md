# Employee Attrition – End-to-End MLOps Project

Production-style ML pipeline for predicting **employee attrition** (who is likely to leave the company), built as part of the IE University MLOps course.

---

## Problem Overview

**Goal:** predict the probability that an employee will leave the company (`AttritionFlag` = 1).

We combine two datasets:

- **IBM_HR-Employee-Attrition.csv** – detailed HR dataset for a single company.  
- **Industry Dataset.csv** – broader industry data, filtered to **Technology sector** and re-aligned to IBM’s schema.

Key steps:

- Clean & harmonise both datasets (text normalisation, mapping ordinal variables, encoding categoricals).
- Standardise numeric features.
- Train an **XGBoost** classifier.
- Evaluate using **ROC-AUC, Precision, Recall, F1**, Confusion Matrix and PR-curve, sending to MLFlow
- Serve predictions via **FastAPI** (single row & CSV).
- CI/CD deployment to render.com

---

## Local setup

Clone repo to local environment and then execute in main directory
```
pip install -r requirements.txt
```

Once you have all requirements, go to attrition directory and execute python file:
```
cd attrition
python train.py
```

After executing train.py script, we can start MLFlow instance and see results:
```
mlflow ui --backend-store-uri mlruns --port 5000
```

You can check results in your browser:
```
http://127.0.0.1:5000/
```

Executing unit tests:
```
python -m pytest -q
```

Building docker image:

```
docker build -t attrition-api:latest .
docker run -p 8000:8000 attrition-api:latest
```

Github actions:
https://github.com/simmyyy/attrition-ie-mlops/actions

API in render.com:
```
https://attrition-api-latest.onrender.com/docs#/
```

---

## Tech Stack

- **Python** (data & orchestration)
- **pandas, scikit-learn, XGBoost** – data processing & modelling
- **MLflow** – experiment tracking, metrics, artifacts (model + plots)
- **FastAPI + Uvicorn** – REST API for online scoring
- **PyYAML** – config management (`configs/train.yaml`)
- **pytest** – unit tests for the API
- **Docker** – containerised API
- **Render** – cloud hosting for the FastAPI app

---

<p align="center">
  <img src="./architecture_diagram.png"
       alt="Attrition MLOps Pipeline"
       width="1400">
</p>

---

## Repository Structure

```text
.
├── .github/workflows/
│   ├── ci-cd.yml                       # github workflow definition for ci-cd part
│   ├── train.yml                       # github workflow definition for training
│
├── attrition/
│   ├── app.py                          # FastAPI application (single-row & CSV prediction)
│   ├── Dockerfile                      # Image for serving API
│   ├── requirements.txt                # All runtime + dev deps (pinned versions)
│   ├── test_app.py                     # pytest tests for the API
│   ├── train.py                        # Training pipeline + MLflow logging
│   ├── xgb_attrition_pipeline.joblib   # Trained model (artifact)
│
├── configs/
│   └── train.yaml                      # Central config: data paths, model params, MLflow settings
│
├── data/
│   ├── IBM_HR-Employee-Attrition.csv   # Company-level HR data
│   └── Industry Dataset.csv            # Industry data (filtered to Technology sector)
│
├── .gitignore
├── architecture_diagram.png
└── README.md