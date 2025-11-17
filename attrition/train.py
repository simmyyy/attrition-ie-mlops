import json
from typing import Tuple, Dict, List

import pandas as pd
from joblib import dump

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
from xgboost import XGBClassifier

IBM_PATH = "IBM_HR-Employee-Attrition.csv"
INDUSTRY_PATH = "Industry Dataset.csv"
MODEL_PATH = "xgb_attrition_pipeline.joblib"
METRICS_PATH = "metrics.json"


def read_raw_data(
    ibm_path: str = IBM_PATH, industry_path: str = INDUSTRY_PATH
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    ibm = pd.read_csv(ibm_path)
    industry = pd.read_csv(industry_path)
    return ibm, industry


def build_common_frame(ibm: pd.DataFrame, industry: pd.DataFrame) -> pd.DataFrame:
    common_cols = [
        "Age",
        "Gender",
        "JobLevel",
        "MonthlyIncome",
        "OverTime",
        "DistanceFromHome",
        "Education",
        "MaritalStatus",
        "JobSatisfaction",
        "WorkLifeBalance",
        "PerformanceRating",
        "YearsAtCompany",
        "Attrition",
    ]

    ibm_common = ibm[[c for c in common_cols if c in ibm.columns]].copy()
    industry_common = industry[[c for c in common_cols if c in industry.columns]].copy()

    for df in (ibm_common, industry_common):
        if "Attrition" in df.columns:
            df["Attrition"] = df["Attrition"].astype(str).str.strip().str.title()
            df["AttritionFlag"] = df["Attrition"].map(
                {
                    "Yes": 1,
                    "No": 0,
                    "Left": 1,
                    "Stayed": 0,
                }
            )

    ibm_common["Source"] = "Company"
    industry_common["Source"] = "Industry"

    df_all = pd.concat([ibm_common, industry_common], ignore_index=True)

    if "AttritionFlag" in df_all.columns:
        df_all = df_all[~df_all["AttritionFlag"].isna()].copy()

    return df_all


def split_X_y(df_all: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    drop_cols = [c for c in ["Attrition", "AttritionFlag"] if c in df_all.columns]
    X = df_all.drop(columns=drop_cols).copy()
    y = df_all["AttritionFlag"].astype(int)
    return X, y


def infer_feature_types(X: pd.DataFrame) -> Tuple[List[str], List[str]]:
    numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = [c for c in X.columns if c not in numeric_cols]
    return numeric_cols, categorical_cols


def build_pipeline(numeric_cols: List[str], categorical_cols: List[str]) -> Pipeline:
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ]
    )

    xgb_model = XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        n_jobs=-1,
        eval_metric="logloss",
        use_label_encoder=False,
    )

    pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", xgb_model)])
    return pipeline


def train_and_save() -> Dict[str, float]:
    ibm, industry = read_raw_data()
    df_all = build_common_frame(ibm, industry)
    X, y = split_X_y(df_all)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42,
    )

    numeric_cols, categorical_cols = infer_feature_types(X_train)
    pipeline = build_pipeline(numeric_cols, categorical_cols)

    # Optional: handle class imbalance
    pos = (y_train == 1).sum()
    neg = (y_train == 0).sum()
    scale_pos_weight = float(neg / max(pos, 1))
    pipeline.named_steps["model"].set_params(scale_pos_weight=scale_pos_weight)

    pipeline.fit(X_train, y_train)

    proba_test = pipeline.predict_proba(X_test)[:, 1]
    y_pred = pipeline.predict(X_test)

    metrics = {
        "auc_test": float(roc_auc_score(y_test, proba_test)),
        "precision_test": float(precision_score(y_test, y_pred)),
        "recall_test": float(recall_score(y_test, y_pred)),
        "f1_test": float(f1_score(y_test, y_pred)),
        "scale_pos_weight": scale_pos_weight,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
    }

    dump(pipeline, MODEL_PATH)
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)

    return metrics


def main():
    metrics = train_and_save()
    print("Training finished. Metrics:")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

# adding more comments
