import os
import json
from typing import Tuple, Dict

import pandas as pd
from joblib import dump

# Sklearn / XGBoost
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    roc_auc_score,
    recall_score,
    precision_score,
    f1_score,
    confusion_matrix,
    classification_report,
    precision_recall_curve,
)

from xgboost import XGBClassifier

# MLflow
import mlflow
from mlflow import sklearn as mlflow_sklearn

# Plots
import matplotlib.pyplot as plt
import seaborn as sns

IBM_PATH = "IBM_HR-Employee-Attrition.csv"
INDUSTRY_PATH = "Industry Dataset.csv"
MODEL_PATH = "xgb_attrition_pipeline.joblib"
METRICS_PATH = "metrics.json"


# ------------------------------------------------------------------------
# Data loading
# ------------------------------------------------------------------------
def read_raw_data(
    ibm_path: str = IBM_PATH, industry_path: str = INDUSTRY_PATH
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Read raw CSV files for IBM and Industry datasets.
    """
    ibm = pd.read_csv(ibm_path)
    industry = pd.read_csv(industry_path)
    return ibm, industry


# ------------------------------------------------------------------------
# Full preprocessing exactly as in the shared notebook
# ------------------------------------------------------------------------
def build_dataset() -> Tuple[pd.DataFrame, pd.Series]:
    """
    Reproduce the full preprocessing from the notebook:

    - Filter Industry dataset to Technology rows only.
    - Rename Industry columns to match IBM naming.
    - Build ibm_common / industry_common.
    - Normalize text and map ordinals to numeric.
    - Label-encode categorical ('Gender', 'OverTime', 'MaritalStatus').
    - Standardize numeric columns using StandardScaler (fitted on IBM).
    - Add Source column and concatenate both datasets.
    - Create AttritionFlag and define X, y.

    Returns:
        X (DataFrame), y (Series)
    """
    ibm, industry = read_raw_data()

    # Now, we should start by making sure that the dataset Industry only
    # contains rows related to Technology by filtering out other sectors
    industry = industry.rename(columns={"Job Role": "Industry"})
    industry["Industry"] = industry["Industry"].astype(str).str.strip().str.lower()

    mask_tech = industry["Industry"].str.contains("technology", na=False)
    industry_tech = industry.loc[mask_tech].copy()

    print(f"Total rows on Industry dataset: {len(industry)}")
    print(f"Rows corresponding to Technology: {len(industry_tech)}")
    print("\nUnique values in industry:")
    print(industry["Industry"].value_counts().head(10))

    # After that, we make sure to rename the variables
    industry = industry_tech.rename(
        columns={
            "Employee ID": "EmployeeID",
            "Monthly Income": "MonthlyIncome",
            "Work-Life Balance": "WorkLifeBalance",
            "Job Satisfaction": "JobSatisfaction",
            "Performance Rating": "PerformanceRating",
            "Overtime": "OverTime",
            "Distance from Home": "DistanceFromHome",
            "Education Level": "Education",
            "Marital Status": "MaritalStatus",
            "Job Level": "JobLevel",
            "Years at Company": "YearsAtCompany",
        }
    )

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
            df["Attrition"] = (
                df["Attrition"]
                .astype(str)
                .str.strip()
                .str.title()  # 'yes'->'Yes', 'left'->'Left', etc.
            )
            df["AttritionFlag"] = df["Attrition"].map(
                {
                    "Yes": 1,
                    "No": 0,  # IBM
                    "Left": 1,
                    "Stayed": 0,  # Industry
                }
            )

    # Before coding and transforming variables, an important step to follow is to make
    # sure all the variables are in the correct type.
    # We normalize the text before assigning a numerical value to each label.
    def normalize_text(df, cols):
        for c in cols:
            if c in df.columns and df[c].dtype == object:
                df[c] = (
                    df[c]
                    .astype(str)
                    .str.strip()
                    .str.replace(r"\s+", " ", regex=True)
                    .str.title()
                )

    normalize_text(
        industry_common,
        [
            "JobLevel",
            "Education",
            "JobSatisfaction",
            "WorkLifeBalance",
            "PerformanceRating",
        ],
    )
    normalize_text(
        ibm_common,
        [
            "JobLevel",
            "Education",
            "JobSatisfaction",
            "WorkLifeBalance",
            "PerformanceRating",
        ],
    )

    # Then, we create a map that contains each label and its numerical equivalent
    joblevel_map = {
        "Entry": 1,
        "Junior": 2,
        "Mid": 3,
        "Middle": 3,
        "Senior": 4,
        "Lead": 5,
        "Manager": 5,
        "Executive": 5,
    }

    education_map = {
        "High School": 1,
        "Highschool": 1,
        "Associate": 2,
        "Associate Degree": 2,
        "Diploma": 2,
        "Bachelor": 3,
        "Bachelors": 3,
        "Bachelor'S": 3,
        "Bachelor’S Degree": 3,
        "Bachelor’S Degree": 3,
        "Undergraduate": 3,
        "Masters": 4,
        "Master'S": 4,
        "Master’S Degree": 4,
        "Postgraduate": 4,
        "Doctorate": 5,
        "Phd": 5,
        "Doctoral": 5,
    }

    jobsat_map = {
        "Very Low": 1,
        "Low": 1,
        "Medium": 2,
        "Average": 2,
        "High": 3,
        "Very High": 4,
    }

    worklife_map = {
        "Poor": 1,
        "Bad": 1,
        "Fair": 2,
        "Average": 2,
        "Good": 3,
        "Excellent": 4,
    }

    perf_map = {
        "Below Average": 1,
        "Average": 2,
        "High": 3,
        "Excellent": 4,
        "Outstanding": 4,
    }

    # Apply this to both datasets
    for df in (ibm_common, industry_common):
        if "JobLevel" in df.columns:
            df["JobLevel"] = df["JobLevel"].replace(joblevel_map)
        if "Education" in df.columns:
            df["Education"] = df["Education"].replace(education_map)
        if "JobSatisfaction" in df.columns:
            df["JobSatisfaction"] = df["JobSatisfaction"].replace(jobsat_map)
        if "WorkLifeBalance" in df.columns:
            df["WorkLifeBalance"] = df["WorkLifeBalance"].replace(worklife_map)
        if "PerformanceRating" in df.columns:
            df["PerformanceRating"] = df["PerformanceRating"].replace(perf_map)

    # We convert to numeric
    for df in (ibm_common, industry_common):
        for c in [
            "JobLevel",
            "Education",
            "JobSatisfaction",
            "WorkLifeBalance",
            "PerformanceRating",
        ]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")

    # Quick check
    print("Unique values after mapping:")
    for c in [
        "JobLevel",
        "Education",
        "JobSatisfaction",
        "WorkLifeBalance",
        "PerformanceRating",
    ]:
        if c in industry_common.columns:
            print(f"{c}: {sorted(industry_common[c].dropna().unique())}")

    # We will start by coding text as numbers for certain variables where the criteria
    # is based on a set of labels.
    categorical_cols = ["Gender", "OverTime", "MaritalStatus"]
    label_encoders = {}

    for col in categorical_cols:
        if col in ibm_common.columns and col in industry_common.columns:
            le = LabelEncoder()
            combined_vals = pd.concat(
                [ibm_common[col].astype(str), industry_common[col].astype(str)], axis=0
            )
            le.fit(combined_vals)

            ibm_common[col] = le.transform(ibm_common[col].astype(str))
            industry_common[col] = le.transform(industry_common[col].astype(str))

            label_encoders[col] = le
            print(f"Coded column: {col}")
        else:
            print(f"The column {col} does nt appear in any dataset")

    # After that, it is necessary to work on mapping our ordinal variables
    numeric_cols = [
        "Age",
        "JobLevel",
        "MonthlyIncome",
        "DistanceFromHome",
        "Education",
        "JobSatisfaction",
        "WorkLifeBalance",
        "PerformanceRating",
        "YearsAtCompany",
    ]

    scaler = StandardScaler()

    for col in numeric_cols:
        if col in ibm_common.columns and col in industry_common.columns:
            scaler.fit(ibm_common[[col]])
            ibm_common[col] = scaler.transform(ibm_common[[col]])
            industry_common[col] = scaler.transform(industry_common[[col]])
        else:
            print(f"The Column {col} does nt appear in any dataset")

    # By adding these columns, we make sure that, during our training of the model,
    # it will be able to distinguish between both datasets
    ibm_common["Source"] = "Company"
    industry_common["Source"] = "Industry"

    df_all = pd.concat([ibm_common, industry_common], ignore_index=True)
    print(f"Merged dataset including rows: {len(df_all)}, columns: {df_all.shape[1]}")
    print(df_all.head())

    # Now we define X and Y
    X = df_all.drop(
        columns=[c for c in ["Attrition", "AttritionFlag"] if c in df_all.columns]
    ).copy()
    y = df_all["AttritionFlag"].astype(int)

    return X, y, df_all


# ------------------------------------------------------------------------
# Plot helpers for MLflow
# ------------------------------------------------------------------------
def create_confusion_matrix_figure(
    y_true, y_pred, title="Confusion Matrix for XGBoost"
):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Prediction")
    ax.set_ylabel("Real")
    fig.tight_layout()
    return fig


def create_pr_curve_figure(y_true, y_proba, title="Precision–Recall Curve"):
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(recall, precision, linewidth=2)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(title)
    ax.grid(True)
    fig.tight_layout()
    return fig


# ------------------------------------------------------------------------
# Training + evaluation (matching notebook, plus MLflow hooks)
# ------------------------------------------------------------------------
def train_and_evaluate():
    X, y, df_all = build_dataset()

    # Classic 80/20 split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,  # 80% train, 20% test
        stratify=y,
        random_state=42,
    )

    print("Train shape:", X_train.shape, "| Test shape:", X_test.shape)
    print(f"Positives proportion for train: {y_train.mean():.3f}")
    print(f"Positives proportion for test: {y_test.mean():.3f}")

    if "Source" in X.columns:
        print("\nType of Source in train:")
        print(X_train["Source"].value_counts())
        print("\nType of Source in test:")
        print(X_test["Source"].value_counts())

    # Make sure the types for each column is correctly updated
    num_cols = [
        "Age",
        "JobLevel",
        "MonthlyIncome",
        "DistanceFromHome",
        "Education",
        "JobSatisfaction",
        "WorkLifeBalance",
        "PerformanceRating",
        "YearsAtCompany",
    ]
    cat_cols = ["Gender", "OverTime", "MaritalStatus", "Source"]

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
            ("num", numeric_transformer, [c for c in num_cols if c in X.columns]),
            ("cat", categorical_transformer, [c for c in cat_cols if c in X.columns]),
        ],
        remainder="drop",
    )

    # XGB model as in notebook
    xgb_model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="auc",
        random_state=42,
        n_estimators=400,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=(y == 0).sum() / max(1, (y == 1).sum()),
        n_jobs=-1,
    )

    pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", xgb_model)])

    print("Finalized pipeline:")
    print(pipeline)

    # Recompute scale_pos_weight on train only
    pos = (y_train == 1).sum()
    neg = (y_train == 0).sum()
    spw = neg / max(1, pos)
    print(f"scale_pos_weight (train) = {spw:.2f}  |  pos={pos}, neg={neg}")

    pipeline.named_steps["model"].set_params(scale_pos_weight=spw)

    pipeline.fit(X_train, y_train)

    # Train AUC
    train_proba = pipeline.predict_proba(X_train)[:, 1]
    print("AUC (train):", round(roc_auc_score(y_train, train_proba), 4))

    # Save model
    dump(pipeline, MODEL_PATH)

    # Evaluation on test
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    y_pred = pipeline.predict(X_test)

    auc = roc_auc_score(y_test, y_proba)
    rec = recall_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print("USED METRICS CALCULATION")
    print(f"ROC-AUC   : {auc:.4f}")
    print(f"Recall    : {rec:.4f}")
    print(f"Precision : {prec:.4f}")
    print(f"F1-score  : {f1:.4f}")
    print("\nClassification report:\n", classification_report(y_test, y_pred, digits=3))

    # IBM-only metrics (Source == Company)
    if "Source" in X_test.columns:
        mask_ibm = X_test["Source"].astype(str).str.lower().eq("company")
        if mask_ibm.any():
            print("IBM METRICS")
            print(classification_report(y_test[mask_ibm], y_pred[mask_ibm], digits=3))
        else:
            print("Error: No IBM rows on the split.")

    metrics = {
        "auc_test": float(auc),
        "precision_test": float(prec),
        "recall_test": float(rec),
        "f1_test": float(f1),
        "scale_pos_weight_train": float(spw),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
    }

    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)

    return pipeline, metrics, y_test, y_pred, y_proba


# ------------------------------------------------------------------------
# Main with MLflow logging
# ------------------------------------------------------------------------
def main():
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns")
    experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME", "attrition-xgb")

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(
        run_name="xgb_merged_notebooks_pipeline",
        tags={
            "stage": "training",
            "notebook": "Merged_Preprocessing_Modelling",
            "model_type": "xgboost",
        },
    ):
        pipeline, metrics, y_test, y_pred, y_proba = train_and_evaluate()

        # Log hyperparameters from XGBoost
        xgb = pipeline.named_steps["model"]
        params_to_log = {
            "n_estimators": xgb.get_params().get("n_estimators"),
            "learning_rate": xgb.get_params().get("learning_rate"),
            "max_depth": xgb.get_params().get("max_depth"),
            "subsample": xgb.get_params().get("subsample"),
            "colsample_bytree": xgb.get_params().get("colsample_bytree"),
            "scale_pos_weight": xgb.get_params().get("scale_pos_weight"),
        }
        for k, v in params_to_log.items():
            mlflow.log_param(k, v)

        # Log metrics
        for k, v in metrics.items():
            mlflow.log_metric(k, v)

        # Log artifacts: model + metrics JSON
        mlflow.log_artifact(MODEL_PATH, artifact_path="model")
        mlflow.log_artifact(METRICS_PATH, artifact_path="metrics")

        # Confusion matrix & PR curve to MLflow
        fig_cm = create_confusion_matrix_figure(y_test, y_pred)
        mlflow.log_figure(fig_cm, "plots/confusion_matrix.png")
        plt.close(fig_cm)

        fig_pr = create_pr_curve_figure(y_test, y_proba)
        mlflow.log_figure(fig_pr, "plots/pr_curve.png")
        plt.close(fig_pr)

        # Also log as MLflow model
        mlflow_sklearn.log_model(
            sk_model=pipeline,
            artifact_path="xgb-pipeline",
            registered_model_name="attrition_xgb",
        )

        print("Training finished. Metrics:")
        print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
