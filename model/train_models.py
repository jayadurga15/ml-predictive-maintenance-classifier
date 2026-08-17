"""
train_models.py
----------------
Trains all 6 classification models on the AI4I 2020 Predictive Maintenance dataset,
saves each as a .pkl file, writes scaler.pkl, metrics_comparison.csv, and ../test_data.csv.

Run with:  python train_models.py
(expects ai4i2020_raw.csv in the same folder)
"""

import os
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
)

from features import build_features_and_target, FEATURE_COLUMNS, TARGET_COL

HERE = os.path.dirname(os.path.abspath(__file__))
RAW_CSV = os.path.join(HERE, "ai4i2020_raw.csv")
TEST_DATA_OUT = os.path.join(HERE, "..", "test_data.csv")
METRICS_OUT = os.path.join(HERE, "metrics_comparison.csv")

RANDOM_STATE = 42


def main():
    df = pd.read_csv(RAW_CSV, encoding="utf-8-sig")
    print(f"Loaded raw data: {df.shape}")

    X, y = build_features_and_target(df)
    print(f"Feature matrix: {X.shape}, features: {list(X.columns)}")
    print(f"Class balance:\n{y.value_counts()}")

    # stratified split preserves the ~3.4% positive-class rate in both splits
    X_train, X_test, y_train, y_test, df_train, df_test = train_test_split(
        X, y, df, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)  # fit on train only — no leakage
    X_test_scaled = scaler.transform(X_test)
    joblib.dump(scaler, os.path.join(HERE, "scaler.pkl"))
    joblib.dump(list(X.columns), os.path.join(HERE, "feature_columns.pkl"))

    # True = use scaled features (distance/probability-based); False = raw (scale-invariant)
    models = {
        "Logistic Regression": (
            LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE),
            True,
        ),
        "Decision Tree": (
            DecisionTreeClassifier(max_depth=8, class_weight="balanced", random_state=RANDOM_STATE),
            False,
        ),
        "kNN": (
            KNeighborsClassifier(n_neighbors=7),
            True,
        ),
        "Naive Bayes": (
            GaussianNB(),
            True,
        ),
        "Random Forest (Ensemble)": (
            RandomForestClassifier(
                n_estimators=300, max_depth=12, class_weight="balanced", random_state=RANDOM_STATE
            ),
            False,
        ),
        "SVM": (
            SVC(kernel="rbf", class_weight="balanced", probability=True, random_state=RANDOM_STATE),
            True,
        ),
    }

    filenames = {
        "Logistic Regression": "logistic_regression.pkl",
        "Decision Tree": "decision_tree.pkl",
        "kNN": "knn.pkl",
        "Naive Bayes": "naive_bayes.pkl",
        "Random Forest (Ensemble)": "random_forest.pkl",
        "SVM": "svm.pkl",
    }

    results = []

    for name, (model, use_scaled) in models.items():
        Xtr = X_train_scaled if use_scaled else X_train
        Xte = X_test_scaled if use_scaled else X_test

        model.fit(Xtr, y_train)
        y_pred = model.predict(Xte)
        y_proba = model.predict_proba(Xte)[:, 1]

        metrics = {
            "ML Model Name": name,
            "Accuracy": round(accuracy_score(y_test, y_pred), 4),
            "AUC": round(roc_auc_score(y_test, y_proba), 4),
            "Precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
            "Recall": round(recall_score(y_test, y_pred), 4),
            "F1": round(f1_score(y_test, y_pred), 4),
            "MCC": round(matthews_corrcoef(y_test, y_pred), 4),
        }
        results.append(metrics)
        print(metrics)

        joblib.dump(model, os.path.join(HERE, filenames[name]))

    results_df = pd.DataFrame(results)
    results_df.to_csv(METRICS_OUT, index=False)
    print(f"\nSaved metrics to {METRICS_OUT}")
    print(results_df.to_string(index=False))

    demo_cols = [
        "Type",
        "Air temperature [K]",
        "Process temperature [K]",
        "Rotational speed [rpm]",
        "Torque [Nm]",
        "Tool wear [min]",
        TARGET_COL,
    ]
    # modest sample size — Streamlit Community Cloud free tier has limited memory
    demo_sample = df_test[demo_cols].sample(n=min(400, len(df_test)), random_state=RANDOM_STATE)
    demo_sample.to_csv(TEST_DATA_OUT, index=False)
    print(f"Saved demo test_data.csv ({demo_sample.shape}) to {TEST_DATA_OUT}")


if __name__ == "__main__":
    main()
