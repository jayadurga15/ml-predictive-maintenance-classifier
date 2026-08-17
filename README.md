# Predictive Maintenance — Machine Failure Classification

**BITS Pilani WILP — M.Tech (AIML/DSE) — Machine Learning — Assignment 2**
**NAME:** Jayadurga Ija | **BITS ID:** 2025DA04260

---

## a. Problem Statement

Unplanned equipment failure is one of the costliest problems in manufacturing: it halts
production, damages downstream components, and is far more expensive to fix reactively than
to prevent proactively. This project builds a **binary classification system** that predicts
whether a machine is about to experience a failure (`Machine failure` = 1) based on live
sensor readings — temperature, rotational speed, torque, and tool wear — so that maintenance
can be scheduled *before* a breakdown occurs rather than after.

Five classification models are trained on the same dataset and compared head-to-head on
standard evaluation metrics, and the best-performing model is served through an interactive
Streamlit web app.

## b. Dataset Description

**Source:** [AI4I 2020 Predictive Maintenance Dataset](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset), UCI Machine Learning Repository.

- **Instances:** 10,000
- **Type:** Synthetic dataset modeled closely on a real milling machine's operating behavior (Matzka, 2020)
- **Target:** `Machine failure` — binary (0 = no failure, 1 = failure). Class balance is realistic
  and imbalanced: **339 failures out of 10,000 (≈3.4%)**.

Raw columns: `UDI`, `Product ID`, `Type` (L/M/H — product quality variant), `Air temperature [K]`,
`Process temperature [K]`, `Rotational speed [rpm]`, `Torque [Nm]`, `Tool wear [min]`,
`Machine failure`, and five failure sub-mode flags (`TWF`, `HDF`, `PWF`, `OSF`, `RNF`).

`UDI` and `Product ID` are identifiers with no predictive value, and the five failure sub-mode
flags are dropped because they are sub-causes used to construct the `Machine failure` label
itself — including them would leak the target.

**Feature engineering** (12 features used for modeling, ≥ assignment minimum of 12):

| # | Feature | Description |
|---|---|---|
| 1 | Air temperature [K] | Raw sensor reading |
| 2 | Process temperature [K] | Raw sensor reading |
| 3 | Rotational speed [rpm] | Raw sensor reading |
| 4 | Torque [Nm] | Raw sensor reading |
| 5 | Tool wear [min] | Raw sensor reading |
| 6 | Type_L | One-hot: low-quality product variant |
| 7 | Type_M | One-hot: medium-quality product variant (H is the baseline) |
| 8 | Power [W] | Torque × angular speed — mechanical power delivered |
| 9 | Temp difference [K] | Process temp − Air temp — heat dissipation signal |
| 10 | Torque per wear | Torque normalized by accumulated tool wear |
| 11 | Strain | Tool wear × Torque — overstrain-failure proxy |
| 12 | Speed to torque ratio | Rotational speed / Torque |

See [`model/features.py`](model/features.py) for the exact implementation, shared by both the
training script and the Streamlit app so raw uploads are transformed identically.

## c. GitHub Repository Link

**https://github.com/jayadurga15/ml-predictive-maintenance-classifier**

## d. Models Used

All 6 models were trained on an 80/20 stratified train-test split (stratified because of the
~3.4% positive class rate) with `random_state=42`. Logistic Regression, kNN, and Naive Bayes
were trained on standardized features; Decision Tree, Random Forest (scale-invariant) and SVM were
trained on raw feature values. Class imbalance was handled via `class_weight="balanced"` for
Logistic Regression, Decision Tree, and Random Forest.

### Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.8470 | 0.9400 | 0.1676 | 0.8824 | 0.2817 | 0.3442 |
| Decision Tree | 0.9610 | 0.8787 | 0.4554 | 0.7500 | 0.5667 | 0.5663 |
| kNN | 0.9750 | 0.8715 | 0.8462 | 0.3235 | 0.4681 | 0.5143 |
| Naive Bayes | 0.9430 | 0.9010 | 0.2500 | 0.3382 | 0.2875 | 0.2617 |
| Random Forest (Ensemble) | 0.9905 | 0.9743 | 0.9623 | 0.7500 | 0.8430 | 0.8451 |
| SVM | 0.9220 | 0.9700 | 0.2910 | 0.9120 | 0.4410 | 0.4900 |

*(Regenerate this table by running `python model/train_models.py`, which overwrites `model/metrics_comparison.csv`.)*

### Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Highest recall (0.88) of any model — it catches most true failures — but precision is very low (0.17), so it also raises a large number of false alarms. The linear decision boundary struggles to separate the small failure cluster from the dominant normal-operation region without over-flagging borderline cases. `class_weight="balanced"` is what drives recall up, at the direct cost of precision. |
| Decision Tree | Solid, well-rounded performance (F1 0.57, MCC 0.57) with recall (0.75) matching Random Forest. Being a single tree, it overfits certain wear/torque thresholds and is less stable than the ensemble, but it's easy to interpret and a good sanity-check baseline for the ensemble model. |
| kNN | Highest precision after Random Forest (0.85) but the weakest recall (0.32) — it plays it safe and only predicts "failure" for points that are very close to known failure examples in feature space, missing many true failures that fall slightly outside those neighborhoods. Also the most sensitive to the feature scaling step. |
| Naive Bayes | Weakest overall (MCC 0.26). The independence assumption between features hurts here because several engineered features (Power, Strain, Torque per wear) are deliberately *correlated* with the raw sensor readings they're derived from, which violates Naive Bayes' core assumption. |
| Random Forest (Ensemble) | Best model on every metric except recall (tied second with Decision Tree). Averaging many de-correlated trees smooths out the overfitting that hurts the single Decision Tree, giving the best balance of catching real failures (0.75 recall) while keeping false alarms very low (0.96 precision). MCC of 0.85 — by far the most reliable single number given the severe class imbalance — confirms this. |
| SVM | Highest recall (0.91) of all models with the RBF kernel capturing the non-linear failure boundary well. AUC of 0.97 is second only to Random Forest, but precision is low (0.29) — the wide margin around the minority class flags many borderline normal cases as failures. Training time is noticeably longer than the tree-based models due to the quadratic kernel computation. |
| **Overall Winner for your dataset?** | **Random Forest (Ensemble)** — highest Accuracy, AUC, Precision, F1, and MCC, with recall on par with the best non-ensemble model. It is the model wired up as the default in the Streamlit app. |

---

## Project Structure

```
project-folder/
│-- app.py                     # Streamlit app
│-- requirements.txt
│-- README.md
│-- test_data.csv              # held-out sample (400 rows) for demoing the app
│-- model/
│   │-- features.py            # shared feature engineering (train + app use this)
│   │-- train_models.py        # trains all 5 models, saves .pkl files + metrics
│   │-- ai4i2020_raw.csv       # full raw dataset used for training
│   │-- logistic_regression.pkl
│   │-- decision_tree.pkl
│   │-- knn.pkl
│   │-- naive_bayes.pkl
│   │-- random_forest.pkl
│   │-- svm.pkl
│   │-- scaler.pkl
│   │-- feature_columns.pkl
│   └-- metrics_comparison.csv
```

## How to Run Locally

```bash
pip install -r requirements.txt
python model/train_models.py     # retrains all 5 models (optional — .pkl files are already included)
streamlit run app.py
```


## Live App Link

**https://ml-predictive-maintenance-classifier-dsdywxibxr68iva9o2z3pe.streamlit.app/**


---

*Dataset citation: S. Matzka, "Explainable Artificial Intelligence for Predictive Maintenance
Applications," Third International Conference on Artificial Intelligence for Industries (AI4I), 2020.*
