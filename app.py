"""
app.py  —  Predictive Maintenance Classifier
Streamlit app for BITS Pilani M.Tech (AIML/DSE) Machine Learning Assignment 2.
"""

import os
import sys

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "model"))
from features import TARGET_COL, build_features_and_target  # noqa: E402

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model")

MODEL_FILES = {
    "Logistic Regression":      ("logistic_regression.pkl", True),
    "Decision Tree":            ("decision_tree.pkl",       False),
    "kNN":                      ("knn.pkl",                 True),
    "Naive Bayes":              ("naive_bayes.pkl",         True),
    "Random Forest (Ensemble)": ("random_forest.pkl",       False),
    "SVM":                      ("svm.pkl",                 True),
}

st.set_page_config(
    page_title="Predictive Maintenance Classifier",
    page_icon="📋",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Source+Serif+4:ital,wght@0,400;0,600;1,400&family=Inter:wght@400;500;600&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    /* Page background */
    .main .block-container { padding-top: 2rem; max-width: 1100px; }

    /* Report title block */
    .report-title {
        border-top: 3px solid #1a1a1a;
        border-bottom: 1px solid #cccccc;
        padding: 1.2rem 0 1rem 0;
        margin-bottom: 1.8rem;
    }
    .report-title h1 {
        font-family: 'Source Serif 4', serif;
        font-size: 1.7rem;
        font-weight: 600;
        color: #1a1a1a;
        margin: 0 0 0.3rem 0;
        line-height: 1.3;
    }
    .report-title .subtitle {
        font-size: 0.82rem;
        color: #555555;
        letter-spacing: 0.03em;
        text-transform: uppercase;
    }

    /* Section headings */
    .section-heading {
        font-family: 'Source Serif 4', serif;
        font-size: 1.05rem;
        font-weight: 600;
        color: #1a1a1a;
        border-bottom: 1px solid #dddddd;
        padding-bottom: 4px;
        margin: 1.4rem 0 0.8rem 0;
    }

    /* Metric row */
    .metric-row {
        display: flex;
        gap: 0;
        border: 1px solid #dddddd;
        border-radius: 6px;
        overflow: hidden;
        margin-bottom: 1rem;
    }
    .metric-cell {
        flex: 1;
        padding: 0.8rem 0.6rem;
        text-align: center;
        border-right: 1px solid #dddddd;
        background: #fafafa;
    }
    .metric-cell:last-child { border-right: none; }
    .metric-cell .mlabel {
        font-size: 0.68rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #777777;
        margin-bottom: 3px;
    }
    .metric-cell .mvalue {
        font-size: 1.25rem;
        font-weight: 600;
        color: #1a1a1a;
    }
    .metric-cell.highlight { background: #f0f4ff; }
    .metric-cell.highlight .mvalue { color: #1d4ed8; }

    /* Caption / note style */
    .note {
        font-size: 0.78rem;
        color: #777777;
        margin-top: -0.5rem;
        margin-bottom: 1rem;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] { background: #f7f7f7; border-right: 1px solid #e0e0e0; }
    section[data-testid="stSidebar"] .stSelectbox label,
    section[data-testid="stSidebar"] .stFileUploader label { font-size: 0.82rem; color: #333; }

    /* Tab bar */
    .stTabs [data-baseweb="tab-list"] { border-bottom: 2px solid #dddddd; gap: 0; }
    .stTabs [data-baseweb="tab"] {
        font-size: 0.82rem;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: #666666;
        padding: 8px 18px;
        background: transparent;
        border-bottom: 2px solid transparent;
        margin-bottom: -2px;
    }
    .stTabs [aria-selected="true"] {
        color: #1a1a1a !important;
        border-bottom: 2px solid #1a1a1a !important;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Report header
# ---------------------------------------------------------------------------
st.markdown("""
<div class="report-title">
  <h1>Predictive Maintenance — Machine Failure Classification</h1>
  <span class="subtitle">
    AI4I 2020 Dataset (UCI ML Repository) &nbsp;·&nbsp;
    Binary Classification &nbsp;·&nbsp;
    BITS Pilani WILP — M.Tech (AIML/DSE) — Machine Learning — Assignment 2
  </span>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("#### Input & Model")
    uploaded_file = st.file_uploader(
        "Upload test data (CSV)",
        type=["csv"],
        help=(
            "Required columns: Type, Air temperature [K], Process temperature [K], "
            "Rotational speed [rpm], Torque [Nm], Tool wear [min]. "
            "Include 'Machine failure' to compute evaluation metrics."
        ),
    )
    model_name = st.selectbox("Classifier", list(MODEL_FILES.keys()))

    use_sample = False
    if uploaded_file is None:
        use_sample = st.checkbox("Use bundled test_data.csv", value=True)

    st.markdown("---")
    st.markdown("""
    **Dataset** \n
    AI4I 2020 Predictive Maintenance \n
    UCI Machine Learning Repository \n
    10,000 instances · 12 features · Binary target
    """)


# ---------------------------------------------------------------------------
# Load artifacts
# ---------------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
    mdls = {name: joblib.load(os.path.join(MODEL_DIR, fname))
            for name, (fname, _) in MODEL_FILES.items()}
    return scaler, mdls

scaler, trained_models = load_artifacts()

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
df_raw = None
if uploaded_file is not None:
    df_raw = pd.read_csv(uploaded_file)
elif use_sample:
    sample_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_data.csv")
    if os.path.exists(sample_path):
        df_raw = pd.read_csv(sample_path)
    else:
        st.sidebar.error("test_data.csv not found.")

if df_raw is None:
    st.info("Upload a CSV or enable 'Use bundled test_data.csv' in the sidebar.")
    st.stop()

# ---------------------------------------------------------------------------
# Feature engineering + prediction
# ---------------------------------------------------------------------------
try:
    X, y = build_features_and_target(df_raw)
except ValueError as e:
    st.error(f"Could not process data: {e}")
    st.stop()

_, use_scaled = MODEL_FILES[model_name]
model = trained_models[model_name]
X_input = scaler.transform(X) if use_scaled else X
y_pred  = model.predict(X_input)
y_proba = model.predict_proba(X_input)[:, 1]

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab1, tab2, tab3 = st.tabs(["Data", "Model Results", "All Models"])

# ── Tab 1: Data ────────────────────────────────────────────────────────────
with tab1:
    st.markdown('<div class="section-heading">Input Data — Preview</div>', unsafe_allow_html=True)
    st.dataframe(df_raw.head(10), use_container_width=True)
    st.markdown(f'<p class="note">{df_raw.shape[0]} rows · {df_raw.shape[1]} columns</p>',
                unsafe_allow_html=True)

    st.markdown(f'<div class="section-heading">Predictions — {model_name}</div>',
                unsafe_allow_html=True)
    out_df = df_raw[["Type", "Air temperature [K]", "Torque [Nm]", "Tool wear [min]"]].copy()
    out_df.insert(0, "Failure Probability", np.round(y_proba, 4))
    out_df.insert(0, "Predicted Failure",   y_pred)
    st.dataframe(out_df.head(15), use_container_width=True)
    failure_pct = y_pred.mean() * 100
    st.markdown(
        f'<p class="note">Predicted {int(y_pred.sum())} failure(s) out of '
        f'{len(y_pred)} records ({failure_pct:.1f}%)</p>',
        unsafe_allow_html=True,
    )

    st.markdown('<div class="section-heading">Failure Probability Distribution</div>',
                unsafe_allow_html=True)
    fig_hist = px.histogram(
        x=y_proba, nbins=40,
        color_discrete_sequence=["#1d4ed8"],
        labels={"x": "Failure Probability", "y": "Count"},
    )
    fig_hist.update_layout(
        plot_bgcolor="white", paper_bgcolor="white",
        font=dict(family="Inter", size=12, color="#333"),
        margin=dict(t=20, b=40, l=40, r=20),
        showlegend=False, height=280,
        xaxis=dict(showgrid=True, gridcolor="#eeeeee", zeroline=False),
        yaxis=dict(showgrid=True, gridcolor="#eeeeee", zeroline=False),
    )
    st.plotly_chart(fig_hist, use_container_width=True)


# ── Tab 2: Model Results ───────────────────────────────────────────────────
with tab2:
    if y is None:
        st.info("Include a 'Machine failure' column in your CSV to compute evaluation metrics.")
        st.stop()

    accuracy  = accuracy_score(y, y_pred)
    try:
        auc = roc_auc_score(y, y_proba)
    except ValueError:
        auc = float("nan")
    precision = precision_score(y, y_pred, zero_division=0)
    recall    = recall_score(y, y_pred, zero_division=0)
    f1        = f1_score(y, y_pred, zero_division=0)
    mcc       = matthews_corrcoef(y, y_pred)

    best_metric = max(accuracy, auc, f1, mcc)

    def cell(label, val, highlight=False):
        cls = "metric-cell highlight" if highlight else "metric-cell"
        display = f"{val:.4f}" if not np.isnan(val) else "—"
        return (f'<div class="{cls}">'
                f'<div class="mlabel">{label}</div>'
                f'<div class="mvalue">{display}</div>'
                f'</div>')

    st.markdown('<div class="section-heading">Evaluation Metrics</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="metric-row">'
        + cell("Accuracy",  accuracy,  accuracy  == best_metric)
        + cell("AUC",       auc,       auc       == best_metric)
        + cell("Precision", precision)
        + cell("Recall",    recall)
        + cell("F1 Score",  f1,        f1        == best_metric)
        + cell("MCC",       mcc,       mcc       == best_metric)
        + '</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<p class="note">Classifier: <strong>{model_name}</strong> · '
        f'Highlighted cell = highest value among Accuracy / AUC / F1 / MCC</p>',
        unsafe_allow_html=True,
    )

    left_col, right_col = st.columns([1, 1])

    with left_col:
        st.markdown('<div class="section-heading">Confusion Matrix</div>', unsafe_allow_html=True)
        cm = confusion_matrix(y, y_pred)
        fig_cm = go.Figure(go.Heatmap(
            z=cm,
            x=["Predicted: No Failure", "Predicted: Failure"],
            y=["Actual: No Failure", "Actual: Failure"],
            colorscale=[[0, "#f0f4ff"], [1, "#1d4ed8"]],
            text=cm, texttemplate="<b>%{text}</b>",
            showscale=False,
        ))
        fig_cm.update_layout(
            plot_bgcolor="white", paper_bgcolor="white",
            font=dict(family="Inter", size=12, color="#333"),
            margin=dict(t=20, b=60, l=100, r=20),
            height=300,
        )
        st.plotly_chart(fig_cm, use_container_width=True)

    with right_col:
        st.markdown('<div class="section-heading">Classification Report</div>', unsafe_allow_html=True)
        report = classification_report(
            y, y_pred, target_names=["No Failure", "Failure"], zero_division=0
        )
        st.code(report, language=None)


# ── Tab 3: All Models ──────────────────────────────────────────────────────
with tab3:
    metrics_path = os.path.join(MODEL_DIR, "metrics_comparison.csv")
    if not os.path.exists(metrics_path):
        st.warning("metrics_comparison.csv not found — run model/train_models.py first.")
        st.stop()

    comp_df = pd.read_csv(metrics_path)

    st.markdown('<div class="section-heading">Comparison Table — Held-Out Test Split</div>',
                unsafe_allow_html=True)
    st.dataframe(
        comp_df.set_index("ML Model Name").style.highlight_max(axis=0, color="#dbeafe"),
        use_container_width=True,
    )
    st.markdown('<p class="note">Blue = highest value in each column</p>',
                unsafe_allow_html=True)

    st.markdown('<div class="section-heading">Visual Comparison</div>', unsafe_allow_html=True)
    metric_cols = ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]
    selected = st.multiselect("Metrics to display", metric_cols,
                              default=["Accuracy", "AUC", "F1", "MCC"])

    if selected:
        melted = comp_df.melt(
            id_vars="ML Model Name", value_vars=selected,
            var_name="Metric", value_name="Score",
        )
        fig_bar = px.bar(
            melted, x="ML Model Name", y="Score", color="Metric",
            barmode="group",
            color_discrete_sequence=["#1d4ed8", "#2563eb", "#3b82f6", "#60a5fa", "#93c5fd", "#bfdbfe"],
            labels={"ML Model Name": "", "Score": "Score"},
        )
        fig_bar.update_layout(
            plot_bgcolor="white", paper_bgcolor="white",
            font=dict(family="Inter", size=12, color="#333"),
            legend=dict(orientation="h", y=-0.2),
            margin=dict(t=20, b=80, l=40, r=20),
            height=360,
            xaxis=dict(showgrid=False, tickangle=-20),
            yaxis=dict(showgrid=True, gridcolor="#eeeeee", range=[0, 1.05]),
        )
        st.plotly_chart(fig_bar, use_container_width=True)

st.markdown("---")
st.markdown(
    '<p class="note">Dataset: S. Matzka, "Explainable Artificial Intelligence for Predictive '
    'Maintenance Applications," AI4I 2020, UCI ML Repository.</p>',
    unsafe_allow_html=True,
)
