"""About page for the Bird Migration Success Prediction project."""

import streamlit as st

from src.dashboard import (
    load_data,
    load_metadata,
    load_metrics,
)
from src.modeling import (
    CATEGORICAL_FEATURES,
    FEATURE_COLUMNS,
    NUMERIC_FEATURES,
    TARGET_COLUMN,
)


# ============================================================
# LOAD DATA
# ============================================================

data = load_data()
metadata = load_metadata()
metrics = load_metrics()


# ============================================================
# PAGE TITLE
# ============================================================

st.title("About the Project")

st.caption(
    "Bird Migration Analytics – Predicting Migration Success"
)


# ============================================================
# PROJECT OVERVIEW
# ============================================================

with st.container(border=True):

    st.subheader("🦅 Bird Migration Success Prediction")

    st.write(
        """
        This project uses machine learning to analyse bird migration
        records and predict whether a migration journey is likely to be
        successful or unsuccessful.

        The analysis combines biological, geographical, environmental,
        flight, behavioural, and temporal information to identify
        important predictive factors associated with migration success.
        """
    )


# ============================================================
# PROJECT OBJECTIVES
# ============================================================

st.subheader("🎯 Project Objectives")

col1, col2 = st.columns(2)

with col1:

    st.markdown(
        """
        **Data Analysis**

        - Analyse bird migration patterns
        - Study environmental conditions
        - Compare species and regions
        - Analyse flight characteristics
        - Investigate behavioural factors
        """
    )

with col2:

    st.markdown(
        """
        **Machine Learning**

        - Predict migration success
        - Compare six classification models
        - Identify important predictive features
        - Evaluate model performance
        - Support migration-risk analysis
        """
    )


# ============================================================
# DATASET SUMMARY
# ============================================================

st.subheader("📊 Dataset Summary")

successful = int(
    (data[TARGET_COLUMN] == 1).sum()
)

failed = int(
    (data[TARGET_COLUMN] == 0).sum()
)

success_rate = (
    successful / len(data)
    if len(data) > 0
    else 0
)

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.metric(
        "Migration Records",
        f"{len(data):,}",
    )

with c2:
    st.metric(
        "Successful",
        f"{successful:,}",
    )

with c3:
    st.metric(
        "Failed",
        f"{failed:,}",
    )

with c4:
    st.metric(
        "Success Rate",
        f"{success_rate:.1%}",
    )


# ============================================================
# FEATURE GROUPS
# ============================================================

st.subheader("🔍 Machine Learning Features")

st.write(
    "The predictive model uses selected features that represent "
    "information available before or during migration."
)

feature_col1, feature_col2 = st.columns(2)

with feature_col1:

    st.markdown("### Categorical Features")

    for feature in CATEGORICAL_FEATURES:
        st.write(f"• {feature}")

with feature_col2:

    st.markdown("### Numerical Features")

    for feature in NUMERIC_FEATURES:
        st.write(f"• {feature}")


# ============================================================
# FEATURE COUNT
# ============================================================

with st.container(border=True):

    st.subheader("Feature Selection")

    st.write(
        f"""
        The final predictive model uses **{len(FEATURE_COLUMNS)} selected
        predictor variables**.

        The target variable is:

        **{TARGET_COLUMN}**

        Target encoding:

        - `1` → Successful migration
        - `0` → Failed migration
        """
    )


# ============================================================
# MODEL COMPARISON
# ============================================================

st.subheader("🤖 Machine Learning Models")

st.write(
    """
    Six classification algorithms are trained and evaluated using the
    same train/test split.
    """
)

models = [
    "Logistic Regression",
    "Decision Tree",
    "K-Nearest Neighbors",
    "Support Vector Machine",
    "Gradient Boosting",
    "Random Forest",
]

for model in models:
    st.write(f"• {model}")


# ============================================================
# BEST MODEL
# ============================================================

if not metrics.empty:

    st.subheader("🏆 Current Model Performance")

    champion = metrics.iloc[0]

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric(
            "Best Model",
            str(champion["model"]),
        )

    with c2:
        st.metric(
            "F1 Score",
            f"{float(champion['f1_score']):.3f}",
        )

    with c3:
        st.metric(
            "ROC-AUC",
            f"{float(champion['roc_auc']):.3f}",
        )

    st.caption(
        "The best model is selected from the actual validation results "
        "rather than assuming Random Forest will always perform best."
    )

else:

    st.info(
        "Model results are not available yet. "
        "Run `python -m src.train` first."
    )


# ============================================================
# LEAKAGE AWARENESS
# ============================================================

st.subheader("🛡️ Data Leakage Consideration")

st.write(
    """
    The project follows a leakage-aware modelling approach.

    Variables that may only become available after a migration outcome
    is known should not be used as primary prediction features.

    Examples include migration interruption information, recovery
    information, nesting outcomes, and other post-outcome indicators.

    This helps ensure that the model learns from information that could
    realistically be available when predicting migration success.
    """
)


# ============================================================
# RESEARCH QUESTIONS
# ============================================================

st.subheader("📚 Research Questions")

questions = [
    "What are the key trends in bird migration behaviour across species, regions, habitats, and seasons?",
    "How do environmental factors influence migration success?",
    "What are the main reasons for bird migration and how do they vary?",
    "How do flight characteristics affect migration success?",
    "How do behavioural factors influence migration outcomes?",
    "Which features have the greatest influence on migration success?",
    "How does Random Forest compare with the other machine learning models?",
]

for number, question in enumerate(questions, start=1):
    st.markdown(f"**RQ{number}.** {question}")


# ============================================================
# DISCLAIMER
# ============================================================

with st.container(border=True):

    st.subheader("⚠️ Interpretation")

    st.write(
        """
        Machine learning feature importance and predictions represent
        predictive associations within the available dataset.

        They should not be interpreted as proof of causation.

        Therefore, the dashboard uses terms such as **predictive drivers**,
        **important contributing factors**, and **factors associated with
        migration success**.
        """
    )


# ============================================================
# PROJECT STATUS
# ============================================================

st.subheader("⚙️ Project Status")

if metadata:

    st.success("Model training artifacts are available.")

    if metadata.get("trained_at_utc"):
        st.caption(
            f"Last training time (UTC): "
            f"{metadata['trained_at_utc']}"
        )

else:

    st.warning(
        "Training metadata is not available yet."
    )