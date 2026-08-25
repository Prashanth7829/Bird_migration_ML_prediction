"""About page for the Bird Migration Analytics project."""

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

st.title("ℹ️ About the Project")

st.caption(
    "Bird Migration Analytics – Predicting Migration Success"
)

st.write(
    "An end-to-end machine learning project for analysing "
    "bird migration patterns, predicting migration outcomes, "
    "and identifying important environmental, geographical, "
    "flight, behavioural, and biological factors."
)

st.divider()


# ============================================================
# PROJECT OVERVIEW
# ============================================================

with st.container(border=True):

    st.subheader("🦅 Bird Migration Analytics")

    st.write(
        """
        Bird Migration Analytics applies machine learning and
        exploratory analysis to understand the conditions associated
        with successful and unsuccessful bird migration journeys.

        The project combines multiple dimensions of migration data,
        including species, geography, environmental conditions,
        migration motivation, flight characteristics, behavioural
        patterns, and seasonal information.

        The resulting system provides both predictive capabilities
        and analytical insights through an interactive Streamlit
        dashboard.
        """
    )


# ============================================================
# PROJECT OBJECTIVES
# ============================================================

st.subheader("🎯 Project Objectives")

objective_col1, objective_col2 = st.columns(2)

with objective_col1:

    with st.container(border=True):

        st.markdown("### 🔎 Data & Migration Analysis")

        st.markdown(
            """
            - Analyse migration patterns across species
            - Explore regional and habitat differences
            - Study seasonal migration behaviour
            - Investigate environmental conditions
            - Analyse flight characteristics
            - Examine behavioural factors
            """
        )


with objective_col2:

    with st.container(border=True):

        st.markdown("### 🤖 Machine Learning")

        st.markdown(
            """
            - Predict migration success or failure
            - Compare multiple classification algorithms
            - Evaluate predictive performance
            - Identify important predictive drivers
            - Support individual migration predictions
            - Provide interpretable model insights
            """
        )


st.divider()


# ============================================================
# DATASET SUMMARY
# ============================================================

st.subheader("📊 Dataset Summary")

st.write(
    "The project is built using the processed bird migration "
    "dataset supplied for the case study."
)

dataset_col1, dataset_col2, dataset_col3 = st.columns(3)

with dataset_col1:

    st.metric(
        "🗃️ Migration Records",
        f"{len(data):,}",
    )

with dataset_col2:

    st.metric(
        "🐦 Species",
        data["Species"].nunique(),
    )

with dataset_col3:

    st.metric(
        "🗺️ Regions",
        data["Region"].nunique(),
    )


# ============================================================
# FEATURE GROUPS
# ============================================================

st.subheader("🔍 Machine Learning Features")

st.write(
    "The predictive model uses selected variables representing "
    "information that can reasonably be available before or "
    "during a migration journey."
)

feature_col1, feature_col2 = st.columns(2)

with feature_col1:

    with st.container(border=True):

        st.markdown("### Categorical Features")

        for feature in CATEGORICAL_FEATURES:

            st.write(
                f"• {feature.replace('_', ' ')}"
            )


with feature_col2:

    with st.container(border=True):

        st.markdown("### Numerical Features")

        for feature in NUMERIC_FEATURES:

            st.write(
                f"• {feature.replace('_', ' ')}"
            )


# ============================================================
# FEATURE SELECTION
# ============================================================

with st.container(border=True):

    st.subheader("🧩 Feature Selection")

    st.write(
        f"""
        The final predictive model uses **{len(FEATURE_COLUMNS)}
        selected predictor variables**.

        The target variable is:

        **{TARGET_COLUMN}**

        Target encoding:

        - `1` → Successful migration
        - `0` → Failed migration
        """
    )

    st.caption(
        "Post-outcome variables and potential leakage variables "
        "are excluded from the primary predictive feature set."
    )


st.divider()


# ============================================================
# MACHINE LEARNING MODELS
# ============================================================

st.subheader("🤖 Machine Learning Models")

st.write(
    """
    Six classification algorithms are trained and evaluated
    using the same modelling workflow and held-out test data.
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

model_col1, model_col2 = st.columns(2)

for index, model in enumerate(models):

    target_col = (
        model_col1
        if index % 2 == 0
        else model_col2
    )

    with target_col:

        st.write(
            f"**{index + 1}.** {model}"
        )


st.divider()


# ============================================================
# CURRENT MODEL PERFORMANCE
# ============================================================

st.subheader("🏆 Current Model Performance")

if not metrics.empty:

    champion = metrics.iloc[0]

    performance_col1, performance_col2, performance_col3 = (
        st.columns(3)
    )

    with performance_col1:

        st.metric(
            "Best Model",
            str(champion["model"]),
        )

    with performance_col2:

        st.metric(
            "F1 Score",
            f"{float(champion['f1_score']):.3f}",
        )

    with performance_col3:

        st.metric(
            "ROC-AUC",
            f"{float(champion['roc_auc']):.3f}",
        )

    st.caption(
        "Performance values are calculated from the trained "
        "model evaluation results."
    )

else:

    st.info(
        "Model results are not available yet. "
        "Run `python -m src.train` first."
    )


st.divider()


# ============================================================
# DATA LEAKAGE AWARENESS
# ============================================================

st.subheader("🛡️ Data Leakage Consideration")

with st.container(border=True):

    st.write(
        """
        A leakage-aware modelling approach is followed throughout
        the project.

        Variables that may only become available after the migration
        outcome is known should not be used as primary prediction
        features.

        Examples include:

        - Migration interruption information
        - Interruption reasons
        - Recovery information
        - Recovery time
        - Nesting outcomes
        - Other post-outcome indicators

        Excluding these variables helps ensure that the model learns
        from information that could realistically be available when
        estimating migration success.
        """
    )


# ============================================================
# RESEARCH QUESTIONS
# ============================================================

st.subheader("📚 Research Questions")

questions = [
    "What are the key trends in bird migration behaviour across species, regions, habitats, and seasons?",
    "How do environmental factors influence migration success?",
    "What are the main reasons for bird migration and how do they vary across species and regions?",
    "How do flight characteristics affect migration success?",
    "How do behavioural factors influence migration outcomes?",
    "Which features have the greatest influence on migration success?",
    "How does these models comparatively perform Random Forest, Logistic Regression, Decision Tree, KNN, SVM, and Gradient Boosting, and which model provides the most effective prediction of bird migration success?"
]

for number, question in enumerate(
    questions,
    start=1,
):

    st.markdown(
        f"**RQ{number}.** {question}"
    )


st.divider()



# ============================================================
# PROJECT STATUS
# ============================================================

st.subheader("⚙️ Project Status")

if metadata:

    st.success(
        "Model training artifacts are available.",
        icon=":material/check_circle:",
    )

    trained_at = metadata.get(
        "trained_at_utc"
    )

    if trained_at:

        st.caption(
            f"Last training time (UTC): {trained_at}"
        )

else:

    st.warning(
        "Training metadata is not available yet. "
        "Run the training pipeline to generate model artifacts."
    )