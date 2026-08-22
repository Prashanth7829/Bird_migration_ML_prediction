"""Machine learning model evaluation and feature importance."""

import altair as alt
import pandas as pd
import streamlit as st

from src.dashboard import (
    load_evaluation,
    load_feature_importance,
    load_metrics,
    load_metadata,
)


# ============================================================
# LOAD
# ============================================================

metrics = load_metrics()
metadata = load_metadata()
evaluation = load_evaluation()
importance = load_feature_importance()


# ============================================================
# HEADER
# ============================================================

st.title("Model Performance & Explainability")

st.caption(
    "Compare six classification models using the same held-out "
    "test set and analyze the strongest predictive drivers."
)


if metrics.empty:

    st.warning(
        "Model results are not available yet. "
        "Run `python -m src.train`.",
        icon=":material/pending:",
    )

    st.stop()


# ============================================================
# Q6 — MODEL PERFORMANCE
# ============================================================

st.header(
    "6. Machine Learning Prediction"
)

st.caption(
    "Research Question: Which features have the greatest influence "
    "on migration success, and how effectively can machine learning "
    "models predict successful and unsuccessful migrations?"
)


# ============================================================
# BEST MODEL
# ============================================================

champion = metrics.iloc[0]

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.metric(
        "Best model",
        champion["model"],
    )

with c2:
    st.metric(
        "Accuracy",
        f"{champion['accuracy']:.3f}",
    )

with c3:
    st.metric(
        "F1-score",
        f"{champion['f1_score']:.3f}",
    )

with c4:
    st.metric(
        "ROC-AUC",
        f"{champion['roc_auc']:.3f}",
    )


# ============================================================
# COMPARISON TABLE
# ============================================================

st.subheader(
    "Model comparison"
)


comparison = metrics[
    [
        "model",
        "accuracy",
        "precision",
        "recall",
        "f1_score",
        "roc_auc",
    ]
].copy()


comparison.columns = [
    "Model",
    "Accuracy",
    "Precision",
    "Recall",
    "F1",
    "ROC-AUC",
]


st.dataframe(
    comparison,
    hide_index=True,
    use_container_width=True,
    column_config={
        "Accuracy":
            st.column_config.NumberColumn(
                format="%.3f"
            ),
        "Precision":
            st.column_config.NumberColumn(
                format="%.3f"
            ),
        "Recall":
            st.column_config.NumberColumn(
                format="%.3f"
            ),
        "F1":
            st.column_config.NumberColumn(
                format="%.3f"
            ),
        "ROC-AUC":
            st.column_config.NumberColumn(
                format="%.3f"
            ),
    },
)


# ============================================================
# MODEL COMPARISON CHART
# ============================================================

metric_long = comparison.melt(
    id_vars="Model",
    value_vars=[
        "Accuracy",
        "Precision",
        "Recall",
        "F1",
        "ROC-AUC",
    ],
    var_name="Metric",
    value_name="Score",
)


chart = (
    alt.Chart(metric_long)
    .mark_bar()
    .encode(
        x=alt.X(
            "Metric:N"
        ),
        y=alt.Y(
            "Score:Q",
            scale=alt.Scale(
                domain=[0, 1]
            ),
        ),
        color="Model:N",
        tooltip=[
            "Model",
            "Metric",
            alt.Tooltip(
                "Score:Q",
                format=".3f",
            ),
        ],
    )
    .properties(
        height=420,
    )
)


st.altair_chart(
    chart,
    use_container_width=True,
)


# ============================================================
# Q7 — RANDOM FOREST
# ============================================================

st.divider()

st.header(
    "7. Random Forest & Feature Importance"
)

st.caption(
    "Random Forest is treated as the primary analytical model, "
    "but the final champion is selected using actual evaluation results."
)


rf_importance = pd.DataFrame(
    importance.get(
        "random_forest",
        [],
    )
)


if rf_importance.empty:

    st.warning(
        "Random Forest feature importance is not available."
    )

else:

    st.subheader(
        "Top predictive drivers"
    )

    top = (
        rf_importance
        .head(15)
        .sort_values(
            "importance"
        )
    )


    chart = (
        alt.Chart(top)
        .mark_bar()
        .encode(
            x=alt.X(
                "importance:Q",
                title="Feature importance",
            ),
            y=alt.Y(
                "feature:N",
                sort=None,
                title=None,
            ),
            tooltip=[
                "feature",
                alt.Tooltip(
                    "importance:Q",
                    format=".5f",
                ),
            ],
        )
        .properties(
            height=500,
        )
    )


    st.altair_chart(
        chart,
        use_container_width=True,
    )


    st.dataframe(
        rf_importance,
        hide_index=True,
        use_container_width=True,
    )


# ============================================================
# CONFUSION MATRIX
# ============================================================

st.divider()

st.subheader(
    "Confusion matrix"
)


selected_model = st.selectbox(
    "Select model",
    list(evaluation.keys()),
)


if selected_model in evaluation:

    result = evaluation[
        selected_model
    ]

    y_true = pd.Series(
        result["y_true"]
    )

    y_pred = pd.Series(
        result["y_pred"]
    )


    matrix = pd.crosstab(
        y_true,
        y_pred,
    )

    matrix.index.name = "Actual"
    matrix.columns.name = "Predicted"


    st.dataframe(
        matrix,
        use_container_width=True,
    )


# ============================================================
# INTERPRETATION
# ============================================================

st.info(
    """
    **Interpretation:** A model with higher ROC-AUC generally provides
    stronger ranking ability across the two classes. F1-score balances
    precision and recall and is useful when both successful and failed
    migrations matter. No model is selected solely because it has the
    highest accuracy.
    """,
    icon=":material/info:",
)