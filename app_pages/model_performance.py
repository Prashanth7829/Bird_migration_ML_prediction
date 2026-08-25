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


# ============================================================
# CHECK MODEL RESULTS
# ============================================================

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

st.header("Machine Learning Prediction")

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
        "Best Model",
        str(champion["model"]),
    )


with c2:
    st.metric(
        "Accuracy",
        f"{float(champion['accuracy']):.2%}",
    )


with c3:
    st.metric(
        "F1 Score",
        f"{float(champion['f1_score']):.2%}",
    )


with c4:
    st.metric(
        "ROC-AUC",
        f"{float(champion['roc_auc']):.2%}",
    )


# ============================================================
# MODEL COMPARISON TABLE
# ============================================================

st.subheader("📊 Model Comparison")

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
    "F1 Score",
    "ROC-AUC",
]


# ------------------------------------------------------------
# Convert metrics from decimal to percentage
# ------------------------------------------------------------

percentage_columns = [
    "Accuracy",
    "Precision",
    "Recall",
    "F1 Score",
    "ROC-AUC",
]

comparison_display = comparison.copy()

for column in percentage_columns:
    comparison_display[column] = (
        comparison_display[column] * 100
    )


st.dataframe(
    comparison_display,
    hide_index=True,
    width="stretch",
    column_config={

        "Accuracy": st.column_config.NumberColumn(
            format="%.2f%%"
        ),

        "Precision": st.column_config.NumberColumn(
            format="%.2f%%"
        ),

        "Recall": st.column_config.NumberColumn(
            format="%.2f%%"
        ),

        "F1 Score": st.column_config.NumberColumn(
            format="%.2f%%"
        ),

        "ROC-AUC": st.column_config.NumberColumn(
            format="%.2f%%"
        ),
    },
)


# ============================================================
# PERFORMANCE INTERPRETATION
# ============================================================

st.subheader("💡 Performance Interpretation")

best_f1_model = metrics.loc[
    metrics["f1_score"].idxmax()
]

best_auc_model = metrics.loc[
    metrics["roc_auc"].idxmax()
]

best_accuracy_model = metrics.loc[
    metrics["accuracy"].idxmax()
]


st.write(
    f"""
    **F1-score:** {best_f1_model["model"]} has the highest F1-score
    at **{float(best_f1_model["f1_score"]):.2%}**, indicating the
    strongest balance between precision and recall among the
    evaluated models.

    **ROC-AUC:** {best_auc_model["model"]} has the highest ROC-AUC
    at **{float(best_auc_model["roc_auc"]):.2%}**, representing the
    strongest ability to distinguish between successful and failed
    migrations.

    **Accuracy:** {best_accuracy_model["model"]} has the highest
    accuracy at **{float(best_accuracy_model["accuracy"]):.2%}**.
    """
)


# ============================================================
# RANDOM FOREST FEATURE IMPORTANCE
# ============================================================

st.divider()

st.header("Random Forest & Feature Importance")

st.caption(
    "Random Forest feature importance is used to identify the "
    "most influential predictive drivers of migration success."
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

    st.subheader("📊 Top Predictive Drivers")

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
                title="Feature Importance",
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
        width="stretch",
    )


    st.subheader("Feature Importance Table")

    st.dataframe(
        rf_importance,
        hide_index=True,
        width="stretch",
    )


# ============================================================
# CONFUSION MATRIX
# ============================================================

st.divider()

st.subheader("🔢 Confusion Matrix")

st.caption(
    "The confusion matrix below is displayed as percentages "
    "of the actual class rather than raw record counts."
)


if evaluation:

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


        # ----------------------------------------------------
        # Raw confusion matrix
        # ----------------------------------------------------

        matrix = pd.crosstab(
            y_true,
            y_pred,
            dropna=False,
        )


        # Ensure both classes always exist
        matrix = matrix.reindex(
            index=[0, 1],
            columns=[0, 1],
            fill_value=0,
        )


        # ----------------------------------------------------
        # Convert each actual class row to percentage
        # ----------------------------------------------------

        percentage_matrix = (
            matrix.div(
                matrix.sum(axis=1),
                axis=0,
            ) * 100
        )


        percentage_matrix.index = [
            "Failed (0)",
            "Successful (1)",
        ]

        percentage_matrix.columns = [
            "Predicted Failed (0)",
            "Predicted Successful (1)",
        ]


        st.dataframe(
            percentage_matrix.round(2),
            width="stretch",
        )


        # ----------------------------------------------------
        # Interpretation
        # ----------------------------------------------------

        failed_correct = percentage_matrix.loc[
            "Failed (0)",
            "Predicted Failed (0)",
        ]

        failed_wrong = percentage_matrix.loc[
            "Failed (0)",
            "Predicted Successful (1)",
        ]

        success_correct = percentage_matrix.loc[
            "Successful (1)",
            "Predicted Successful (1)",
        ]

        success_wrong = percentage_matrix.loc[
            "Successful (1)",
            "Predicted Failed (0)",
        ]


        col1, col2 = st.columns(2)


        with col1:

            st.markdown(
                f"""
                **Failed migrations**

                - Correctly predicted as failed:
                  **{failed_correct:.2f}%**
                - Incorrectly predicted as successful:
                  **{failed_wrong:.2f}%**
                """
            )


        with col2:

            st.markdown(
                f"""
                **Successful migrations**

                - Correctly predicted as successful:
                  **{success_correct:.2f}%**
                - Incorrectly predicted as failed:
                  **{success_wrong:.2f}%**
                """
            )


else:

    st.info(
        "Confusion matrix information is not available yet."
    )


# ============================================================
# FINAL INTERPRETATION
# ============================================================

st.divider()

st.subheader("📝 Overall Interpretation")

st.info(
    """
    Model performance should be interpreted using multiple metrics
    rather than accuracy alone.

    **Accuracy** represents the overall proportion of correctly
    classified migrations.

    **Precision** indicates how reliable the model's positive
    migration predictions are.

    **Recall** indicates how effectively the model identifies
    successful migrations.

    **F1-score** provides a balance between precision and recall.

    **ROC-AUC** measures how well the model separates successful
    and failed migration outcomes across different classification
    thresholds.

    Feature importance represents predictive association within
    the dataset and should not be interpreted as proof of biological
    causation.
    """,
    icon=":material/info:",
)