"""Interactive prediction and model comparison page."""

from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st

from src.dashboard import (
    batch_predictions,
    categorical_columns,
    format_column,
    input_defaults,
    load_data,
    load_feature_importance,
    load_metrics,
    load_models,
    load_metadata,
    models_ready,
    numeric_columns,
    prediction_table,
)
from src.modeling import MODEL_LABELS, build_prediction_template


# ============================================================
# LOAD DATA
# ============================================================

data = load_data()
metrics = load_metrics()
metadata = load_metadata()


# ============================================================
# PAGE HEADER
# ============================================================

st.title("🦅 Predict & Compare")

st.caption(
    "Enter migration conditions and compare predictions "
    "from all six trained machine-learning models."
)


# ============================================================
# MODEL CHECK
# ============================================================

if not models_ready():
    st.warning(
        "Trained models are not available yet. "
        "Run `python -m src.train` first.",
        icon="⚠️",
    )
    st.stop()


models = load_models()
importance = load_feature_importance()
defaults = input_defaults(data)


if not models:
    st.error("No trained model pipelines could be loaded.")
    st.stop()


# ============================================================
# MODEL INFORMATION
# ============================================================

model_options = list(MODEL_LABELS.keys())

champion_key = str(
    metadata.get(
        "champion_key",
        model_options[0],
    )
)

model_display = {
    "🏆 Best validation model": champion_key
}

for key, label in MODEL_LABELS.items():
    model_display[label] = key


# ============================================================
# INPUT MODE
# ============================================================

mode = st.segmented_control(
    "Prediction input",
    ["Single record", "CSV upload"],
    default="Single record",
    required=True,
)


# ============================================================
# SINGLE RECORD
# ============================================================

if mode == "Single record":

    form_col, result_col, importance_col = st.columns(
        [1.08, 0.9, 1.12],
        vertical_alignment="top",
    )

    # --------------------------------------------------------
    # FORM
    # --------------------------------------------------------

    with form_col:

        with st.form(
            "single_prediction_form",
            clear_on_submit=False,
            border=True,
        ):

            st.subheader(
                "🧭 Migration conditions",
                anchor=False,
            )

            values: dict[str, object] = {}

            # -----------------------------
            # CATEGORICAL FEATURES
            # -----------------------------

            st.markdown("#### Biological & environmental")

            for column in categorical_columns():

                options = sorted(
                    data[column]
                    .dropna()
                    .astype(str)
                    .unique()
                    .tolist()
                )

                if not options:
                    continue

                default = str(
                    defaults.get(
                        column,
                        options[0],
                    )
                )

                if default not in options:
                    default = options[0]

                values[column] = st.selectbox(
                    format_column(column),
                    options,
                    index=options.index(default),
                )

            # -----------------------------
            # NUMERICAL FEATURES
            # -----------------------------

            with st.expander(
                "✈️ Flight, environment & behaviour",
                expanded=True,
            ):

                for column in numeric_columns():

                    series = pd.to_numeric(
                        data[column],
                        errors="coerce",
                    ).dropna()

                    if series.empty:
                        continue

                    lower = float(series.quantile(0.01))
                    upper = float(series.quantile(0.99))

                    if lower == upper:
                        lower = float(series.min())
                        upper = float(series.max())

                    default_value = float(
                        defaults.get(
                            column,
                            series.median(),
                        )
                    )

                    default_value = min(
                        max(default_value, lower),
                        upper,
                    )

                    values[column] = st.number_input(
                        format_column(column),
                        min_value=lower,
                        max_value=upper,
                        value=default_value,
                    )

            # -----------------------------
            # MODEL SELECTION
            # -----------------------------

            selected_display = st.selectbox(
                "Model to highlight",
                list(model_display.keys()),
            )

            # =================================================
            # IMPORTANT:
            # submit button MUST remain inside st.form()
            # =================================================

            submitted = st.form_submit_button(
                "🚀 Predict all models",
                type="primary",
            )

    # ========================================================
    # PROCESS SUBMISSION
    # ========================================================

    if submitted:

        st.session_state["single_prediction"] = {
            "frame": pd.DataFrame([values]),
            "selected_key": model_display[selected_display],
        }

    # ========================================================
    # DISPLAY RESULT
    # ========================================================

    payload = st.session_state.get(
        "single_prediction"
    )

    if payload:

        selected_key = payload["selected_key"]

        results = prediction_table(
            models,
            payload["frame"],
        )

        if metrics.empty:
            merged = results
        else:
            merged = results.merge(
                metrics,
                on="model_key",
                how="left",
            )

        selected_rows = merged[
            merged["model_key"] == selected_key
        ]

        if selected_rows.empty:
            st.error(
                "Selected model is not available."
            )
            st.stop()

        selected = selected_rows.iloc[0]

        # ----------------------------------------------------
        # SELECTED MODEL RESULT
        # ----------------------------------------------------

        with result_col:

            with st.container(border=True):

                st.subheader(
                    "🎯 Selected model",
                    anchor=False,
                )

                st.caption(
                    selected["Model"]
                )

                prediction = selected[
                    "Prediction"
                ]

                probability = float(
                    selected[
                        "Success probability"
                    ]
                )

                if prediction == "Successful":

                    st.success(
                        "MIGRATION SUCCESSFUL",
                        icon="✅",
                    )

                else:

                    st.error(
                        "MIGRATION FAILED",
                        icon="❌",
                    )

                st.metric(
                    "Probability of success",
                    f"{probability:.1%}",
                )

                st.metric(
                    "Probability of failure",
                    f"{1 - probability:.1%}",
                )

                st.caption(
                    "The probability represents the model's "
                    "prediction for the entered migration conditions."
                )

        # ----------------------------------------------------
        # FEATURE IMPORTANCE
        # ----------------------------------------------------

        with importance_col:

            with st.container(border=True):

                st.subheader(
                    "📊 Feature importance",
                    anchor=False,
                )

                selected_importance = pd.DataFrame(
                    importance.get(
                        selected_key,
                        [],
                    )
                )

                if selected_importance.empty:

                    st.info(
                        "Feature importance is unavailable "
                        "for this model."
                    )

                else:

                    chart_data = (
                        selected_importance
                        .head(10)
                        .copy()
                    )

                    chart = (
                        alt.Chart(chart_data)
                        .mark_bar(
                            cornerRadiusTopRight=5,
                            cornerRadiusBottomRight=5,
                        )
                        .encode(
                            x=alt.X(
                                "importance:Q",
                                title="Importance",
                            ),
                            y=alt.Y(
                                "feature:N",
                                sort="-x",
                                title=None,
                            ),
                            tooltip=[
                                "feature",
                                alt.Tooltip(
                                    "importance:Q",
                                    format=".3f",
                                ),
                            ],
                        )
                        .properties(
                            height=320,
                        )
                    )

                    st.altair_chart(
                        chart,
                        width="stretch",
                    )

        # ====================================================
        # ALL MODEL COMPARISON
        # ====================================================

        with st.container(border=True):

            st.subheader(
                "📈 All-model comparison",
                anchor=False,
            )

            columns = [
                "Model",
                "Prediction",
                "Success probability",
                "accuracy",
                "precision",
                "recall",
                "f1_score",
                "roc_auc",
            ]

            available_columns = [
                column
                for column in columns
                if column in merged.columns
            ]

            display = merged[
                available_columns
            ].copy()

            display = display.rename(
                columns={
                    "accuracy": "Accuracy",
                    "precision": "Precision",
                    "recall": "Recall",
                    "f1_score": "F1 score",
                    "roc_auc": "ROC-AUC",
                }
            )

            st.dataframe(
                display,
                hide_index=True,
                width="stretch",
                column_config={
                    "Success probability":
                        st.column_config.NumberColumn(
                            format="%.1%",
                        ),
                    "Accuracy":
                        st.column_config.NumberColumn(
                            format="%.3f",
                        ),
                    "Precision":
                        st.column_config.NumberColumn(
                            format="%.3f",
                        ),
                    "Recall":
                        st.column_config.NumberColumn(
                            format="%.3f",
                        ),
                    "F1 score":
                        st.column_config.NumberColumn(
                            format="%.3f",
                        ),
                    "ROC-AUC":
                        st.column_config.NumberColumn(
                            format="%.3f",
                        ),
                },
            )

    else:

        with result_col:

            st.info(
                "Enter migration conditions and click "
                "**Predict all models**.",
                icon="ℹ️",
            )

        with importance_col:

            st.info(
                "Feature importance for the selected "
                "model will appear after prediction.",
                icon="📊",
            )


# ============================================================
# CSV BATCH PREDICTION
# ============================================================

else:

    with st.container(border=True):

        st.subheader(
            "📂 Batch CSV prediction",
            anchor=False,
        )

        st.write(
            "Upload a CSV containing the approved model "
            "input features. The trained pipelines will "
            "generate predictions for every available model."
        )

        # ----------------------------------------------------
        # TEMPLATE
        # ----------------------------------------------------

        template = (
            build_prediction_template(data)
            .to_csv(index=False)
            .encode("utf-8")
        )

        st.download_button(
            "⬇️ Download input template",
            data=template,
            file_name=(
                "bird_migration_prediction_template.csv"
            ),
            mime="text/csv",
        )

        uploaded = st.file_uploader(
            "Upload prediction CSV",
            type=["csv"],
            max_upload_size=50,
            key="batch_upload",
        )

        if uploaded is not None:

            try:

                uploaded_data = pd.read_csv(
                    uploaded
                )

                results = batch_predictions(
                    models,
                    uploaded_data,
                )

                st.success(
                    f"Generated predictions for "
                    f"{len(results):,} record(s).",
                    icon="✅",
                )

                st.dataframe(
                    results.head(100),
                    hide_index=True,
                    width="stretch",
                )

                st.download_button(
                    "⬇️ Download predictions",
                    data=(
                        results
                        .to_csv(index=False)
                        .encode("utf-8")
                    ),
                    file_name=(
                        "bird_migration_model_predictions.csv"
                    ),
                    mime="text/csv",
                )

            except (
                ValueError,
                pd.errors.ParserError,
                KeyError,
            ) as error:

                st.error(
                    f"Prediction failed: {error}",
                    icon="❌",
                )