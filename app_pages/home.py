import altair as alt
import pandas as pd
import streamlit as st

from src.dashboard import (
    load_data,
    load_metadata,
    load_metrics,
    load_feature_importance,
)


# ============================================================
# LOAD DATA
# ============================================================

data = load_data()
metadata = load_metadata()
metrics = load_metrics()
feature_importance = load_feature_importance()


# ============================================================
# PAGE TITLE
# ============================================================

st.title("🦅 Bird Migration Analytics")

st.write(
    "A machine learning system designed to predict migration "
    "success and explore the environmental, geographical, "
    "flight, and behavioural factors associated with bird migration."
)

st.caption(
    "MACHINE LEARNING  •  MIGRATION ANALYTICS  •  PREDICTIVE INSIGHTS"
)

st.divider()


# ============================================================
# DATASET OVERVIEW
# ============================================================

st.subheader("📊 Dataset Overview")

total_records = len(data)
species_names = sorted(
    data["Species"].dropna().astype(str).unique().tolist()
)
region_names = sorted(
    data["Region"].dropna().astype(str).unique().tolist()
)

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "🗃️ Migration Records",
        f"{total_records:,}",
    )

with col2:
    st.metric(
        "🐦 Species",
        data["Species"].nunique(),
    )

with col3:
    st.metric(
        "🗺️ Regions",
        data["Region"].nunique(),
    )


# ============================================================
# SPECIES AND REGION NAMES
# ============================================================

species_col, region_col = st.columns(2)

with species_col:

    with st.container(border=True):

        st.markdown("### 🐦 Species Covered")

        st.write(
            "Bird species represented in the migration dataset:"
        )

        st.write(
            " • ".join(species_names)
        )


with region_col:

    with st.container(border=True):

        st.markdown("### 🗺️ Regions Covered")

        st.write(
            "Geographic regions represented in the migration records:"
        )

        st.write(
            " • ".join(region_names)
        )


st.divider()


# ============================================================
# MODEL PERFORMANCE
# ============================================================

st.subheader("🏆 Current Model Performance")

if metrics.empty:

    st.warning(
        "Models have not been trained yet. "
        "Run `python -m src.train` to generate the model results."
    )

else:

    champion = metrics.iloc[0]

    model_col1, model_col2, model_col3 = st.columns(3)

    with model_col1:

        st.metric(
            "Best Validation Model",
            str(champion["model"]),
        )

    with model_col2:

        st.metric(
            "F1 Score",
            f"{float(champion['f1_score']):.3f}",
        )

    with model_col3:

        st.metric(
            "ROC-AUC",
            f"{float(champion['roc_auc']):.3f}",
        )

    if metadata:

        train_rows = metadata.get("train_rows")
        test_rows = metadata.get("test_rows")

        if train_rows is not None and test_rows is not None:

            st.caption(
                f"Training records: {int(train_rows):,} "
                f"| Test records: {int(test_rows):,}"
            )


st.divider()


# ============================================================
# WEATHER ANALYSIS
# ============================================================

weather_col, species_success_col = st.columns(2)


with weather_col:

    with st.container(border=True):

        st.subheader(
            "🌦️ Migration Success by Weather"
        )

        st.caption(
            "Observed migration success across different weather conditions."
        )

        weather = (
            data.groupby(
                "Weather_Condition",
                as_index=False,
            )["Migration_Success_Num"]
            .mean()
            .rename(
                columns={
                    "Migration_Success_Num": "Success Rate"
                }
            )
        )

        chart = (
            alt.Chart(weather)
            .mark_bar(
                cornerRadiusTopRight=5,
                cornerRadiusBottomRight=5,
            )
            .encode(
                x=alt.X(
                    "Success Rate:Q",
                    title="Success Rate",
                    axis=alt.Axis(format="%"),
                ),
                y=alt.Y(
                    "Weather_Condition:N",
                    title=None,
                    sort="-x",
                ),
                tooltip=[
                    alt.Tooltip(
                        "Weather_Condition:N",
                        title="Weather",
                    ),
                    alt.Tooltip(
                        "Success Rate:Q",
                        title="Success Rate",
                        format=".1%",
                    ),
                ],
            )
            .properties(height=300)
        )

        st.altair_chart(
            chart,
            width="stretch",
        )


# ============================================================
# SPECIES SUCCESS
# ============================================================

with species_success_col:

    with st.container(border=True):

        st.subheader(
            "🐦 Migration Success by Species"
        )

        st.caption(
            "Observed success rates across the bird species in the dataset."
        )

        species = (
            data.groupby(
                "Species",
                as_index=False,
            )["Migration_Success_Num"]
            .mean()
            .rename(
                columns={
                    "Migration_Success_Num": "Success Rate"
                }
            )
        )

        chart = (
            alt.Chart(species)
            .mark_bar(
                cornerRadiusTopRight=5,
                cornerRadiusBottomRight=5,
            )
            .encode(
                x=alt.X(
                    "Success Rate:Q",
                    title="Success Rate",
                    axis=alt.Axis(format="%"),
                ),
                y=alt.Y(
                    "Species:N",
                    title=None,
                    sort="-x",
                ),
                tooltip=[
                    alt.Tooltip(
                        "Species:N",
                        title="Species",
                    ),
                    alt.Tooltip(
                        "Success Rate:Q",
                        title="Success Rate",
                        format=".1%",
                    ),
                ],
            )
            .properties(height=300)
        )

        st.altair_chart(
            chart,
            width="stretch",
        )


st.divider()


# ============================================================
# TOP PREDICTIVE DRIVERS
# ============================================================

st.subheader("📊 Top Predictive Drivers")

st.caption(
    "The features identified as most influential by the trained model."
)

if feature_importance and not metrics.empty:

    champion_key = str(
        metadata.get(
            "champion_key",
            metrics.iloc[0]["model_key"],
        )
    )

    top_features = pd.DataFrame(
        feature_importance.get(
            champion_key,
            [],
        )
    )

    if not top_features.empty:

        top_features = top_features.head(10).copy()

        chart = (
            alt.Chart(top_features)
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
                    title=None,
                    sort="-x",
                ),
                tooltip=[
                    alt.Tooltip(
                        "feature:N",
                        title="Feature",
                    ),
                    alt.Tooltip(
                        "importance:Q",
                        title="Importance",
                        format=".3f",
                    ),
                ],
            )
            .properties(height=350)
        )

        st.altair_chart(
            chart,
            width="stretch",
        )

    else:

        st.info(
            "Feature importance is not available yet."
        )

else:

    st.info(
        "Train the models to display predictive drivers."
    )


st.divider()


# ============================================================
# PROJECT OVERVIEW
# ============================================================

st.subheader("📋 Project Overview")

st.write(
    """
    Bird Migration Analytics uses machine learning to study the
    conditions associated with successful and unsuccessful bird
    migration journeys.

    The project brings together biological, geographical,
    environmental, flight, behavioural, and temporal information
    to build predictive models and identify the factors that
    provide the strongest predictive signals.

    The dashboard provides an interactive environment for
    exploring migration patterns, comparing machine learning
    models, making individual predictions, and understanding
    the key factors behind model predictions.
    """
)

st.markdown(
    """
    **The project focuses on three main goals:**

    - 🔎 **Understand migration patterns** across species,
      regions, seasons, environments, and habitats.
    - 🤖 **Predict migration outcomes** using multiple
      classification algorithms.
    - 💡 **Explain predictive drivers** and identify the
      environmental, flight, and behavioural factors most
      strongly associated with migration success.
    """
)


# ============================================================
# PROJECT NOTE
# ============================================================

st.info(
    "The model identifies predictive patterns within the dataset. "
    "These findings represent associations and should not be "
    "interpreted as proof of biological causation."
)
