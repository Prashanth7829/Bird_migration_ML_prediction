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

st.title("🦅 Bird Migration Success Prediction")

st.write(
    "Predict, compare, and understand migration outcomes "
    "using environmental, geographical, flight and behavioural data."
)

st.divider()


# ============================================================
# KEY METRICS
# ============================================================

total_records = len(data)

successful = int(
    data["Migration_Success_Num"].sum()
)

failed = total_records - successful

success_rate = (
    successful / total_records
    if total_records > 0
    else 0
)

species_count = data["Species"].nunique()
region_count = data["Region"].nunique()


col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric(
        "Migration Records",
        f"{total_records:,}",
    )

with col2:
    st.metric(
        "Successful",
        f"{successful:,}",
    )

with col3:
    st.metric(
        "Failed",
        f"{failed:,}",
    )

with col4:
    st.metric(
        "Success Rate",
        f"{success_rate:.1%}",
    )

with col5:
    st.metric(
        "Species / Regions",
        f"{species_count} / {region_count}",
    )


st.divider()


# ============================================================
# MODEL STATUS
# ============================================================

st.subheader("🏆 Model Status")

if metrics.empty:

    st.warning(
        "Models have not been trained yet."
    )

else:

    champion = metrics.iloc[0]

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Best Model",
            champion["model"],
        )

    with col2:
        st.metric(
            "F1 Score",
            f"{champion['f1_score']:.3f}",
        )

    with col3:
        st.metric(
            "ROC-AUC",
            f"{champion['roc_auc']:.3f}",
        )

    if metadata:

        st.caption(
            f"Training records: "
            f"{metadata.get('train_rows', 0):,} | "
            f"Test records: "
            f"{metadata.get('test_rows', 0):,}"
        )


st.divider()


# ============================================================
# WEATHER SUCCESS RATE
# ============================================================

col1, col2 = st.columns(2)


with col1:

    st.subheader(
        "🌦️ Success Rate by Weather Condition"
    )

    weather = (
        data.groupby(
            "Weather_Condition",
            as_index=False,
        )["Migration_Success_Num"]
        .mean()
    )

    weather["Success Rate"] = (
        weather["Migration_Success_Num"]
    )

    chart = (
        alt.Chart(weather)
        .mark_bar()
        .encode(
            x=alt.X(
                "Success Rate:Q",
                axis=alt.Axis(
                    format="%",
                ),
            ),
            y=alt.Y(
                "Weather_Condition:N",
                sort="-x",
                title=None,
            ),
            tooltip=[
                "Weather_Condition",
                alt.Tooltip(
                    "Success Rate:Q",
                    format=".1%",
                ),
            ],
        )
        .properties(
            height=300,
        )
    )

    st.altair_chart(
        chart,
        width="stretch",
    )


# ============================================================
# SPECIES SUCCESS RATE
# ============================================================

with col2:

    st.subheader(
        "🐦 Success Rate by Species"
    )

    species = (
        data.groupby(
            "Species",
            as_index=False,
        )["Migration_Success_Num"]
        .mean()
    )

    species["Success Rate"] = (
        species["Migration_Success_Num"]
    )

    chart = (
        alt.Chart(species)
        .mark_bar()
        .encode(
            x=alt.X(
                "Success Rate:Q",
                axis=alt.Axis(
                    format="%",
                ),
            ),
            y=alt.Y(
                "Species:N",
                sort="-x",
                title=None,
            ),
            tooltip=[
                "Species",
                alt.Tooltip(
                    "Success Rate:Q",
                    format=".1%",
                ),
            ],
        )
        .properties(
            height=300,
        )
    )

    st.altair_chart(
        chart,
        width="stretch",
    )


st.divider()


# ============================================================
# TOP PREDICTIVE FEATURES
# ============================================================

st.subheader("📊 Top Predictive Drivers")

if feature_importance and metrics.empty is False:

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

        top_features = top_features.head(10)

        chart = (
            alt.Chart(top_features)
            .mark_bar()
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
                height=350,
            )
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
        "Train the models to display feature importance."
    )


st.divider()


# ============================================================
# PROJECT OVERVIEW
# ============================================================

st.subheader("📋 Project Overview")

st.write(
    """
    This project uses machine learning to predict whether a
    bird migration will be successful or unsuccessful.

    The analysis considers environmental, geographical,
    flight, behavioural, biological and temporal factors.

    Six classification algorithms are evaluated:

    • Logistic Regression  
    • Decision Tree  
    • K-Nearest Neighbors  
    • Support Vector Machine  
    • Gradient Boosting  
    • Random Forest

    The best-performing model is selected using evaluation
    metrics such as F1-score and ROC-AUC rather than assuming
    Random Forest will always be the winner.
    """
)


# ============================================================
# IMPORTANT NOTE
# ============================================================

st.info(
    "Model predictions represent predictive associations "
    "in the dataset and should not be interpreted as proof "
    "of biological causation."
)