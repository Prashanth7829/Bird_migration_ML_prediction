"""Interactive exploratory data analysis for bird migration success."""

from __future__ import annotations

import numpy as np
import pandas as pd
import altair as alt
import streamlit as st

from src.dashboard import load_data


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.title("Data Insights")
st.caption(
    "Interactive exploration of the factors associated with bird migration success."
)

data = load_data().copy()


# ============================================================
# HELPER FUNCTIONS
# ============================================================

SUCCESS_COL = "Migration_Success_Num"
LABEL_COL = "Success label"


def available(column: str) -> bool:
    """Check whether a column exists in the dataset."""
    return column in data.columns


def clean_numeric(column: str) -> pd.Series:
    """Return a numeric version of a column."""
    return pd.to_numeric(data[column], errors="coerce")


def success_rate(frame: pd.DataFrame) -> float:
    """Calculate migration success rate."""
    if frame.empty:
        return np.nan

    values = pd.to_numeric(
        frame[SUCCESS_COL],
        errors="coerce",
    ).dropna()

    if values.empty:
        return np.nan

    return float(values.mean())


def format_percent(value: float) -> str:
    """Format a proportion as a percentage."""
    if pd.isna(value):
        return "N/A"

    return f"{value:.1%}"


def numeric_interpretation(
    frame: pd.DataFrame,
    column: str,
) -> str:
    """
    Generate an interpretation comparing the numeric feature
    between successful and failed migrations.
    """

    temp = frame[[column, SUCCESS_COL]].copy()
    temp[column] = pd.to_numeric(
        temp[column],
        errors="coerce",
    )

    temp = temp.dropna()

    if temp.empty:
        return "There is not enough valid data to interpret this feature."

    failed = temp.loc[
        temp[SUCCESS_COL] == 0,
        column,
    ]

    successful = temp.loc[
        temp[SUCCESS_COL] == 1,
        column,
    ]

    if failed.empty or successful.empty:
        return "Both migration outcome groups are not sufficiently represented."

    failed_median = failed.median()
    successful_median = successful.median()

    difference = successful_median - failed_median

    if difference > 0:
        direction = "higher"
    elif difference < 0:
        direction = "lower"
    else:
        direction = "similar"

    return (
        f"The median {column.replace('_', ' ')} was "
        f"{abs(difference):.2f} units {direction} for successful "
        f"migrations compared with failed migrations "
        f"({successful_median:.2f} vs {failed_median:.2f}). "
        f"This indicates an observed association between {column.replace('_', ' ')} "
        f"and migration outcome in this dataset. "
        f"It should not be interpreted as proof of causation."
    )


def binned_success_data(
    frame: pd.DataFrame,
    column: str,
    bins: int = 6,
) -> pd.DataFrame:
    """Create quantile-based bins and calculate success rate."""

    temp = frame[[column, SUCCESS_COL]].copy()

    temp[column] = pd.to_numeric(
        temp[column],
        errors="coerce",
    )

    temp = temp.dropna()

    if temp.empty or temp[column].nunique() < 2:
        return pd.DataFrame()

    try:
        temp["Range"] = pd.qcut(
            temp[column],
            q=bins,
            duplicates="drop",
        )
    except ValueError:
        return pd.DataFrame()

    result = (
        temp.groupby(
            "Range",
            observed=True,
        )[SUCCESS_COL]
        .agg(
            success_rate="mean",
            records="count",
        )
        .reset_index()
    )

    result["Range"] = result["Range"].astype(str)

    result["Success rate"] = result["success_rate"]

    return result


def binned_interpretation(
    frame: pd.DataFrame,
    column: str,
) -> str:
    """Explain the highest and lowest success-rate ranges."""

    result = binned_success_data(
        frame,
        column,
    )

    if result.empty:
        return "There is not enough variation to create meaningful ranges."

    highest = result.loc[
        result["success_rate"].idxmax()
    ]

    lowest = result.loc[
        result["success_rate"].idxmin()
    ]

    return (
        f"The highest observed success rate occurs in the range "
        f"**{highest['Range']}** at **{highest['success_rate']:.1%}**, "
        f"while the lowest occurs in **{lowest['Range']}** at "
        f"**{lowest['success_rate']:.1%}**. "
        f"This suggests that migration outcomes vary across different "
        f"{column.replace('_', ' ')} ranges."
    )


def categorical_success_data(
    frame: pd.DataFrame,
    column: str,
) -> pd.DataFrame:
    """Calculate success rate for categorical values."""

    temp = frame[[column, SUCCESS_COL]].copy()

    temp[column] = temp[column].astype(str)

    result = (
        temp.groupby(column)[SUCCESS_COL]
        .agg(
            success_rate="mean",
            records="count",
        )
        .reset_index()
        .sort_values(
            "success_rate",
            ascending=False,
        )
    )

    result["Success rate"] = result["success_rate"]

    return result


def categorical_interpretation(
    frame: pd.DataFrame,
    column: str,
) -> str:
    """Generate interpretation for categorical variables."""

    result = categorical_success_data(
        frame,
        column,
    )

    if result.empty:
        return "No valid data is available."

    highest = result.iloc[0]
    lowest = result.iloc[-1]

    return (
        f"**{highest[column]}** has the highest observed success rate "
        f"at **{highest['success_rate']:.1%}**, while "
        f"**{lowest[column]}** has the lowest at "
        f"**{lowest['success_rate']:.1%}**. "
        f"These differences represent patterns observed in the dataset "
        f"and should not be interpreted as causal effects."
    )


# ============================================================
# SIDEBAR FILTERS
# ============================================================

st.sidebar.header("Insight Filters")

filtered_data = data.copy()

if available("Species"):
    species_options = sorted(
        data["Species"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    selected_species = st.sidebar.multiselect(
        "Species",
        species_options,
        default=species_options,
    )

    if selected_species:
        filtered_data = filtered_data[
            filtered_data["Species"]
            .astype(str)
            .isin(selected_species)
        ]


if available("Region"):
    region_options = sorted(
        data["Region"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    selected_regions = st.sidebar.multiselect(
        "Region",
        region_options,
        default=region_options,
    )

    if selected_regions:
        filtered_data = filtered_data[
            filtered_data["Region"]
            .astype(str)
            .isin(selected_regions)
        ]


if available("Habitat"):
    habitat_options = sorted(
        data["Habitat"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    selected_habitats = st.sidebar.multiselect(
        "Habitat",
        habitat_options,
        default=habitat_options,
    )

    if selected_habitats:
        filtered_data = filtered_data[
            filtered_data["Habitat"]
            .astype(str)
            .isin(selected_habitats)
        ]


st.sidebar.divider()

st.sidebar.metric(
    "Filtered records",
    f"{len(filtered_data):,}",
)

st.sidebar.metric(
    "Filtered success rate",
    format_percent(
        success_rate(filtered_data)
    ),
)


# ============================================================
# OVERVIEW
# ============================================================

st.header("1. Migration Overview")

st.write(
    "Use the filters on the left to explore how migration outcomes "
    "change across species, regions, and habitats."
)

overview_cols = st.columns(4)

with overview_cols[0]:
    st.metric(
        "Records",
        f"{len(filtered_data):,}",
    )

with overview_cols[1]:
    successful = int(
        filtered_data[SUCCESS_COL].sum()
    )

    st.metric(
        "Successful",
        f"{successful:,}",
    )

with overview_cols[2]:
    failed = len(filtered_data) - successful

    st.metric(
        "Failed",
        f"{failed:,}",
    )

with overview_cols[3]:
    st.metric(
        "Success rate",
        format_percent(
            success_rate(filtered_data)
        ),
    )


# ============================================================
# ENVIRONMENTAL FACTORS
# ============================================================

st.header("2. Environmental Factors")

st.caption(
    "Research Question: How do environmental factors influence migration success?"
)


environmental_features = [
    (
        "Temperature_C",
        "Temperature",
        "°C",
    ),
    (
        "Wind_Speed_kmph",
        "Wind Speed",
        "km/h",
    ),
    (
        "Humidity_%",
        "Humidity",
        "%",
    ),
    (
        "Visibility_km",
        "Visibility",
        "km",
    ),
    (
        "Pressure_hPa",
        "Pressure",
        "hPa",
    ),
    (
        "Max_Altitude_m",
        "Maximum Altitude",
        "m",
    ),
]


available_environmental = [
    item
    for item in environmental_features
    if available(item[0])
]


environmental_tabs = st.tabs(
    [
        item[1]
        for item in available_environmental
    ]
)


for tab, feature_info in zip(
    environmental_tabs,
    available_environmental,
):

    column, label, unit = feature_info

    with tab:

        st.subheader(
            f"{label} and migration outcome"
        )

        chart_data = filtered_data[
            [
                column,
                SUCCESS_COL,
                LABEL_COL,
            ]
        ].copy()

        chart_data[column] = pd.to_numeric(
            chart_data[column],
            errors="coerce",
        )

        chart_data = chart_data.dropna()

        if chart_data.empty:
            st.warning(
                "No valid data is available for this filter."
            )
            continue

        # ----------------------------------------------------
        # Distribution chart
        # ----------------------------------------------------

        st.markdown(
            f"**Distribution of {label.lower()} by migration outcome**"
        )

        distribution = (
            alt.Chart(chart_data)
            .transform_calculate(
                Outcome="datum['"
                + LABEL_COL
                + "']"
            )
            .mark_circle(
                opacity=0.35,
                size=45,
            )
            .encode(
                x=alt.X(
                    f"{column}:Q",
                    title=f"{label} ({unit})",
                ),
                y=alt.Y(
                    "Outcome:N",
                    title="Migration outcome",
                ),
                color=alt.Color(
                    "Outcome:N",
                    scale=alt.Scale(
                        domain=[
                            "Failed",
                            "Successful",
                        ],
                        range=[
                            "#C2410C",
                            "#15803D",
                        ],
                    ),
                ),
                tooltip=[
                    alt.Tooltip(
                        f"{column}:Q",
                        title=label,
                        format=".2f",
                    ),
                    alt.Tooltip(
                        "Outcome:N",
                        title="Outcome",
                    ),
                ],
            )
            .properties(
                height=220,
            )
            .interactive()
        )

        st.altair_chart(
            distribution,
            width="stretch",
        )

        st.info(
            numeric_interpretation(
                filtered_data,
                column,
            ),
            icon="💡",
        )

        # ----------------------------------------------------
        # Binned success rate
        # ----------------------------------------------------

        st.markdown(
            f"**Success rate across {label.lower()} ranges**"
        )

        binned = binned_success_data(
            filtered_data,
            column,
        )

        if not binned.empty:

            bar_chart = (
                alt.Chart(binned)
                .mark_bar(
                    cornerRadiusTopLeft=5,
                    cornerRadiusTopRight=5,
                )
                .encode(
                    x=alt.X(
                        "Range:N",
                        title=f"{label} range",
                        sort=None,
                        axis=alt.Axis(
                            labelAngle=-35
                        ),
                    ),
                    y=alt.Y(
                        "Success rate:Q",
                        title="Success rate",
                        axis=alt.Axis(
                            format="%"
                        ),
                        scale=alt.Scale(
                            domain=[
                                0,
                                1,
                            ]
                        ),
                    ),
                    tooltip=[
                        alt.Tooltip(
                            "Range:N",
                            title="Range",
                        ),
                        alt.Tooltip(
                            "Success rate:Q",
                            title="Success rate",
                            format=".1%",
                        ),
                        alt.Tooltip(
                            "records:Q",
                            title="Records",
                        ),
                    ],
                )
                .properties(
                    height=300,
                )
                .interactive()
            )

            st.altair_chart(
                bar_chart,
                width="stretch",
            )

            st.success(
                binned_interpretation(
                    filtered_data,
                    column,
                ),
                icon="📊",
            )


# ============================================================
# WEATHER CONDITIONS
# ============================================================

if available("Weather_Condition"):

    st.header("3. Weather Conditions")

    st.caption(
        "Compare migration outcomes across observed weather conditions."
    )

    weather = categorical_success_data(
        filtered_data,
        "Weather_Condition",
    )

    if not weather.empty:

        weather_chart = (
            alt.Chart(weather)
            .mark_bar(
                cornerRadiusTopLeft=5,
                cornerRadiusTopRight=5,
            )
            .encode(
                x=alt.X(
                    "Weather_Condition:N",
                    title="Weather condition",
                    sort="-y",
                ),
                y=alt.Y(
                    "Success rate:Q",
                    title="Success rate",
                    axis=alt.Axis(
                        format="%"
                    ),
                    scale=alt.Scale(
                        domain=[
                            0,
                            1,
                        ]
                    ),
                ),
                color=alt.Color(
                    "Success rate:Q",
                    scale=alt.Scale(
                        scheme="greens"
                    ),
                    legend=None,
                ),
                tooltip=[
                    "Weather_Condition",
                    alt.Tooltip(
                        "Success rate:Q",
                        format=".1%",
                    ),
                    alt.Tooltip(
                        "records:Q",
                        title="Records",
                    ),
                ],
            )
            .properties(
                height=350,
            )
            .interactive()
        )

        st.altair_chart(
            weather_chart,
            width="stretch",
        )

        st.info(
            categorical_interpretation(
                filtered_data,
                "Weather_Condition",
            ),
            icon="🌦️",
        )


# ============================================================
# SPECIES ANALYSIS
# ============================================================

if available("Species"):

    st.header("4. Species Success Overview")

    st.caption(
        "Research Question: What are the key migration trends across species?"
    )

    species = categorical_success_data(
        filtered_data,
        "Species",
    )

    species_chart = (
        alt.Chart(species)
        .mark_bar(
            cornerRadiusTopLeft=5,
            cornerRadiusTopRight=5,
        )
        .encode(
            x=alt.X(
                "Species:N",
                sort="-y",
                title="Species",
            ),
            y=alt.Y(
                "Success rate:Q",
                title="Success rate",
                axis=alt.Axis(
                    format="%"
                ),
                scale=alt.Scale(
                    domain=[
                        0,
                        1,
                    ]
                ),
            ),
            tooltip=[
                "Species",
                alt.Tooltip(
                    "Success rate:Q",
                    format=".1%",
                ),
                alt.Tooltip(
                    "records:Q",
                    title="Records",
                ),
            ],
        )
        .properties(
            height=350,
        )
        .interactive()
    )

    st.altair_chart(
        species_chart,
        width="stretch",
    )

    if not species.empty:

        highest = species.iloc[0]
        lowest = species.iloc[-1]

        st.info(
            f"**{highest['Species']}** has the highest observed "
            f"success rate at **{highest['success_rate']:.1%}**, "
            f"while **{lowest['Species']}** has the lowest at "
            f"**{lowest['success_rate']:.1%}**. "
            f"Species-level differences may reflect a combination "
            f"of environmental, geographical and behavioural factors.",
            icon="🐦",
        )


# ============================================================
# REGION ANALYSIS
# ============================================================

if available("Region"):

    st.header("5. Regional Migration Patterns")

    region = categorical_success_data(
        filtered_data,
        "Region",
    )

    region_chart = (
        alt.Chart(region)
        .mark_bar(
            cornerRadiusTopLeft=5,
            cornerRadiusTopRight=5,
        )
        .encode(
            x=alt.X(
                "Region:N",
                sort="-y",
                title="Region",
            ),
            y=alt.Y(
                "Success rate:Q",
                title="Success rate",
                axis=alt.Axis(
                    format="%"
                ),
                scale=alt.Scale(
                    domain=[
                        0,
                        1,
                    ]
                ),
            ),
            tooltip=[
                "Region",
                alt.Tooltip(
                    "Success rate:Q",
                    format=".1%",
                ),
                alt.Tooltip(
                    "records:Q",
                    title="Records",
                ),
            ],
        )
        .properties(
            height=350,
        )
        .interactive()
    )

    st.altair_chart(
        region_chart,
        width="stretch",
    )

    st.info(
        categorical_interpretation(
            filtered_data,
            "Region",
        ),
        icon="🗺️",
    )


# ============================================================
# HABITAT ANALYSIS
# ============================================================

if available("Habitat"):

    st.header("6. Habitat Patterns")

    habitat = categorical_success_data(
        filtered_data,
        "Habitat",
    )

    habitat_chart = (
        alt.Chart(habitat)
        .mark_bar(
            cornerRadiusTopLeft=5,
            cornerRadiusTopRight=5,
        )
        .encode(
            x=alt.X(
                "Habitat:N",
                sort="-y",
                title="Habitat",
            ),
            y=alt.Y(
                "Success rate:Q",
                title="Success rate",
                axis=alt.Axis(
                    format="%"
                ),
                scale=alt.Scale(
                    domain=[
                        0,
                        1,
                    ]
                ),
            ),
            tooltip=[
                "Habitat",
                alt.Tooltip(
                    "Success rate:Q",
                    format=".1%",
                ),
                alt.Tooltip(
                    "records:Q",
                    title="Records",
                ),
            ],
        )
        .properties(
            height=350,
        )
        .interactive()
    )

    st.altair_chart(
        habitat_chart,
        width="stretch",
    )

    st.info(
        categorical_interpretation(
            filtered_data,
            "Habitat",
        ),
        icon="🌳",
    )


# ============================================================
# FLIGHT CHARACTERISTICS
# ============================================================

st.header("7. Flight Characteristics")

st.caption(
    "Research Question: How do flight characteristics relate to migration success?"
)


flight_features = [
    (
        "Flight_Distance_km",
        "Flight Distance",
        "km",
    ),
    (
        "Flight_Duration_hours",
        "Flight Duration",
        "hours",
    ),
    (
        "Average_Speed_kmph",
        "Average Speed",
        "km/h",
    ),
    (
        "Max_Altitude_m",
        "Maximum Altitude",
        "m",
    ),
    (
        "Min_Altitude_m",
        "Minimum Altitude",
        "m",
    ),
]


available_flight = [
    item
    for item in flight_features
    if available(item[0])
]


flight_tabs = st.tabs(
    [
        item[1]
        for item in available_flight
    ]
)


for tab, feature_info in zip(
    flight_tabs,
    available_flight,
):

    column, label, unit = feature_info

    with tab:

        flight_data = filtered_data[
            [
                column,
                SUCCESS_COL,
                LABEL_COL,
            ]
        ].copy()

        flight_data[column] = pd.to_numeric(
            flight_data[column],
            errors="coerce",
        )

        flight_data = flight_data.dropna()

        if flight_data.empty:
            st.warning(
                "No valid data available."
            )
            continue

        chart = (
            alt.Chart(flight_data)
            .mark_circle(
                opacity=0.35,
                size=45,
            )
            .encode(
                x=alt.X(
                    f"{column}:Q",
                    title=f"{label} ({unit})",
                ),
                y=alt.Y(
                    "Migration_Success_Num:Q",
                    title="Migration success",
                    scale=alt.Scale(
                        domain=[
                            0,
                            1,
                        ]
                    ),
                ),
                color=alt.Color(
                    "Success label:N",
                    scale=alt.Scale(
                        domain=[
                            "Failed",
                            "Successful",
                        ],
                        range=[
                            "#C2410C",
                            "#15803D",
                        ],
                    ),
                ),
                tooltip=[
                    alt.Tooltip(
                        f"{column}:Q",
                        title=label,
                        format=".2f",
                    ),
                    "Success label",
                ],
            )
            .properties(
                height=320,
            )
            .interactive()
        )

        st.altair_chart(
            chart,
            width="stretch",
        )

        st.info(
            numeric_interpretation(
                filtered_data,
                column,
            ),
            icon="✈️",
        )


# ============================================================
# BEHAVIOURAL FACTORS
# ============================================================

st.header("8. Behavioural Factors")

st.caption(
    "Research Question: How do behavioural factors influence migration outcomes?"
)


behaviour_features = [
    (
        "Flock_Size",
        "Flock Size",
        "birds",
    ),
    (
        "Rest_Stops",
        "Rest Stops",
        "stops",
    ),
    (
        "Predator_Sightings",
        "Predator Sightings",
        "sightings",
    ),
]


available_behaviour = [
    item
    for item in behaviour_features
    if available(item[0])
]


behaviour_tabs = st.tabs(
    [
        item[1]
        for item in available_behaviour
    ]
)


for tab, feature_info in zip(
    behaviour_tabs,
    available_behaviour,
):

    column, label, unit = feature_info

    with tab:

        behaviour_data = filtered_data[
            [
                column,
                SUCCESS_COL,
                LABEL_COL,
            ]
        ].copy()

        behaviour_data[column] = pd.to_numeric(
            behaviour_data[column],
            errors="coerce",
        )

        behaviour_data = behaviour_data.dropna()

        chart = (
            alt.Chart(behaviour_data)
            .mark_circle(
                opacity=0.35,
                size=45,
            )
            .encode(
                x=alt.X(
                    f"{column}:Q",
                    title=f"{label} ({unit})",
                ),
                y=alt.Y(
                    "Migration_Success_Num:Q",
                    title="Migration success",
                    scale=alt.Scale(
                        domain=[
                            0,
                            1,
                        ]
                    ),
                ),
                color=alt.Color(
                    "Success label:N",
                    scale=alt.Scale(
                        domain=[
                            "Failed",
                            "Successful",
                        ],
                        range=[
                            "#C2410C",
                            "#15803D",
                        ],
                    ),
                ),
                tooltip=[
                    alt.Tooltip(
                        f"{column}:Q",
                        title=label,
                    ),
                    "Success label",
                ],
            )
            .properties(
                height=320,
            )
            .interactive()
        )

        st.altair_chart(
            chart,
            width="stretch",
        )

        st.info(
            numeric_interpretation(
                filtered_data,
                column,
            ),
            icon="🦅",
        )


# ============================================================
# MIGRATION REASONS
# ============================================================

if available("Migration_Reason"):

    st.header("9. Migration Reasons")

    st.caption(
        "Research Question: What are the main reasons for migration "
        "and how do they vary across species and regions?"
    )

    reason = categorical_success_data(
        filtered_data,
        "Migration_Reason",
    )

    reason_chart = (
        alt.Chart(reason)
        .mark_bar(
            cornerRadiusTopLeft=5,
            cornerRadiusTopRight=5,
        )
        .encode(
            x=alt.X(
                "Migration_Reason:N",
                sort="-y",
                title="Migration reason",
            ),
            y=alt.Y(
                "Success rate:Q",
                title="Success rate",
                axis=alt.Axis(
                    format="%"
                ),
                scale=alt.Scale(
                    domain=[
                        0,
                        1,
                    ]
                ),
            ),
            tooltip=[
                "Migration_Reason",
                alt.Tooltip(
                    "Success rate:Q",
                    format=".1%",
                ),
                alt.Tooltip(
                    "records:Q",
                    title="Records",
                ),
            ],
        )
        .properties(
            height=350,
        )
        .interactive()
    )

    st.altair_chart(
        reason_chart,
        width="stretch",
    )

    st.info(
        categorical_interpretation(
            filtered_data,
            "Migration_Reason",
        ),
        icon="🧭",
    )


# ============================================================
# CORRELATION OVERVIEW
# ============================================================

st.header("10. Numerical Feature Relationships")

st.caption(
    "Explore how numerical variables move together. "
    "Correlation should be interpreted as association, not causation."
)


numeric_columns = [
    column
    for column in [
        "Temperature_C",
        "Wind_Speed_kmph",
        "Humidity_%",
        "Visibility_km",
        "Pressure_hPa",
        "Flight_Distance_km",
        "Flight_Duration_hours",
        "Average_Speed_kmph",
        "Max_Altitude_m",
        "Min_Altitude_m",
        "Rest_Stops",
        "Predator_Sightings",
        "Flock_Size",
        SUCCESS_COL,
    ]
    if available(column)
]


if len(numeric_columns) >= 2:

    correlation = (
        filtered_data[numeric_columns]
        .apply(
            pd.to_numeric,
            errors="coerce",
        )
        .corr()
    )

    correlation_display = (
        correlation
        .reset_index()
        .melt(
            id_vars="index",
            var_name="Feature",
            value_name="Correlation",
        )
        .rename(
            columns={
                "index": "Feature 1"
            }
        )
    )

    heatmap = (
        alt.Chart(correlation_display)
        .mark_rect()
        .encode(
            x=alt.X(
                "Feature 1:N",
                title=None,
            ),
            y=alt.Y(
                "Feature:N",
                title=None,
            ),
            color=alt.Color(
                "Correlation:Q",
                scale=alt.Scale(
                    domain=[
                        -1,
                        1,
                    ],
                    scheme="redblue",
                ),
            ),
            tooltip=[
                "Feature 1",
                "Feature",
                alt.Tooltip(
                    "Correlation:Q",
                    format=".2f",
                ),
            ],
        )
        .properties(
            height=500,
        )
        .interactive()
    )

    st.altair_chart(
        heatmap,
        width="stretch",
    )


# ============================================================
# FINAL INTERPRETATION
# ============================================================

st.header("11. Overall Interpretation")

overall_rate = success_rate(
    filtered_data
)

st.info(
    f"""
### What the exploratory analysis tells us

The current filtered dataset contains **{len(filtered_data):,} migration records**,
with an observed migration success rate of **{format_percent(overall_rate)}**.

The charts above highlight differences in migration outcomes across:

- Environmental conditions
- Weather
- Species
- Regions
- Habitats
- Flight characteristics
- Behavioural factors
- Migration reasons

These observations are useful for identifying **patterns and potential predictive
drivers** for the machine-learning models.

However, an observed difference in success rate does **not** establish causation.
The machine-learning stage evaluates whether these variables improve prediction
of migration success when considered together.
""",
    icon="🧠",
)