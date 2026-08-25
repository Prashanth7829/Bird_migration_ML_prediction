"""
Shared modelling configuration and prediction helpers.

Bird Migration Analytics
Predicting Migration Success

This module defines:
    - Approved ML features
    - Target variable
    - Leakage / post-outcome exclusions
    - Preprocessing pipeline
    - Prediction input validation
    - Prediction template generation
    - Model artifact paths
"""

from __future__ import annotations

from pathlib import Path
import re

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "Bird_Migration_Analysis_Improved_Synthetic.csv"
)

MODELS_DIR = PROJECT_ROOT / "models"


# ============================================================
# TARGET VARIABLE
# ============================================================

TARGET_COLUMN = "Migration_Success_Num"


# ============================================================
# APPROVED ML FEATURES
# ============================================================
#
# These are selected from the research-question feature groups.
#
# We intentionally DO NOT use every column in the dataset.
#
# The primary model should use variables that are available
# before or during migration rather than variables that are
# consequences of the final migration outcome.
# ============================================================


# ------------------------------------------------------------
# CATEGORICAL FEATURES
# ------------------------------------------------------------

CATEGORICAL_FEATURES = [
    # Biological
    "Species",

    # Geographical
    "Region",
    "Habitat",
    "Origin",

    # Migration motivation
    "Migration_Reason",

    # Environmental condition
    "Weather_Condition",

    # Temporal
    "Migration_Start_Month",
    "Migration_End_Month",

    # Behavioural
    "Migrated_in_Flock",
    "Food_Supply_Level",
]


# ------------------------------------------------------------
# NUMERICAL FEATURES
# ------------------------------------------------------------

NUMERIC_FEATURES = [
    # Geographical
    "Start_Latitude",
    "Start_Longitude",
    "End_Latitude",
    "End_Longitude",

    # Environmental
    "Temperature_C",
    "Wind_Speed_kmph",
    "Humidity_%",
    "Pressure_hPa",
    "Visibility_km",

    # Flight characteristics
    "Flight_Distance_km",
    "Flight_Duration_hours",
    "Average_Speed_kmph",
    "Max_Altitude_m",
    "Min_Altitude_m",

    # Behavioural
    "Rest_Stops",
    "Predator_Sightings",
    "Flock_Size",
]


# ============================================================
# FINAL FEATURE LIST
# ============================================================

FEATURE_COLUMNS = (
    CATEGORICAL_FEATURES
    + NUMERIC_FEATURES
)


# ============================================================
# FEATURE GROUPS
# ============================================================
#
# Used by dashboard / analysis pages to explain how features
# relate to the research questions.
# ============================================================

FEATURE_GROUPS = {
    "Biological": [
        "Species",
    ],

    "Geographical": [
        "Region",
        "Habitat",
        "Origin",
        "Start_Latitude",
        "Start_Longitude",
        "End_Latitude",
        "End_Longitude",
    ],

    "Migration Motivation": [
        "Migration_Reason",
    ],

    "Environmental": [
        "Weather_Condition",
        "Temperature_C",
        "Wind_Speed_kmph",
        "Humidity_%",
        "Pressure_hPa",
        "Visibility_km",
    ],

    "Flight Characteristics": [
        "Flight_Distance_km",
        "Flight_Duration_hours",
        "Average_Speed_kmph",
        "Max_Altitude_m",
        "Min_Altitude_m",
    ],

    "Behavioural": [
        "Rest_Stops",
        "Predator_Sightings",
        "Migrated_in_Flock",
        "Flock_Size",
        "Food_Supply_Level",
    ],

    "Temporal": [
        "Migration_Start_Month",
        "Migration_End_Month",
    ],
}


# ============================================================
# EXCLUDED / LEAKAGE / ANALYSIS-ONLY COLUMNS
# ============================================================
#
# These columns are deliberately NOT used as ML predictors.
#
# Reasons:
#   1. Target variables
#   2. Post-outcome information
#   3. Possible leakage
#   4. Derived / redundant representations
#   5. Tracking / data-quality variables
#   6. Unnecessary clustering / analysis columns
# ============================================================

LEAKAGE_OR_ANALYSIS_ONLY_COLUMNS = [
    # --------------------------------------------------------
    # TARGET
    # --------------------------------------------------------

    "Migration_Success",
    "Migration_Success_Num",

    # --------------------------------------------------------
    # POST-OUTCOME / POSSIBLE DATA LEAKAGE
    # --------------------------------------------------------

    "Migration_Interrupted",
    "Interrupted_Reason",
    "Recovery_Location_Known",
    "Recovery_Time_days",
    "Nesting_Success",

    # --------------------------------------------------------
    # TRACKING / DATA-QUALITY VARIABLES
    # --------------------------------------------------------
    #
    # These may describe the quality of tracking rather than
    # ecological causes of migration success.
    #

    "Tag_Battery_Level_%",
    "Tracking_Quality",

    # --------------------------------------------------------
    # DERIVED / REDUNDANT REPRESENTATIONS
    # --------------------------------------------------------

    "Food_Supply_Num",
    "Flock_Size_Band",
    "Flock_Size_Num",
    "Flight_Distance_Category",
    "Speed_Category",
    "Altitude_Category",
    "Temperature_Category",
    "Wind_Speed_Category",
    "Visibility_Category",
    "Humidity_Category",

    # --------------------------------------------------------
    # AUTOMATICALLY GENERATED / ANALYSIS-ONLY VARIABLES
    # --------------------------------------------------------

    "Cluster",
    "Migration_Behavior_Cluster",

    # --------------------------------------------------------
    # OTHER ANALYSIS / PATH REPRESENTATIONS
    # --------------------------------------------------------

    "Path",
]


# ============================================================
# EXCLUSION REASONS
# ============================================================
#
# Useful for the About page / documentation.
# ============================================================

FEATURE_EXCLUSION_REASONS = {
    "Migration_Success": (
        "Target representation. Must never be used as a predictor."
    ),

    "Migration_Success_Num": (
        "Primary target variable. Used as y, not as an input feature."
    ),

    "Migration_Interrupted": (
        "Potential post-outcome variable. May directly describe "
        "the migration outcome."
    ),

    "Interrupted_Reason": (
        "Post-outcome information describing why a migration "
        "was interrupted."
    ),

    "Recovery_Location_Known": (
        "Potentially known only after the migration outcome."
    ),

    "Recovery_Time_days": (
        "Post-outcome information and therefore unsuitable "
        "for primary prediction."
    ),

    "Nesting_Success": (
        "Outcome occurring after migration and therefore "
        "potentially leaks future information."
    ),

    "Tag_Battery_Level_%": (
        "Tracking/data-quality variable rather than a primary "
        "ecological predictor."
    ),

    "Tracking_Quality": (
        "Tracking/data-quality information rather than a "
        "direct migration factor."
    ),

    "Food_Supply_Num": (
        "Numerical duplicate/derived representation of "
        "Food_Supply_Level."
    ),

    "Flock_Size_Band": (
        "Manually binned representation of Flock_Size. "
        "Continuous Flock_Size is preferred."
    ),

    "Flock_Size_Num": (
        "Potential duplicate numerical representation of "
        "Flock_Size."
    ),

    "Flight_Distance_Category": (
        "Binned representation of Flight_Distance_km. "
        "Continuous distance is preferred."
    ),

    "Speed_Category": (
        "Binned representation of Average_Speed_kmph."
    ),

    "Altitude_Category": (
        "Binned representation of altitude variables."
    ),

    "Temperature_Category": (
        "Binned representation of Temperature_C."
    ),

    "Wind_Speed_Category": (
        "Binned representation of Wind_Speed_kmph."
    ),

    "Visibility_Category": (
        "Binned representation of Visibility_km."
    ),

    "Humidity_Category": (
        "Binned representation of Humidity_%."
    ),

    "Cluster": (
        "Automatically generated analytical cluster. "
        "Not a primary ecological predictor."
    ),

    "Migration_Behavior_Cluster": (
        "Automatically generated behavioural cluster. "
        "Excluded to avoid circular/derived information."
    ),

    "Path": (
        "Path representation is excluded because the underlying "
        "geographical variables are already represented explicitly."
    ),
}


# ============================================================
# MODEL DISPLAY LABELS
# ============================================================

MODEL_LABELS = {
    "logistic_regression": "Logistic Regression",
    "decision_tree": "Decision Tree",
    "knn": "KNN",
    "svm": "SVM",
    "gradient_boosting": "Gradient Boosting",
    "random_forest": "Random Forest",
}


# ============================================================
# MODEL KEY HELPERS
# ============================================================

def slugify_model(name: str) -> str:
    """
    Convert a model display name into the persistent model key.

    Example:
        "Random Forest" -> "random_forest"
    """

    return re.sub(
        r"[^a-z0-9]+",
        "_",
        name.lower(),
    ).strip("_")


def model_path(model_key: str) -> Path:
    """
    Return the path of a saved model pipeline.
    """

    return MODELS_DIR / f"{model_key}.joblib"


# ============================================================
# PREPROCESSOR
# ============================================================

def build_preprocessor() -> ColumnTransformer:
    """
    Build the common preprocessing pipeline.

    Numerical:
        - Median imputation
        - Standard scaling

    Categorical:
        - Most-frequent imputation
        - One-hot encoding

    The preprocessing is fitted only on the training data because
    it is part of the sklearn Pipeline.
    """

    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                ),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent"
                ),
            ),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_pipeline,
                NUMERIC_FEATURES,
            ),
            (
                "categorical",
                categorical_pipeline,
                CATEGORICAL_FEATURES,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )


# ============================================================
# INPUT VALIDATION
# ============================================================

def validate_feature_columns(
    frame: pd.DataFrame,
) -> None:
    """
    Validate that all approved model features exist.

    Extra columns are allowed because prediction CSV files may
    contain target or analysis columns. Only FEATURE_COLUMNS
    are passed into the model.
    """

    missing = [
        column
        for column in FEATURE_COLUMNS
        if column not in frame.columns
    ]

    if missing:
        raise ValueError(
            "The uploaded CSV is missing required model "
            "features:\n\n"
            + "\n".join(
                f"- {column}"
                for column in missing
            )
        )


# ============================================================
# PREPARE PREDICTION DATA
# ============================================================

def prepare_prediction_frame(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    """
    Validate and standardise input data before prediction.

    Only approved FEATURE_COLUMNS are retained.

    Target columns, leakage columns, derived columns and
    unrelated columns are ignored if present.
    """

    if not isinstance(frame, pd.DataFrame):
        raise TypeError(
            "Prediction input must be a pandas DataFrame."
        )

    validate_feature_columns(frame)

    # Keep ONLY approved ML features.
    prepared = frame.loc[
        :,
        FEATURE_COLUMNS,
    ].copy()

    # --------------------------------------------------------
    # Numerical conversion
    # --------------------------------------------------------

    for column in NUMERIC_FEATURES:

        prepared[column] = pd.to_numeric(
            prepared[column],
            errors="coerce",
        )

    # --------------------------------------------------------
    # Categorical conversion
    # --------------------------------------------------------

    for column in CATEGORICAL_FEATURES:

        prepared[column] = (
            prepared[column]
            .astype("string")
        )

    return prepared


# ============================================================
# PREDICTION TEMPLATE
# ============================================================

def build_prediction_template(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create a valid one-row prediction template.

    Categorical variables:
        Most frequent value

    Numerical variables:
        Median value

    This is used by the Streamlit prediction interface and
    downloadable CSV template.
    """

    validate_feature_columns(data)

    values: dict[str, object] = {}

    # --------------------------------------------------------
    # Categorical defaults
    # --------------------------------------------------------

    for column in CATEGORICAL_FEATURES:

        series = (
            data[column]
            .dropna()
            .astype(str)
        )

        mode = series.mode()

        if not mode.empty:
            values[column] = mode.iloc[0]
        else:
            values[column] = "Unknown"

    # --------------------------------------------------------
    # Numerical defaults
    # --------------------------------------------------------

    for column in NUMERIC_FEATURES:

        numeric = pd.to_numeric(
            data[column],
            errors="coerce",
        )

        median = numeric.median()

        if pd.isna(median):
            values[column] = 0.0
        else:
            values[column] = round(
                float(median),
                2,
            )

    return pd.DataFrame(
        [values],
        columns=FEATURE_COLUMNS,
    )


# ============================================================
# DISPLAY HELPERS
# ============================================================

def clean_feature_name(
    name: str,
) -> str:
    """
    Convert a technical feature name into a dashboard-friendly
    label.
    """

    name = name.replace(
        "_",
        " ",
    )

    name = name.replace(
        "kmph",
        "km/h",
    )

    return name


def format_column(
    column: str,
) -> str:
    """
    Convert dataset column names into readable labels.
    """

    return clean_feature_name(column)


# ============================================================
# FEATURE ACCESS HELPERS
# ============================================================

def predictor_columns() -> list[str]:
    """
    Return the final approved predictor list.
    """

    return FEATURE_COLUMNS.copy()


def categorical_columns() -> list[str]:
    """
    Return approved categorical predictors.
    """

    return CATEGORICAL_FEATURES.copy()


def numeric_columns() -> list[str]:
    """
    Return approved numerical predictors.
    """

    return NUMERIC_FEATURES.copy()


def feature_group_columns(
    group: str,
) -> list[str]:
    """
    Return features belonging to a particular research group.
    """

    if group not in FEATURE_GROUPS:
        raise KeyError(
            f"Unknown feature group: {group}"
        )

    return FEATURE_GROUPS[group].copy()


# ============================================================
# DATASET FEATURE AUDIT
# ============================================================

def audit_dataset_columns(
    data: pd.DataFrame,
) -> dict[str, list[str]]:
    """
    Categorise dataset columns into:

        approved_features
        target
        excluded
        unused

    This is useful for documenting the feature-selection
    decision in the project.
    """

    columns = set(data.columns)

    approved = [
        column
        for column in FEATURE_COLUMNS
        if column in columns
    ]

    target = [
        column
        for column in [
            "Migration_Success",
            "Migration_Success_Num",
        ]
        if column in columns
    ]

    excluded = [
        column
        for column in LEAKAGE_OR_ANALYSIS_ONLY_COLUMNS
        if column in columns
    ]

    known = set(
        approved
        + target
        + excluded
    )

    unused = [
        column
        for column in data.columns
        if column not in known
    ]

    return {
        "approved_features": approved,
        "target": target,
        "excluded": excluded,
        "unused": unused,
    }


# ============================================================
# FEATURE SUMMARY
# ============================================================

def feature_summary() -> pd.DataFrame:
    """
    Return a structured summary of the final ML features.

    Useful for the About page or documentation.
    """

    rows: list[dict[str, str]] = []

    for group, features in FEATURE_GROUPS.items():

        for feature in features:

            feature_type = (
                "Categorical"
                if feature in CATEGORICAL_FEATURES
                else "Numerical"
            )

            rows.append(
                {
                    "Feature": feature,
                    "Group": group,
                    "Type": feature_type,
                    "Role": "ML Predictor",
                }
            )

    return pd.DataFrame(rows)