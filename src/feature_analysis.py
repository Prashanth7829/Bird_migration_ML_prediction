"""
Bird Migration Success Prediction
Feature Influence & Prediction Effectiveness Analysis

This script answers Research Question 6:

"Which features have the greatest influence on migration success,
and how effectively can machine learning models predict successful
and unsuccessful migrations?"

Analysis performed:
1. Dataset validation
2. Success-rate analysis
3. Numerical feature correlation
4. Lift analysis for categorical features
5. Lift analysis for numerical features using quantile bins
6. Gradient Boosting feature importance
7. Random Forest feature importance
8. Permutation importance using the original 27 features
9. Combined feature-importance ranking
10. Model performance comparison
11. Champion model confusion matrix
12. Prediction probability analysis
13. Saves CSV results and plots

The script uses the existing trained models and does NOT retrain them.
"""

from pathlib import Path
import json
import warnings

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)

warnings.filterwarnings("ignore")


# =============================================================================
# CONFIGURATION
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "Bird_Migration_Analysis_Improved_Synthetic.csv"
)

MODEL_DIR = PROJECT_ROOT / "models"

OUTPUT_DIR = MODEL_DIR / "feature_analysis"
PLOT_DIR = OUTPUT_DIR / "plots"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)


TARGET = "Migration_Success_Num"


# =============================================================================
# APPROVED FEATURE SET
# =============================================================================

APPROVED_FEATURES = [
    "Species",
    "Region",
    "Habitat",
    "Origin",
    "Migration_Reason",
    "Weather_Condition",
    "Migration_Start_Month",
    "Migration_End_Month",
    "Migrated_in_Flock",
    "Food_Supply_Level",
    "Start_Latitude",
    "Start_Longitude",
    "End_Latitude",
    "End_Longitude",
    "Temperature_C",
    "Wind_Speed_kmph",
    "Humidity_%",
    "Pressure_hPa",
    "Visibility_km",
    "Flight_Distance_km",
    "Flight_Duration_hours",
    "Average_Speed_kmph",
    "Max_Altitude_m",
    "Min_Altitude_m",
    "Rest_Stops",
    "Predator_Sightings",
    "Flock_Size",
]


CATEGORICAL_FEATURES = [
    "Species",
    "Region",
    "Habitat",
    "Origin",
    "Migration_Reason",
    "Weather_Condition",
    "Migration_Start_Month",
    "Migration_End_Month",
    "Migrated_in_Flock",
    "Food_Supply_Level",
]


NUMERIC_FEATURES = [
    "Start_Latitude",
    "Start_Longitude",
    "End_Latitude",
    "End_Longitude",
    "Temperature_C",
    "Wind_Speed_kmph",
    "Humidity_%",
    "Pressure_hPa",
    "Visibility_km",
    "Flight_Distance_km",
    "Flight_Duration_hours",
    "Average_Speed_kmph",
    "Max_Altitude_m",
    "Min_Altitude_m",
    "Rest_Stops",
    "Predator_Sightings",
    "Flock_Size",
]


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def section(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def save_dataframe(df, filename):
    path = OUTPUT_DIR / filename
    df.to_csv(path, index=False)
    print(f"Saved: {path}")
    return path


def save_plot(filename):
    path = PLOT_DIR / filename
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved plot: {path}")


# =============================================================================
# LOAD DATA
# =============================================================================

section("BIRD MIGRATION FEATURE INFLUENCE ANALYSIS")

print(f"Dataset path : {DATA_PATH}")

df = pd.read_csv(DATA_PATH)

print(f"Dataset shape: {df.shape}")

# Validate target
if TARGET not in df.columns:
    raise ValueError(f"Target column '{TARGET}' not found.")

# Validate features
missing_features = [
    feature for feature in APPROVED_FEATURES
    if feature not in df.columns
]

if missing_features:
    raise ValueError(
        f"Missing approved features: {missing_features}"
    )

X = df[APPROVED_FEATURES].copy()
y = df[TARGET].astype(int)

print(f"Approved features: {len(APPROVED_FEATURES)}")
print(f"Target: {TARGET}")

print("\nTarget distribution:")
print(y.value_counts().sort_index())

baseline_success_rate = y.mean()

print(
    f"\nOverall migration success rate: "
    f"{baseline_success_rate:.2%}"
)


# =============================================================================
# DATA VALIDATION
# =============================================================================

section("DATA VALIDATION")

print(f"Rows                  : {len(df):,}")
print(f"Features              : {len(APPROVED_FEATURES)}")
print(f"Duplicate rows        : {df.duplicated().sum():,}")
print(f"Missing target values : {df[TARGET].isna().sum():,}")

print("\nMissing values in approved features:")

missing_summary = (
    X.isna()
    .sum()
    .sort_values(ascending=False)
)

print(missing_summary[missing_summary > 0])

# Save missing-value report
save_dataframe(
    missing_summary.reset_index(
        name="missing_count"
    ).rename(
        columns={"index": "feature"}
    ),
    "missing_value_analysis.csv",
)


# =============================================================================
# 1. SUCCESS RATE ANALYSIS
# =============================================================================

section("SUCCESS RATE ANALYSIS")

success_summary = pd.DataFrame({
    "Outcome": [
        "Unsuccessful Migration",
        "Successful Migration",
    ],
    "Count": [
        int((y == 0).sum()),
        int((y == 1).sum()),
    ],
})

success_summary["Percentage"] = (
    success_summary["Count"]
    / len(y)
    * 100
)

print(success_summary)

save_dataframe(
    success_summary,
    "overall_success_rate.csv",
)


# =============================================================================
# 2. NUMERICAL FEATURE CORRELATION
# =============================================================================

section("NUMERICAL FEATURE RELATIONSHIP WITH MIGRATION SUCCESS")

correlation_results = []

for feature in NUMERIC_FEATURES:

    correlation = df[feature].corr(y)

    success_mean = df.loc[y == 1, feature].mean()
    failure_mean = df.loc[y == 0, feature].mean()

    correlation_results.append({
        "feature": feature,
        "correlation_with_success": correlation,
        "success_group_mean": success_mean,
        "failure_group_mean": failure_mean,
        "absolute_correlation": abs(correlation),
    })


correlation_df = (
    pd.DataFrame(correlation_results)
    .sort_values(
        "absolute_correlation",
        ascending=False
    )
)

print(
    correlation_df[
        [
            "feature",
            "correlation_with_success",
            "success_group_mean",
            "failure_group_mean",
        ]
    ].to_string(index=False)
)

save_dataframe(
    correlation_df,
    "numerical_feature_correlation.csv",
)


# =============================================================================
# CORRELATION PLOT
# =============================================================================

top_corr = correlation_df.head(15).sort_values(
    "correlation_with_success"
)

plt.figure(figsize=(10, 7))

plt.barh(
    top_corr["feature"],
    top_corr["correlation_with_success"],
)

plt.axvline(
    0,
    linestyle="--",
    linewidth=1,
)

plt.xlabel("Correlation with Migration Success")
plt.ylabel("Feature")
plt.title(
    "Top Numerical Features Associated with Migration Success"
)

save_plot("numerical_feature_correlation.png")


# =============================================================================
# 3. LIFT ANALYSIS — CATEGORICAL FEATURES
# =============================================================================

section("LIFT ANALYSIS — CATEGORICAL FEATURES")

categorical_lift_results = []

for feature in CATEGORICAL_FEATURES:

    grouped = (
        df.groupby(feature, dropna=False)[TARGET]
        .agg(
            sample_count="count",
            success_count="sum",
            success_rate="mean",
        )
        .reset_index()
    )

    grouped["failure_count"] = (
        grouped["sample_count"]
        - grouped["success_count"]
    )

    grouped["baseline_success_rate"] = baseline_success_rate

    grouped["lift"] = (
        grouped["success_rate"]
        / baseline_success_rate
    )

    grouped["feature"] = feature

    categorical_lift_results.append(
        grouped
    )


categorical_lift_df = pd.concat(
    categorical_lift_results,
    ignore_index=True,
)

categorical_lift_df = categorical_lift_df[
    [
        "feature",
        *CATEGORICAL_FEATURES[:0],
        "sample_count",
        "success_count",
        "failure_count",
        "success_rate",
        "baseline_success_rate",
        "lift",
    ]
    if False
    else [
        "feature",
        "sample_count",
        "success_count",
        "failure_count",
        "success_rate",
        "baseline_success_rate",
        "lift",
    ]
]

categorical_lift_df = categorical_lift_df.sort_values(
    "lift",
    ascending=False
)

print(
    categorical_lift_df.head(30).to_string(
        index=False
    )
)

save_dataframe(
    categorical_lift_df,
    "categorical_lift_analysis.csv",
)


# =============================================================================
# TOP LIFTED CATEGORIES
# =============================================================================

top_lift_categories = (
    categorical_lift_df[
        categorical_lift_df["sample_count"] >= 30
    ]
    .head(15)
    .sort_values("lift")
)

plt.figure(figsize=(11, 8))

labels = (
    top_lift_categories["feature"]
    + " = "
    + top_lift_categories.index.astype(str)
)

# Use feature/value labels directly
labels = [
    f"{row.feature}"
    for row in top_lift_categories.itertuples()
]

plt.barh(
    labels,
    top_lift_categories["lift"],
)

plt.axvline(
    1,
    linestyle="--",
    linewidth=1,
    label="Baseline lift = 1",
)

plt.xlabel("Lift")
plt.ylabel("Feature")
plt.title(
    "Highest-Lift Categorical Feature Groups"
)

plt.legend()

save_plot("categorical_lift.png")


# =============================================================================
# 4. NUMERICAL FEATURE LIFT ANALYSIS
# =============================================================================

section("LIFT ANALYSIS — NUMERICAL FEATURES")

numeric_lift_results = []

for feature in NUMERIC_FEATURES:

    try:
        bins = pd.qcut(
            df[feature],
            q=4,
            duplicates="drop",
        )

        temp = pd.DataFrame({
            "bin": bins,
            TARGET: y,
        })

        grouped = (
            temp.groupby(
                "bin",
                observed=True
            )[TARGET]
            .agg(
                sample_count="count",
                success_count="sum",
                success_rate="mean",
            )
            .reset_index()
        )

        grouped["feature"] = feature

        grouped["baseline_success_rate"] = (
            baseline_success_rate
        )

        grouped["lift"] = (
            grouped["success_rate"]
            / baseline_success_rate
        )

        numeric_lift_results.append(
            grouped
        )

    except Exception as e:
        print(
            f"Skipping numerical lift for {feature}: {e}"
        )


numeric_lift_df = pd.concat(
    numeric_lift_results,
    ignore_index=True,
)

numeric_lift_df = numeric_lift_df[
    [
        "feature",
        "bin",
        "sample_count",
        "success_count",
        "success_rate",
        "baseline_success_rate",
        "lift",
    ]
]

numeric_lift_df = numeric_lift_df.sort_values(
    "lift",
    ascending=False,
)

print(
    numeric_lift_df.head(30).to_string(
        index=False
    )
)

save_dataframe(
    numeric_lift_df,
    "numerical_lift_analysis.csv",
)


# =============================================================================
# 5. LOAD TRAINED MODELS
# =============================================================================

section("LOADING TRAINED MODELS")

gradient_boosting_path = (
    MODEL_DIR / "gradient_boosting.joblib"
)

random_forest_path = (
    MODEL_DIR / "random_forest.joblib"
)

if not gradient_boosting_path.exists():
    raise FileNotFoundError(
        f"Gradient Boosting model not found:\n"
        f"{gradient_boosting_path}"
    )

if not random_forest_path.exists():
    raise FileNotFoundError(
        f"Random Forest model not found:\n"
        f"{random_forest_path}"
    )


gradient_boosting = joblib.load(
    gradient_boosting_path
)

random_forest = joblib.load(
    random_forest_path
)

print(
    f"Loaded Gradient Boosting: "
    f"{gradient_boosting_path}"
)

print(
    f"Loaded Random Forest: "
    f"{random_forest_path}"
)


# =============================================================================
# 6. HELPER — GET CLASSIFIER FROM PIPELINE
# =============================================================================

def get_classifier(model):

    if hasattr(model, "named_steps"):

        if "classifier" in model.named_steps:
            return model.named_steps["classifier"]

        # fallback: last pipeline component
        return list(
            model.named_steps.values()
        )[-1]

    return model


gb_classifier = get_classifier(
    gradient_boosting
)

rf_classifier = get_classifier(
    random_forest
)


print(
    f"\nGradient Boosting classifier: "
    f"{type(gb_classifier).__name__}"
)

print(
    f"Random Forest classifier: "
    f"{type(rf_classifier).__name__}"
)


# =============================================================================
# 7. TREE-BASED FEATURE IMPORTANCE
# =============================================================================

section("TREE-BASED FEATURE IMPORTANCE")


def extract_feature_importance(
    model,
    original_features,
):
    """
    Extract feature importance from a pipeline.

    If preprocessing produces one-hot encoded features,
    the transformed feature importances are aggregated back
    to the original 27 features.
    """

    classifier = get_classifier(model)

    if not hasattr(
        classifier,
        "feature_importances_"
    ):
        raise ValueError(
            "Model does not provide feature_importances_."
        )

    raw_importance = (
        classifier.feature_importances_
    )

    # Try to access preprocessing step
    preprocessor = None

    if hasattr(model, "named_steps"):

        for step_name, step in model.named_steps.items():

            if (
                hasattr(step, "transformers_")
                and hasattr(step, "get_feature_names_out")
            ):
                preprocessor = step
                break

    # If preprocessing does not change dimensions
    if len(raw_importance) == len(original_features):

        result = pd.DataFrame({
            "feature": original_features,
            "importance": raw_importance,
        })

        return result.sort_values(
            "importance",
            ascending=False,
        )

    # If preprocessing generated transformed features
    if preprocessor is not None:

        try:
            transformed_names = (
                preprocessor
                .get_feature_names_out()
            )

            if len(transformed_names) != len(
                raw_importance
            ):
                raise ValueError(
                    "Feature name and importance lengths differ."
                )

            aggregated = {
                feature: 0.0
                for feature in original_features
            }

            for transformed_name, importance in zip(
                transformed_names,
                raw_importance,
            ):

                clean_name = transformed_name

                # Remove transformer prefixes
                for feature in original_features:

                    if (
                        clean_name.endswith(
                            feature
                        )
                        or clean_name.startswith(
                            f"{feature}_"
                        )
                        or f"__{feature}" in clean_name
                        or clean_name.startswith(
                            f"num__{feature}"
                        )
                        or clean_name.startswith(
                            f"cat__{feature}"
                        )
                    ):
                        aggregated[feature] += (
                            float(importance)
                        )
                        break

            result = pd.DataFrame({
                "feature": list(
                    aggregated.keys()
                ),
                "importance": list(
                    aggregated.values()
                ),
            })

            return result.sort_values(
                "importance",
                ascending=False,
            )

        except Exception as e:

            print(
                "Could not aggregate transformed "
                f"feature names: {e}"
            )

    # Fallback
    print(
        "Warning: transformed feature names "
        "could not be mapped to original features."
    )

    return pd.DataFrame({
        "feature": [
            f"transformed_feature_{i}"
            for i in range(
                len(raw_importance)
            )
        ],
        "importance": raw_importance,
    }).sort_values(
        "importance",
        ascending=False,
    )


gb_importance = extract_feature_importance(
    gradient_boosting,
    APPROVED_FEATURES,
)

rf_importance = extract_feature_importance(
    random_forest,
    APPROVED_FEATURES,
)


gb_importance = gb_importance.rename(
    columns={
        "importance":
        "gradient_boosting_importance"
    }
)

rf_importance = rf_importance.rename(
    columns={
        "importance":
        "random_forest_importance"
    }
)


print("\nGradient Boosting feature importance:")
print(
    gb_importance.head(20).to_string(
        index=False
    )
)

print("\nRandom Forest feature importance:")
print(
    rf_importance.head(20).to_string(
        index=False
    )
)


# =============================================================================
# 8. PERMUTATION IMPORTANCE
# =============================================================================

section("PERMUTATION IMPORTANCE")

print(
    "Calculating permutation importance for "
    "Gradient Boosting..."
)

gb_permutation = permutation_importance(
    gradient_boosting,
    X,
    y,
    scoring="f1",
    n_repeats=10,
    random_state=42,
    n_jobs=2,
)

permutation_df = pd.DataFrame({
    "feature": APPROVED_FEATURES,
    "permutation_importance_mean":
        gb_permutation.importances_mean,
    "permutation_importance_std":
        gb_permutation.importances_std,
})

permutation_df[
    "absolute_permutation_importance"
] = (
    permutation_df[
        "permutation_importance_mean"
    ].abs()
)

permutation_df = permutation_df.sort_values(
    "permutation_importance_mean",
    ascending=False,
)

print(
    permutation_df.head(20).to_string(
        index=False
    )
)

save_dataframe(
    permutation_df,
    "permutation_importance.csv",
)


# =============================================================================
# PERMUTATION IMPORTANCE PLOT
# =============================================================================

top_perm = (
    permutation_df
    .head(15)
    .sort_values(
        "permutation_importance_mean"
    )
)

plt.figure(figsize=(11, 8))

plt.barh(
    top_perm["feature"],
    top_perm[
        "permutation_importance_mean"
    ],
)

plt.xlabel(
    "Mean decrease in F1-score"
)

plt.ylabel("Feature")

plt.title(
    "Top Features — Gradient Boosting Permutation Importance"
)

save_plot(
    "gradient_boosting_permutation_importance.png"
)


# =============================================================================
# 9. COMBINED FEATURE IMPORTANCE
# =============================================================================

section("COMBINED FEATURE IMPORTANCE")

combined = pd.DataFrame({
    "feature": APPROVED_FEATURES
})

combined = combined.merge(
    gb_importance,
    on="feature",
    how="left",
)

combined = combined.merge(
    rf_importance,
    on="feature",
    how="left",
)

combined = combined.merge(
    permutation_df[
        [
            "feature",
            "permutation_importance_mean",
            "permutation_importance_std",
        ]
    ],
    on="feature",
    how="left",
)

combined = combined.merge(
    correlation_df[
        [
            "feature",
            "correlation_with_success",
            "absolute_correlation",
        ]
    ],
    on="feature",
    how="left",
)


# Normalize importance values
def normalize_series(series):

    series = series.fillna(0)

    min_value = series.min()
    max_value = series.max()

    if max_value == min_value:
        return pd.Series(
            np.ones(len(series)),
            index=series.index,
        )

    return (
        (series - min_value)
        / (max_value - min_value)
    )


combined[
    "gb_importance_normalized"
] = normalize_series(
    combined[
        "gradient_boosting_importance"
    ]
)

combined[
    "rf_importance_normalized"
] = normalize_series(
    combined[
        "random_forest_importance"
    ]
)

combined[
    "permutation_importance_normalized"
] = normalize_series(
    combined[
        "permutation_importance_mean"
    ].abs()
)

combined[
    "correlation_normalized"
] = normalize_series(
    combined[
        "absolute_correlation"
    ]
)


# Weighted overall importance
#
# Model importance is given the highest weight because
# Question 6 asks which features influence predictive success.
#
# Permutation importance receives strong weight because
# it evaluates the actual effect on model performance.
#
# Correlation is included as supporting statistical evidence.

combined["overall_importance_score"] = (
    0.30
    * combined[
        "gb_importance_normalized"
    ]
    +
    0.25
    * combined[
        "rf_importance_normalized"
    ]
    +
    0.35
    * combined[
        "permutation_importance_normalized"
    ]
    +
    0.10
    * combined[
        "correlation_normalized"
    ]
)


combined = combined.sort_values(
    "overall_importance_score",
    ascending=False,
).reset_index(drop=True)


combined.insert(
    0,
    "rank",
    range(1, len(combined) + 1),
)


print(
    combined.head(20).to_string(
        index=False
    )
)

save_dataframe(
    combined,
    "combined_feature_importance.csv",
)


# =============================================================================
# TOP FEATURE PLOT
# =============================================================================

top_combined = (
    combined.head(15)
    .sort_values(
        "overall_importance_score"
    )
)

plt.figure(figsize=(11, 8))

plt.barh(
    top_combined["feature"],
    top_combined[
        "overall_importance_score"
    ],
)

plt.xlabel(
    "Combined Importance Score"
)

plt.ylabel("Feature")

plt.title(
    "Top Features Influencing Migration Success"
)

save_plot(
    "combined_feature_importance.png"
)


# =============================================================================
# 10. FEATURE CATEGORY ANALYSIS
# =============================================================================

section("FEATURE CATEGORY ANALYSIS")

feature_categories = {}

for feature in APPROVED_FEATURES:

    if feature in [
        "Species",
        "Region",
        "Habitat",
        "Origin",
        "Migration_Reason",
        "Start_Latitude",
        "Start_Longitude",
        "End_Latitude",
        "End_Longitude",
    ]:
        category = "Geographic & Migration Context"

    elif feature in [
        "Weather_Condition",
        "Temperature_C",
        "Wind_Speed_kmph",
        "Humidity_%",
        "Pressure_hPa",
        "Visibility_km",
    ]:
        category = "Environmental"

    elif feature in [
        "Flight_Distance_km",
        "Flight_Duration_hours",
        "Average_Speed_kmph",
        "Max_Altitude_m",
        "Min_Altitude_m",
    ]:
        category = "Flight Characteristics"

    elif feature in [
        "Migrated_in_Flock",
        "Flock_Size",
        "Rest_Stops",
        "Predator_Sightings",
        "Food_Supply_Level",
    ]:
        category = "Behavioral"

    elif feature in [
        "Migration_Start_Month",
        "Migration_End_Month",
    ]:
        category = "Seasonal"

    else:
        category = "Other"

    feature_categories[feature] = category


combined["feature_category"] = (
    combined["feature"]
    .map(feature_categories)
)


category_importance = (
    combined.groupby(
        "feature_category"
    )["overall_importance_score"]
    .agg(
        total_importance="sum",
        average_importance="mean",
        feature_count="count",
    )
    .reset_index()
    .sort_values(
        "total_importance",
        ascending=False,
    )
)

print(category_importance)

save_dataframe(
    category_importance,
    "feature_category_importance.csv",
)


# =============================================================================
# CATEGORY IMPORTANCE PLOT
# =============================================================================

category_plot = category_importance.sort_values(
    "total_importance"
)

plt.figure(figsize=(10, 6))

plt.barh(
    category_plot["feature_category"],
    category_plot["total_importance"],
)

plt.xlabel(
    "Total Combined Importance"
)

plt.ylabel("Feature Category")

plt.title(
    "Migration Success Importance by Feature Category"
)

save_plot(
    "feature_category_importance.png"
)


# =============================================================================
# 11. LOAD MODEL COMPARISON RESULTS
# =============================================================================

section("MODEL PREDICTION EFFECTIVENESS")

metrics_path = MODEL_DIR / "model_metrics.csv"

if metrics_path.exists():

    model_metrics = pd.read_csv(
        metrics_path
    )

    print(
        model_metrics.to_string(
            index=False
        )
    )

    save_dataframe(
        model_metrics,
        "model_prediction_effectiveness.csv",
    )

else:

    print(
        "model_metrics.csv not found."
    )

    model_metrics = None


# =============================================================================
# 12. RE-EVALUATE CHAMPION MODEL
# =============================================================================

section("GRADIENT BOOSTING CHAMPION EVALUATION")

gb_predictions = (
    gradient_boosting.predict(X)
)

gb_probabilities = (
    gradient_boosting.predict_proba(X)[:, 1]
    if hasattr(
        gradient_boosting,
        "predict_proba"
    )
    else None
)


gb_accuracy = accuracy_score(
    y,
    gb_predictions,
)

gb_precision = precision_score(
    y,
    gb_predictions,
    zero_division=0,
)

gb_recall = recall_score(
    y,
    gb_predictions,
    zero_division=0,
)

gb_f1 = f1_score(
    y,
    gb_predictions,
    zero_division=0,
)

print(
    f"Accuracy : {gb_accuracy:.4f}"
)

print(
    f"Precision: {gb_precision:.4f}"
)

print(
    f"Recall   : {gb_recall:.4f}"
)

print(
    f"F1-score : {gb_f1:.4f}"
)

if gb_probabilities is not None:

    gb_roc_auc = roc_auc_score(
        y,
        gb_probabilities,
    )

    print(
        f"ROC-AUC  : {gb_roc_auc:.4f}"
    )

else:

    gb_roc_auc = None


# =============================================================================
# 13. CONFUSION MATRIX
# =============================================================================

section("CONFUSION MATRIX — GRADIENT BOOSTING")

cm = confusion_matrix(
    y,
    gb_predictions,
)

print(
    "\nConfusion Matrix:"
)

print(cm)

tn, fp, fn, tp = cm.ravel()

print(
    f"\nTrue Negatives : {tn:,}"
)

print(
    f"False Positives: {fp:,}"
)

print(
    f"False Negatives: {fn:,}"
)

print(
    f"True Positives  : {tp:,}"
)


plt.figure(figsize=(7, 6))

plt.imshow(cm)

plt.title(
    "Gradient Boosting Confusion Matrix"
)

plt.xlabel(
    "Predicted Class"
)

plt.ylabel(
    "Actual Class"
)

plt.xticks(
    [0, 1],
    [
        "Unsuccessful",
        "Successful",
    ],
)

plt.yticks(
    [0, 1],
    [
        "Unsuccessful",
        "Successful",
    ],
)

for i in range(2):

    for j in range(2):

        plt.text(
            j,
            i,
            cm[i, j],
            ha="center",
            va="center",
            fontsize=14,
        )

save_plot(
    "gradient_boosting_confusion_matrix.png"
)


# =============================================================================
# 14. CLASSIFICATION REPORT
# =============================================================================

report = classification_report(
    y,
    gb_predictions,
    target_names=[
        "Unsuccessful Migration",
        "Successful Migration",
    ],
    output_dict=True,
)

classification_report_df = (
    pd.DataFrame(report)
    .transpose()
)

save_dataframe(
    classification_report_df.reset_index(
        names="class"
    ),
    "gradient_boosting_classification_report.csv",
)


# =============================================================================
# 15. PREDICTION PROBABILITY DISTRIBUTION
# =============================================================================

if gb_probabilities is not None:

    section(
        "PREDICTION PROBABILITY ANALYSIS"
    )

    success_probabilities = (
        gb_probabilities[y == 1]
    )

    failure_probabilities = (
        gb_probabilities[y == 0]
    )

    print(
        "Average predicted probability "
        "of success:"
    )

    print(
        f"Actual successful migrations : "
        f"{success_probabilities.mean():.4f}"
    )

    print(
        f"Actual unsuccessful migrations: "
        f"{failure_probabilities.mean():.4f}"
    )


    plt.figure(figsize=(10, 6))

    plt.hist(
        failure_probabilities,
        bins=20,
        alpha=0.6,
        label="Actual Unsuccessful",
    )

    plt.hist(
        success_probabilities,
        bins=20,
        alpha=0.6,
        label="Actual Successful",
    )

    plt.xlabel(
        "Predicted Probability of Migration Success"
    )

    plt.ylabel(
        "Number of Records"
    )

    plt.title(
        "Gradient Boosting Prediction Probability Distribution"
    )

    plt.legend()

    save_plot(
        "prediction_probability_distribution.png"
    )


# =============================================================================
# 16. SAVE FEATURE SUMMARY
# =============================================================================

section("TOP FEATURES SUMMARY")

top_features = combined.head(10).copy()

print(
    "\nTop 10 features influencing migration success:\n"
)

for _, row in top_features.iterrows():

    print(
        f"{int(row['rank']):2d}. "
        f"{row['feature']:<30} "
        f"Score = "
        f"{row['overall_importance_score']:.4f}"
    )


summary = {
    "research_question": 6,
    "question": (
        "Which features have the greatest influence "
        "on migration success, and how effectively can "
        "machine learning models predict successful "
        "and unsuccessful migrations?"
    ),
    "dataset_records": int(len(df)),
    "approved_features": len(APPROVED_FEATURES),
    "baseline_success_rate": float(
        baseline_success_rate
    ),
    "champion_model": "Gradient Boosting",
    "champion_accuracy": float(
        gb_accuracy
    ),
    "champion_precision": float(
        gb_precision
    ),
    "champion_recall": float(
        gb_recall
    ),
    "champion_f1": float(
        gb_f1
    ),
    "champion_roc_auc": (
        float(gb_roc_auc)
        if gb_roc_auc is not None
        else None
    ),
    "top_10_features": top_features[
        [
            "rank",
            "feature",
            "feature_category",
            "overall_importance_score",
        ]
    ].to_dict(
        orient="records"
    ),
}


summary_path = (
    OUTPUT_DIR
    / "question_6_summary.json"
)

with open(
    summary_path,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        summary,
        f,
        indent=4,
    )

print(
    f"\nSaved: {summary_path}"
)


# =============================================================================
# FINAL OUTPUT
# =============================================================================

section("QUESTION 6 ANALYSIS COMPLETED")

print(
    "\nResearch Question 6 has been analyzed using:"
)

print(
    "  ✓ Numerical correlation analysis"
)

print(
    "  ✓ Categorical lift analysis"
)

print(
    "  ✓ Numerical lift analysis"
)

print(
    "  ✓ Gradient Boosting feature importance"
)

print(
    "  ✓ Random Forest feature importance"
)

print(
    "  ✓ Gradient Boosting permutation importance"
)

print(
    "  ✓ Combined feature importance ranking"
)

print(
    "  ✓ Feature-category importance"
)

print(
    "  ✓ Model prediction effectiveness"
)

print(
    "  ✓ Gradient Boosting confusion matrix"
)

print(
    "  ✓ Classification report"
)

print(
    "\nOutput directory:"
)

print(
    OUTPUT_DIR
)

print(
    "\nTop 10 influential features:"
)

for _, row in combined.head(10).iterrows():

    print(
        f"{int(row['rank']):2d}. "
        f"{row['feature']} "
        f"({row['feature_category']})"
    )

print(
    "\nAnalysis completed successfully."
)