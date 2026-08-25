"""
Optimized Random Forest training for Bird Migration Success Prediction.

This script:
1. Loads the processed dataset
2. Uses only the approved ML features from src.modeling
3. Performs a stratified train/test split
4. Builds a leakage-safe preprocessing pipeline
5. Trains a baseline Random Forest
6. Tunes Random Forest using RandomizedSearchCV
7. Evaluates baseline and tuned models
8. Saves the best Random Forest pipeline
9. Saves metrics and feature importance
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    RandomizedSearchCV,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


from src.modeling import (
    DATA_PATH,
    MODELS_DIR,
    TARGET_COLUMN,
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    FEATURE_COLUMNS,
    prepare_prediction_frame,
)


# ============================================================
# CONFIGURATION
# ============================================================

RANDOM_STATE = 42

TEST_SIZE = 0.20

# Number of random hyperparameter combinations.
# 20 is reasonable for an 8 GB RAM / i3 machine.
N_ITER = 20

# CV folds.
CV_FOLDS = 3

# Primary tuning metric.
SCORING = "f1"

MODEL_KEY = "random_forest"

MODEL_NAME = "Random Forest"


# ============================================================
# PREPROCESSOR
# ============================================================

def build_preprocessor() -> ColumnTransformer:
    """
    Build preprocessing pipeline.

    Numerical:
        Missing values -> median

    Categorical:
        Missing values -> most frequent
        One-hot encoding

    Random Forest does not require feature scaling,
    therefore StandardScaler is intentionally NOT used.
    """

    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median"),
            )
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

    preprocessor = ColumnTransformer(
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
    )

    return preprocessor


# ============================================================
# BASELINE MODEL
# ============================================================

def build_baseline_model() -> Pipeline:
    """
    Build baseline Random Forest pipeline.
    """

    model = RandomForestClassifier(
        n_estimators=300,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        class_weight="balanced",
    )

    pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                build_preprocessor(),
            ),
            (
                "model",
                model,
            ),
        ]
    )

    return pipeline


# ============================================================
# TUNING MODEL
# ============================================================

def build_search() -> RandomizedSearchCV:
    """
    Build RandomizedSearchCV for Random Forest.

    Search space is intentionally moderate so it can run
    on an 8 GB RAM / i3 machine.
    """

    pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                build_preprocessor(),
            ),
            (
                "model",
                RandomForestClassifier(
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                    class_weight="balanced",
                ),
            ),
        ]
    )

    param_distributions = {
        "model__n_estimators": [
            200,
            300,
            400,
            500,
        ],

        "model__max_depth": [
            None,
            8,
            12,
            16,
            20,
            25,
        ],

        "model__min_samples_split": [
            2,
            5,
            10,
            15,
        ],

        "model__min_samples_leaf": [
            1,
            2,
            4,
            6,
            8,
        ],

        "model__max_features": [
            "sqrt",
            "log2",
            0.5,
            0.7,
        ],

        "model__bootstrap": [
            True,
        ],
    }

    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_distributions,
        n_iter=N_ITER,
        scoring=SCORING,
        cv=CV_FOLDS,
        verbose=2,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        return_train_score=True,
    )

    return search


# ============================================================
# EVALUATION
# ============================================================

def evaluate_model(
    model: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict:
    """
    Calculate all required evaluation metrics.
    """

    predictions = model.predict(X_test)

    probabilities = model.predict_proba(
        X_test
    )[:, 1]

    accuracy = accuracy_score(
        y_test,
        predictions,
    )

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0,
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0,
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0,
    )

    roc_auc = roc_auc_score(
        y_test,
        probabilities,
    )

    cm = confusion_matrix(
        y_test,
        predictions,
    )

    report = classification_report(
        y_test,
        predictions,
        output_dict=True,
        zero_division=0,
    )

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "roc_auc": float(roc_auc),
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
        "y_true": y_test.astype(int).tolist(),
        "y_pred": predictions.astype(int).tolist(),
        "y_score": probabilities.tolist(),
    }


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

def get_feature_importance(
    pipeline: Pipeline,
) -> pd.DataFrame:
    """
    Extract Random Forest feature importance.

    One-hot encoded categorical variables are grouped back
    to their original source columns.
    """

    preprocessor = pipeline.named_steps[
        "preprocessor"
    ]

    model = pipeline.named_steps[
        "model"
    ]

    transformed_names = (
        preprocessor
        .get_feature_names_out()
    )

    importances = model.feature_importances_

    rows = []

    for name, importance in zip(
        transformed_names,
        importances,
    ):

        clean_name = name.split(
            "__",
            maxsplit=1,
        )[-1]

        source_feature = clean_name

        # Match original feature name.
        for feature in sorted(
            FEATURE_COLUMNS,
            key=len,
            reverse=True,
        ):
            if (
                clean_name == feature
                or clean_name.startswith(
                    feature + "_"
                )
            ):
                source_feature = feature
                break

        rows.append(
            {
                "feature": source_feature,
                "raw_feature": clean_name,
                "importance": float(
                    importance
                ),
            }
        )

    importance_df = pd.DataFrame(
        rows
    )

    # Group one-hot encoded columns.
    importance_df = (
        importance_df
        .groupby(
            "feature",
            as_index=False,
        )["importance"]
        .sum()
        .sort_values(
            "importance",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    importance_df[
        "importance"
    ] = importance_df[
        "importance"
    ].round(6)

    return importance_df


# ============================================================
# MAIN TRAINING
# ============================================================

def train_random_forest():
    """
    Complete Random Forest training workflow.
    """

    print("\n")
    print("=" * 70)
    print("BIRD MIGRATION - RANDOM FOREST OPTIMIZATION")
    print("=" * 70)

    start_time = time.time()

    # --------------------------------------------------------
    # LOAD DATA
    # --------------------------------------------------------

    if not DATA_PATH.exists():

        raise FileNotFoundError(
            f"Dataset not found:\n{DATA_PATH}"
        )

    print("\nLoading dataset...")

    data = pd.read_csv(
        DATA_PATH
    )

    print(
        f"Dataset shape: {data.shape}"
    )

    # --------------------------------------------------------
    # TARGET
    # --------------------------------------------------------

    if TARGET_COLUMN not in data.columns:

        raise ValueError(
            f"Target column '{TARGET_COLUMN}' "
            "not found."
        )

    y = (
        pd.to_numeric(
            data[TARGET_COLUMN],
            errors="raise",
        )
        .astype(int)
    )

    # --------------------------------------------------------
    # FEATURES
    # --------------------------------------------------------

    X = prepare_prediction_frame(
        data
    )

    print(
        f"Number of ML features: "
        f"{len(FEATURE_COLUMNS)}"
    )

    print(
        "\nCategorical features:"
    )

    for feature in CATEGORICAL_FEATURES:
        print(
            f"  - {feature}"
        )

    print(
        "\nNumerical features:"
    )

    for feature in NUMERIC_FEATURES:
        print(
            f"  - {feature}"
        )

    # --------------------------------------------------------
    # CLASS DISTRIBUTION
    # --------------------------------------------------------

    print(
        "\nTarget distribution:"
    )

    print(
        y.value_counts()
    )

    print(
        "\nTarget percentages:"
    )

    print(
        y.value_counts(
            normalize=True
        ).round(4)
    )

    # --------------------------------------------------------
    # TRAIN TEST SPLIT
    # --------------------------------------------------------

    print(
        "\nCreating stratified train/test split..."
    )

    X_train, X_test, y_train, y_test = (
        train_test_split(
            X,
            y,
            test_size=TEST_SIZE,
            stratify=y,
            random_state=RANDOM_STATE,
        )
    )

    print(
        f"Training records: {len(X_train):,}"
    )

    print(
        f"Testing records: {len(X_test):,}"
    )

    # --------------------------------------------------------
    # BASELINE RANDOM FOREST
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("1. BASELINE RANDOM FOREST")
    print("=" * 70)

    baseline = build_baseline_model()

    baseline_start = time.time()

    baseline.fit(
        X_train,
        y_train,
    )

    baseline_time = (
        time.time()
        - baseline_start
    )

    baseline_metrics = evaluate_model(
        baseline,
        X_test,
        y_test,
    )

    print(
        f"\nBaseline training time: "
        f"{baseline_time:.2f} seconds"
    )

    print(
        f"Accuracy : "
        f"{baseline_metrics['accuracy']:.4f}"
    )

    print(
        f"Precision: "
        f"{baseline_metrics['precision']:.4f}"
    )

    print(
        f"Recall   : "
        f"{baseline_metrics['recall']:.4f}"
    )

    print(
        f"F1-score : "
        f"{baseline_metrics['f1_score']:.4f}"
    )

    print(
        f"ROC-AUC  : "
        f"{baseline_metrics['roc_auc']:.4f}"
    )

    # --------------------------------------------------------
    # RANDOMIZED SEARCH
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("2. RANDOM FOREST HYPERPARAMETER TUNING")
    print("=" * 70)

    print(
        f"\nRandomized search:"
    )

    print(
        f"{N_ITER} combinations "
        f"x {CV_FOLDS}-fold CV"
    )

    print(
        f"Total fits: "
        f"{N_ITER * CV_FOLDS}"
    )

    print(
        f"Optimization metric: "
        f"{SCORING}"
    )

    search = build_search()

    search_start = time.time()

    search.fit(
        X_train,
        y_train,
    )

    search_time = (
        time.time()
        - search_start
    )

    print(
        "\nHyperparameter tuning complete."
    )

    print(
        f"Tuning time: "
        f"{search_time / 60:.2f} minutes"
    )

    # --------------------------------------------------------
    # BEST PARAMETERS
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("BEST RANDOM FOREST PARAMETERS")
    print("=" * 70)

    for parameter, value in (
        search.best_params_.items()
    ):
        print(
            f"{parameter}: {value}"
        )

    print(
        f"\nBest CV F1-score: "
        f"{search.best_score_:.4f}"
    )

    # --------------------------------------------------------
    # TUNED MODEL TEST EVALUATION
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("3. TUNED RANDOM FOREST TEST PERFORMANCE")
    print("=" * 70)

    tuned_model = search.best_estimator_

    tuned_metrics = evaluate_model(
        tuned_model,
        X_test,
        y_test,
    )

    print(
        f"\nAccuracy : "
        f"{tuned_metrics['accuracy']:.4f}"
    )

    print(
        f"Precision: "
        f"{tuned_metrics['precision']:.4f}"
    )

    print(
        f"Recall   : "
        f"{tuned_metrics['recall']:.4f}"
    )

    print(
        f"F1-score : "
        f"{tuned_metrics['f1_score']:.4f}"
    )

    print(
        f"ROC-AUC  : "
        f"{tuned_metrics['roc_auc']:.4f}"
    )

    print(
        "\nConfusion Matrix:"
    )

    print(
        np.array(
            tuned_metrics[
                "confusion_matrix"
            ]
        )
    )

    # --------------------------------------------------------
    # COMPARE BASELINE VS TUNED
    # --------------------------------------------------------

    comparison = pd.DataFrame(
        [
            {
                "Model": "Random Forest - Baseline",
                "Accuracy": baseline_metrics[
                    "accuracy"
                ],
                "Precision": baseline_metrics[
                    "precision"
                ],
                "Recall": baseline_metrics[
                    "recall"
                ],
                "F1": baseline_metrics[
                    "f1_score"
                ],
                "ROC-AUC": baseline_metrics[
                    "roc_auc"
                ],
                "Training_Time_sec": baseline_time,
            },
            {
                "Model": "Random Forest - Tuned",
                "Accuracy": tuned_metrics[
                    "accuracy"
                ],
                "Precision": tuned_metrics[
                    "precision"
                ],
                "Recall": tuned_metrics[
                    "recall"
                ],
                "F1": tuned_metrics[
                    "f1_score"
                ],
                "ROC-AUC": tuned_metrics[
                    "roc_auc"
                ],
                "Training_Time_sec": (
                    search_time
                ),
            },
        ]
    )

    comparison[
        [
            "Accuracy",
            "Precision",
            "Recall",
            "F1",
            "ROC-AUC",
        ]
    ] = comparison[
        [
            "Accuracy",
            "Precision",
            "Recall",
            "F1",
            "ROC-AUC",
        ]
    ].round(4)

    # --------------------------------------------------------
    # FEATURE IMPORTANCE
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("4. RANDOM FOREST FEATURE IMPORTANCE")
    print("=" * 70)

    importance_df = get_feature_importance(
        tuned_model
    )

    print(
        "\nTop predictive drivers:"
    )

    print(
        importance_df.head(20).to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # SAVE ARTIFACTS
    # --------------------------------------------------------

    MODELS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Save tuned model.
    rf_path = (
        MODELS_DIR
        / "random_forest.joblib"
    )

    joblib.dump(
        tuned_model,
        rf_path,
    )

    # Save comparison.
    comparison.to_csv(
        MODELS_DIR
        / "random_forest_comparison.csv",
        index=False,
    )

    # Save feature importance.
    importance_df.to_csv(
        MODELS_DIR
        / "random_forest_feature_importance.csv",
        index=False,
    )

    # Save evaluation.
    evaluation = {
        "baseline": baseline_metrics,
        "tuned": tuned_metrics,
    }

    joblib.dump(
        evaluation,
        MODELS_DIR
        / "random_forest_evaluation.joblib",
    )

    # Save best parameters.
    with open(
        MODELS_DIR
        / "random_forest_best_params.json",
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            search.best_params_,
            file,
            indent=4,
            default=str,
        )

    # Save metadata.
    metadata = {
        "model": MODEL_NAME,
        "model_key": MODEL_KEY,
        "target": TARGET_COLUMN,
        "features": FEATURE_COLUMNS,
        "categorical_features": CATEGORICAL_FEATURES,
        "numeric_features": NUMERIC_FEATURES,
        "train_rows": int(
            len(X_train)
        ),
        "test_rows": int(
            len(X_test)
        ),
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "cv_folds": CV_FOLDS,
        "n_iter": N_ITER,
        "scoring": SCORING,
        "baseline_metrics": {
            key: value
            for key, value
            in baseline_metrics.items()
            if key in [
                "accuracy",
                "precision",
                "recall",
                "f1_score",
                "roc_auc",
            ]
        },
        "tuned_metrics": {
            key: value
            for key, value
            in tuned_metrics.items()
            if key in [
                "accuracy",
                "precision",
                "recall",
                "f1_score",
                "roc_auc",
            ]
        },
        "best_cv_f1": float(
            search.best_score_
        ),
        "trained_at_utc": (
            datetime.now(
                timezone.utc
            ).isoformat()
        ),
    }

    with open(
        MODELS_DIR
        / "random_forest_metadata.json",
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            metadata,
            file,
            indent=4,
        )

    # --------------------------------------------------------
    # FINAL SUMMARY
    # --------------------------------------------------------

    total_time = (
        time.time()
        - start_time
    )

    print("\n")
    print("=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)

    print(
        f"\nModel saved to:"
        f"\n{rf_path}"
    )

    print(
        f"\nTotal execution time:"
        f" {total_time / 60:.2f} minutes"
    )

    print("\nFinal tuned Random Forest:")

    print(
        f"Accuracy : "
        f"{tuned_metrics['accuracy']:.4f}"
    )

    print(
        f"Precision: "
        f"{tuned_metrics['precision']:.4f}"
    )

    print(
        f"Recall   : "
        f"{tuned_metrics['recall']:.4f}"
    )

    print(
        f"F1-score : "
        f"{tuned_metrics['f1_score']:.4f}"
    )

    print(
        f"ROC-AUC  : "
        f"{tuned_metrics['roc_auc']:.4f}"
    )

    print("\nComparison:")
    print(
        comparison.to_string(
            index=False
        )
    )

    return (
        tuned_model,
        comparison,
        importance_df,
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    train_random_forest()