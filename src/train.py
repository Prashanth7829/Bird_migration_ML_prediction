"""Optimized training, tuning, evaluation, and persistence for bird migration models."""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
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
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from src.modeling import (
    DATA_PATH,
    FEATURE_COLUMNS,
    MODEL_LABELS,
    MODELS_DIR,
    TARGET_COLUMN,
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    prepare_prediction_frame,
    model_path,
)


# ============================================================
# GLOBAL CONFIGURATION
# ============================================================

RANDOM_STATE = 42

TEST_SIZE = 0.20

# 3-fold is substantially faster than 5-fold while still
# providing reasonable hyperparameter validation.
CV_FOLDS = 3

# Number of random parameter combinations per model.
# This is deliberately kept moderate for faster training.
N_ITER_SEARCH = 8


# ============================================================
# PREPROCESSOR
# ============================================================

def build_preprocessor() -> ColumnTransformer:
    """
    Build preprocessing pipeline.

    Numeric:
        - Median imputation
        - Standard scaling

    Categorical:
        - Most-frequent imputation
        - One-hot encoding
    """

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", __import__("sklearn").impute.SimpleImputer(
                strategy="median"
            )),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", __import__("sklearn").impute.SimpleImputer(
                strategy="most_frequent"
            )),
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
# MODEL CONFIGURATION
# ============================================================

def get_model_configs() -> dict[str, dict]:
    """
    Return optimized model definitions and search spaces.

    RandomizedSearchCV is used instead of GridSearchCV to
    drastically reduce the number of model fits.
    """

    return {

        # ----------------------------------------------------
        # 1. LOGISTIC REGRESSION
        # ----------------------------------------------------

        "logistic_regression": {
            "estimator": LogisticRegression(
                max_iter=1000,
                solver="liblinear",
                random_state=RANDOM_STATE,
            ),

            "params": {
                "model__C": [
                    0.01,
                    0.1,
                    1.0,
                    10.0,
                ],

                "model__penalty": [
                    "l1",
                    "l2",
                ],
            },
        },

        # ----------------------------------------------------
        # 2. DECISION TREE
        # ----------------------------------------------------

        "decision_tree": {
            "estimator": DecisionTreeClassifier(
                random_state=RANDOM_STATE,
            ),

            "params": {
                "model__max_depth": [
                    3,
                    5,
                    7,
                    10,
                    15,
                    None,
                ],

                "model__min_samples_split": [
                    2,
                    5,
                    10,
                    20,
                ],

                "model__min_samples_leaf": [
                    1,
                    2,
                    5,
                    10,
                ],

                "model__criterion": [
                    "gini",
                    "entropy",
                ],
            },
        },

        # ----------------------------------------------------
        # 3. KNN
        # ----------------------------------------------------

        "knn": {
            "estimator": KNeighborsClassifier(
                n_jobs=-1,
            ),

            "params": {
                "model__n_neighbors": [
                    3,
                    5,
                    7,
                    11,
                    15,
                ],

                "model__weights": [
                    "uniform",
                    "distance",
                ],

                "model__metric": [
                    "euclidean",
                    "manhattan",
                ],
            },
        },

        # ----------------------------------------------------
        # 4. SVM
        # ----------------------------------------------------

        "svm": {
            "estimator": SVC(
                probability=True,
                random_state=RANDOM_STATE,
            ),

            "params": {
                "model__C": [
                    0.1,
                    1,
                    10,
                ],

                "model__kernel": [
                    "rbf",
                    "linear",
                ],

                "model__gamma": [
                    "scale",
                    "auto",
                ],
            },
        },

        # ----------------------------------------------------
        # 5. GRADIENT BOOSTING
        # ----------------------------------------------------

        "gradient_boosting": {
            "estimator": GradientBoostingClassifier(
                random_state=RANDOM_STATE,
            ),

            "params": {
                "model__n_estimators": [
                    50,
                    100,
                    150,
                ],

                "model__learning_rate": [
                    0.03,
                    0.05,
                    0.1,
                ],

                "model__max_depth": [
                    2,
                    3,
                    4,
                ],

                "model__subsample": [
                    0.8,
                    1.0,
                ],
            },
        },

        # ----------------------------------------------------
        # 6. RANDOM FOREST
        # ----------------------------------------------------

        "random_forest": {
            "estimator": RandomForestClassifier(
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),

            "params": {
                "model__n_estimators": [
                    150,
                    250,
                    350,
                ],

                "model__max_depth": [
                    None,
                    10,
                    15,
                    20,
                ],

                "model__min_samples_split": [
                    2,
                    5,
                    10,
                ],

                "model__min_samples_leaf": [
                    1,
                    2,
                    4,
                ],

                "model__max_features": [
                    "sqrt",
                    "log2",
                ],
            },
        },
    }


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

def source_feature_name(transformed_name: str) -> str:
    """
    Map one-hot encoded feature back to original source column.
    """

    name = transformed_name.split(
        "__",
        maxsplit=1,
    )[-1]

    for source in sorted(
        FEATURE_COLUMNS,
        key=len,
        reverse=True,
    ):
        if (
            name == source
            or name.startswith(f"{source}_")
        ):
            return source

    return name


def extract_feature_importance(
    pipeline: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_key: str,
) -> list[dict[str, float | str]]:
    """
    Extract grouped feature importance.

    Tree models:
        feature_importances_

    Logistic/SVM:
        absolute coefficient magnitude

    KNN:
        permutation importance
    """

    estimator = pipeline.named_steps["model"]

    preprocessor = pipeline.named_steps["preprocessor"]

    feature_names = (
        preprocessor.get_feature_names_out()
    )

    grouped: dict[str, float] = {}

    # --------------------------------------------------------
    # TREE MODELS
    # --------------------------------------------------------

    if hasattr(
        estimator,
        "feature_importances_",
    ):

        values = np.asarray(
            estimator.feature_importances_
        )

        for name, value in zip(
            feature_names,
            values,
        ):

            source = source_feature_name(name)

            grouped[source] = (
                grouped.get(source, 0.0)
                + float(value)
            )

    # --------------------------------------------------------
    # LINEAR / SVM
    # --------------------------------------------------------

    elif hasattr(estimator, "coef_"):

        values = np.abs(
            np.asarray(
                estimator.coef_
            )
        ).mean(axis=0)

        for name, value in zip(
            feature_names,
            values,
        ):

            source = source_feature_name(name)

            grouped[source] = (
                grouped.get(source, 0.0)
                + float(value)
            )

    # --------------------------------------------------------
    # KNN
    # --------------------------------------------------------

    else:

        sample_size = min(
            500,
            len(X_test),
        )

        sample = X_test.sample(
            sample_size,
            random_state=RANDOM_STATE,
        )

        sample_y = y_test.loc[
            sample.index
        ]

        permutation = permutation_importance(
            pipeline,
            sample,
            sample_y,
            scoring="roc_auc",
            n_repeats=2,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )

        grouped = {
            column: max(
                float(value),
                0.0,
            )
            for column, value in zip(
                FEATURE_COLUMNS,
                permutation.importances_mean,
            )
        }

    importance = [
        {
            "feature": feature,
            "importance": round(
                value,
                6,
            ),
        }

        for feature, value in sorted(
            grouped.items(),
            key=lambda item: item[1],
            reverse=True,
        )
    ]

    return importance


# ============================================================
# TRAIN ONE MODEL
# ============================================================

def train_model(
    model_key: str,
    estimator,
    param_distributions: dict,
    X_train: pd.DataFrame,
    y_train: pd.Series,
):
    """
    Train one model using RandomizedSearchCV.
    """

    pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                build_preprocessor(),
            ),

            (
                "model",
                estimator,
            ),
        ]
    )

    print()
    print("=" * 80)
    print(
        f"TRAINING: {MODEL_LABELS[model_key]}"
    )
    print("=" * 80)

    print(
        f"Randomized search: "
        f"{N_ITER_SEARCH} combinations × "
        f"{CV_FOLDS}-fold CV"
    )

    start_time = time.time()

    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_distributions,
        n_iter=N_ITER_SEARCH,
        scoring="roc_auc",
        cv=CV_FOLDS,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=1,
        refit=True,
    )

    search.fit(
        X_train,
        y_train,
    )

    elapsed = time.time() - start_time

    print(
        f"Best CV ROC-AUC: "
        f"{search.best_score_:.4f}"
    )

    print(
        f"Training time: "
        f"{elapsed / 60:.2f} minutes"
    )

    print(
        "Best parameters:"
    )

    for parameter, value in search.best_params_.items():
        print(
            f"  {parameter}: {value}"
        )

    return (
        search.best_estimator_,
        search.best_score_,
        search.best_params_,
        elapsed,
    )


# ============================================================
# MAIN TRAINING FUNCTION
# ============================================================

def train_all_models() -> pd.DataFrame:

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Processed dataset not found: "
            f"{DATA_PATH}"
        )

    print()
    print("=" * 80)
    print("BIRD MIGRATION ML TRAINING")
    print("=" * 80)

    # --------------------------------------------------------
    # LOAD DATA
    # --------------------------------------------------------

    print(
        f"\nLoading dataset:\n{DATA_PATH}"
    )

    data = pd.read_csv(
        DATA_PATH
    )

    print(
        f"Dataset shape: "
        f"{data.shape}"
    )

    # --------------------------------------------------------
    # PREPARE FEATURES
    # --------------------------------------------------------

    X = prepare_prediction_frame(
        data
    )

    y = (
        pd.to_numeric(
            data[TARGET_COLUMN],
            errors="raise",
        )
        .astype(int)
    )

    print(
        f"\nSelected ML features: "
        f"{len(FEATURE_COLUMNS)}"
    )

    print(
        "\nCategorical features:"
    )

    for column in CATEGORICAL_FEATURES:
        print(
            f"  - {column}"
        )

    print(
        "\nNumerical features:"
    )

    for column in NUMERIC_FEATURES:
        print(
            f"  - {column}"
        )

    print(
        "\nTarget distribution:"
    )

    print(
        y.value_counts(
            normalize=False
        )
    )

    print(
        "\nTarget percentage:"
    )

    print(
        y.value_counts(
            normalize=True
        ).round(4)
    )

    # --------------------------------------------------------
    # TRAIN / TEST SPLIT
    # --------------------------------------------------------

    X_train, X_test, y_train, y_test = (
        train_test_split(
            X,
            y,
            test_size=TEST_SIZE,
            stratify=y,
            random_state=RANDOM_STATE,
        )
    )

    print()
    print(
        f"Training rows: "
        f"{len(X_train):,}"
    )

    print(
        f"Testing rows: "
        f"{len(X_test):,}"
    )

    # --------------------------------------------------------
    # OUTPUT DIRECTORY
    # --------------------------------------------------------

    MODELS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    metrics_rows = []

    evaluation = {}

    feature_importances = {}

    best_parameters = {}

    training_times = {}

    cv_scores = {}

    # --------------------------------------------------------
    # MODEL LOOP
    # --------------------------------------------------------

    configs = get_model_configs()

    for model_key, config in configs.items():

        pipeline, cv_score, params, elapsed = (
            train_model(
                model_key,
                config["estimator"],
                config["params"],
                X_train,
                y_train,
            )
        )

        # ----------------------------------------------------
        # PREDICTIONS
        # ----------------------------------------------------

        predictions = (
            pipeline.predict(
                X_test
            ).astype(int)
        )

        probabilities = (
            pipeline.predict_proba(
                X_test
            )[:, 1]
        )

        # ----------------------------------------------------
        # METRICS
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # CONFUSION MATRIX
        # ----------------------------------------------------

        cm = confusion_matrix(
            y_test,
            predictions,
        )

        # ----------------------------------------------------
        # STORE METRICS
        # ----------------------------------------------------

        metrics_rows.append(
            {
                "model_key": model_key,
                "model": MODEL_LABELS[
                    model_key
                ],
                "accuracy": round(
                    float(accuracy),
                    4,
                ),
                "precision": round(
                    float(precision),
                    4,
                ),
                "recall": round(
                    float(recall),
                    4,
                ),
                "f1_score": round(
                    float(f1),
                    4,
                ),
                "roc_auc": round(
                    float(roc_auc),
                    4,
                ),
                "cv_roc_auc": round(
                    float(cv_score),
                    4,
                ),
                "training_time_seconds": round(
                    float(elapsed),
                    2,
                ),
            }
        )

        # ----------------------------------------------------
        # EVALUATION DATA
        # ----------------------------------------------------

        evaluation[
            model_key
        ] = {
            "y_true": y_test.tolist(),

            "y_pred": predictions.tolist(),

            "y_score": probabilities.tolist(),

            "confusion_matrix": cm.tolist(),

            "classification_report":
                classification_report(
                    y_test,
                    predictions,
                    output_dict=True,
                    zero_division=0,
                ),
        }

        # ----------------------------------------------------
        # FEATURE IMPORTANCE
        # ----------------------------------------------------

        print(
            f"\nCalculating feature importance "
            f"for {MODEL_LABELS[model_key]}..."
        )

        feature_importances[
            model_key
        ] = extract_feature_importance(
            pipeline,
            X_test,
            y_test,
            model_key,
        )

        # ----------------------------------------------------
        # SAVE MODEL
        # ----------------------------------------------------

        output_path = model_path(
            model_key
        )

        joblib.dump(
            pipeline,
            output_path,
        )

        print(
            f"Saved model: "
            f"{output_path}"
        )

        # ----------------------------------------------------
        # STORE ADDITIONAL INFORMATION
        # ----------------------------------------------------

        best_parameters[
            model_key
        ] = params

        training_times[
            model_key
        ] = elapsed

        cv_scores[
            model_key
        ] = cv_score

        print()
        print(
            f"{MODEL_LABELS[model_key]}"
        )
        print(
            f"Accuracy : {accuracy:.4f}"
        )
        print(
            f"Precision: {precision:.4f}"
        )
        print(
            f"Recall   : {recall:.4f}"
        )
        print(
            f"F1 Score : {f1:.4f}"
        )
        print(
            f"ROC-AUC  : {roc_auc:.4f}"
        )

    # ========================================================
    # MODEL COMPARISON
    # ========================================================

    metrics = pd.DataFrame(
        metrics_rows
    )

    metrics = metrics.sort_values(
        [
            "roc_auc",
            "f1_score",
        ],
        ascending=False,
    ).reset_index(
        drop=True
    )

    # Rank models
    metrics.insert(
        0,
        "rank",
        range(
            1,
            len(metrics) + 1,
        ),
    )

    # --------------------------------------------------------
    # SAVE METRICS
    # --------------------------------------------------------

    metrics_path = (
        MODELS_DIR
        / "model_metrics.csv"
    )

    metrics.to_csv(
        metrics_path,
        index=False,
    )

    # --------------------------------------------------------
    # SAVE EVALUATION
    # --------------------------------------------------------

    evaluation_path = (
        MODELS_DIR
        / "evaluation_results.joblib"
    )

    joblib.dump(
        evaluation,
        evaluation_path,
    )

    # --------------------------------------------------------
    # SAVE FEATURE IMPORTANCE
    # --------------------------------------------------------

    importance_path = (
        MODELS_DIR
        / "feature_importance.json"
    )

    importance_path.write_text(
        json.dumps(
            feature_importances,
            indent=2,
        ),
        encoding="utf-8",
    )

    # --------------------------------------------------------
    # SAVE HYPERPARAMETERS
    # --------------------------------------------------------

    parameters_path = (
        MODELS_DIR
        / "best_parameters.json"
    )

    parameters_path.write_text(
        json.dumps(
            best_parameters,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )

    # ========================================================
    # CHAMPION MODEL
    # ========================================================

    champion = metrics.iloc[0]

    champion_key = str(
        champion["model_key"]
    )

    champion_model = str(
        champion["model"]
    )

    # --------------------------------------------------------
    # METADATA
    # --------------------------------------------------------

    metadata = {

        "target_column":
            TARGET_COLUMN,

        "feature_columns":
            FEATURE_COLUMNS,

        "categorical_features":
            CATEGORICAL_FEATURES,

        "numeric_features":
            NUMERIC_FEATURES,

        "train_rows":
            int(len(X_train)),

        "test_rows":
            int(len(X_test)),

        "test_size":
            TEST_SIZE,

        "cv_folds":
            CV_FOLDS,

        "random_search_iterations":
            N_ITER_SEARCH,

        "random_state":
            RANDOM_STATE,

        "trained_at_utc":
            datetime.now(
                timezone.utc
            ).isoformat(),

        "champion_key":
            champion_key,

        "champion_model":
            champion_model,

        "champion_accuracy":
            float(
                champion["accuracy"]
            ),

        "champion_precision":
            float(
                champion["precision"]
            ),

        "champion_recall":
            float(
                champion["recall"]
            ),

        "champion_f1":
            float(
                champion["f1_score"]
            ),

        "champion_roc_auc":
            float(
                champion["roc_auc"]
            ),

        "excluded_post_outcome_features": [
            "Migration_Success",
            "Migration_Interrupted",
            "Interrupted_Reason",
            "Recovery_Location_Known",
            "Recovery_Time_days",
            "Nesting_Success",
        ],

        "excluded_analysis_features": [
            "Cluster",
            "Migration_Behavior_Cluster",
            "Path",
        ],

        "excluded_redundant_features": [
            "Migration_Success",
            "Migration_Success_Num",
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
        ],

        "tracking_features_reviewed": [
            "Tag_Battery_Level_%",
            "Tracking_Quality",
        ],
    }

    metadata_path = (
        MODELS_DIR
        / "model_metadata.json"
    )

    metadata_path.write_text(
        json.dumps(
            metadata,
            indent=2,
        ),
        encoding="utf-8",
    )

    # ========================================================
    # FINAL OUTPUT
    # ========================================================

    print()
    print("=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)

    print()
    print(
        "MODEL COMPARISON"
    )

    print(
        metrics[
            [
                "rank",
                "model",
                "accuracy",
                "precision",
                "recall",
                "f1_score",
                "roc_auc",
                "cv_roc_auc",
                "training_time_seconds",
            ]
        ].to_string(
            index=False
        )
    )

    print()
    print(
        "CHAMPION MODEL"
    )

    print(
        f"Model    : {champion_model}"
    )

    print(
        f"Accuracy : "
        f"{champion['accuracy']:.4f}"
    )

    print(
        f"Precision: "
        f"{champion['precision']:.4f}"
    )

    print(
        f"Recall   : "
        f"{champion['recall']:.4f}"
    )

    print(
        f"F1 Score : "
        f"{champion['f1_score']:.4f}"
    )

    print(
        f"ROC-AUC  : "
        f"{champion['roc_auc']:.4f}"
    )

    print()
    print(
        "Saved artifacts:"
    )

    print(
        f"  {metrics_path}"
    )

    print(
        f"  {evaluation_path}"
    )

    print(
        f"  {importance_path}"
    )

    print(
        f"  {parameters_path}"
    )

    print(
        f"  {metadata_path}"
    )

    print(
        f"  {MODELS_DIR}/*.joblib"
    )

    print("=" * 80)

    return metrics


# ============================================================
# COMMAND-LINE ENTRY POINT
# ============================================================

if __name__ == "__main__":

    results = train_all_models()

    print()
    print(
        "Done."
    )