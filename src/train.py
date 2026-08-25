"""
End-to-end training pipeline for Bird Migration Success Prediction.

Models:
1. Logistic Regression
2. Decision Tree
3. K-Nearest Neighbors
4. Support Vector Machine
5. Gradient Boosting
6. Random Forest

The pipeline:
- Loads the processed dataset
- Uses only approved predictors from src.modeling
- Performs stratified train/test split
- Builds preprocessing pipelines
- Performs efficient hyperparameter tuning
- Evaluates all models on the same untouched test set
- Saves trained models and evaluation artifacts
"""

from __future__ import annotations

import json
import time
import warnings
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    train_test_split,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
)

from src.modeling import (
    CATEGORICAL_FEATURES,
    DATA_PATH,
    FEATURE_COLUMNS,
    MODEL_LABELS,
    MODELS_DIR,
    NUMERIC_FEATURES,
    TARGET_COLUMN,
    model_path,
)


warnings.filterwarnings("ignore")


# ============================================================
# CONFIGURATION
# ============================================================

RANDOM_STATE = 42

TEST_SIZE = 0.20

# 3-fold is deliberately used because of the user's
# 8 GB RAM + i3 laptop.
CV_FOLDS = 3

# Keep this at 2 for an 8 GB machine.
N_JOBS = 2

RANDOM_SEARCH_ITERATIONS = 12

PLOTS_DIR = MODELS_DIR / "plots"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# PRINT HELPERS
# ============================================================

def print_header(title: str) -> None:
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


# ============================================================
# LOAD DATA
# ============================================================

def load_dataset() -> pd.DataFrame:

    print_header("LOADING DATA")

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found:\n{DATA_PATH}"
        )

    data = pd.read_csv(DATA_PATH)

    print(f"Dataset path : {DATA_PATH}")
    print(f"Shape        : {data.shape}")
    print()

    print("Target distribution:")
    print(data[TARGET_COLUMN].value_counts(dropna=False))

    return data


# ============================================================
# DATA VALIDATION
# ============================================================

def validate_dataset(data: pd.DataFrame) -> None:

    print_header("DATA VALIDATION")

    missing_features = [
        col
        for col in FEATURE_COLUMNS
        if col not in data.columns
    ]

    if missing_features:
        raise ValueError(
            "The following approved features are missing:\n"
            + "\n".join(missing_features)
        )

    if TARGET_COLUMN not in data.columns:
        raise ValueError(
            f"Target column '{TARGET_COLUMN}' not found."
        )

    print(f"Target column        : {TARGET_COLUMN}")
    print(f"Approved features    : {len(FEATURE_COLUMNS)}")
    print(f"Categorical features : {len(CATEGORICAL_FEATURES)}")
    print(f"Numeric features     : {len(NUMERIC_FEATURES)}")

    duplicates = data.duplicated().sum()

    print(f"Duplicate rows       : {duplicates:,}")

    missing_target = data[TARGET_COLUMN].isna().sum()

    print(f"Missing target       : {missing_target:,}")

    if missing_target > 0:
        raise ValueError(
            "Target contains missing values. "
            "Fix the dataset before training."
        )

    unique_target = sorted(
        data[TARGET_COLUMN].dropna().unique().tolist()
    )

    print(f"Target classes       : {unique_target}")

    if not set(unique_target).issubset({0, 1}):
        raise ValueError(
            "Target must contain only 0 and 1."
        )


# ============================================================
# PREPROCESSING
# ============================================================

def build_preprocessor() -> ColumnTransformer:

    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median"),
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
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=True,
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
# MODEL DEFINITIONS
# ============================================================

def build_model_searches() -> dict[str, tuple]:

    searches = {}

    # --------------------------------------------------------
    # 1. LOGISTIC REGRESSION
    # --------------------------------------------------------

    logistic_pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                build_preprocessor(),
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=3000,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )

    logistic_params = {
        "classifier__C": [
            0.01,
            0.1,
            0.5,
            1,
            2,
            5,
            10,
            20,
        ],
        "classifier__class_weight": [
            None,
            "balanced",
        ],
        "classifier__solver": [
            "lbfgs",
        ],
    }

    searches["logistic_regression"] = (
        logistic_pipeline,
        logistic_params,
    )

    # --------------------------------------------------------
    # 2. DECISION TREE
    # --------------------------------------------------------

    tree_pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                build_preprocessor(),
            ),
            (
                "classifier",
                DecisionTreeClassifier(
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )

    tree_params = {
        "classifier__max_depth": [
            3,
            5,
            7,
            10,
            15,
            20,
            None,
        ],
        "classifier__min_samples_split": [
            2,
            5,
            10,
            20,
        ],
        "classifier__min_samples_leaf": [
            1,
            2,
            4,
            8,
        ],
        "classifier__criterion": [
            "gini",
            "entropy",
            "log_loss",
        ],
        "classifier__class_weight": [
            None,
            "balanced",
        ],
    }

    searches["decision_tree"] = (
        tree_pipeline,
        tree_params,
    )

    # --------------------------------------------------------
    # 3. KNN
    # --------------------------------------------------------

    knn_pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                build_preprocessor(),
            ),
            (
                "classifier",
                KNeighborsClassifier(),
            ),
        ]
    )

    knn_params = {
        "classifier__n_neighbors": [
            3,
            5,
            7,
            9,
            11,
            15,
            21,
            31,
        ],
        "classifier__weights": [
            "uniform",
            "distance",
        ],
        "classifier__p": [
            1,
            2,
        ],
    }

    searches["knn"] = (
        knn_pipeline,
        knn_params,
    )

    # --------------------------------------------------------
    # 4. SVM
    # --------------------------------------------------------

    svm_pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                build_preprocessor(),
            ),
            (
                "classifier",
                SVC(
                    probability=True,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )

    svm_params = {
        "classifier__C": [
            0.1,
            0.5,
            1,
            2,
            5,
            10,
        ],
        "classifier__gamma": [
            "scale",
            "auto",
        ],
        "classifier__kernel": [
            "rbf",
        ],
        "classifier__class_weight": [
            None,
            "balanced",
        ],
    }

    searches["svm"] = (
        svm_pipeline,
        svm_params,
    )

    # --------------------------------------------------------
    # 5. GRADIENT BOOSTING
    # --------------------------------------------------------

    gb_pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                build_preprocessor(),
            ),
            (
                "classifier",
                GradientBoostingClassifier(
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )

    gb_params = {
        "classifier__n_estimators": [
            50,
            100,
            150,
            200,
        ],
        "classifier__learning_rate": [
            0.01,
            0.03,
            0.05,
            0.1,
            0.15,
        ],
        "classifier__max_depth": [
            2,
            3,
            4,
            5,
        ],
        "classifier__min_samples_split": [
            2,
            5,
            10,
        ],
        "classifier__min_samples_leaf": [
            1,
            2,
            4,
        ],
        "classifier__subsample": [
            0.8,
            1.0,
        ],
    }

    searches["gradient_boosting"] = (
        gb_pipeline,
        gb_params,
    )

    # --------------------------------------------------------
    # 6. RANDOM FOREST
    # --------------------------------------------------------

    rf_pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                build_preprocessor(),
            ),
            (
                "classifier",
                RandomForestClassifier(
                    random_state=RANDOM_STATE,
                    n_jobs=N_JOBS,
                ),
            ),
        ]
    )

    rf_params = {
        "classifier__n_estimators": [
            100,
            200,
            300,
            400,
        ],
        "classifier__max_depth": [
            None,
            10,
            15,
            20,
            30,
        ],
        "classifier__min_samples_split": [
            2,
            5,
            10,
        ],
        "classifier__min_samples_leaf": [
            1,
            2,
            4,
        ],
        "classifier__max_features": [
            "sqrt",
            "log2",
            None,
        ],
        "classifier__class_weight": [
            None,
            "balanced",
            "balanced_subsample",
        ],
    }

    searches["random_forest"] = (
        rf_pipeline,
        rf_params,
    )

    return searches


# ============================================================
# EVALUATION
# ============================================================

def evaluate_model(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> tuple[dict, dict]:

    predictions = model.predict(X_test)

    probabilities = model.predict_proba(X_test)[:, 1]

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

    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "roc_auc": roc_auc,
    }

    evaluation = {
        "y_true": np.asarray(y_test),
        "y_pred": np.asarray(predictions),
        "y_score": np.asarray(probabilities),
        "confusion_matrix": cm,
        "classification_report": report,
    }

    return metrics, evaluation


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

def extract_feature_importance(
    pipeline,
) -> list[dict[str, object]]:

    classifier = pipeline.named_steps["classifier"]

    preprocessor = pipeline.named_steps[
        "preprocessor"
    ]

    if hasattr(
        preprocessor,
        "get_feature_names_out",
    ):
        feature_names = (
            preprocessor
            .get_feature_names_out()
        )
    else:
        feature_names = np.array(
            FEATURE_COLUMNS
        )

    if hasattr(
        classifier,
        "feature_importances_",
    ):
        importances = (
            classifier.feature_importances_
        )

    elif hasattr(
        classifier,
        "coef_",
    ):
        importances = np.abs(
            classifier.coef_[0]
        )

    else:
        return []

    if len(feature_names) != len(importances):
        return []

    result = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": importances,
        }
    )

    result = (
        result
        .sort_values(
            "importance",
            ascending=False,
        )
        .head(20)
    )

    return result.to_dict(
        orient="records"
    )


# ============================================================
# FEATURE IMPORTANCE PLOT
# ============================================================

def save_feature_importance_plot(
    model,
    model_key: str,
) -> None:

    importance = extract_feature_importance(
        model
    )

    if not importance:
        return

    df = pd.DataFrame(
        importance
    ).sort_values(
        "importance",
        ascending=True,
    )

    plt.figure(
        figsize=(9, 6)
    )

    plt.barh(
        df["feature"],
        df["importance"],
    )

    plt.xlabel(
        "Importance"
    )

    plt.ylabel(
        "Feature"
    )

    plt.title(
        f"{MODEL_LABELS[model_key]} - Top Predictive Features"
    )

    plt.tight_layout()

    output = (
        PLOTS_DIR
        / f"{model_key}_feature_importance.png"
    )

    plt.savefig(
        output,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close()


# ============================================================
# CONFUSION MATRIX PLOT
# ============================================================

def save_confusion_matrix_plot(
    cm,
    model_key: str,
) -> None:

    plt.figure(
        figsize=(5, 4)
    )

    plt.imshow(cm)

    plt.title(
        f"{MODEL_LABELS[model_key]} - Confusion Matrix"
    )

    plt.xlabel(
        "Predicted"
    )

    plt.ylabel(
        "Actual"
    )

    plt.xticks(
        [0, 1],
        ["Failed", "Successful"],
    )

    plt.yticks(
        [0, 1],
        ["Failed", "Successful"],
    )

    for i in range(2):
        for j in range(2):
            plt.text(
                j,
                i,
                cm[i, j],
                ha="center",
                va="center",
            )

    plt.tight_layout()

    output = (
        PLOTS_DIR
        / f"{model_key}_confusion_matrix.png"
    )

    plt.savefig(
        output,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close()


# ============================================================
# ROC CURVE
# ============================================================

def save_roc_curves(
    evaluations: dict,
) -> None:

    plt.figure(
        figsize=(9, 7)
    )

    for model_key, evaluation in evaluations.items():

        y_true = evaluation["y_true"]

        y_score = evaluation["y_score"]

        fpr, tpr, _ = roc_curve(
            y_true,
            y_score,
        )

        auc = roc_auc_score(
            y_true,
            y_score,
        )

        plt.plot(
            fpr,
            tpr,
            label=(
                f"{MODEL_LABELS[model_key]} "
                f"(AUC={auc:.3f})"
            ),
        )

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
    )

    plt.xlabel(
        "False Positive Rate"
    )

    plt.ylabel(
        "True Positive Rate"
    )

    plt.title(
        "ROC Curve Comparison"
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        PLOTS_DIR / "roc_curve_comparison.png",
        dpi=150,
        bbox_inches="tight",
    )

    plt.close()


# ============================================================
# MAIN TRAINING FUNCTION
# ============================================================

def main():

    overall_start = time.time()

    print_header(
        "BIRD MIGRATION SUCCESS PREDICTION"
    )

    print(
        "Efficient hyperparameter tuning configuration:"
    )

    print(
        f"CV folds              : {CV_FOLDS}"
    )

    print(
        f"Random search trials  : {RANDOM_SEARCH_ITERATIONS}"
    )

    print(
        f"Parallel workers      : {N_JOBS}"
    )

    print(
        f"Test size             : {TEST_SIZE}"
    )

    # --------------------------------------------------------
    # LOAD DATA
    # --------------------------------------------------------

    data = load_dataset()

    validate_dataset(
        data
    )

    # --------------------------------------------------------
    # REMOVE DUPLICATES
    # --------------------------------------------------------

    duplicate_count = data.duplicated().sum()

    if duplicate_count > 0:

        print(
            f"\nRemoving {duplicate_count:,} duplicate rows..."
        )

        data = data.drop_duplicates().reset_index(
            drop=True
        )

    # --------------------------------------------------------
    # PREPARE X / Y
    # --------------------------------------------------------

    X = data[
        FEATURE_COLUMNS
    ].copy()

    y = data[
        TARGET_COLUMN
    ].astype(int)

    print_header(
        "APPROVED FEATURE SET"
    )

    print(
        f"Using {len(FEATURE_COLUMNS)} features:"
    )

    for feature in FEATURE_COLUMNS:
        print(
            f"  - {feature}"
        )

    print()

    print(
        "Target:"
    )

    print(
        y.value_counts()
    )

    print(
        f"\nSuccess rate: {y.mean():.2%}"
    )

    # --------------------------------------------------------
    # TRAIN / TEST SPLIT
    # --------------------------------------------------------

    X_train, X_test, y_train, y_test = (
        train_test_split(
            X,
            y,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            stratify=y,
        )
    )

    print_header(
        "TRAIN / TEST SPLIT"
    )

    print(
        f"Training records : {len(X_train):,}"
    )

    print(
        f"Testing records  : {len(X_test):,}"
    )

    print(
        f"Training success : {y_train.mean():.2%}"
    )

    print(
        f"Testing success  : {y_test.mean():.2%}"
    )

    # --------------------------------------------------------
    # MODEL SEARCH
    # --------------------------------------------------------

    searches = build_model_searches()

    cv = StratifiedKFold(
        n_splits=CV_FOLDS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    results = []

    evaluations = {}

    feature_importance = {}

    best_parameters = {}

    trained_models = {}

    # --------------------------------------------------------
    # TRAIN EACH MODEL
    # --------------------------------------------------------

    for model_key, (
        pipeline,
        parameter_space,
    ) in searches.items():

        print_header(
            f"TRAINING: {MODEL_LABELS[model_key]}"
        )

        print(
            f"Randomized search: "
            f"{RANDOM_SEARCH_ITERATIONS} sampled combinations "
            f"x {CV_FOLDS}-fold CV"
        )

        start = time.time()

        search = RandomizedSearchCV(
            estimator=pipeline,
            param_distributions=parameter_space,
            n_iter=RANDOM_SEARCH_ITERATIONS,
            scoring="roc_auc",
            cv=cv,
            verbose=1,
            random_state=RANDOM_STATE,
            n_jobs=N_JOBS,
            refit=True,
        )

        search.fit(
            X_train,
            y_train,
        )

        elapsed = (
            time.time() - start
        )

        best_model = search.best_estimator_

        metrics, evaluation = (
            evaluate_model(
                best_model,
                X_test,
                y_test,
            )
        )

        trained_models[
            model_key
        ] = best_model

        evaluations[
            model_key
        ] = evaluation

        best_parameters[
            model_key
        ] = search.best_params_

        importance = (
            extract_feature_importance(
                best_model
            )
        )

        feature_importance[
            model_key
        ] = importance

        save_confusion_matrix_plot(
            evaluation[
                "confusion_matrix"
            ],
            model_key,
        )

        save_feature_importance_plot(
            best_model,
            model_key,
        )

        row = {
            "model_key": model_key,
            "model": MODEL_LABELS[
                model_key
            ],
            "accuracy": metrics[
                "accuracy"
            ],
            "precision": metrics[
                "precision"
            ],
            "recall": metrics[
                "recall"
            ],
            "f1_score": metrics[
                "f1_score"
            ],
            "roc_auc": metrics[
                "roc_auc"
            ],
            "training_time_seconds": elapsed,
            "cv_best_score": search.best_score_,
        }

        results.append(
            row
        )

        print()

        print(
            f"Training time : {elapsed:.2f} seconds"
        )

        print(
            f"Accuracy      : {metrics['accuracy']:.4f}"
        )

        print(
            f"Precision     : {metrics['precision']:.4f}"
        )

        print(
            f"Recall        : {metrics['recall']:.4f}"
        )

        print(
            f"F1-score      : {metrics['f1_score']:.4f}"
        )

        print(
            f"ROC-AUC       : {metrics['roc_auc']:.4f}"
        )

        print()

        print(
            "Best parameters:"
        )

        print(
            search.best_params_
        )

        print()

        # Save model
        output_model_path = model_path(
            model_key
        )

        joblib.dump(
            best_model,
            output_model_path,
            compress=3,
        )

        print(
            f"Saved model: {output_model_path}"
        )

    # --------------------------------------------------------
    # MODEL COMPARISON
    # --------------------------------------------------------

    metrics_df = pd.DataFrame(
        results
    )

    # Sort primarily by F1,
    # then ROC-AUC,
    # then accuracy.
    metrics_df = (
        metrics_df
        .sort_values(
            by=[
                "f1_score",
                "roc_auc",
                "accuracy",
            ],
            ascending=False,
        )
        .reset_index(drop=True)
    )

    print_header(
        "FINAL MODEL COMPARISON"
    )

    display_columns = [
        "model",
        "accuracy",
        "precision",
        "recall",
        "f1_score",
        "roc_auc",
        "training_time_seconds",
    ]

    print(
        metrics_df[
            display_columns
        ].to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # CHAMPION MODEL
    # --------------------------------------------------------

    champion = metrics_df.iloc[0]

    champion_key = champion[
        "model_key"
    ]

    print_header(
        "BEST MODEL"
    )

    print(
        f"Best model: "
        f"{champion['model']}"
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
        f"F1-score : "
        f"{champion['f1_score']:.4f}"
    )

    print(
        f"ROC-AUC  : "
        f"{champion['roc_auc']:.4f}"
    )

    # --------------------------------------------------------
    # ROC CURVE
    # --------------------------------------------------------

    save_roc_curves(
        evaluations
    )

    # --------------------------------------------------------
    # SAVE METRICS
    # --------------------------------------------------------

    metrics_path = (
        MODELS_DIR
        / "model_metrics.csv"
    )

    metrics_df.to_csv(
        metrics_path,
        index=False,
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
            feature_importance,
            indent=2,
            default=float,
        ),
        encoding="utf-8",
    )

    # --------------------------------------------------------
    # SAVE BEST PARAMETERS
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

    # --------------------------------------------------------
    # SAVE EVALUATION RESULTS
    # --------------------------------------------------------

    evaluation_path = (
        MODELS_DIR
        / "evaluation_results.joblib"
    )

    joblib.dump(
        evaluations,
        evaluation_path,
        compress=3,
    )

    # --------------------------------------------------------
    # SAVE METADATA
    # --------------------------------------------------------

    metadata = {
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "cv_folds": CV_FOLDS,
        "random_search_iterations": (
            RANDOM_SEARCH_ITERATIONS
        ),
        "n_jobs": N_JOBS,
        "train_rows": int(
            len(X_train)
        ),
        "test_rows": int(
            len(X_test)
        ),
        "total_rows": int(
            len(data)
        ),
        "feature_count": int(
            len(FEATURE_COLUMNS)
        ),
        "features": FEATURE_COLUMNS,
        "categorical_features": (
            CATEGORICAL_FEATURES
        ),
        "numeric_features": (
            NUMERIC_FEATURES
        ),
        "target_column": TARGET_COLUMN,
        "champion_key": champion_key,
        "champion_model": champion[
            "model"
        ],
        "champion_accuracy": float(
            champion["accuracy"]
        ),
        "champion_precision": float(
            champion["precision"]
        ),
        "champion_recall": float(
            champion["recall"]
        ),
        "champion_f1": float(
            champion["f1_score"]
        ),
        "champion_roc_auc": float(
            champion["roc_auc"]
        ),
    }

    metadata_path = (
        MODELS_DIR
        / "model_metadata.json"
    )

    metadata_path.write_text(
        json.dumps(
            metadata,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )

    # --------------------------------------------------------
    # FINAL SUMMARY
    # --------------------------------------------------------

    total_time = (
        time.time()
        - overall_start
    )

    print_header(
        "TRAINING COMPLETED"
    )

    print(
        f"Total training time: "
        f"{total_time / 60:.2f} minutes"
    )

    print()

    print(
        f"Champion model: "
        f"{champion['model']}"
    )

    print(
        f"Champion F1: "
        f"{champion['f1_score']:.4f}"
    )

    print(
        f"Champion ROC-AUC: "
        f"{champion['roc_auc']:.4f}"
    )

    print()

    print(
        "Saved artifacts:"
    )

    print(
        f"  - {metrics_path}"
    )

    print(
        f"  - {importance_path}"
    )

    print(
        f"  - {parameters_path}"
    )

    print(
        f"  - {evaluation_path}"
    )

    print(
        f"  - {metadata_path}"
    )

    print(
        f"  - {PLOTS_DIR}"
    )

    print()

    print(
        "Training finished successfully."
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()