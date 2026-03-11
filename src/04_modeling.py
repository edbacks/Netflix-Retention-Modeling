"""
04_modeling.py
Netflix Retention Modeling - Modeling

This script:
1. Loads user-level features
2. Validates the modeling dataset
3. Splits data into train and test sets
4. Trains baseline and machine learning models
5. Compares model performance
6. Saves predictions and visual outputs
"""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    ConfusionMatrixDisplay,
    RocCurveDisplay,
)

# ─────────────────────────────────────────
# Project Paths
# ─────────────────────────────────────────

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

INPUT_PATH = DATA_DIR / "user_features.csv"

MODEL_PERFORMANCE_PATH = OUTPUT_DIR / "model_performance.csv"
TEST_PREDICTIONS_PATH = OUTPUT_DIR / "test_predictions.csv"
ROC_CURVE_PATH = OUTPUT_DIR / "roc_curve_comparison.png"
BEST_CONFUSION_MATRIX_PATH = OUTPUT_DIR / "best_model_confusion_matrix.png"
FEATURE_IMPORTANCE_PATH = OUTPUT_DIR / "feature_importance.csv"
FEATURE_IMPORTANCE_PLOT_PATH = OUTPUT_DIR / "feature_importance.png"

RANDOM_STATE = 42
TEST_SIZE = 0.2


def load_data(path: Path) -> pd.DataFrame:
    """Load user-level feature dataset."""
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    return pd.read_csv(path)


def validate_input_data(df: pd.DataFrame) -> None:
    """Validate required columns and target values."""
    required_cols = {"user_id", "churned"}
    missing_cols = required_cols - set(df.columns)

    if missing_cols:
        raise ValueError(f"Missing required columns: {sorted(missing_cols)}")

    if df["churned"].isna().any():
        raise ValueError("Target column 'churned' contains missing values.")

    unique_targets = set(df["churned"].dropna().unique())
    if not unique_targets.issubset({0, 1}):
        raise ValueError(
            f"Target column 'churned' must contain only 0 and 1. Current values: {sorted(unique_targets)}"
        )


def prepare_features_and_target(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series, list[str], pd.Series]:
    """
    Split the dataset into features, target, feature names, and user IDs.
    Use numeric columns only for modeling stability.
    """
    target_col = "churned"
    id_col = "user_id"

    candidate_features = [col for col in df.columns if col not in [id_col, target_col]]
    numeric_features = df[candidate_features].select_dtypes(include=[np.number]).columns.tolist()

    if not numeric_features:
        raise ValueError("No numeric features are available for modeling.")

    X = df[numeric_features].copy()
    y = df[target_col].astype(int).copy()
    user_ids = df[id_col].copy()

    return X, y, numeric_features, user_ids


def split_data(X: pd.DataFrame, y: pd.Series, user_ids: pd.Series):
    """Split data into training and test sets with stratification."""
    return train_test_split(
        X,
        y,
        user_ids,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )


def build_models() -> dict:
    """Create baseline and machine learning model pipelines."""
    models = {
        "Baseline (Most Frequent)": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", DummyClassifier(strategy="most_frequent")),
        ]),
        "Logistic Regression": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(
                random_state=RANDOM_STATE,
                max_iter=1000,
                class_weight="balanced",
            )),
        ]),
        "Random Forest": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", RandomForestClassifier(
                random_state=RANDOM_STATE,
                n_estimators=300,
                min_samples_split=5,
                min_samples_leaf=2,
                class_weight="balanced",
                n_jobs=-1,
            )),
        ]),
    }
    return models


def evaluate_model(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> tuple[dict, np.ndarray, np.ndarray | None]:
    """Evaluate a fitted model on the test set."""
    y_pred = model.predict(X_test)

    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
    else:
        y_prob = None

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1_score": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_prob) if y_prob is not None else np.nan,
        "pr_auc": average_precision_score(y_test, y_prob) if y_prob is not None else np.nan,
    }

    return metrics, y_pred, y_prob


def train_and_compare_models(
    models: dict,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> tuple[pd.DataFrame, dict, dict]:
    """
    Train all models and compare performance.

    Returns:
        results_df: performance summary table
        fitted_models: trained model dictionary
        predictions_dict: predictions and probabilities by model
    """
    results = []
    fitted_models = {}
    predictions_dict = {}

    print("\n" + "=" * 60)
    print("4. Model Training and Evaluation")
    print("=" * 60)

    for model_name, model in models.items():
        print(f"\n[{model_name}]")

        model.fit(X_train, y_train)
        fitted_models[model_name] = model

        metrics, y_pred, y_prob = evaluate_model(model, X_test, y_test)
        metrics["model"] = model_name
        results.append(metrics)

        predictions_dict[model_name] = {
            "y_pred": y_pred,
            "y_prob": y_prob,
        }

        print(
            f"accuracy={metrics['accuracy']:.4f} | "
            f"precision={metrics['precision']:.4f} | "
            f"recall={metrics['recall']:.4f} | "
            f"f1={metrics['f1_score']:.4f} | "
            f"roc_auc={metrics['roc_auc']:.4f} | "
            f"pr_auc={metrics['pr_auc']:.4f}"
        )

    results_df = pd.DataFrame(results)[
        ["model", "accuracy", "precision", "recall", "f1_score", "roc_auc", "pr_auc"]
    ].sort_values(by=["roc_auc", "f1_score"], ascending=False)

    return results_df, fitted_models, predictions_dict


def save_model_performance(results_df: pd.DataFrame, path: Path) -> None:
    """Save model comparison results."""
    results_df.to_csv(path, index=False)

    print("\n" + "=" * 60)
    print("5. Model Performance Summary")
    print("=" * 60)
    print(results_df.round(4).to_string(index=False))
    print(f"\nSaved file: {path}")


def save_test_predictions(
    best_model_name: str,
    user_ids_test: pd.Series,
    y_test: pd.Series,
    predictions_dict: dict,
    path: Path,
) -> None:
    """Save test-set predictions from the best-performing model."""
    y_pred = predictions_dict[best_model_name]["y_pred"]
    y_prob = predictions_dict[best_model_name]["y_prob"]

    pred_df = pd.DataFrame({
        "user_id": user_ids_test.values,
        "actual_churned": y_test.values,
        "predicted_churned": y_pred,
    })

    if y_prob is not None:
        pred_df["predicted_probability"] = y_prob

    pred_df.to_csv(path, index=False)
    print(f"Saved file: {path}")


def plot_confusion_matrix_for_best_model(
    best_model_name: str,
    fitted_models: dict,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    path: Path,
) -> None:
    """Save the confusion matrix for the best-performing model."""
    model = fitted_models[best_model_name]
    y_pred = model.predict(X_test)

    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=ax)
    ax.set_title(f"Confusion Matrix - {best_model_name}")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Saved file: {path}")


def plot_roc_curve(
    fitted_models: dict,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    path: Path,
) -> None:
    """Save ROC curve comparison plot."""
    fig, ax = plt.subplots(figsize=(7, 6))

    for model_name, model in fitted_models.items():
        if hasattr(model, "predict_proba"):
            RocCurveDisplay.from_estimator(model, X_test, y_test, ax=ax, name=model_name)

    ax.set_title("ROC Curve Comparison")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Saved file: {path}")


def save_feature_importance_if_available(
    best_model_name: str,
    fitted_models: dict,
    feature_cols: list[str],
    csv_path: Path,
    plot_path: Path,
) -> None:
    """
    Save feature importance if supported by the best model.
    Currently supports tree-based models with feature_importances_.
    """
    model = fitted_models[best_model_name]
    estimator = model.named_steps["model"]

    if not hasattr(estimator, "feature_importances_"):
        print("Skipping feature importance export because the best model does not support feature_importances_.")
        return

    importances = pd.DataFrame({
        "feature": feature_cols,
        "importance": estimator.feature_importances_,
    }).sort_values(by="importance", ascending=False)

    importances.to_csv(csv_path, index=False)
    print(f"Saved file: {csv_path}")

    top_importances = importances.head(10).sort_values(by="importance", ascending=True)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(top_importances["feature"], top_importances["importance"])
    ax.set_title(f"Top 10 Feature Importances - {best_model_name}")
    ax.set_xlabel("Importance")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Saved file: {plot_path}")


def print_data_summary(y_train: pd.Series, y_test: pd.Series) -> None:
    """Print train/test class balance summary."""
    print("\n" + "=" * 60)
    print("3. Train/Test Split")
    print("=" * 60)
    print(f"Train size        : {len(y_train):,}")
    print(f"Test size         : {len(y_test):,}")
    print(f"Train churn ratio : {y_train.mean() * 100:.1f}%")
    print(f"Test churn ratio  : {y_test.mean() * 100:.1f}%")


def print_key_takeaways(results_df: pd.DataFrame) -> None:
    """Print a concise summary of final results."""
    best_model_row = results_df.iloc[0]

    print("\n" + "=" * 60)
    print("6. Key Takeaways")
    print("=" * 60)
    print(
        f"Best model: {best_model_row['model']} | "
        f"ROC-AUC = {best_model_row['roc_auc']:.4f} | "
        f"PR-AUC = {best_model_row['pr_auc']:.4f} | "
        f"F1 = {best_model_row['f1_score']:.4f}"
    )
    print("Saved test predictions and confusion matrix for the best model.")
    print("Feature importance was also saved if the best model was tree-based.")


def main() -> None:
    print("=" * 60)
    print("1. Loading and Validating Data")
    print("=" * 60)

    df = load_data(INPUT_PATH)
    validate_input_data(df)

    print(f"Total users   : {len(df):,}")
    print(f"Total columns : {df.shape[1]}")
    print(f"Churn rate    : {df['churned'].mean() * 100:.1f}%")

    print("\n" + "=" * 60)
    print("2. Preparing Features and Target")
    print("=" * 60)

    X, y, feature_cols, user_ids = prepare_features_and_target(df)

    print(f"Number of features: {len(feature_cols)}")
    print("Feature columns:")
    for col in feature_cols:
        print(f"  - {col}")

    X_train, X_test, y_train, y_test, user_ids_train, user_ids_test = split_data(X, y, user_ids)
    print_data_summary(y_train, y_test)

    models = build_models()

    print("\nAvailable models:")
    for model_name in models.keys():
        print(f"  - {model_name}")

    results_df, fitted_models, predictions_dict = train_and_compare_models(
        models,
        X_train,
        X_test,
        y_train,
        y_test,
    )

    save_model_performance(results_df, MODEL_PERFORMANCE_PATH)

    best_model_name = results_df.iloc[0]["model"]
    print(f"\nSelected best model: {best_model_name}")

    save_test_predictions(
        best_model_name,
        user_ids_test,
        y_test,
        predictions_dict,
        TEST_PREDICTIONS_PATH,
    )

    plot_confusion_matrix_for_best_model(
        best_model_name,
        fitted_models,
        X_test,
        y_test,
        BEST_CONFUSION_MATRIX_PATH,
    )

    plot_roc_curve(
        fitted_models,
        X_test,
        y_test,
        ROC_CURVE_PATH,
    )

    save_feature_importance_if_available(
        best_model_name,
        fitted_models,
        feature_cols,
        FEATURE_IMPORTANCE_PATH,
        FEATURE_IMPORTANCE_PLOT_PATH,
    )

    print_key_takeaways(results_df)


if __name__ == "__main__":
    main()
