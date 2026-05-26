# -*- coding: utf-8 -*-
"""
classification.py
Classification modeling utilities for the Hotel Booking Demand project.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier


def _resolve_project_path(path_value: str, project_root: Path) -> Path:
    """Resolve relative paths from the current directory first, then the project root."""
    path = Path(path_value)
    if path.is_absolute() or path.exists():
        return path
    return project_root / path


def run_classification_modeling(
    data_path="data/processed/hotel_bookings_clf.csv",
    output_dir="reports/figures",
    test_size=0.2,
    n_splits=5,
    random_state=42,
    save_figures=True,
    run_tuning=True,
    drop_assigned_room_type=True,
):
    """
    Train and evaluate cancellation classification models for the hotel booking project.

    Parameters
    ----------
    data_path : str
        Path to the preprocessed classification CSV file.
    output_dir : str
        Directory where classification figures will be saved.
    test_size : float
        Ratio of data used for the final test set.
    n_splits : int
        Number of folds for Stratified K-fold cross validation.
    random_state : int
        Random seed used for repeatable train/test split and models.
    save_figures : bool
        If True, save model comparison, confusion matrix, and feature importance plots.
    run_tuning : bool
        If True, tune the best baseline model with GridSearchCV.
    drop_assigned_room_type : bool
        If True, remove assigned_room_type_* columns to reduce prediction-time leakage risk.

    Returns
    -------
    dict
        Dictionary containing CV results, test results, confusion matrix,
        feature importance, selected model name, selected fitted model,
        feature columns, and dropped columns.
    """
    plt.style.use("default")
    sns.set_theme(style="whitegrid")

    project_root = Path(__file__).resolve().parents[1]
    data_file = _resolve_project_path(data_path, project_root)
    figure_dir = _resolve_project_path(output_dir, project_root)
    if save_figures:
        figure_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("STARTING CLASSIFICATION MODELING PIPELINE")
    print("=" * 60)
    print("Data path:", data_file)

    df = pd.read_csv(data_file)
    if "is_canceled" not in df.columns:
        raise ValueError("The classification dataset must include the target column 'is_canceled'.")

    print("Shape:", df.shape)
    print("Missing values:", int(df.isnull().sum().sum()))
    print("Object columns:", len(df.select_dtypes(include="object").columns))
    print("\nTarget count:")
    print(df["is_canceled"].value_counts().sort_index())
    print("\nTarget ratio:")
    print((df["is_canceled"].value_counts(normalize=True).sort_index() * 100).round(2))

    # Feature and target setup
    target = "is_canceled"
    assigned_cols = [col for col in df.columns if col.startswith("assigned_room_type_")]
    drop_cols = [target, "adr"]
    if drop_assigned_room_type:
        drop_cols.extend(assigned_cols)
    dropped_columns = [col for col in drop_cols if col in df.columns]

    X = df.drop(columns=dropped_columns, errors="ignore")
    y = df[target]

    print("\nTarget:", target)
    print("Dropped columns:", len(dropped_columns))
    print("Assigned room columns removed:", len(assigned_cols) if drop_assigned_room_type else 0)
    print("Feature shape:", X.shape)
    print("Target shape:", y.shape)

    # Hold out a stratified test set before cross validation and final evaluation.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    print("\nTrain shape:", X_train.shape)
    print("Test shape:", X_test.shape)
    print("\nTrain target ratio:")
    print((y_train.value_counts(normalize=True).sort_index() * 100).round(2))
    print("\nTest target ratio:")
    print((y_test.value_counts(normalize=True).sort_index() * 100).round(2))

    # Keep model candidates within the course-covered scikit-learn models.
    models = {
        "Logistic Regression": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(max_iter=1000, random_state=random_state)),
            ]
        ),
        "Decision Tree": DecisionTreeClassifier(max_depth=10, random_state=random_state),
        "Random Forest": RandomForestClassifier(
            n_estimators=50,
            max_depth=12,
            random_state=random_state,
            n_jobs=1,
        ),
    }

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    scoring = ["accuracy", "precision", "recall", "f1"]
    parallel_jobs = -1

    print("\nModels:", list(models.keys()))
    print("CV:", cv)
    print("\nRunning cross validation...")

    cv_rows = []
    for name, model in models.items():
        print(f"Running CV for {name}...")
        score = cross_validate(
            model,
            X_train,
            y_train,
            cv=cv,
            scoring=scoring,
            return_train_score=False,
            n_jobs=parallel_jobs,
        )
        cv_rows.append(
            {
                "model": name,
                "cv_accuracy": score["test_accuracy"].mean(),
                "cv_precision": score["test_precision"].mean(),
                "cv_recall": score["test_recall"].mean(),
                "cv_f1": score["test_f1"].mean(),
            }
        )

    cv_result = pd.DataFrame(cv_rows).sort_values("cv_f1", ascending=False).reset_index(drop=True)
    print("\n[Cross Validation Results]")
    print(cv_result.round(4).to_string(index=False))

    if save_figures:
        fig, ax = plt.subplots(figsize=(8, 5))
        plot_df = cv_result.melt(
            id_vars="model",
            value_vars=["cv_accuracy", "cv_precision", "cv_recall", "cv_f1"],
        )
        sns.barplot(data=plot_df, x="model", y="value", hue="variable", ax=ax)
        ax.set_ylim(0, 1)
        ax.set_title(f"{n_splits}-Fold Cross Validation Model Comparison")
        ax.set_xlabel("Model")
        ax.set_ylabel("Score")
        ax.legend(title="Metric", loc="lower right")
        ax.tick_params(axis="x", rotation=15)
        plt.tight_layout()
        comparison_path = figure_dir / "classification_model_comparison.png"
        plt.savefig(comparison_path, dpi=300, bbox_inches="tight")
        plt.close()
        print("Saved figure:", comparison_path)

    # Hyperparameter tuning is applied to the best baseline model by CV F1.
    best_name = cv_result.loc[0, "model"]
    print(f"\nInitial best model for tuning: {best_name}")

    if run_tuning:
        if best_name == "Decision Tree":
            param_grid = {
                "max_depth": [8, 10, 12, 15],
                "min_samples_split": [2, 5, 10],
                "criterion": ["gini", "entropy"],
            }
            tuning_model = DecisionTreeClassifier(random_state=random_state)
        elif best_name == "Random Forest":
            param_grid = {
                "n_estimators": [50, 100],
                "max_depth": [10, 12, 15],
                "min_samples_split": [2, 5],
            }
            tuning_model = RandomForestClassifier(random_state=random_state, n_jobs=1)
        else:
            param_grid = {
                "model__C": [0.1, 1.0, 10.0],
            }
            tuning_model = models["Logistic Regression"]

        print("Running GridSearchCV...")
        print("GridSearchCV parameter grid:", param_grid)
        grid_search = GridSearchCV(
            estimator=tuning_model,
            param_grid=param_grid,
            cv=cv,
            scoring="f1",
            n_jobs=parallel_jobs,
        )
        grid_search.fit(X_train, y_train)

        print("Best parameters found by GridSearchCV:", grid_search.best_params_)
        print(f"Best CV F1-score: {grid_search.best_score_:.4f}")
        models[best_name] = grid_search.best_estimator_
    else:
        print("Skipping GridSearchCV because run_tuning=False.")

    # Final test evaluation
    best_model = models[best_name]
    safe_name = best_name.lower().replace(" ", "_")

    print(f"\nFitting final model: {best_name}")
    best_model.fit(X_train, y_train)
    y_pred = best_model.predict(X_test)

    test_result = pd.DataFrame(
        [
            {
                "model": best_name,
                "test_accuracy": accuracy_score(y_test, y_pred),
                "test_precision": precision_score(y_test, y_pred),
                "test_recall": recall_score(y_test, y_pred),
                "test_f1": f1_score(y_test, y_pred),
            }
        ]
    )

    print("\nBest model after tuning:", best_name if run_tuning else best_name)
    print(test_result.round(4).to_string(index=False))
    print("\nClassification report:")
    print(classification_report(y_test, y_pred, target_names=["Not canceled", "Canceled"]))

    cm = confusion_matrix(y_test, y_pred)
    cm_table = pd.DataFrame(
        cm,
        index=["Actual not canceled", "Actual canceled"],
        columns=["Predicted not canceled", "Predicted canceled"],
    )

    print("\nConfusion matrix:")
    print(cm_table.to_string())

    if save_figures:
        fig, ax = plt.subplots(figsize=(6, 5))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Not canceled", "Canceled"])
        disp.plot(cmap="Blues", values_format="d", ax=ax)
        ax.set_title(f"Confusion Matrix: {best_name}")
        plt.tight_layout()
        confusion_path = figure_dir / f"confusion_matrix_{safe_name}.png"
        plt.savefig(confusion_path, dpi=300, bbox_inches="tight")
        plt.close()
        print("Saved figure:", confusion_path)

    # Feature interpretation
    model_part = best_model.named_steps["model"] if hasattr(best_model, "named_steps") else best_model

    if hasattr(model_part, "feature_importances_"):
        importance_values = model_part.feature_importances_
        importance_label = "Feature Importance"
    elif hasattr(model_part, "coef_"):
        importance_values = np.abs(model_part.coef_[0])
        importance_label = "Absolute Coefficient"
    else:
        importance_values = np.zeros(len(X.columns))
        importance_label = "Importance"

    importance = (
        pd.DataFrame({"feature": X.columns, "importance": importance_values})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )

    print("\n[Top 15 Features]")
    print(importance.head(15).round(4).to_string(index=False))

    if save_figures:
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.barplot(data=importance.head(15), x="importance", y="feature", ax=ax, color="#4C72B0")
        ax.set_title(f"Top 15 Features: {best_name}")
        ax.set_xlabel(importance_label)
        ax.set_ylabel("Feature")
        plt.tight_layout()
        importance_path = figure_dir / "classification_feature_importance.png"
        plt.savefig(importance_path, dpi=300, bbox_inches="tight")
        plt.close()
        print("Saved figure:", importance_path)

    print("\n" + "=" * 50)
    print("Final selected model:", best_name)
    print("Number of input features:", X.shape[1])
    print("Excluded adr:", "adr" in dropped_columns)
    print("Excluded assigned_room_type columns:", len(assigned_cols) if drop_assigned_room_type else 0)
    print("=" * 50)

    return {
        "cv_result": cv_result,
        "test_result": test_result,
        "confusion_matrix": cm_table,
        "feature_importance": importance,
        "best_model_name": best_name,
        "best_model": best_model,
        "feature_columns": list(X.columns),
        "dropped_columns": dropped_columns,
    }


def main():
    """Run classification modeling with default project settings."""
    result = run_classification_modeling()
    print("Best model:", result["best_model_name"])
    print(result["cv_result"].round(4).to_string(index=False))
    print(result["test_result"].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
