# src/regression.py
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor


def run_regression_modeling(
    data_path="data/processed/hotel_bookings_reg.csv",
    output_dir="reports/figures",
    test_size=0.2,
    n_splits=5,
    random_state=42,
    save_figures=True,
):
    """
    Train and evaluate ADR regression models for the hotel booking project.

    Parameters
    ----------
    data_path : str
        Path to the preprocessed regression CSV file.
    output_dir : str
        Directory where regression figures will be saved.
    test_size : float
        Ratio of data used for the final test set.
    n_splits : int
        Number of folds for K-fold cross validation.
    random_state : int
        Random seed used for repeatable train/test split and models.
    save_figures : bool
        If True, save model comparison and interpretation plots.

    Returns
    -------
    dict
        Dictionary containing CV results, test results, feature importance,
        selected model name, and selected fitted model.
    """
    # Resolve paths from either the project root or explicit user input.
    project_root = Path(__file__).resolve().parents[1]
    data_file = Path(data_path)
    if not data_file.exists() and not data_file.is_absolute():
        data_file = project_root / data_path

    figure_dir = Path(output_dir)
    if not figure_dir.is_absolute():
        figure_dir = project_root / output_dir
    figure_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(data_file)
    if "adr" not in df.columns:
        raise ValueError("The regression dataset must include the target column 'adr'.")

    # Remove target, constant cancellation flag, and possible prediction-time leakage columns.
    assigned_cols = [col for col in df.columns if col.startswith("assigned_room_type_")]
    drop_cols = ["adr", "is_canceled"] + assigned_cols
    X = df.drop(columns=drop_cols, errors="ignore")
    y = df["adr"]

    # Hold out a test set before cross validation and final evaluation.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
    )

    models = {
        "Linear Regression": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", LinearRegression()),
            ]
        ),
        "Decision Tree depth 8": DecisionTreeRegressor(max_depth=8, random_state=random_state),
        "Decision Tree depth 12": DecisionTreeRegressor(max_depth=12, random_state=random_state),
        "Decision Tree depth 16": DecisionTreeRegressor(max_depth=16, random_state=random_state),
        "Random Forest depth 14": RandomForestRegressor(
            n_estimators=50,
            max_depth=14,
            random_state=random_state,
            n_jobs=-1,
        ),
    }

    # Use the same evaluation metrics as the notebook and regression summary.
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    scoring = {
        "mae": "neg_mean_absolute_error",
        "mse": "neg_mean_squared_error",
        "r2": "r2",
    }

    cv_rows = []
    for name, model in models.items():
        # Cross validation is run only on the training set.
        score = cross_validate(
            model,
            X_train,
            y_train,
            cv=cv,
            scoring=scoring,
            return_train_score=False,
        )
        mse = -score["test_mse"].mean()
        cv_rows.append(
            {
                "model": name,
                "cv_mae": -score["test_mae"].mean(),
                "cv_mse": mse,
                "cv_rmse": np.sqrt(mse),
                "cv_r2": score["test_r2"].mean(),
            }
        )

    cv_result = pd.DataFrame(cv_rows).sort_values("cv_rmse").reset_index(drop=True)
    top5_result = cv_result.head(5)

    best_name = cv_result.loc[0, "model"]
    best_model = models[best_name]
    best_model.fit(X_train, y_train)
    y_pred = best_model.predict(X_test)

    # Test metrics show how well the selected model works on unseen rows.
    test_result = pd.DataFrame(
        [
            {
                "model": best_name,
                "test_mae": mean_absolute_error(y_test, y_pred),
                "test_mse": mean_squared_error(y_test, y_pred),
                "test_rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
                "test_r2": r2_score(y_test, y_pred),
            }
        ]
    )

    model_part = best_model.named_steps["model"] if hasattr(best_model, "named_steps") else best_model
    if hasattr(model_part, "feature_importances_"):
        importance_values = model_part.feature_importances_
    elif hasattr(model_part, "coef_"):
        importance_values = np.abs(model_part.coef_)
    else:
        importance_values = np.zeros(len(X.columns))

    importance = (
        pd.DataFrame({"feature": X.columns, "importance": importance_values})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )

    if save_figures:
        # Save plots for the report and final presentation.
        plot_df = top5_result.melt(id_vars="model", value_vars=["cv_mae", "cv_rmse"])
        fig, ax = plt.subplots(figsize=(9, 5))
        sns.barplot(data=plot_df, x="model", y="value", hue="variable", ax=ax)
        ax.set_title("Top 5 Regression Model Comparison")
        ax.set_xlabel("Model / Parameter Combination")
        ax.set_ylabel("Error")
        ax.tick_params(axis="x", rotation=30)
        plt.tight_layout()
        plt.savefig(figure_dir / "regression_model_comparison.png", dpi=300, bbox_inches="tight")
        plt.close()

        pred_df = pd.DataFrame(
            {
                "actual_adr": y_test,
                "predicted_adr": y_pred,
                "residual": y_test - y_pred,
            }
        )
        plot_sample = pred_df.sample(n=min(5000, len(pred_df)), random_state=random_state)

        fig, ax = plt.subplots(figsize=(7, 6))
        sns.scatterplot(data=plot_sample, x="actual_adr", y="predicted_adr", alpha=0.35, s=18, ax=ax)
        min_val = min(plot_sample["actual_adr"].min(), plot_sample["predicted_adr"].min())
        max_val = max(plot_sample["actual_adr"].max(), plot_sample["predicted_adr"].max())
        ax.plot([min_val, max_val], [min_val, max_val], color="red", linestyle="--", linewidth=2)
        ax.set_title(f"Actual vs Predicted ADR: {best_name}")
        ax.set_xlabel("Actual ADR")
        ax.set_ylabel("Predicted ADR")
        plt.tight_layout()
        plt.savefig(figure_dir / "regression_actual_vs_predicted.png", dpi=300, bbox_inches="tight")
        plt.close()

        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        sns.histplot(pred_df["residual"], bins=50, kde=True, ax=axes[0], color="#4C72B0")
        axes[0].axvline(0, color="red", linestyle="--")
        axes[0].set_title("Residual Distribution")
        axes[0].set_xlabel("Actual ADR - Predicted ADR")
        sns.scatterplot(data=plot_sample, x="predicted_adr", y="residual", alpha=0.35, s=18, ax=axes[1])
        axes[1].axhline(0, color="red", linestyle="--")
        axes[1].set_title("Residuals vs Predicted ADR")
        axes[1].set_xlabel("Predicted ADR")
        axes[1].set_ylabel("Residual")
        plt.tight_layout()
        plt.savefig(figure_dir / "regression_residual_analysis.png", dpi=300, bbox_inches="tight")
        plt.close()

        fig, ax = plt.subplots(figsize=(8, 6))
        sns.barplot(data=importance.head(15), x="importance", y="feature", ax=ax, color="#55A868")
        ax.set_title(f"Top 15 Features: {best_name}")
        ax.set_xlabel("Feature Importance")
        ax.set_ylabel("Feature")
        plt.tight_layout()
        plt.savefig(figure_dir / "regression_feature_importance.png", dpi=300, bbox_inches="tight")
        plt.close()

    return {
        "cv_result": cv_result,
        "top5_result": top5_result,
        "test_result": test_result,
        "feature_importance": importance,
        "best_model_name": best_name,
        "best_model": best_model,
        "feature_columns": list(X.columns),
        "dropped_columns": drop_cols,
    }


if __name__ == "__main__":
    result = run_regression_modeling()
    print("Best model:", result["best_model_name"])
    print(result["top5_result"].round(4).to_string(index=False))
    print(result["test_result"].round(4).to_string(index=False))
