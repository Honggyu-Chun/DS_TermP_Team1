"""
pipeline.py
Project-level pipeline runner for the Hotel Booking Demand term project.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.classification import run_classification_modeling
from src.processing import preprocess_hotel_booking_data
from src.regression import run_regression_modeling


def run_full_pipeline(
    raw_data_path="data/raw/hotel_bookings.csv",
    processed_dir="data/processed",
    figure_dir="reports/figures",
    run_classification=True,
    run_regression=True,
    run_classification_tuning=True,
    save_figures=True,
):
    """
    Run preprocessing and optional modeling steps from a single top-level function.

    Parameters
    ----------
    raw_data_path : str
        Path to the raw Kaggle Hotel Booking Demand CSV file.
    processed_dir : str
        Directory where processed classification and regression datasets are saved.
    figure_dir : str
        Directory where modeling figures are saved.
    run_classification : bool
        If True, run cancellation classification modeling.
    run_regression : bool
        If True, run ADR regression modeling.
    run_classification_tuning : bool
        If True, run GridSearchCV in the classification stage.
    save_figures : bool
        If True, save modeling figures.

    Returns
    -------
    dict
        Dictionary containing preprocessing output and optional modeling outputs.
    """
    print("=" * 60)
    print("STARTING FULL HOTEL BOOKING PIPELINE")
    print("=" * 60)

    preprocessing_result = preprocess_hotel_booking_data(
        raw_data_path=raw_data_path,
        output_dir=processed_dir,
    )

    result = {"preprocessing": preprocessing_result}

    if run_classification:
        result["classification"] = run_classification_modeling(
            data_path=preprocessing_result["classification_path"],
            output_dir=figure_dir,
            save_figures=save_figures,
            run_tuning=run_classification_tuning,
        )

    if run_regression:
        result["regression"] = run_regression_modeling(
            data_path=preprocessing_result["regression_path"],
            output_dir=figure_dir,
            save_figures=save_figures,
        )

    print("=" * 60)
    print("FULL HOTEL BOOKING PIPELINE FINISHED")
    print("=" * 60)

    return result


if __name__ == "__main__":
    run_full_pipeline()
