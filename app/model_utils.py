"""
Model loading and prediction logic.

The model must be loaded ONCE at module level, NOT inside the predict function.
"""

from pathlib import Path

import joblib
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
pipeline = joblib.load(BASE_DIR / "pipeline" / "pipeline.joblib")

FEATURE_COLUMNS = [
    "CreditScore",
    "Geography",
    "Gender",
    "Age",
    "Tenure",
    "Balance",
    "NumOfProducts",
    "HasCrCard",
    "IsActiveMember",
    "EstimatedSalary",
]


def predict_churn(features: list[float | str]) -> int:
    """
    Takes a list of raw feature values and returns a churn prediction (0 or 1).
    """
    input_df = pd.DataFrame([features], columns=FEATURE_COLUMNS)
    prediction = pipeline.predict(input_df)

    return int(prediction[0])


if __name__ == "__main__":
    sample = [619, "France", "Female", 42, 2, 0, 1, 1, 1, 101348.88]
    print(f"Input:      {sample}")
    print(f"Prediction: {predict_churn(sample)}")
