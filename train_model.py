"""
train_model.py

TODO:
- Load diabetes dataset
- Train a RandomForestRegressor (or any regression model)
- Save model to models/diabetes_model.pkl
- Print evaluation metrics
"""

import os
# TODO: import necessary libraries (numpy, pandas, sklearn, joblib, etc.)
import json
from pathlib import Path
import joblib
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split


def train_model():
    # TODO: Load diabetes dataset
    X, y = load_diabetes(return_X_y=True)
    feature_names = ["age", "sex", "bmi", "bp", "s1", "s2", "s3", "s4", "s5", "s6"]

    # TODO: Split data into train/test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # TODO: Train a regression model
    model = RandomForestRegressor(
        n_estimators=300,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    # TODO: Evaluate model (R^2, MAE, RMSE)
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = (mean_squared_error(y_test, y_pred)) ** 0.5  # compatible everywhere

    print("=== Evaluation on test set ===")
    print(f"R^2:  {r2:.4f}")
    print(f"MAE:  {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")

    # TODO: Save model to models/diabetes_model.pkl
    os.makedirs("models", exist_ok=True)
    # joblib.dump(model, "models/diabetes_model.pkl")
    model_path = Path("models") / "diabetes_model.pkl"
    joblib.dump(model, model_path)

    metadata = {
        "model": "RandomForestRegressor(n_estimators=300, random_state=42)",
        "feature_names": feature_names,
        "test_metrics": {"r2": r2, "mae": mae, "rmse": rmse},
        "notes": "Inputs must be the 10 standardized features of the sklearn diabetes dataset.",
    }
    with open(Path("models") / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved model to: {model_path.resolve()}")


if __name__ == "__main__":
    train_model()
