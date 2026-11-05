from pathlib import Path

import joblib


from ml.scenarios.features import (
    SCENARIO_FEATURE_COLUMNS,
    TARGET_COLUMN,
)
from ml.training.dataset import load_dataset
from ml.training.model_factory import create_model


ML_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = ML_DIR / "models"
MODEL_FILE = MODEL_DIR / "scenario_xgboost.joblib"



def main():
    print("Loading scenario training dataset...")

    dataframe = load_dataset()

    print(f"Records: {len(dataframe)}")
    print(f"Years: {dataframe['yield_year'].min()}-{dataframe['yield_year'].max()}")

    X = dataframe[SCENARIO_FEATURE_COLUMNS]
    y = dataframe[TARGET_COLUMN]

    model = create_model(
        n_estimators=400,
        learning_rate=0.05,
        max_depth=3,
        subsample=0.8,
        colsample_bytree=1.0,
        min_child_weight=1,
        reg_lambda=5.0,
    )

    print("\nTraining production scenario XGBoost...")
    model.fit(X, y)

    artifact = {
        "model": model,
        "feature_columns": SCENARIO_FEATURE_COLUMNS,
        "training_year_min": int(dataframe["yield_year"].min()),
        "training_year_max": int(dataframe["yield_year"].max()),
        "model_type": "scenario_xgboost_actual_weather",
    }

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, MODEL_FILE)

    print("Training completed.")
    print(f"Model saved to:\n{MODEL_FILE}")


if __name__ == "__main__":
    main()