from pathlib import Path

import joblib
import pandas as pd

from ml.data_loader import TARGET_COLUMN, get_connection
from ml.scenarios.features import SCENARIO_FEATURE_COLUMNS
from ml.train.train_xgboost import create_model


ML_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = ML_DIR / "models"
MODEL_FILE = MODEL_DIR / "scenario_xgboost.joblib"


def load_dataset():
    query = """
        SELECT
            t.yield_year,
            t.county_name,
            t.crop,
            t.soil_ph,
            t.soil_soc_g_kg,
            t.soil_clay_pct,
            w.season_mean_temperature_c AS season_temperature_c,
            w.season_precipitation_mm AS season_precipitation_mm,
            t.recent_yield_mean_t_ha,
            t.average_yield_t_ha
        FROM ml.training_dataset t
        JOIN processed.weather_seasonal w
          ON w.season_start_year = t.season_start_year
         AND w.county_name = t.county_name
         AND w.crop = t.crop
        ORDER BY t.yield_year, t.county_name, t.crop;
    """

    connection = get_connection()

    try:
        with connection.cursor() as cursor:
            cursor.execute(query)
            rows = cursor.fetchall()
            columns = [description.name for description in cursor.description]
    finally:
        connection.close()

    dataframe = pd.DataFrame(rows, columns=columns)

    numeric_columns = [
        "soil_ph",
        "soil_soc_g_kg",
        "soil_clay_pct",
        "season_temperature_c",
        "season_precipitation_mm",
        "recent_yield_mean_t_ha",
        TARGET_COLUMN,
    ]

    for column in numeric_columns:
        dataframe[column] = pd.to_numeric(dataframe[column])

    return dataframe


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