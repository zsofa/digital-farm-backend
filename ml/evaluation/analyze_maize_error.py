# diagnostical script for checking maize problems
# final: precipitation in 2025 has +95 mm difference (expected - actual)

from pathlib import Path

import pandas as pd

from ml.data_loader import get_connection


BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "results"

PREDICTIONS_FILE = RESULTS_DIR / "final_xgboost_test_predictions.csv"
OUTPUT_FILE = RESULTS_DIR / "maize_error_analysis.csv"


def load_weather_data():
    query = """
        SELECT
            t.yield_year,
            t.county_name,
            t.recent_yield_mean_t_ha,
            t.expected_season_temperature_c,
            t.expected_season_precipitation_mm,
            w.season_mean_temperature_c AS actual_temperature_c,
            w.season_precipitation_mm AS actual_precipitation_mm
        FROM ml.training_dataset t
        JOIN processed.weather_seasonal w
          ON w.season_start_year = t.season_start_year
         AND w.county_name = t.county_name
         AND w.crop = t.crop
        WHERE t.crop = 'maize';
    """

    connection = get_connection()

    try:
        with connection.cursor() as cursor:
            cursor.execute(query)
            rows = cursor.fetchall()
            columns = [description.name for description in cursor.description]

        return pd.DataFrame(rows, columns=columns)

    finally:
        connection.close()


def main():
    print("Loading final XGBoost predictions...")

    predictions = pd.read_csv(PREDICTIONS_FILE)
    predictions = predictions[predictions["crop"] == "maize"].copy()

    weather = load_weather_data()

    dataframe = predictions.merge(
        weather,
        on=["yield_year", "county_name"],
        how="left",
    )

    numeric_columns = [
        "actual_yield_t_ha",
        "predicted_yield_t_ha",
        "recent_yield_mean_t_ha",
        "expected_season_temperature_c",
        "expected_season_precipitation_mm",
        "actual_temperature_c",
        "actual_precipitation_mm",
    ]

    for column in numeric_columns:
        dataframe[column] = pd.to_numeric(dataframe[column])

    dataframe["prediction_error_t_ha"] = (
        dataframe["predicted_yield_t_ha"] - dataframe["actual_yield_t_ha"]
    )

    dataframe["temperature_error_c"] = (
        dataframe["expected_season_temperature_c"] - dataframe["actual_temperature_c"]
    )

    dataframe["precipitation_error_mm"] = (
        dataframe["expected_season_precipitation_mm"]
        - dataframe["actual_precipitation_mm"]
    )

    print("\n" + "=" * 80)
    print("MAIZE ERROR ANALYSIS BY YEAR")
    print("=" * 80)

    yearly = dataframe.groupby("yield_year").agg(
        actual_yield=("actual_yield_t_ha", "mean"),
        predicted_yield=("predicted_yield_t_ha", "mean"),
        prediction_bias=("prediction_error_t_ha", "mean"),
        recent_yield=("recent_yield_mean_t_ha", "mean"),
        expected_temperature=("expected_season_temperature_c", "mean"),
        actual_temperature=("actual_temperature_c", "mean"),
        temperature_error=("temperature_error_c", "mean"),
        expected_precipitation=("expected_season_precipitation_mm", "mean"),
        actual_precipitation=("actual_precipitation_mm", "mean"),
        precipitation_error=("precipitation_error_mm", "mean"),
    )

    print(yearly.round(3).to_string())

    print("\n" + "=" * 80)
    print("10 LARGEST MAIZE PREDICTION ERRORS")
    print("=" * 80)

    worst = dataframe.sort_values(
        "absolute_error_t_ha",
        ascending=False,
    ).head(10)

    columns = [
        "yield_year",
        "county_name",
        "actual_yield_t_ha",
        "predicted_yield_t_ha",
        "prediction_error_t_ha",
        "recent_yield_mean_t_ha",
        "expected_season_precipitation_mm",
        "actual_precipitation_mm",
        "expected_season_temperature_c",
        "actual_temperature_c",
    ]

    print(worst[columns].round(3).to_string(index=False))

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(OUTPUT_FILE, index=False)

    print(f"\nFull analysis saved to:\n{OUTPUT_FILE}")


if __name__ == "__main__":
    main()