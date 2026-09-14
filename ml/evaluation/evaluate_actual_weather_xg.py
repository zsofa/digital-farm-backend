# Since the ml-s do not predict better then the baseline
# checking what would happer if we know the weather exactly

# expected weather: MAE 0.950, R2: -0.296
# actual weather:  MAE 0.633, R2: +0.432!!!!!!!

from pathlib import Path

import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from ml.data_loader import FEATURE_COLUMNS, TARGET_COLUMN, get_connection, get_complete_years
from ml.train.train_xgboost import create_model


BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "results"
OUTPUT_FILE = RESULTS_DIR / "actual_weather_xgboost_predictions.csv"

TEST_YEAR_COUNT = 3


def load_actual_weather_dataset():
    query = """
        SELECT
            t.yield_year,
            t.season_start_year,
            t.county_name,
            t.crop,
            t.soil_ph,
            t.soil_soc_g_kg,
            t.soil_clay_pct,

            w.season_mean_temperature_c AS expected_season_temperature_c,
            w.season_precipitation_mm AS expected_season_precipitation_mm,

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

        dataframe = pd.DataFrame(rows, columns=columns)

    finally:
        connection.close()

    numeric_columns = [
        "soil_ph",
        "soil_soc_g_kg",
        "soil_clay_pct",
        "expected_season_temperature_c",
        "expected_season_precipitation_mm",
        "recent_yield_mean_t_ha",
        "average_yield_t_ha",
    ]

    for column in numeric_columns:
        dataframe[column] = pd.to_numeric(dataframe[column])

    return dataframe


def calculate_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    r2 = r2_score(y_true, y_pred)

    return mae, rmse, r2


def print_metrics(title, dataframe):
    mae, rmse, r2 = calculate_metrics(
        dataframe["actual_yield_t_ha"],
        dataframe["predicted_yield_t_ha"],
    )

    print(f"\n{title}")
    print(f"Records: {len(dataframe)}")
    print(f"MAE:  {mae:.3f} t/ha")
    print(f"RMSE: {rmse:.3f} t/ha")
    print(f"R²:   {r2:.3f}")


def main():
    print("Loading dataset with ACTUAL seasonal weather...")

    dataframe = load_actual_weather_dataset()

    print(f"Records loaded: {len(dataframe)}")

    complete_years = get_complete_years(dataframe)
    test_years = complete_years[-TEST_YEAR_COUNT:]
    train_years = complete_years[:-TEST_YEAR_COUNT]

    train = dataframe[dataframe["yield_year"].isin(train_years)].copy()
    test = dataframe[dataframe["yield_year"].isin(test_years)].copy()

    print(f"Train years: {train_years[0]}-{train_years[-1]}")
    print(f"Test years:  {test_years}")
    print(f"Train records: {len(train)}")
    print(f"Test records:  {len(test)}")

    X_train = train[FEATURE_COLUMNS]
    y_train = train[TARGET_COLUMN]
    X_test = test[FEATURE_COLUMNS]

    model = create_model(
        n_estimators=400,
        learning_rate=0.05,
        max_depth=3,
        subsample=0.8,
        colsample_bytree=1.0,
        min_child_weight=1,
        reg_lambda=5.0,
    )

    print("\nTraining XGBoost with actual seasonal weather...")
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    results = test[["yield_year", "county_name", "crop", TARGET_COLUMN]].copy()
    results = results.rename(columns={TARGET_COLUMN: "actual_yield_t_ha"})
    results["predicted_yield_t_ha"] = predictions
    results["absolute_error_t_ha"] = (
        results["actual_yield_t_ha"] - results["predicted_yield_t_ha"]
    ).abs()

    print("\n" + "=" * 65)
    print("ACTUAL-WEATHER XGBOOST DIAGNOSTIC")
    print("=" * 65)

    print_metrics("OVERALL", results)

    print("\n" + "=" * 65)
    print("BY CROP")
    print("=" * 65)

    for crop in sorted(results["crop"].unique()):
        print_metrics(crop.upper(), results[results["crop"] == crop])

    print("\n" + "=" * 65)
    print("BY YEAR")
    print("=" * 65)

    for year in sorted(results["yield_year"].unique()):
        print_metrics(str(year), results[results["yield_year"] == year])

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUTPUT_FILE, index=False)

    print(f"\nDetailed results saved to:\n{OUTPUT_FILE}")
    print("\nThis is a diagnostic experiment, NOT a production forecasting model.")


if __name__ == "__main__":
    main()