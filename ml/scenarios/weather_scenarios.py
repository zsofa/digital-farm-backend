import argparse
from datetime import date

import numpy as np

from ml.data_loader import get_connection


TEMPERATURE_HISTORY_YEARS = 15
PRECIPITATION_HISTORY_YEARS = 10


def get_next_season_start_year(crop, today=None):
    if today is None:
        today = date.today()

    if crop in ("wheat", "barley"):
        season_start = date(today.year, 9, 1)
    elif crop == "maize":
        season_start = date(today.year, 3, 1)
    else:
        raise ValueError(f"Unsupported crop: {crop}")

    return today.year if today < season_start else today.year + 1

# Get county, crop before targer year
def load_weather_history(county_name, crop, target_season_start_year):
    query = """
        SELECT
            season_start_year,
            season_mean_temperature_c,
            season_precipitation_mm
        FROM processed.weather_seasonal
        WHERE county_name = %s
          AND crop = %s
          AND season_start_year < %s
        ORDER BY season_start_year;
    """

    connection = get_connection()

    try:
        with connection.cursor() as cursor:
            cursor.execute(query, (county_name, crop, target_season_start_year))
            rows = cursor.fetchall()
    finally:
        connection.close()

    if not rows:
        raise ValueError(
            f"No historical weather found for {county_name}, {crop}."
        )

    return [
        {
            "year": int(year),
            "temperature": float(temperature),
            "precipitation": float(precipitation),
        }
        for year, temperature, precipitation in rows
    ]


# Get last 15 -> linear trend -> extrapolation for targer year
# 20. p -> cool devation; 80. p -> hot deviation
# dry hot: expected + hot d.; expected = trend; wet cool: expected + cool d.
def calculate_temperature_scenarios(history, target_year):
    recent = history[-TEMPERATURE_HISTORY_YEARS:]

    if len(recent) < TEMPERATURE_HISTORY_YEARS:
        raise ValueError(
            f"At least {TEMPERATURE_HISTORY_YEARS} temperature seasons are required."
        )

    years = np.array([row["year"] for row in recent], dtype=float)
    temperatures = np.array(
        [row["temperature"] for row in recent], dtype=float
    )

    slope, intercept = np.polyfit(years, temperatures, 1)

    expected_temperature = slope * target_year + intercept

    fitted_temperatures = slope * years + intercept
    residuals = temperatures - fitted_temperatures

    cool_residual = np.percentile(residuals, 20)
    hot_residual = np.percentile(residuals, 80)

    return {
        "dry_hot": expected_temperature + hot_residual,
        "expected": expected_temperature,
        "wet_cool": expected_temperature + cool_residual,
    }


# Get last 10 -> linear trend -> extrapolation for targer year
# dry hot: 20.p; expected = avg; wet cool: 80.p
def calculate_precipitation_scenarios(history):
    recent = history[-PRECIPITATION_HISTORY_YEARS:]

    if len(recent) < PRECIPITATION_HISTORY_YEARS:
        raise ValueError(
            f"At least {PRECIPITATION_HISTORY_YEARS} precipitation seasons are required."
        )

    precipitation = np.array(
        [row["precipitation"] for row in recent], dtype=float
    )

    return {
        "dry_hot": np.percentile(precipitation, 20),
        "expected": np.mean(precipitation),
        "wet_cool": np.percentile(precipitation, 80),
    }


def create_weather_scenarios(county_name, crop, target_season_start_year):
    history = load_weather_history(
        county_name, crop, target_season_start_year
    )

    temperature = calculate_temperature_scenarios(
        history, target_season_start_year
    )
    precipitation = calculate_precipitation_scenarios(history)

    scenarios = {}

    for scenario_name in ("dry_hot", "expected", "wet_cool"):
        scenarios[scenario_name] = {
            "temperature_c": round(float(temperature[scenario_name]), 3),
            "precipitation_mm": round(float(precipitation[scenario_name]), 2),
        }

    return scenarios


def main():
    parser = argparse.ArgumentParser(
        description="Generate seasonal weather scenarios."
    )

    parser.add_argument("--county", required=True)
    parser.add_argument(
        "--crop",
        required=True,
        choices=["wheat", "barley", "maize"],
    )
    parser.add_argument("--target-year", type=int)

    args = parser.parse_args()

    target_year = args.target_year

    if target_year is None:
        target_year = get_next_season_start_year(args.crop)

    scenarios = create_weather_scenarios(
        args.county,
        args.crop,
        target_year,
    )

    print("\nWeather scenarios")
    print(f"County: {args.county}")
    print(f"Crop: {args.crop}")
    print(f"Target season start year: {target_year}")

    for name, values in scenarios.items():
        print(
            f"\n{name.upper()}"
            f"\nTemperature: {values['temperature_c']:.3f} °C"
            f"\nPrecipitation: {values['precipitation_mm']:.2f} mm"
        )


if __name__ == "__main__":
    main()