import argparse
from pathlib import Path

import joblib
import pandas as pd

from ml.data_loader import get_connection
from ml.scenarios.features import SCENARIO_FEATURE_COLUMNS
from ml.scenarios.weather_scenarios import (
    create_weather_scenarios,
    get_next_season_start_year,
)


ML_DIR = Path(__file__).resolve().parent.parent
MODEL_FILE = ML_DIR / "models" / "scenario_xgboost.joblib"


def load_soil(county_name):
    query = """
        SELECT soil_ph, soil_soc_g_kg, soil_clay_pct
        FROM processed.soil
        WHERE county_name = %s;
    """

    connection = get_connection()

    try:
        with connection.cursor() as cursor:
            cursor.execute(query, (county_name,))
            row = cursor.fetchone()
    finally:
        connection.close()

    if row is None:
        raise ValueError(f"No soil data found for {county_name}.")

    return {
        "soil_ph": float(row[0]),
        "soil_soc_g_kg": float(row[1]),
        "soil_clay_pct": float(row[2]),
    }


def load_recent_yield(county_name, crop, target_season_start_year):
    target_yield_year = (
        target_season_start_year
        if crop == "maize"
        else target_season_start_year + 1
    )

    query = """
        SELECT year, average_yield_t_ha
        FROM processed.crop_production
        WHERE county_name = %s
          AND crop = %s
          AND year < %s
          AND average_yield_t_ha IS NOT NULL
        ORDER BY year DESC
        LIMIT 5;
    """

    connection = get_connection()

    try:
        with connection.cursor() as cursor:
            cursor.execute(query, (county_name, crop, target_yield_year))
            rows = cursor.fetchall()
    finally:
        connection.close()

    if len(rows) < 5:
        raise ValueError("At least 5 historical yield records are required.")

    values = [float(row[1]) for row in rows]
    years = sorted(int(row[0]) for row in rows)

    return sum(values) / len(values), years


def main():
    parser = argparse.ArgumentParser(description="Run yield scenario simulation.")
    parser.add_argument("--county", required=True)
    parser.add_argument("--crop", required=True, choices=["wheat", "barley", "maize"])
    parser.add_argument("--target-year", type=int)
    args = parser.parse_args()

    target_year = args.target_year
    if target_year is None:
        target_year = get_next_season_start_year(args.crop)

    artifact = joblib.load(MODEL_FILE)
    model = artifact["model"]

    soil = load_soil(args.county)
    recent_yield, recent_years = load_recent_yield(
        args.county, args.crop, target_year
    )

    scenarios = create_weather_scenarios(
        args.county, args.crop, target_year
    )

    rows = []

    for scenario_name in ("dry_hot", "expected", "wet_cool"):
        weather = scenarios[scenario_name]

        rows.append({
            "scenario": scenario_name,
            "crop": args.crop,
            **soil,
            "season_temperature_c": weather["temperature_c"],
            "season_precipitation_mm": weather["precipitation_mm"],
            "recent_yield_mean_t_ha": recent_yield,
        })

    dataframe = pd.DataFrame(rows)
    dataframe["predicted_yield_t_ha"] = model.predict(
        dataframe[SCENARIO_FEATURE_COLUMNS]
    )

    print("\n" + "=" * 65)
    print("YIELD SCENARIO SIMULATION")
    print("=" * 65)

    print(f"County: {args.county}")
    print(f"Crop: {args.crop}")
    print(f"Target season start year: {target_year}")
    print(f"Recent yield years: {recent_years}")
    print(f"Recent yield mean: {recent_yield:.3f} t/ha")

    print(
        f"Soil: pH={soil['soil_ph']:.2f}, "
        f"SOC={soil['soil_soc_g_kg']:.2f} g/kg, "
        f"clay={soil['soil_clay_pct']:.2f}%"
    )

    print("\nScenario results:")

    for _, row in dataframe.iterrows():
        print(
            f"\n{row['scenario'].upper()}"
            f"\nTemperature: {row['season_temperature_c']:.3f} °C"
            f"\nPrecipitation: {row['season_precipitation_mm']:.2f} mm"
            f"\nPredicted yield: {row['predicted_yield_t_ha']:.3f} t/ha"
        )


if __name__ == "__main__":
    main()