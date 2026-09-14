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

_model_artifact = None


def get_model_artifact():
    global _model_artifact

    if _model_artifact is None:
        _model_artifact = joblib.load(MODEL_FILE)

    return _model_artifact


def load_county_soil(county_name):
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
        raise ValueError(f"No soil data found for county: {county_name}")

    return {
        "soil_ph": float(row[0]),
        "soil_soc_g_kg": float(row[1]),
        "soil_clay_pct": float(row[2]),
    }


def calculate_recent_yield(county_name, crop, target_season_start_year):
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


def simulate_yield_scenarios(
    county_name,
    crop,
    target_season_start_year=None,
    soil=None,
    area_ha=None,
):
    if target_season_start_year is None:
        target_season_start_year = get_next_season_start_year(crop)

    if soil is None:
        soil = load_county_soil(county_name)

    recent_yield, recent_yield_years = calculate_recent_yield(
        county_name, crop, target_season_start_year
    )

    weather_scenarios = create_weather_scenarios(
        county_name, crop, target_season_start_year
    )

    rows = []

    for scenario_name in ("dry_hot", "expected", "wet_cool"):
        weather = weather_scenarios[scenario_name]

        rows.append({
            "scenario": scenario_name,
            "crop": crop,
            **soil,
            "season_temperature_c": weather["temperature_c"],
            "season_precipitation_mm": weather["precipitation_mm"],
            "recent_yield_mean_t_ha": recent_yield,
        })

    dataframe = pd.DataFrame(rows)

    artifact = get_model_artifact()
    model = artifact["model"]

    dataframe["predicted_yield_t_ha"] = model.predict(
        dataframe[SCENARIO_FEATURE_COLUMNS]
    )

    scenarios = {}

    for _, row in dataframe.iterrows():
        predicted_yield = round(float(row["predicted_yield_t_ha"]), 3)

        scenario = {
            "temperature_c": round(float(row["season_temperature_c"]), 3),
            "precipitation_mm": round(float(row["season_precipitation_mm"]), 2),
            "predicted_yield_t_ha": predicted_yield,
        }

        if area_ha is not None:
            scenario["estimated_total_t"] = round(
                predicted_yield * float(area_ha), 3
            )

        scenarios[row["scenario"]] = scenario

    return {
        "county_name": county_name,
        "crop": crop,
        "target_season_start_year": target_season_start_year,
        "recent_yield_mean_t_ha": round(recent_yield, 3),
        "recent_yield_years": recent_yield_years,
        "soil": soil,
        "scenarios": scenarios,
    }