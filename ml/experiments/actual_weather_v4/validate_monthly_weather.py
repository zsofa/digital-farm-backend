import calendar
from pathlib import Path

import numpy as np
import pandas as pd

from ml.data_loader import get_connection


MONTHLY_FILE = Path(__file__).resolve().parent / "data" / "monthly_weather.csv"

# Database values are rounded to 3 and 2 decimal places.
TEMPERATURE_TOLERANCE = 0.000501
PRECIPITATION_TOLERANCE = 0.005001

KEYS = ["season_start_year", "county_name", "crop"]


def load_monthly():
    dataframe = pd.read_csv(MONTHLY_FILE)

    keys = ["year", "month", "county_name"]

    if dataframe[keys].isna().any().any():
        raise ValueError("Missing monthly keys.")

    if dataframe.duplicated(keys).any():
        raise ValueError("Duplicate county-month rows.")

    for column in [
        "mean_temperature_c",
        "precipitation_mm",
        "temperature_days",
        "precipitation_days",
    ]:
        dataframe[column] = pd.to_numeric(dataframe[column])

        if not np.isfinite(dataframe[column]).all():
            raise ValueError(f"Invalid values in {column}.")

    for row in dataframe.itertuples(index=False):
        required_days = calendar.monthrange(
            int(row.year), int(row.month)
        )[1]

        if (
            row.temperature_days != required_days
            or row.precipitation_days != required_days
        ):
            raise ValueError(
                f"Incomplete month: {row.county_name}, "
                f"{row.year}-{row.month}."
            )

    return dataframe.set_index(keys)


def load_seasonal():
    query = """
        SELECT
            season_start_year,
            county_name,
            crop,
            season_mean_temperature_c,
            season_precipitation_mm
        FROM processed.weather_seasonal
        WHERE
            (
                crop = 'maize'
                AND season_start_year BETWEEN 1971 AND 2025
            )
            OR
            (
                crop IN ('wheat', 'barley')
                AND season_start_year BETWEEN 1971 AND 2024
            )
        ORDER BY season_start_year, county_name, crop;
    """

    connection = get_connection()

    try:
        with connection.cursor() as cursor:
            cursor.execute(query)
            rows = cursor.fetchall()
            columns = [column.name for column in cursor.description]
    finally:
        connection.close()

    dataframe = pd.DataFrame(rows, columns=columns)

    for column in [
        "season_mean_temperature_c",
        "season_precipitation_mm",
    ]:
        dataframe[column] = pd.to_numeric(
            dataframe[column]
        ).astype(float)

        if not np.isfinite(dataframe[column]).all():
            raise ValueError(f"Invalid stored values in {column}.")

    if dataframe.duplicated(KEYS).any():
        raise ValueError("Duplicate seasonal rows.")

    return dataframe


def season_months(crop, start_year):
    if crop == "maize":
        return [(start_year, month) for month in range(3, 11)]

    if crop in {"wheat", "barley"}:
        return (
            [(start_year, month) for month in range(9, 13)]
            + [(start_year + 1, month) for month in range(1, 8)]
        )

    raise ValueError(f"Unknown crop: {crop}")


def reconstruct_season(monthly, county, crop, start_year):
    temperature_total = 0.0
    precipitation_total = 0.0
    day_count = 0

    for year, month in season_months(crop, start_year):
        key = (year, month, county)

        if key not in monthly.index:
            raise ValueError(f"Missing monthly row: {key}")

        row = monthly.loc[key]
        days = calendar.monthrange(year, month)[1]

        temperature_total += float(row["mean_temperature_c"]) * days
        precipitation_total += float(row["precipitation_mm"])
        day_count += days

    return temperature_total / day_count, precipitation_total


def main():
    monthly = load_monthly()
    seasonal = load_seasonal()

    counties = sorted(
        monthly.index.get_level_values("county_name").unique()
    )

    if len(counties) != 19:
        raise ValueError("Expected 19 counties.")

    expected_keys = {
        (year, county, crop)
        for crop in ["barley", "maize", "wheat"]
        for year in range(1971, 2026 if crop == "maize" else 2025)
        for county in counties
    }

    actual_keys = set(
        seasonal[KEYS].itertuples(index=False, name=None)
    )

    if actual_keys != expected_keys:
        raise ValueError(
            "Seasonal coverage mismatch: "
            f"{len(expected_keys - actual_keys)} missing, "
            f"{len(actual_keys - expected_keys)} unexpected rows."
        )

    results = []

    for row in seasonal.itertuples(index=False):
        temperature, precipitation = reconstruct_season(
            monthly,
            row.county_name,
            row.crop,
            int(row.season_start_year),
        )

        results.append({
            "season_start_year": row.season_start_year,
            "county_name": row.county_name,
            "crop": row.crop,
            "temperature_difference": abs(
                temperature - row.season_mean_temperature_c
            ),
            "precipitation_difference": abs(
                precipitation - row.season_precipitation_mm
            ),
        })

    comparison = pd.DataFrame(results)

    comparison["temperature_mismatch"] = (
        comparison["temperature_difference"] > TEMPERATURE_TOLERANCE
    )
    comparison["precipitation_mismatch"] = (
        comparison["precipitation_difference"] > PRECIPITATION_TOLERANCE
    )

    summary = comparison.groupby("crop").agg(
        compared_rows=("crop", "size"),
        temperature_mismatches=("temperature_mismatch", "sum"),
        precipitation_mismatches=("precipitation_mismatch", "sum"),
        max_temperature_difference=("temperature_difference", "max"),
        max_precipitation_difference=("precipitation_difference", "max"),
    )

    print("=" * 100)
    print("MONTHLY VS STORED SEASONAL WEATHER")
    print("=" * 100)
    print(summary.round(6).to_string())

    mismatches = comparison[
        comparison["temperature_mismatch"]
        | comparison["precipitation_mismatch"]
    ]

    if not mismatches.empty:
        print("\nFirst 20 mismatches:")
        print(mismatches.head(20).round(6).to_string(index=False))
        raise ValueError("Seasonal reconstruction check failed.")

    print("\nPASSED: all seasonal values match within rounding tolerance.")
    print("No files, database tables or models were modified.")


if __name__ == "__main__":
    main()