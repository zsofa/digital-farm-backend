import calendar
import csv
import os
from collections import defaultdict
from pathlib import Path

import psycopg
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent

WEATHER_DIR = BASE_DIR / "data" / "raw" / "hungaroMet"

TEMPERATURE_DIR = WEATHER_DIR / "temp" / "ta_grid_19712025.txt"
PRECIPITATION_DIR = WEATHER_DIR / "precipitation" / "r_grid_19712025.txt"

MAPPING_FILE = (
    BASE_DIR / "data" / "processed" / "gridpoint_county_mapping.csv"
)

START_YEAR = 2000
END_YEAR = 2025

EXPECTED_GRIDPOINTS = 1233
EXPECTED_USABLE_GRIDPOINTS = 1097
EXPECTED_COUNTIES = 19
EXPECTED_RECORDS = 19 * 26

def load_mapping():
    grid_to_county = {}
    county_gridpoint_counts = defaultdict(int)

    with open(
        MAPPING_FILE,
        "r",
        encoding="utf-8",
        newline=""
    ) as file:

        reader = csv.DictReader(file)

        for row in reader:
            grid_index = int(row["grid_index"])
            county_name = row["county_name"]

            if grid_index in grid_to_county:
                raise ValueError(
                    f"Duplicate grid index in mapping: {grid_index}"
                )

            grid_to_county[grid_index] = county_name
            county_gridpoint_counts[county_name] += 1

    if len(grid_to_county) != EXPECTED_USABLE_GRIDPOINTS:
        raise ValueError(
            f"Expected {EXPECTED_USABLE_GRIDPOINTS} usable gridpoints, "
            f"found {len(grid_to_county)}."
        )

    if len(county_gridpoint_counts) != EXPECTED_COUNTIES:
        raise ValueError(
            f"Expected {EXPECTED_COUNTIES} counties, "
            f"found {len(county_gridpoint_counts)}."
        )

    print(
        f"Mapping loaded: {len(grid_to_county)} gridpoints, "
        f"{len(county_gridpoint_counts)} counties."
    )

    return grid_to_county, dict(county_gridpoint_counts)


def build_county_positions(header_grid_indices, grid_to_county):
    county_positions = defaultdict(list)

    for position, grid_index in enumerate(header_grid_indices):
        county_name = grid_to_county.get(grid_index)

        if county_name is not None:
            county_positions[county_name].append(position)

    if len(county_positions) != EXPECTED_COUNTIES:
        raise ValueError(
            f"Only {len(county_positions)} counties found "
            f"in weather grid."
        )

    return county_positions


def process_weather_file(
    file_path,
    grid_to_county,
    aggregation
):
    annual_values = defaultdict(float)
    year_day_counts = defaultdict(int)

    print(f"Processing: {file_path.name}")

    with open(
        file_path,
        "r",
        encoding="ascii"
    ) as file:

        # First line contains grid indices: 1 ... 1233
        header = file.readline().split()

        header_grid_indices = [
            int(value)
            for value in header
        ]

        if len(header_grid_indices) != EXPECTED_GRIDPOINTS:
            raise ValueError(
                f"{file_path.name}: expected "
                f"{EXPECTED_GRIDPOINTS} gridpoints, "
                f"found {len(header_grid_indices)}."
            )

        county_positions = build_county_positions(
            header_grid_indices,
            grid_to_county
        )

        for line in file:

            if not line.strip():
                continue

            # HungaroMet uses fixed-width 8-character date field
            year = int(line[0:4])

            if year < START_YEAR:
                continue

            if year > END_YEAR:
                break

            month = int(line[4:6].strip())
            day = int(line[6:8].strip())

            values = [
                float(value)
                for value in line[8:].split()
            ]

            if len(values) != EXPECTED_GRIDPOINTS:
                raise ValueError(
                    f"{file_path.name}: wrong number of grid values "
                    f"on {year}-{month:02d}-{day:02d}. "
                    f"Expected {EXPECTED_GRIDPOINTS}, "
                    f"found {len(values)}."
                )

            # First of all, aggregate all gridpoints of each county
            # into one daily county-level value
            for county_name, positions in county_positions.items():

                daily_mean = (
                    sum(values[position] for position in positions)
                    / len(positions)
                )

                annual_values[
                    (year, county_name)
                ] += daily_mean

            year_day_counts[year] += 1

    validate_day_counts(
        file_path,
        year_day_counts
    )

    # temperature: mean of all daily county means
    if aggregation == "mean":

        result = {
            (year, county_name):
                total / year_day_counts[year]

            for (year, county_name), total
            in annual_values.items()
        }

    #precipitation: sum of daily county-average precipitation
    elif aggregation == "sum":

        result = dict(annual_values)

    else:
        raise ValueError(
            f"Unknown aggregation: {aggregation}"
        )

    if len(result) != EXPECTED_RECORDS:
        raise ValueError(
            f"{file_path.name}: expected "
            f"{EXPECTED_RECORDS} annual county values, "
            f"found {len(result)}."
        )

    print(
        f"{file_path.name}: {len(result)} "
        f"annual county values calculated."
    )

    return result


def validate_day_counts(file_path, year_day_counts):
    expected_years = set(
        range(START_YEAR, END_YEAR + 1)
    )

    if set(year_day_counts.keys()) != expected_years:
        raise ValueError(
            f"{file_path.name}: wrong year range."
        )

    for year, number_of_days in year_day_counts.items():

        expected_days = (
            366
            if calendar.isleap(year)
            else 365
        )

        if number_of_days != expected_days:
            raise ValueError(
                f"{file_path.name}: {year} contains "
                f"{number_of_days} days instead of "
                f"{expected_days}."
            )


def build_records(
    temperature,
    precipitation,
    county_gridpoint_counts
):
    temperature_keys = set(temperature.keys())
    precipitation_keys = set(precipitation.keys())

    if temperature_keys != precipitation_keys:
        raise ValueError(
            "Temperature and precipitation keys do not match."
        )

    if len(temperature_keys) != EXPECTED_RECORDS:
        raise ValueError(
            f"Expected {EXPECTED_RECORDS} weather records, "
            f"found {len(temperature_keys)}."
        )

    records = []

    for year, county_name in sorted(
        temperature_keys
    ):
        records.append(
            (
                year,
                county_name,
                round(
                    temperature[(year, county_name)],
                    3
                ),
                round(
                    precipitation[(year, county_name)],
                    2
                ),
                county_gridpoint_counts[county_name]
            )
        )

    return records


def get_connection():
    load_dotenv(BASE_DIR / ".env")

    return psycopg.connect(
        host=os.getenv("DB_HOST"),
        port=os.getenv("DB_PORT"),
        dbname=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PSW"),
    )


def import_records(connection, records):
    with connection.cursor() as cursor:

        cursor.execute(
            """
            DELETE FROM processed.weather;
            """
        )

        cursor.executemany(
            """
            INSERT INTO processed.weather (
                year,
                county_name,
                annual_mean_temperature_c,
                annual_precipitation_mm,
                gridpoint_count
            )
            VALUES (
                %s,
                %s,
                %s,
                %s,
                %s
            );
            """,
            records
        )


def main():
    print("Importing HungaroMet weather data...")

    required_files = [
        TEMPERATURE_DIR,
        PRECIPITATION_DIR,
        MAPPING_FILE,
    ]

    for file_path in required_files:
        if not file_path.exists():
            raise FileNotFoundError(
                f"File does not exist: {file_path}"
            )

    grid_to_county, county_gridpoint_counts = (
        load_mapping()
    )

    temperature = process_weather_file(
        TEMPERATURE_DIR,
        grid_to_county,
        aggregation="mean"
    )

    precipitation = process_weather_file(
        PRECIPITATION_DIR,
        grid_to_county,
        aggregation="sum"
    )

    records = build_records(
        temperature,
        precipitation,
        county_gridpoint_counts
    )

    connection = get_connection()

    try:
        import_records(
            connection,
            records
        )

        connection.commit()

        print("\nWeather import was successful.")
        print(f"Imported records: {len(records)}")

    except Exception:
        connection.rollback()
        raise

    finally:
        connection.close()


if __name__ == "__main__":
    main()