# -*- coding: cp1250 -*-

import csv
import os
from decimal import Decimal
from pathlib import Path

import psycopg
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "raw" / "ksh" / "irrigation"
FILE_PATH = DATA_DIR / "stadat-mez0094-19.1.2.26-en.csv"

METRIC_HEADERS = {
    "Irrigated area, ha": ("irrigated_area", "ha"),
    "Amount of irrigated water per hectare, m3/ha": ("irrigated_water_per_ha", "m3/ha")
}

def parse_value(value):
    value = value.strip()
    value = value.replace("\xa0", "")
    value = value.replace(" ", "")

    # Handle missing values
    if value in {"", "-", "–", "..", ":"}:
        return None

    value = value.replace(",", ".")

    return Decimal(value)


def parse_csv_file(file_path):
    # Process csv
    # Get only 19 county, only the needed units and 2004-2025 time interval

    records = []
    years = []
    active_metric = None
    active_unit = None

    with open(file_path, "r", encoding="cp1250", newline="") as file:
        reader = csv.reader(file, delimiter=";")

        for row in reader:

            if not row: continue

            first_column = row[0].strip()
            second_column = (
                row[1].strip()
                if len(row) > 1
                else ""
            )

            # Year header
            if first_column == "Name of territorial unit":
                years = [
                    int(year.strip())
                    for year in row[2:]
                    if year.strip()
                ]

                continue

            # Check if new metric unit started
            if first_column in METRIC_HEADERS:
                active_metric, active_unit = (
                    METRIC_HEADERS[first_column]
                )

                continue

            # Other header
            if first_column and not second_column:
                active_metric = None
                active_unit = None

                continue

            # Only process the line if the metric unit needed
            if active_metric is None:
                continue

            # Only megye -> county in the level name
            if "county" not in second_column.lower():
                continue

            county_name = first_column
            values = row[2:2 + len(years)]
            for year, value in zip(years, values):
                records.append(
                    (
                        active_metric,
                        county_name,
                        year,
                        parse_value(value),
                        active_unit,
                        file_path.name
                    )
                )
    return records


def validate_records(records):
    counties = {
        record[1]
        for record in records
    }

    years = {
        record[2]
        for record in records
    }

    metrics = {
        record[0]
        for record in records
    }

    expected_metrics = {
        "irrigated_area",
        "irrigated_water_per_ha"
    }

    if len(counties) != 19:
        raise ValueError(
            f"Instead of 19 counties, there are {len(counties)}"
        )

    if years != set(range(2004, 2026)):
        raise ValueError(f"Wrong time range.")

    print(metrics, expected_metrics)
    if metrics != expected_metrics:
        raise ValueError(
            f"Wrong metrics: {metrics}"
        )

    expected_record_count = 19 * 22 * 2

    if len(records) != expected_record_count:
        raise ValueError(
            f"Instead of {expected_record_count} number of records, there are {len(records)}"
        )

    missing_values = sum(1 for record in records if record[3] is None)

    print(f"Counties: {len(counties)}")
    print(f"Years: {len(years)}")
    print(f"Metrics: {len(metrics)}")
    print(f"Missing values: {missing_values}")

# Create postgreSQl connection
def get_connection():
    load_dotenv(BASE_DIR / ".env")

    return psycopg.connect(
        host=os.getenv("DB_HOST"),
        port=os.getenv("DB_PORT"),
        dbname=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PSW"),
    )

# Load records into db
def import_records(connection, records):
    with connection.cursor() as cursor:
        # If I run the script again,
        # the old records of the crop will be removed to avoid duplication

        cursor.execute(
            """
            DELETE FROM raw.ksh_irrigation;
            """
        )

        cursor.executemany(
            """
            INSERT INTO raw.ksh_irrigation (
                metric,
                county_name,
                year,
                value,
                unit,
                source_file
            )
            VALUES (
                %s,
                %s,
                %s,
                %s,
                %s,
                %s
            );
            """,
            records,
        )


def main():
    print("Importing records from KSH Irrigation...")

    if not FILE_PATH.exists():
        raise FileNotFoundError(f"File {FILE_PATH} does not exist.")

    records = parse_csv_file(FILE_PATH)
    validate_records(records)

    print(f"{len(records)} records imported.")

    connection = get_connection()

    try:
        import_records(connection, records)
        connection.commit()

        print("Import was successful.")
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


if __name__ == "__main__":
    main()
