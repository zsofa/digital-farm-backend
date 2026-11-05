import csv
import os
from decimal import Decimal
from pathlib import Path

import psycopg
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "raw" / "ksh" / "crop"
FILES = {
    "wheat": DATA_DIR / "stadat-mez0071-19.1.2.4-en.csv",
    "maize": DATA_DIR / "stadat-mez0072-19.1.2.5-en.csv",
    "barley": DATA_DIR / "stadat-mez0073-19.1.2.6-en.csv",
}

# KSH CSV match data to own data
METRIC_HEADERS = {
    # Wheat
    "Harvested area, hectare": ("harvested_area", "ha"),
    "Total harvested production, ton": ("harvested_production", "t"),

    # Maize
    "Harvested area, hectares": ("harvested_area", "ha"),
    "Total harvested production, tonnes": ("harvested_production", "t"),

    # Wheat + maize
    "Average yield, kg/hectare": ("average_yield", "kg/ha"),

    # Barley
    "Harvested area of barley, total, hectare": ("harvested_area", "ha"),
    "Total harvested production of barley, total, ton": ("harvested_production", "t"),
    "Average yield of barley, total, kg/hectare": ("average_yield", "kg/ha"),
}


def parse_value(value):
    # Numerical values -> 58 788 -> to Decimal('58788')
    value = value.strip()
    value = value.replace("\xa0", "")
    value = value.replace(" ", "")

    # Handle missing values
    if value in {"", "-", "–", "..", ":"}:
        return None

    value = value.replace(",", ".")

    return Decimal(value)


def parse_csv_file(crop, file_path):
    # Process csv
    # Get only 19 county, only the needed units and 2000-2025 time interval

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

            if first_column == "Name of territorial units":
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

            # Winter or spring barley are not processed, should be skipped
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
                        crop,
                        active_metric,
                        county_name,
                        year,
                        parse_value(value),
                        active_unit,
                        file_path.name
                    )
                )
    return records

def validate_records(crop_type, records):
    counties = {
        record[2]
        for record in records
    }

    years = {
        record[3]
        for record in records
    }

    metrics = {
        record[1]
        for record in records
    }

    expected_metrics = {
        "harvested_area",
        "harvested_production",
        "average_yield",
    }

    if len(counties) != 19:
        raise ValueError(
            f"{crop_type}: Instead of 19 counties, there are {len(counties)}"
        )

    if years != set(range(2000,2026)):
        raise ValueError(f"{crop_type}: Wrong time range.")


    if metrics != expected_metrics:
        raise ValueError(
            f"{crop_type}: Wrong metrics: {metrics}"
        )

    expected_record_count = 19 * 26 * 3

    if len(records) != expected_record_count:
        raise ValueError(
            f"{crop_type}: Instead of {expected_record_count} number of records, there are {len(records)}"
        )


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
def import_records(connection, crop, records):
    with connection.cursor() as cursor:
        # If I run the script again,
        # the old records of the crop will be removed to avoid duplication

        cursor.execute(
            """
            DELETE FROM raw.ksh_crop_production
            WHERE crop = %s;
            """,
            (crop,)
        )

        cursor.executemany(
            """
            INSERT INTO raw.ksh_crop_production (
                crop,
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
                %s,
                %s
            );
            """,
            records,
        )

def main():
    print("Importing records from KSH crops...")
    connection = get_connection()

    try:
        total_records = 0

        for crop, file_path in FILES.items():
            print(f"\nProcessing: {crop}")

            if not file_path.exists():
                raise ValueError(f"File {file_path} does not exist.")

            records = parse_csv_file(crop, file_path)
            validate_records(crop, records)
            import_records(connection, crop, records)
            total_records += len(records)

            print( f"{crop}: {len(records)} record preprocessed.")

        connection.commit()
        print("\nImport was successful.")

    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()

if __name__ == "__main__":
    main()




