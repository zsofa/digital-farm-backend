import os
from pathlib import Path

import numpy as np
import psycopg
from dotenv import load_dotenv


BASE_DIR = Path(__file__).resolve().parent.parent

HISTORY_LENGTH = 5


def get_connection():
    load_dotenv(BASE_DIR / ".env")

    return psycopg.connect(
        host=os.getenv("DB_HOST"),
        port=os.getenv("DB_PORT"),
        dbname=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PSW"),
    )


def load_crop_yields(connection):
    """
    load allhistorical KSH yield records.

    {
        ("Győr-Moson-Sopron", "maize"): [
            {"year": 2000, "yield": 5.123},
            {"year": 2001, "yield": 5.456},
            ...
        ]
    }
    """

    with connection.cursor() as cursor:
        cursor.execute(
            """
            SELECT
                year,
                county_name,
                crop,
                average_yield_t_ha
            FROM processed.crop_production
            WHERE average_yield_t_ha IS NOT NULL
            ORDER BY
                county_name,
                crop,
                year;
            """
        )

        rows = cursor.fetchall()

    yield_history = {}

    for (
        year,
        county_name,
        crop,
        yield_t_ha,
    ) in rows:

        key = (
            county_name,
            crop,
        )

        if key not in yield_history:
            yield_history[key] = []

        yield_history[key].append(
            {
                "year": int(year),
                "yield": float(yield_t_ha),
            }
        )

    print(
        f"County-crop yield histories loaded: "
        f"{len(yield_history)}"
    )

    return yield_history


def calculate_recent_yield(
    historical_yields,
    target_year,
):
    """
    Calculate mean yield of the previous 5 years before target_year
    """

    available = [
        row
        for row in historical_yields
        if row["year"] < target_year
    ]

    available = available[
        -HISTORY_LENGTH:
    ]

    if len(available) < HISTORY_LENGTH:
        return None

    values = [
        row["yield"]
        for row in available
    ]

    return float(
        np.mean(values)
    )


def build_records(yield_history):
    records = []

    for (
        county_name,
        crop,
    ), historical_yields in (
        yield_history.items()
    ):

        target_years = [
            row["year"]
            for row in historical_yields
        ]

        for target_year in target_years:

            recent_mean = (
                calculate_recent_yield(
                    historical_yields,
                    target_year,
                )
            )

            if recent_mean is None:
                continue

            records.append(
                (
                    target_year,
                    county_name,
                    crop,
                    round(
                        recent_mean,
                        3,
                    ),
                )
            )

    return records


def validate_records(records):
    if not records:
        raise ValueError(
            "No recent yield records created."
        )

    years = {
        record[0]
        for record in records
    }

    counties = {
        record[1]
        for record in records
    }

    crops = {
        record[2]
        for record in records
    }

    print(
        f"Target years: "
        f"{min(years)}-{max(years)}"
    )

    print(
        f"Counties: {len(counties)}"
    )

    print(
        f"Crops: {len(crops)}"
    )

    print(
        f"Recent yield records prepared: "
        f"{len(records)}"
    )

    if len(counties) != 19:
        raise ValueError(
            f"Expected 19 counties, "
            f"found {len(counties)}."
        )

    if crops != {
        "wheat",
        "barley",
        "maize",
    }:
        raise ValueError(
            f"Unexpected crop set: {crops}"
        )


def import_records(
    connection,
    records,
):
    with connection.cursor() as cursor:

        cursor.execute(
            """
            DELETE FROM processed.recent_yield;
            """
        )

        cursor.executemany(
            """
            INSERT INTO processed.recent_yield (
                target_year,
                county_name,
                crop,
                recent_yield_mean_t_ha
            )
            VALUES (
                %s,
                %s,
                %s,
                %s
            );
            """,
            records,
        )


def main():
    print(
        "Creating historical recent yield features..."
    )

    connection = get_connection()

    try:
        yield_history = (
            load_crop_yields(
                connection
            )
        )

        records = build_records(
            yield_history
        )

        validate_records(
            records
        )

        import_records(
            connection,
            records
        )

        connection.commit()

        print(
            "\nRecent yield import "
            "was successful."
        )

        print(
            f"Imported records: "
            f"{len(records)}"
        )

    except Exception:
        connection.rollback()
        raise

    finally:
        connection.close()


if __name__ == "__main__":
    main()