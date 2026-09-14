import calendar
import csv
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import psycopg
from dotenv import load_dotenv


BASE_DIR = Path(__file__).resolve().parent.parent

WEATHER_DIR = (BASE_DIR / "data" / "raw"/ "hungaroMet")

TEMPERATURE_FILE = (WEATHER_DIR / "temp" / "ta_grid_19712025.txt")

PRECIPITATION_FILE = (WEATHER_DIR / "precipitation" / "r_grid_19712025.txt")

MAPPING_FILE = (BASE_DIR/ "data" / "processed" / "gridpoint_county_mapping.csv")


EXPECTED_GRIDPOINTS = 1233
EXPECTED_USABLE_GRIDPOINTS = 1097
EXPECTED_COUNTIES = 19

MIN_SEASON_YEAR = 1971

MAX_MAIZE_SEASON_YEAR = 2025
MAX_WINTER_CROP_SEASON_YEAR = 2024

EXPECTED_RECORDS = 3097


def load_mapping():
    grid_to_county = {}

    with open(
MAPPING_FILE,
        "r",
        encoding="utf-8",
        newline=""
    ) as file:

        reader = csv.DictReader(file)

        for row in reader:
            grid_index = int(
                row["grid_index"]
            )

            county_name = row[
                "county_name"
            ]

            if grid_index in grid_to_county:
                raise ValueError(
                    f"Duplicate grid index: "
                    f"{grid_index}"
                )

            grid_to_county[
                grid_index
            ] = county_name

    if (
        len(grid_to_county)
        != EXPECTED_USABLE_GRIDPOINTS
    ):
        raise ValueError(
            f"Expected "
            f"{EXPECTED_USABLE_GRIDPOINTS} "
            f"usable gridpoints, found "
            f"{len(grid_to_county)}."
        )

    counties = set(
        grid_to_county.values()
    )

    if len(counties) != EXPECTED_COUNTIES:
        raise ValueError(
            f"Expected {EXPECTED_COUNTIES} "
            f"counties, found "
            f"{len(counties)}."
        )

    print(
        f"Mapping loaded: "
        f"{len(grid_to_county)} gridpoints, "
        f"{len(counties)} counties."
    )

    return grid_to_county


def build_county_positions(
    header_grid_indices,
    grid_to_county
):
    county_positions = defaultdict(
        list
    )

    for position, grid_index in enumerate(
        header_grid_indices
    ):
        county_name = (
            grid_to_county.get(
                grid_index
            )
        )

        if county_name is not None:
            county_positions[
                county_name
            ].append(position)

    if (
        len(county_positions)
        != EXPECTED_COUNTIES
    ):
        raise ValueError(
            f"Expected "
            f"{EXPECTED_COUNTIES} counties "
            f"in weather grid, found "
            f"{len(county_positions)}."
        )

    return county_positions


def get_seasons_for_date(
    year,
    month,
    day
):
    """
    Returns the crop + season_start_year to which a date belongs
    """

    seasons = []

    # Maize:
    # Március 1 - október 31
    if 3 <= month <= 10:

        if (
            MIN_SEASON_YEAR
            <= year
            <= MAX_MAIZE_SEASON_YEAR
        ):
            seasons.append(
                (
                    "maize",
                    year
                )
            )

    # Wheat and barley:
    # Szeptember 1 - Július 31

    winter_season_year = None

    if month >= 9:

        winter_season_year = year

    elif month <= 7:

        winter_season_year = (
            year - 1
        )

    if winter_season_year is not None:

        if (
            MIN_SEASON_YEAR
            <= winter_season_year
            <= MAX_WINTER_CROP_SEASON_YEAR
        ):
            seasons.append(
                (
                    "wheat",
                    winter_season_year
                )
            )

            seasons.append(
                (
                    "barley",
                    winter_season_year
                )
            )

    return seasons


def expected_days(
    crop,
    season_start_year
):
    if crop == "maize":

        # March 1 - October 31
        return 245

    if crop in (
        "wheat",
        "barley"
    ):

        # Szeptember 1 - Július 31.
        next_year = (
            season_start_year + 1
        )

        if calendar.isleap(
            next_year
        ):
            return 335

        return 334

    raise ValueError(
        f"Unknown crop: {crop}"
    )


def process_weather_file(
    file_path,
    grid_to_county,
    aggregation
):
    totals = defaultdict(float)
    day_counts = defaultdict(int)

    print(
        f"\nProcessing: "
        f"{file_path.name}"
    )

    with open(
        file_path,
        "r",
        encoding="ascii"
    ) as file:

        # First row contains gridpoint indices 1..1233.
        header = file.readline().split()

        header_grid_indices = [
            int(value)
            for value in header
        ]

        if (
            len(header_grid_indices)
            != EXPECTED_GRIDPOINTS
        ):
            raise ValueError(
                f"{file_path.name}: "
                f"expected "
                f"{EXPECTED_GRIDPOINTS} "
                f"gridpoints, found "
                f"{len(header_grid_indices)}."
            )

        county_positions = (
            build_county_positions(
                header_grid_indices,
                grid_to_county
            )
        )

        for line in file:

            if not line.strip():
                continue

            # HungaroMet date field uses the first 8 characters
            year = int(
                line[0:4]
            )

            month = int(
                line[4:6].strip()
            )

            day = int(
                line[6:8].strip()
            )

            seasons = (
                get_seasons_for_date(
                    year,
                    month,
                    day
                )
            )

            # Date does not belong to any usable crop season
            if not seasons:
                continue

            values = np.asarray(
                [
                    float(value)
                    for value
                    in line[8:].split()
                ],
                dtype=float
            )

            if (
                len(values)
                != EXPECTED_GRIDPOINTS
            ):
                raise ValueError(
                    f"{file_path.name}: "
                    f"wrong number of values "
                    f"on "
                    f"{year}-{month:02d}-{day:02d}. "
                    f"Expected "
                    f"{EXPECTED_GRIDPOINTS}, "
                    f"found {len(values)}."
                )

            values[
                values <= -900
            ] = np.nan

            # First aggregate gridpoints to daily county values
            for (
                county_name,
                positions
            ) in county_positions.items():

                county_values = values[
                    positions
                ]

                if np.all(
                    np.isnan(
                        county_values
                    )
                ):
                    raise ValueError(
                        f"No valid weather values "
                        f"for {county_name} on "
                        f"{year}-{month:02d}-{day:02d}."
                    )

                daily_mean = float(
                    np.nanmean(
                        county_values
                    )
                )

                for (
                    crop,
                    season_start_year
                ) in seasons:

                    key = (
                        season_start_year,
                        county_name,
                        crop
                    )

                    totals[key] += (
                        daily_mean
                    )

                    day_counts[key] += 1

    validate_day_counts(
        file_path,
        day_counts
    )

    if aggregation == "mean":

        result = {
            key:
                total
                / day_counts[key]

            for key, total
            in totals.items()
        }

    elif aggregation == "sum":

        result = dict(totals)

    else:
        raise ValueError(
            f"Unknown aggregation: "
            f"{aggregation}"
        )

    if (
        len(result)
        != EXPECTED_RECORDS
    ):
        raise ValueError(
            f"{file_path.name}: "
            f"expected "
            f"{EXPECTED_RECORDS} "
            f"seasonal county-crop "
            f"values, found "
            f"{len(result)}."
        )

    print(
        f"{file_path.name}: "
        f"{len(result)} seasonal "
        f"county-crop values calculated."
    )

    return result


def validate_day_counts(
    file_path,
    day_counts
):
    for (
        season_start_year,
        county_name,
        crop
    ), actual_days in (
        day_counts.items()
    ):

        required_days = (
            expected_days(
                crop,
                season_start_year
            )
        )

        if actual_days != required_days:
            raise ValueError(
                f"{file_path.name}: "
                f"{crop}, "
                f"{county_name}, "
                f"season {season_start_year}: "
                f"{actual_days} days found "
                f"instead of "
                f"{required_days}."
            )


def build_records(
    temperature,
    precipitation
):
    temperature_keys = set(
        temperature.keys()
    )

    precipitation_keys = set(
        precipitation.keys()
    )

    if (
        temperature_keys
        != precipitation_keys
    ):
        raise ValueError(
            "Temperature and "
            "precipitation season keys "
            "do not match."
        )

    if (
        len(temperature_keys)
        != EXPECTED_RECORDS
    ):
        raise ValueError(
            f"Expected "
            f"{EXPECTED_RECORDS} records, "
            f"found "
            f"{len(temperature_keys)}."
        )

    records = []

    for (
        season_start_year,
        county_name,
        crop
    ) in sorted(
        temperature_keys
    ):

        records.append(
            (
                season_start_year,
                county_name,
                crop,
                round(
                    temperature[
                        (
                            season_start_year,
                            county_name,
                            crop
                        )
                    ],
                    3
                ),
                round(
                    precipitation[
                        (
                            season_start_year,
                            county_name,
                            crop
                        )
                    ],
                    2
                ),
            )
        )

    return records


def get_connection():
    load_dotenv(
        BASE_DIR
        / ".env"
    )

    return psycopg.connect(
        host=os.getenv(
            "DB_HOST"
        ),
        port=os.getenv(
            "DB_PORT"
        ),
        dbname=os.getenv(
            "DB_NAME"
        ),
        user=os.getenv(
            "DB_USER"
        ),
        password=os.getenv(
            "DB_PSW"
        ),
    )


def import_records(
    connection,
    records
):
    with connection.cursor() as cursor:

        cursor.execute(
            """
            DELETE FROM
                processed.weather_seasonal;
            """
        )

        cursor.executemany(
            """
            INSERT INTO
                processed.weather_seasonal (
                    season_start_year,
                    county_name,
                    crop,
                    season_mean_temperature_c,
                    season_precipitation_mm
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
    print(
        "Importing seasonal "
        "HungaroMet weather data..."
    )

    required_files = [
        TEMPERATURE_FILE,
        PRECIPITATION_FILE,
        MAPPING_FILE,
    ]

    for file_path in required_files:

        if not file_path.exists():
            raise FileNotFoundError(
                f"File does not exist: "
                f"{file_path}"
            )

    grid_to_county = (
        load_mapping()
    )

    temperature = (
        process_weather_file(
            TEMPERATURE_FILE,
            grid_to_county,
            aggregation="mean"
        )
    )

    precipitation = (
        process_weather_file(
            PRECIPITATION_FILE,
            grid_to_county,
            aggregation="sum"
        )
    )

    records = build_records(
        temperature,
        precipitation
    )

    print(
        f"\nRecords prepared: "
        f"{len(records)}"
    )

    connection = get_connection()

    try:

        import_records(
            connection,
            records
        )

        connection.commit()

        print(
            "\nSeasonal weather "
            "import was successful."
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