import os
from pathlib import Path

import numpy as np
import psycopg
from dotenv import load_dotenv


BASE_DIR = Path(__file__).resolve().parent.parent

# Temperature:
# Use the previous 15 completed seasons and use a linear trend
TEMPERATURE_HISTORY_LENGTH = 15

# Precipitation:
# Use the mean of the previous 10 completed seasons
PRECIPITATION_HISTORY_LENGTH = 10


def get_connection():
    load_dotenv(BASE_DIR / ".env")

    return psycopg.connect(
        host=os.getenv("DB_HOST"),
        port=os.getenv("DB_PORT"),
        dbname=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PSW"),
    )


def load_seasonal_weather(connection):
    """
    Load actual historical seasonal weather data.

    Result structure:

    {
        ("Győr-Moson-Sopron", "maize"): [
            {
                "year": 1971,
                "temperature": ...,
                "precipitation": ...
            },
            ...
        ]
    }
    """

    with connection.cursor() as cursor:
        cursor.execute(
            """
            SELECT
                season_start_year,
                county_name,
                crop,
                season_mean_temperature_c,
                season_precipitation_mm
            FROM processed.weather_seasonal
            WHERE
                season_mean_temperature_c IS NOT NULL
                AND season_precipitation_mm IS NOT NULL
            ORDER BY
                county_name,
                crop,
                season_start_year;
            """
        )

        rows = cursor.fetchall()

    weather = {}

    for (
        season_start_year,
        county_name,
        crop,
        temperature,
        precipitation,
    ) in rows:

        key = (
            county_name,
            crop,
        )

        if key not in weather:
            weather[key] = []

        weather[key].append(
            {
                "year": int(season_start_year),
                "temperature": float(temperature),
                "precipitation": float(precipitation),
            }
        )

    print(
        f"County-crop weather histories loaded: "
        f"{len(weather)}"
    )

    return weather


def get_historical_training_targets(connection):
    """
    Determine historical ML training seasons from
    the actually available KSH yield records.

    Examples:

    KSH maize yield 2025
        -> season_start_year = 2025
        -> 2025-03-01 ... 2025-10-31

    KSH wheat yield 2025
        -> season_start_year = 2024
        -> 2024-09-01 ... 2025-07-31

    Barley and wheat uses the same.
    """

    with connection.cursor() as cursor:
        cursor.execute(
            """
            SELECT DISTINCT
                year,
                crop
            FROM processed.crop_production
            WHERE average_yield_t_ha IS NOT NULL
            ORDER BY
                year,
                crop;
            """
        )

        rows = cursor.fetchall()

    targets = []

    for yield_year, crop in rows:

        yield_year = int(yield_year)

        # Maize is sown and harvested
        # within the same calendar year.
        if crop == "maize":
            season_start_year = yield_year

        # Wheat / barley season starts
        # in the previous calendar year.
        elif crop in (
            "wheat",
            "barley",
        ):
            season_start_year = (
                yield_year - 1
            )

        else:
            raise ValueError(
                f"Unknown crop in "
                f"processed.crop_production: "
                f"{crop}"
            )

        targets.append(
            (
                crop,
                season_start_year,
            )
        )

    targets = sorted(
        set(targets),
        key=lambda value: (
            value[1],
            value[0],
        ),
    )

    print(
        f"Historical training targets found: "
        f"{len(targets)}"
    )

    return targets


def get_available_history(
    historical_seasons,
    target_year,
):
    return [
        row
        for row in historical_seasons
        if row["year"] < target_year
    ]


def calculate_expected_temperature(
    historical_seasons,
    target_year,
):
    """
    1. Take the last 15 completed seasons
       before the target season
    2. Fit a linear temperature trend
    3. Extrapolate the trend to the target year
    """

    available = get_available_history(
        historical_seasons,
        target_year,
    )

    available = available[
        -TEMPERATURE_HISTORY_LENGTH:
    ]

    if (
        len(available)
        < TEMPERATURE_HISTORY_LENGTH
    ):
        raise ValueError(
            f"Not enough temperature history "
            f"for target season {target_year}. "
            f"Required "
            f"{TEMPERATURE_HISTORY_LENGTH}, "
            f"found {len(available)}."
        )

    years = np.array(
        [
            row["year"]
            for row in available
        ],
        dtype=float,
    )

    temperatures = np.array(
        [
            row["temperature"]
            for row in available
        ],
        dtype=float,
    )

    # Linear model:
    # temperature =
    # slope * season_start_year + intercept

    slope, intercept = np.polyfit(
        years,
        temperatures,
        1,
    )

    expected_temperature = (
        slope * target_year
        + intercept
    )

    return float(
        expected_temperature
    )


def calculate_expected_precipitation(
    historical_seasons,
    target_year,
):
    """
    the mean precipitation of the
    previous 10 completed seasons
    """

    available = get_available_history(
        historical_seasons,
        target_year,
    )

    available = available[
        -PRECIPITATION_HISTORY_LENGTH:
    ]

    if (
        len(available)
        < PRECIPITATION_HISTORY_LENGTH
    ):
        raise ValueError(
            f"Not enough precipitation history "
            f"for target season {target_year}. "
            f"Required "
            f"{PRECIPITATION_HISTORY_LENGTH}, "
            f"found {len(available)}."
        )

    precipitation_values = [
        row["precipitation"]
        for row in available
    ]

    expected_precipitation = np.mean(
        precipitation_values
    )

    return float(
        expected_precipitation
    )


def build_records(
    weather,
    historical_targets,
):

    records = []

    counties = sorted(
        {
            county_name
            for county_name, _
            in weather.keys()
        }
    )

    if not counties:
        raise ValueError(
            "No counties found in "
            "seasonal weather data."
        )

    print(
        f"Counties found: "
        f"{len(counties)}"
    )

    for (
        crop,
        target_year,
    ) in historical_targets:

        for county_name in counties:

            key = (
                county_name,
                crop,
            )

            if key not in weather:
                raise ValueError(
                    f"No historical weather "
                    f"found for "
                    f"{county_name}, {crop}."
                )

            historical_seasons = (
                weather[key]
            )

            expected_temperature = (
                calculate_expected_temperature(
                    historical_seasons,
                    target_year,
                )
            )

            expected_precipitation = (
                calculate_expected_precipitation(
                    historical_seasons,
                    target_year,
                )
            )

            records.append(
                (
                    target_year,
                    county_name,
                    crop,
                    round(
                        expected_temperature,
                        3,
                    ),
                    round(
                        expected_precipitation,
                        2,
                    ),
                )
            )

    expected_record_count = (
        len(historical_targets)
        * len(counties)
    )

    if (
        len(records)
        != expected_record_count
    ):
        raise ValueError(
            f"Expected "
            f"{expected_record_count} records, "
            f"but prepared "
            f"{len(records)}."
        )

    return records


def import_records(
    connection,
    records,
):
    """
    Replace the historical expected-weather
    dataset with the newly calculated records.
    """

    with connection.cursor() as cursor:

        cursor.execute(
            """
            DELETE FROM
                processed.weather_expected;
            """
        )

        cursor.executemany(
            """
            INSERT INTO
                processed.weather_expected (
                    target_season_start_year,
                    county_name,
                    crop,
                    expected_season_temperature_c,
                    expected_season_precipitation_mm
                )
            VALUES (
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
    print(
        "Creating historical expected "
        "seasonal weather..."
    )

    connection = get_connection()

    try:

        # 1. Actual historical seasonal weather
        weather = load_seasonal_weather(
            connection
        )

        # 2. Historical seasons for whichreal KSH yield targets exist
        historical_targets = (
            get_historical_training_targets(
                connection
            )
        )

        # 3. Calculate what weather could
        #    have been expected BEFORE each
        #    historical target season
        records = build_records(
            weather,
            historical_targets,
        )

        print(
            f"Expected weather records "
            f"prepared: {len(records)}"
        )

        # 4. Save historical ML features
        import_records(
            connection,
            records,
        )

        connection.commit()

        print(
            "\nHistorical expected weather "
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