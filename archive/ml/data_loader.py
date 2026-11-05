# Common dataloader, all modell gets the data from this file

import os
from pathlib import Path

import pandas as pd
import psycopg
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent

FEATURE_COLUMNS = [
    "crop",
    "soil_ph",
    "soil_soc_g_kg",
    "soil_clay_pct",
    "expected_season_temperature_c",
    "expected_season_precipitation_mm",
    "recent_yield_mean_t_ha",
]

TARGET_COLUMN = "average_yield_t_ha"

def get_connection():
    load_dotenv(BASE_DIR / ".env")

    return psycopg.connect(
        host=os.getenv("DB_HOST"),
        port=os.getenv("DB_PORT"),
        dbname=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PSW"),
    )

# Postgresql -> sql select -> python rows -> pandas dataframe -> NUMERIC python numeric típus
def load_training_dataset():
    query = """
        SELECT
            yield_year,
            season_start_year,
            county_name,
            crop,

            soil_ph,
            soil_soc_g_kg,
            soil_clay_pct,

            expected_season_temperature_c,
            expected_season_precipitation_mm,

            recent_yield_mean_t_ha,

            average_yield_t_ha

        FROM ml.training_dataset

        ORDER BY
            yield_year,
            county_name,
            crop;
    """

    connection = get_connection()

    try:
        with connection.cursor() as cursor:
            cursor.execute(query)

            rows = cursor.fetchall()

            columns = [
                description.name
                for description in cursor.description
            ]

        dataframe = pd.DataFrame(
            rows,
            columns=columns,
        )

    finally:
        connection.close()

    numeric_columns = [
        "soil_ph",
        "soil_soc_g_kg",
        "soil_clay_pct",
        "expected_season_temperature_c",
        "expected_season_precipitation_mm",
        "recent_yield_mean_t_ha",
        "average_yield_t_ha",
    ]

    for column in numeric_columns:
        dataframe[column] = pd.to_numeric(
            dataframe[column]
        )

    return dataframe

# Train: 2005-2025, validation: 2021-2022, test: 2023-2025 /DINAMIC SPLIT, NOT HARDCODED/
def split_dataset(dataframe):
    """
    Dynamic split.

    Test:
        latest 3 complete years

    Validation:
        2 complete years before test

    Train:
        all earlier complete years
    """

    expected_records_per_year = (
        dataframe["county_name"].nunique()
        * dataframe["crop"].nunique()
    )

    records_per_year = (
        dataframe
        .groupby("yield_year")
        .size()
    )

    complete_years = (
        records_per_year[
            records_per_year
            == expected_records_per_year
        ]
        .index
        .sort_values()
        .tolist()
    )

    if len(complete_years) < 6:
        raise ValueError(
            "Not enough complete years "
            "for train/validation/test split."
        )

    test_years = complete_years[-3:]

    validation_years = complete_years[-5:-3]

    train_years = complete_years[:-5]

    train = dataframe[
        dataframe["yield_year"].isin(
            train_years
        )
    ].copy()

    validation = dataframe[
        dataframe["yield_year"].isin(
            validation_years
        )
    ].copy()

    test = dataframe[
        dataframe["yield_year"].isin(
            test_years
        )
    ].copy()

    return train, validation, test

# pl: 2026 nem teljes év
def get_complete_years(dataframe):
    """
    Return years for which every county-crop
    combination is present.
    """

    expected_records_per_year = (
        dataframe["county_name"].nunique()
        * dataframe["crop"].nunique()
    )

    records_per_year = (
        dataframe
        .groupby("yield_year")
        .size()
    )

    complete_years = (
        records_per_year[
            records_per_year
            == expected_records_per_year
        ]
        .index
        .sort_values()
        .tolist()
    )

    return complete_years

# current:
# fold1: t: 2005-2014 v: 2015-16
# fold2: t: 2005-2016 v: 2017-18 -> etc
# final test: 2023-2025
def expanding_window_splits(
    dataframe,
    test_year_count=3,
    validation_year_count=2,
    minimum_train_year_count=10,
):
    """
    Generate expanding-window validation folds

    The latest test_year_count complete years are always reserved as final test data

    Earlier years are divided into:
        expanding training window
        +
        following validation window
    """

    complete_years = get_complete_years(dataframe)

    required_years = (
        test_year_count
        + validation_year_count
        + minimum_train_year_count
    )

    if len(complete_years) < required_years:
        raise ValueError(
            "Not enough complete years "
            "for expanding-window validation."
        )

    test_years = complete_years[-test_year_count:]

    development_years = complete_years[:-test_year_count]

    folds = []

    train_end = minimum_train_year_count

    while (
        train_end
        + validation_year_count
        <= len(development_years)
    ):
        train_years = development_years[:train_end]

        validation_years = (
            development_years[
                train_end:
                train_end
                + validation_year_count
            ]
        )

        train = dataframe[
            dataframe["yield_year"].isin(
                train_years
            )
        ].copy()

        validation = dataframe[
            dataframe["yield_year"].isin(
                validation_years
            )
        ].copy()

        folds.append(
            (
                train,
                validation,
                train_years,
                validation_years,
            )
        )

        train_end += validation_year_count

    return folds, test_years