
from db.connection import get_connection
import pandas as pd
from ml.scenarios.features import TARGET_COLUMN


def load_dataset():
    query = """
        SELECT
            t.yield_year,
            t.county_name,
            t.crop,
            t.soil_ph,
            t.soil_soc_g_kg,
            t.soil_clay_pct,
            w.season_mean_temperature_c AS season_temperature_c,
            w.season_precipitation_mm AS season_precipitation_mm,
            t.recent_yield_mean_t_ha,
            t.average_yield_t_ha
        FROM ml.training_dataset t
        JOIN processed.weather_seasonal w
          ON w.season_start_year = t.season_start_year
         AND w.county_name = t.county_name
         AND w.crop = t.crop
        ORDER BY t.yield_year, t.county_name, t.crop;
    """

    connection = get_connection()

    try:
        with connection.cursor() as cursor:
            cursor.execute(query)
            rows = cursor.fetchall()
            columns = [description.name for description in cursor.description]
    finally:
        connection.close()

    dataframe = pd.DataFrame(rows, columns=columns)

    numeric_columns = [
        "soil_ph",
        "soil_soc_g_kg",
        "soil_clay_pct",
        "season_temperature_c",
        "season_precipitation_mm",
        "recent_yield_mean_t_ha",
        TARGET_COLUMN,
    ]

    for column in numeric_columns:
        dataframe[column] = pd.to_numeric(dataframe[column])

    return dataframe