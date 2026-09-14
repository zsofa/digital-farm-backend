import os
from pathlib import Path

import geopandas as gpd
import numpy as np
import psycopg
import rasterio
from dotenv import load_dotenv
from rasterio.mask import mask
from pyproj import CRS


BASE_DIR = Path(__file__).resolve().parent.parent

SOIL_DIR = (
    BASE_DIR/ "data" / "raw" / "soilgrids"
)

NUTS_FILE = (
    BASE_DIR / "data" / "raw" / "geography" / "NUTS_RG_01M_2024_4326.geojson"
)

SOILGRIDS_PROJ = (
    "+proj=igh "
    "+lat_0=0 "
    "+lon_0=0 "
    "+datum=WGS84 "
    "+units=m "
    "+no_defs"
)

SOILGRIDS_CRS = CRS.from_proj4(
    SOILGRIDS_PROJ
)


LAYERS = {
    "phh2o": {
        "0-5cm": SOIL_DIR / "phh2o_0-5cm_mean.tif",
        "5-15cm": SOIL_DIR / "phh2o_5-15cm_mean.tif",
        "15-30cm": SOIL_DIR / "phh2o_15-30cm_mean.tif",
    },
    "soc": {
        "0-5cm": SOIL_DIR / "soc_0-5cm_mean.tif",
        "5-15cm": SOIL_DIR / "soc_5-15cm_mean.tif",
        "15-30cm": SOIL_DIR / "soc_15-30cm_mean.tif",
    },
    "clay": {
        "0-5cm": SOIL_DIR / "clay_0-5cm_mean.tif",
        "5-15cm": SOIL_DIR / "clay_5-15cm_mean.tif",
        "15-30cm": SOIL_DIR / "clay_15-30cm_mean.tif",
    },
}


# Thickness of SoilGrids depth intervals in cm
DEPTH_WEIGHTS = {
    "0-5cm": 5,
    "5-15cm": 10,
    "15-30cm": 15,
}


def load_counties():
    nuts = gpd.read_file(NUTS_FILE)

    counties = nuts[
        (nuts["CNTR_CODE"] == "HU")
        & (nuts["LEVL_CODE"] == 3)
        & (nuts["NUTS_NAME"] != "Budapest")
    ].copy()

    if len(counties) != 19:
        raise ValueError(
            f"Expected 19 counties, found {len(counties)}."
        )

    print(f"Counties loaded: {len(counties)}")

    return counties


def calculate_county_mean(
    raster_file,
    counties
):
    results = {}

    with rasterio.open(raster_file) as src:
        print(f"Raster crs data: {src.crs}")

        # County polygons must use the same CRs as the SoilGrids raster
        counties_projected = counties.to_crs(
            SOILGRIDS_CRS
        )

        for _, county in counties_projected.iterrows():

            county_name = county["NUTS_NAME"]

            masked, _ = mask(
                src,
                [county.geometry],
                crop=True,
                filled=False,
            )

            values = masked[0]

            valid_values = values.compressed()

            if len(valid_values) == 0:
                raise ValueError(
                    f"No valid soil values for {county_name} "
                    f"in {raster_file.name}."
                )

            results[county_name] = float(
                np.mean(valid_values)
            )

    return results


def calculate_property(
    property_name,
    counties
):
    print(f"\nProcessing property: {property_name}")

    depth_results = {}

    for depth, raster_file in (
        LAYERS[property_name].items()
    ):

        if not raster_file.exists():
            raise FileNotFoundError(
                f"Missing file: {raster_file}"
            )

        print(
            f"Processing {raster_file.name}..."
        )

        depth_results[depth] = (
            calculate_county_mean(
                raster_file,
                counties
            )
        )

    counties_names = depth_results[
        "0-5cm"
    ].keys()

    weighted_results = {}

    for county_name in counties_names:

        weighted_sum = 0.0
        total_depth = 0

        for depth, weight in (
            DEPTH_WEIGHTS.items()
        ):

            weighted_sum += (
                depth_results[depth][county_name]
                * weight
            )

            total_depth += weight

        # 0-30 cm thickness-weighted value
        raw_value = (
            weighted_sum
            / total_depth
        )

        # SoilGrids conversion factor:
        # phh2o -> pH
        # soc -> g/kg
        # clay -> %
        converted_value = raw_value / 10.0

        weighted_results[county_name] = (
            converted_value
        )

    return weighted_results


def build_records(
    ph,
    soc,
    clay
):
    if not (
        set(ph.keys())
        == set(soc.keys())
        == set(clay.keys())
    ):
        raise ValueError(
            "County sets do not match between soil properties."
        )

    if len(ph) != 19:
        raise ValueError(
            f"Expected 19 counties, found {len(ph)}."
        )

    records = []

    for county_name in sorted(ph.keys()):

        records.append(
            (
                county_name,
                round(ph[county_name], 2),
                round(soc[county_name], 2),
                round(clay[county_name], 2),
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


def import_records(
    connection,
    records
):
    with connection.cursor() as cursor:

        cursor.execute(
            """
            DELETE FROM processed.soil;
            """
        )

        cursor.executemany(
            """
            INSERT INTO processed.soil (
                county_name,
                soil_ph,
                soil_soc_g_kg,
                soil_clay_pct
            )
            VALUES (
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
        "Processing SoilGrids data..."
    )

    counties = load_counties()

    ph = calculate_property("phh2o", counties)

    soc = calculate_property("soc", counties)

    clay = calculate_property("clay", counties)

    records = build_records(ph, soc, clay)

    print(f"\nSoil records prepared: {len(records)}")

    for record in records:
        print(record)

    connection = get_connection()

    try:
        import_records(connection, records)

        connection.commit()

        print("\nSoilGrids import was successful.")
        print(f"Imported records: {len(records)}")

    except Exception:
        connection.rollback()
        raise

    finally:
        connection.close()


if __name__ == "__main__":
    main()