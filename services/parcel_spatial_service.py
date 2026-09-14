from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from pyproj import CRS
from rasterio.mask import mask
from shapely import wkt
import json
from ml.data_loader import get_connection


BASE_DIR = Path(__file__).resolve().parent.parent
SOIL_DIR = BASE_DIR / "data" / "raw" / "soilgrids"

SOILGRIDS_PROJ = (
    "+proj=igh "
    "+lat_0=0 "
    "+lon_0=0 "
    "+datum=WGS84 "
    "+units=m "
    "+no_defs"
)

SOILGRIDS_CRS = CRS.from_proj4(SOILGRIDS_PROJ)

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

DEPTH_WEIGHTS = {
    "0-5cm": 5,
    "5-15cm": 10,
    "15-30cm": 15,
}


def load_parcel_geometry(parcel_id):
    query = """
        SELECT ST_AsText(geometry)
        FROM app.parcel
        WHERE id = %s;
    """

    connection = get_connection()

    try:
        with connection.cursor() as cursor:
            cursor.execute(query, (parcel_id,))
            row = cursor.fetchone()
    finally:
        connection.close()

    if row is None:
        raise ValueError(f"Parcel not found: {parcel_id}")

    return wkt.loads(row[0])


def transform_to_soilgrids_crs(geometry):
    dataframe = gpd.GeoDataFrame(
        geometry=[geometry],
        crs="EPSG:4326",
    )

    dataframe = dataframe.to_crs(SOILGRIDS_CRS)

    return dataframe.geometry.iloc[0]


def calculate_raster_mean(raster_file, geometry):
    if not raster_file.exists():
        raise FileNotFoundError(f"Missing SoilGrids file: {raster_file}")

    with rasterio.open(raster_file) as src:
        masked, _ = mask(
            src,
            [geometry],
            crop=True,
            filled=False,
            all_touched=True,
        )

    valid_values = masked[0].compressed()

    if len(valid_values) == 0:
        raise ValueError(
            f"No valid SoilGrids values found in {raster_file.name}."
        )

    return float(np.mean(valid_values))


def calculate_property(property_name, geometry):
    weighted_sum = 0.0
    total_depth = 0

    for depth, raster_file in LAYERS[property_name].items():
        value = calculate_raster_mean(
            raster_file,
            geometry,
        )

        weight = DEPTH_WEIGHTS[depth]

        weighted_sum += value * weight
        total_depth += weight

    raw_value = weighted_sum / total_depth

    return raw_value / 10.0


def calculate_parcel_soil(parcel_id):
    geometry = load_parcel_geometry(parcel_id)
    geometry = transform_to_soilgrids_crs(geometry)

    return {
        "soil_ph": round(
            calculate_property("phh2o", geometry),
            2,
        ),
        "soil_soc_g_kg": round(
            calculate_property("soc", geometry),
            2,
        ),
        "soil_clay_pct": round(
            calculate_property("clay", geometry),
            2,
        ),
    }


def save_soilgrids_result(parcel_id, soil):
    delete_query = """
        DELETE FROM app.parcel_soil
        WHERE parcel_id = %s
          AND source = 'soilgrids';
    """

    insert_query = """
        INSERT INTO app.parcel_soil (
            parcel_id,
            source,
            soil_ph,
            soil_soc_g_kg,
            soil_clay_pct
        )
        VALUES (
            %s,
            'soilgrids',
            %s,
            %s,
            %s
        );
    """

    connection = get_connection()

    try:
        with connection.cursor() as cursor:
            cursor.execute(delete_query, (parcel_id,))

            cursor.execute(
                insert_query,
                (
                    parcel_id,
                    soil["soil_ph"],
                    soil["soil_soc_g_kg"],
                    soil["soil_clay_pct"],
                ),
            )

        connection.commit()

    except Exception:
        connection.rollback()
        raise

    finally:
        connection.close()


def generate_parcel_soil(parcel_id):
    soil = calculate_parcel_soil(parcel_id)
    save_soilgrids_result(parcel_id, soil)

    return soil


def resolve_parcel_county(parcel_id):
    query = """
        SELECT cb.county_name
        FROM app.parcel p
        JOIN processed.county_boundary cb
          ON ST_Covers(
              cb.geometry,
              ST_PointOnSurface(p.geometry)
          )
        WHERE p.id = %s;
    """

    connection = get_connection()

    try:
        with connection.cursor() as cursor:
            cursor.execute(query, (parcel_id,))
            row = cursor.fetchone()
    finally:
        connection.close()

    if row is None:
        raise ValueError(
            f"County could not be resolved for parcel {parcel_id}."
        )

    return row[0]

def resolve_geojson_county(geometry):
    query = """
        SELECT county_name
        FROM processed.county_boundary
        WHERE ST_Covers(
            geometry,
            ST_PointOnSurface(
                ST_SetSRID(ST_GeomFromGeoJSON(%s), 4326)
            )
        )
        LIMIT 1;
    """

    connection = get_connection()

    try:
        with connection.cursor() as cursor:
            cursor.execute(
                query,
                (json.dumps(geometry),),
            )
            row = cursor.fetchone()
    finally:
        connection.close()

    if row is None:
        raise ValueError(
            "The parcel must be located within a supported Hungarian county."
        )

    return row[0]

