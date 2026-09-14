from datetime import date
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from pyproj import CRS
from rasterio.mask import mask
from shapely import wkt

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

SOILGRIDS_CRS = CRS.from_proj4(
    SOILGRIDS_PROJ
)


PH_FILES = [
    SOIL_DIR / "phh2o_0-5cm_mean.tif",
    SOIL_DIR / "phh2o_5-15cm_mean.tif",
    SOIL_DIR / "phh2o_15-30cm_mean.tif",
]

SOC_FILES = [
    SOIL_DIR / "soc_0-5cm_mean.tif",
    SOIL_DIR / "soc_5-15cm_mean.tif",
    SOIL_DIR / "soc_15-30cm_mean.tif",
]

CLAY_FILES = [
    SOIL_DIR / "clay_0-5cm_mean.tif",
    SOIL_DIR / "clay_5-15cm_mean.tif",
    SOIL_DIR / "clay_15-30cm_mean.tif",
]

DEPTH_WEIGHTS = np.array(
    [5, 10, 15],
    dtype=float,
)


def load_parcel_geometry(parcel_id):
    query = """
        SELECT ST_AsText(geometry)
        FROM app.parcel
        WHERE id = %s;
    """

    connection = get_connection()

    try:
        with connection.cursor() as cursor:
            cursor.execute(
                query,
                (parcel_id,),
            )

            row = cursor.fetchone()

    finally:
        connection.close()

    if row is None:
        raise ValueError(
            f"Parcel not found: {parcel_id}"
        )

    return wkt.loads(
        row[0]
    )


def transform_geometry_to_soilgrids(
    geometry,
):
    geodataframe = gpd.GeoDataFrame(
        geometry=[geometry],
        crs="EPSG:4326",
    )

    transformed = (
        geodataframe.to_crs(
            SOILGRIDS_CRS
        )
    )

    return transformed.geometry.iloc[0]


def calculate_raster_mean(
    raster_file,
    geometry,
):
    with rasterio.open(
        raster_file
    ) as raster:
        data, _ = mask(
            raster,
            [
                geometry.__geo_interface__
            ],
            crop=True,
            filled=False,
            all_touched=True,
        )

    values = data[0]

    if np.ma.is_masked(values):
        values = values.compressed()
    else:
        values = values.flatten()

    values = values[
        np.isfinite(values)
    ]

    if len(values) == 0:
        raise ValueError(
            "No SoilGrids pixels found in "
            f"{raster_file.name}."
        )

    return float(
        np.mean(values)
    )


def calculate_weighted_soil_value(
    files,
    geometry,
):
    layer_values = [
        calculate_raster_mean(
            file,
            geometry,
        )
        for file in files
    ]

    return float(
        np.average(
            layer_values,
            weights=DEPTH_WEIGHTS,
        )
    )


def calculate_parcel_soil(
    parcel_id,
):
    geometry = load_parcel_geometry(
        parcel_id
    )

    geometry = (
        transform_geometry_to_soilgrids(
            geometry
        )
    )

    soil_ph = (
        calculate_weighted_soil_value(
            PH_FILES,
            geometry,
        )
    )

    soil_soc = (
        calculate_weighted_soil_value(
            SOC_FILES,
            geometry,
        )
    )

    soil_clay = (
        calculate_weighted_soil_value(
            CLAY_FILES,
            geometry,
        )
    )

    return {
        "soil_ph": round(
            soil_ph / 10,
            2,
        ),
        "soil_soc_g_kg": round(
            soil_soc / 10,
            2,
        ),
        "soil_clay_pct": round(
            soil_clay / 10,
            2,
        ),
    }


def save_soilgrids_result(
    parcel_id,
    soil,
):
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
            cursor.execute(
                delete_query,
                (parcel_id,),
            )

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


def generate_parcel_soil(
    parcel_id,
):
    soil = calculate_parcel_soil(
        parcel_id
    )

    save_soilgrids_result(
        parcel_id,
        soil,
    )

    return soil


def resolve_parcel_soil(
    parcel_id,
):
    query = """
        SELECT
            source,
            soil_ph,
            soil_soc_g_kg,
            soil_clay_pct,
            measured_at
        FROM app.parcel_soil
        WHERE parcel_id = %s
        ORDER BY
            CASE
                WHEN source = 'user' THEN 1
                WHEN source = 'soilgrids' THEN 2
            END,
            measured_at DESC NULLS LAST,
            created_at DESC
        LIMIT 1;
    """

    connection = get_connection()

    try:
        with connection.cursor() as cursor:
            cursor.execute(
                query,
                (parcel_id,),
            )

            row = cursor.fetchone()

    finally:
        connection.close()

    if row is None:
        try:
            soil = generate_parcel_soil(
                parcel_id
            )

            return {
                "source": "soilgrids",
                **soil,
                "measured_at": None,
            }

        except ValueError as error:
            print(
                "SoilGrids unavailable "
                f"for parcel {parcel_id}: "
                f"{error}"
            )

            return None

    return {
        "source": row[0],
        "soil_ph": float(row[1]),
        "soil_soc_g_kg": float(row[2]),
        "soil_clay_pct": float(row[3]),
        "measured_at": (
            row[4].isoformat()
            if row[4] is not None
            else None
        ),
    }


def validate_user_soil_data(
    data,
):
    required_fields = [
        "soil_ph",
        "soil_soc_g_kg",
        "soil_clay_pct",
    ]

    for field in required_fields:
        if data.get(field) is None:
            raise ValueError(
                f"{field} is required."
            )

    try:
        soil_ph = float(
            data["soil_ph"]
        )

        soil_soc = float(
            data["soil_soc_g_kg"]
        )

        soil_clay = float(
            data["soil_clay_pct"]
        )

    except (
        TypeError,
        ValueError,
    ) as error:
        raise ValueError(
            "Soil values must be numeric."
        ) from error

    if not 0 <= soil_ph <= 14:
        raise ValueError(
            "Soil pH must be between 0 and 14."
        )

    if soil_soc < 0:
        raise ValueError(
            "Soil organic carbon cannot be negative."
        )

    if not 0 <= soil_clay <= 100:
        raise ValueError(
            "Clay content must be between 0 and 100%."
        )

    measured_at = data.get(
        "measured_at"
    )

    if measured_at in (
        "",
        None,
    ):
        measured_at = None

    elif isinstance(
        measured_at,
        date,
    ):
        pass

    else:
        try:
            measured_at = (
                date.fromisoformat(
                    str(measured_at)
                )
            )

        except ValueError as error:
            raise ValueError(
                "Measurement date must use "
                "YYYY-MM-DD format."
            ) from error

    return {
        "soil_ph": soil_ph,
        "soil_soc_g_kg": soil_soc,
        "soil_clay_pct": soil_clay,
        "measured_at": measured_at,
    }


def soil_input_changed(
    current_soil,
    new_soil,
):
    if current_soil is None:
        return True

    if (
        current_soil["source"]
        != "user"
    ):
        return True

    fields = [
        "soil_ph",
        "soil_soc_g_kg",
        "soil_clay_pct",
    ]

    for field in fields:
        difference = abs(
            float(current_soil[field])
            - float(new_soil[field])
        )

        if difference > 0.000001:
            return True

    return False


def save_user_soil(
    parcel_id,
    data,
):
    validated = (
        validate_user_soil_data(
            data
        )
    )

    current_soil = (
        resolve_parcel_soil(
            parcel_id
        )
    )

    revision_changed = (
        soil_input_changed(
            current_soil,
            validated,
        )
    )

    connection = get_connection()

    try:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT id
                FROM app.parcel_soil
                WHERE parcel_id = %s
                  AND source = 'user'
                ORDER BY
                    measured_at DESC
                        NULLS LAST,
                    created_at DESC
                LIMIT 1;
                """,
                (parcel_id,),
            )

            existing = (
                cursor.fetchone()
            )

            if existing is None:
                cursor.execute(
                    """
                    INSERT INTO app.parcel_soil (
                        parcel_id,
                        source,
                        soil_ph,
                        soil_soc_g_kg,
                        soil_clay_pct,
                        measured_at
                    )
                    VALUES (
                        %s,
                        'user',
                        %s,
                        %s,
                        %s,
                        %s
                    )
                    RETURNING
                        id,
                        parcel_id,
                        source,
                        soil_ph,
                        soil_soc_g_kg,
                        soil_clay_pct,
                        measured_at,
                        created_at;
                    """,
                    (
                        parcel_id,
                        validated["soil_ph"],
                        validated[
                            "soil_soc_g_kg"
                        ],
                        validated[
                            "soil_clay_pct"
                        ],
                        validated[
                            "measured_at"
                        ],
                    ),
                )

            else:
                user_soil_id = (
                    existing[0]
                )

                cursor.execute(
                    """
                    UPDATE app.parcel_soil
                    SET
                        soil_ph = %s,
                        soil_soc_g_kg = %s,
                        soil_clay_pct = %s,
                        measured_at = %s
                    WHERE id = %s
                    RETURNING
                        id,
                        parcel_id,
                        source,
                        soil_ph,
                        soil_soc_g_kg,
                        soil_clay_pct,
                        measured_at,
                        created_at;
                    """,
                    (
                        validated["soil_ph"],
                        validated[
                            "soil_soc_g_kg"
                        ],
                        validated[
                            "soil_clay_pct"
                        ],
                        validated[
                            "measured_at"
                        ],
                        user_soil_id,
                    ),
                )

            row = cursor.fetchone()

            # Remove possible legacy duplicate
            # user measurements. The latest
            # user measurement is the active one.
            cursor.execute(
                """
                DELETE FROM app.parcel_soil
                WHERE parcel_id = %s
                  AND source = 'user'
                  AND id <> %s;
                """,
                (
                    parcel_id,
                    row[0],
                ),
            )

            if revision_changed:
                cursor.execute(
                    """
                    UPDATE app.parcel
                    SET
                        revision =
                            revision + 1,
                        updated_at = NOW()
                    WHERE id = %s;
                    """,
                    (parcel_id,),
                )

        connection.commit()

    except Exception:
        connection.rollback()
        raise

    finally:
        connection.close()

    return {
        "id": row[0],
        "parcel_id": row[1],
        "source": row[2],
        "soil_ph": float(row[3]),
        "soil_soc_g_kg": float(row[4]),
        "soil_clay_pct": float(row[5]),
        "measured_at": (
            row[6].isoformat()
            if row[6] is not None
            else None
        ),
        "created_at": (
            row[7].isoformat()
        ),
    }


def delete_user_soil(
    parcel_id,
):
    query = """
        DELETE FROM app.parcel_soil
        WHERE parcel_id = %s
          AND source = 'user';
    """

    connection = get_connection()

    try:
        with connection.cursor() as cursor:
            cursor.execute(
                query,
                (parcel_id,),
            )

            deleted_count = (
                cursor.rowcount
            )

            if deleted_count > 0:
                cursor.execute(
                    """
                    UPDATE app.parcel
                    SET
                        revision =
                            revision + 1,
                        updated_at = NOW()
                    WHERE id = %s;
                    """,
                    (parcel_id,),
                )

        connection.commit()

    except Exception:
        connection.rollback()
        raise

    finally:
        connection.close()

    return deleted_count > 0


def delete_soilgrids_result(
    parcel_id,
):
    query = """
        DELETE FROM app.parcel_soil
        WHERE parcel_id = %s
          AND source = 'soilgrids';
    """

    connection = get_connection()

    try:
        with connection.cursor() as cursor:
            cursor.execute(
                query,
                (parcel_id,),
            )

        connection.commit()

    except Exception:
        connection.rollback()
        raise

    finally:
        connection.close()