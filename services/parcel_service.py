import json

from ml.data_loader import get_connection

from services.parcel_season_service import (
    list_parcel_seasons,
)

from services.parcel_soil_service import (
    resolve_parcel_soil,
)

from services.parcel_spatial_service import (
    resolve_geojson_county,
    resolve_parcel_county,
)


def parcel_row_to_dict(row):
    return {
        "id": row[0],
        "farm_id": row[1],
        "name": row[2],
        "official_area_ha": float(row[3]),
        "geometry": json.loads(row[4]),
        "created_at": row[5].isoformat(),
        "updated_at": row[6].isoformat(),
        "is_active": row[7],
        "deactivated_at": (
            row[8].isoformat()
            if row[8] is not None
            else None
        ),
    }


def validate_create_parcel_data(data):
    if not data.get("name"):
        raise ValueError(
            "Parcel name is required."
        )

    if data.get("official_area_ha") is None:
        raise ValueError(
            "Official area is required."
        )

    if float(data["official_area_ha"]) <= 0:
        raise ValueError(
            "Official area must be greater than 0."
        )

    geometry = data.get("geometry")

    if not geometry:
        raise ValueError(
            "Parcel geometry is required."
        )

    if geometry.get("type") != "Polygon":
        raise ValueError(
            "Geometry must be a GeoJSON Polygon."
        )


def validate_update_parcel_data(data):
    name = data.get("name")

    if not name:
        raise ValueError(
            "Parcel name is required."
        )


def ensure_farm_owned_by_user(
    farm_id,
    user_id,
):
    query = """
        SELECT 1
        FROM app.farm
        WHERE id = %s
          AND user_id = %s;
    """

    connection = get_connection()

    try:
        with connection.cursor() as cursor:
            cursor.execute(
                query,
                (
                    farm_id,
                    user_id,
                ),
            )

            row = cursor.fetchone()

    finally:
        connection.close()

    if row is None:
        raise ValueError(
            f"Farm not found: {farm_id}"
        )


def enrich_parcel(
    parcel,
    county_name,
):
    soil = resolve_parcel_soil(
        parcel["id"]
    )

    return {
        **parcel,
        "county_name": county_name,
        "soil": soil,
    }


def create_parcel(
    user_id,
    farm_id,
    data,
):
    validate_create_parcel_data(data)

    ensure_farm_owned_by_user(
        farm_id,
        user_id,
    )

    county_name = resolve_geojson_county(
        data["geometry"]
    )

    query = """
        INSERT INTO app.parcel (
            farm_id,
            name,
            geometry,
            official_area_ha
        )
        VALUES (
            %s,
            %s,
            ST_SetSRID(
                ST_GeomFromGeoJSON(%s),
                4326
            ),
            %s
        )
        RETURNING
            id,
            farm_id,
            name,
            official_area_ha,
            ST_AsGeoJSON(geometry),
            created_at,
            updated_at,
            is_active,
            deactivated_at;
    """

    connection = get_connection()

    try:
        with connection.cursor() as cursor:
            cursor.execute(
                query,
                (
                    farm_id,
                    data["name"].strip(),
                    json.dumps(
                        data["geometry"]
                    ),
                    data["official_area_ha"],
                ),
            )

            row = cursor.fetchone()

        connection.commit()

    except Exception:
        connection.rollback()
        raise

    finally:
        connection.close()

    parcel = parcel_row_to_dict(
        row
    )

    return enrich_parcel(
        parcel,
        county_name,
    )


def get_parcel(
    parcel_id,
    user_id,
):
    query = """
        SELECT
            p.id,
            p.farm_id,
            p.name,
            p.official_area_ha,
            ST_AsGeoJSON(p.geometry),
            p.created_at,
            p.updated_at,
            p.is_active,
            p.deactivated_at
        FROM app.parcel p
        JOIN app.farm f
          ON f.id = p.farm_id
        WHERE p.id = %s
          AND f.user_id = %s;
    """

    connection = get_connection()

    try:
        with connection.cursor() as cursor:
            cursor.execute(
                query,
                (
                    parcel_id,
                    user_id,
                ),
            )

            row = cursor.fetchone()

    finally:
        connection.close()

    if row is None:
        raise ValueError(
            f"Parcel not found: {parcel_id}"
        )

    return parcel_row_to_dict(
        row
    )


def get_parcel_details(
    parcel_id,
    user_id,
):
    parcel = get_parcel(
        parcel_id,
        user_id,
    )

    county_name = resolve_parcel_county(
        parcel_id
    )

    soil = resolve_parcel_soil(
        parcel_id
    )

    seasons = list_parcel_seasons(
        parcel_id,
        user_id,
    )

    return {
        **parcel,
        "county_name": county_name,
        "soil": soil,
        "seasons": seasons,
    }


def list_parcels(
    farm_id,
    user_id,
):
    query = """
        SELECT
            p.id,
            p.farm_id,
            p.name,
            p.official_area_ha,
            ST_AsGeoJSON(p.geometry),
            p.created_at,
            p.updated_at,
            p.is_active,
            p.deactivated_at
        FROM app.parcel p
        JOIN app.farm f
          ON f.id = p.farm_id
        WHERE p.farm_id = %s
          AND f.user_id = %s
        ORDER BY
            p.is_active DESC,
            p.name;
    """

    connection = get_connection()

    try:
        with connection.cursor() as cursor:
            cursor.execute(
                query,
                (
                    farm_id,
                    user_id,
                ),
            )

            rows = cursor.fetchall()

    finally:
        connection.close()

    return [
        parcel_row_to_dict(row)
        for row in rows
    ]


def update_parcel(
    parcel_id,
    user_id,
    data,
):
    validate_update_parcel_data(
        data
    )

    query = """
        UPDATE app.parcel
        SET
            name = %s,
            updated_at = NOW()
        WHERE id = %s
          AND farm_id IN (
              SELECT id
              FROM app.farm
              WHERE user_id = %s
          )
        RETURNING
            id,
            farm_id,
            name,
            official_area_ha,
            ST_AsGeoJSON(geometry),
            created_at,
            updated_at,
            is_active,
            deactivated_at;
    """

    connection = get_connection()

    try:
        with connection.cursor() as cursor:
            cursor.execute(
                query,
                (
                    data["name"].strip(),
                    parcel_id,
                    user_id,
                ),
            )

            row = cursor.fetchone()

        if row is None:
            connection.rollback()

            raise ValueError(
                f"Parcel not found: {parcel_id}"
            )

        connection.commit()

    except Exception:
        connection.rollback()
        raise

    finally:
        connection.close()

    parcel = parcel_row_to_dict(
        row
    )

    county_name = resolve_parcel_county(
        parcel_id
    )

    return enrich_parcel(
        parcel,
        county_name,
    )


def set_parcel_active(
    parcel_id,
    user_id,
    is_active,
):
    query = """
        UPDATE app.parcel
        SET
            is_active = %s,
            deactivated_at =
                CASE
                    WHEN %s = FALSE
                    THEN NOW()
                    ELSE NULL
                END,
            updated_at = NOW()
        WHERE id = %s
          AND farm_id IN (
              SELECT id
              FROM app.farm
              WHERE user_id = %s
          )
        RETURNING
            id,
            farm_id,
            name,
            official_area_ha,
            ST_AsGeoJSON(geometry),
            created_at,
            updated_at,
            is_active,
            deactivated_at;
    """

    connection = get_connection()

    try:
        with connection.cursor() as cursor:
            cursor.execute(
                query,
                (
                    is_active,
                    is_active,
                    parcel_id,
                    user_id,
                ),
            )

            row = cursor.fetchone()

        if row is None:
            connection.rollback()

            raise ValueError(
                f"Parcel not found: {parcel_id}"
            )

        connection.commit()

    except Exception:
        connection.rollback()
        raise

    finally:
        connection.close()

    parcel = parcel_row_to_dict(
        row
    )

    county_name = resolve_parcel_county(
        parcel_id
    )

    return enrich_parcel(
        parcel,
        county_name,
    )


def parcel_has_history(
    parcel_id,
):
    query = """
        SELECT EXISTS (
            SELECT 1
            FROM app.parcel_season
            WHERE parcel_id = %s
        );
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

    return bool(row[0])


def delete_parcel(
    parcel_id,
    user_id,
):
    parcel = get_parcel(
        parcel_id,
        user_id,
    )

    if parcel_has_history(
        parcel_id
    ):
        raise ValueError(
            "This parcel has historical farming data. "
            "Set the parcel inactive instead of deleting it."
        )

    query = """
        DELETE FROM app.parcel
        WHERE id = %s
          AND farm_id IN (
              SELECT id
              FROM app.farm
              WHERE user_id = %s
          )
        RETURNING id;
    """

    connection = get_connection()

    try:
        with connection.cursor() as cursor:
            cursor.execute(
                query,
                (
                    parcel_id,
                    user_id,
                ),
            )

            row = cursor.fetchone()

        if row is None:
            connection.rollback()

            raise ValueError(
                f"Parcel not found: {parcel_id}"
            )

        connection.commit()

    except Exception:
        connection.rollback()
        raise

    finally:
        connection.close()