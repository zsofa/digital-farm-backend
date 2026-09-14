import json

from ml.data_loader import get_connection
from services.parcel_spatial_service import resolve_parcel_county


ALLOWED_ACTIVITY_TYPES = {
    "crop_production",
    "livestock",
    "mixed",
}

ALLOWED_LEGAL_FORMS = {
    "individual",
    "family_farm",
    "company",
    "other",
}


def validate_farm_data(data):
    if not data.get("name"):
        raise ValueError(
            "Farm name is required."
        )

    if not data.get("county_name"):
        raise ValueError(
            "County is required."
        )

    activity_type = data.get(
        "activity_type"
    )

    if (
        activity_type
        not in ALLOWED_ACTIVITY_TYPES
    ):
        raise ValueError(
            "Invalid activity type."
        )

    legal_form = data.get(
        "legal_form"
    )

    if (
        legal_form is not None
        and legal_form not in ALLOWED_LEGAL_FORMS
    ):
        raise ValueError(
            "Invalid legal form."
        )


def create_farm(
    user_id,
    data,
):
    validate_farm_data(data)

    query = """
        INSERT INTO app.farm (
            user_id,
            name,
            county_name,
            activity_type,
            legal_form,
            description
        )
        VALUES (
            %s,
            %s,
            %s,
            %s,
            %s,
            %s
        )
        RETURNING
            id,
            name,
            county_name,
            activity_type,
            legal_form,
            description,
            created_at,
            updated_at;
    """

    connection = get_connection()

    try:
        with connection.cursor() as cursor:
            cursor.execute(
                query,
                (
                    user_id,
                    data["name"],
                    data["county_name"],
                    data["activity_type"],
                    data.get("legal_form"),
                    data.get("description"),
                ),
            )

            row = cursor.fetchone()

        connection.commit()

    except Exception:
        connection.rollback()
        raise

    finally:
        connection.close()

    return farm_row_to_dict(
        row
    )


def get_farm(
    farm_id,
    user_id,
):
    query = """
        SELECT
            id,
            name,
            county_name,
            activity_type,
            legal_form,
            description,
            created_at,
            updated_at
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

    return farm_row_to_dict(
        row
    )


def list_farms(
    user_id,
):
    query = """
        SELECT
            id,
            name,
            county_name,
            activity_type,
            legal_form,
            description,
            created_at,
            updated_at
        FROM app.farm
        WHERE user_id = %s
        ORDER BY name;
    """

    connection = get_connection()

    try:
        with connection.cursor() as cursor:
            cursor.execute(
                query,
                (user_id,),
            )

            rows = cursor.fetchall()

    finally:
        connection.close()

    return [
        farm_row_to_dict(row)
        for row in rows
    ]


def update_farm(
    farm_id,
    user_id,
    data,
):
    validate_farm_data(data)

    query = """
        UPDATE app.farm
        SET
            name = %s,
            county_name = %s,
            activity_type = %s,
            legal_form = %s,
            description = %s,
            updated_at = NOW()
        WHERE id = %s
          AND user_id = %s
        RETURNING
            id,
            name,
            county_name,
            activity_type,
            legal_form,
            description,
            created_at,
            updated_at;
    """

    connection = get_connection()

    try:
        with connection.cursor() as cursor:
            cursor.execute(
                query,
                (
                    data["name"],
                    data["county_name"],
                    data["activity_type"],
                    data.get("legal_form"),
                    data.get("description"),
                    farm_id,
                    user_id,
                ),
            )

            row = cursor.fetchone()

        if row is None:
            connection.rollback()

            raise ValueError(
                f"Farm not found: {farm_id}"
            )

        connection.commit()

    except Exception:
        connection.rollback()
        raise

    finally:
        connection.close()

    return farm_row_to_dict(
        row
    )


def delete_farm(
    farm_id,
    user_id,
):
    query = """
        DELETE FROM app.farm
        WHERE id = %s
          AND user_id = %s
        RETURNING id;
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

        if row is None:
            connection.rollback()

            raise ValueError(
                f"Farm not found: {farm_id}"
            )

        connection.commit()

    except Exception:
        connection.rollback()
        raise

    finally:
        connection.close()


def get_farm_details(
    farm_id,
    user_id,
):
    farm = get_farm(
        farm_id,
        user_id,
    )

    query = """
        SELECT
            id,
            name,
            official_area_ha,
            ST_AsGeoJSON(geometry),
            is_active,
            deactivated_at
        FROM app.parcel
        WHERE farm_id = %s
        ORDER BY
            is_active DESC,
            name;
    """

    connection = get_connection()

    try:
        with connection.cursor() as cursor:
            cursor.execute(
                query,
                (farm_id,),
            )

            rows = cursor.fetchall()

    finally:
        connection.close()

    parcels = []

    for row in rows:
        parcel_id = row[0]

        parcels.append({
            "id": parcel_id,
            "name": row[1],
            "official_area_ha": float(
                row[2]
            ),
            "county_name":
                resolve_parcel_county(
                    parcel_id
                ),
            "geometry": json.loads(
                row[3]
            ),
            "is_active": row[4],
            "deactivated_at": (
                row[5].isoformat()
                if row[5] is not None
                else None
            ),
        })

    total_area_ha = sum(
        parcel["official_area_ha"]
        for parcel in parcels
    )

    return {
        **farm,
        "parcel_count": len(
            parcels
        ),
        "total_area_ha": round(
            total_area_ha,
            3,
        ),
        "parcels": parcels,
    }


def farm_row_to_dict(
    row,
):
    return {
        "id": row[0],
        "name": row[1],
        "county_name": row[2],
        "activity_type": row[3],
        "legal_form": row[4],
        "description": row[5],
        "created_at": (
            row[6].isoformat()
        ),
        "updated_at": (
            row[7].isoformat()
        ),
    }