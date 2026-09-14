from ml.data_loader import get_connection

from services.authorization_service import (
    ensure_parcel_owned_by_user,
    ensure_parcel_season_owned_by_user,
)
from services.crop_season_service import (
    SUPPORTED_CROPS,
    resolve_next_full_season,
)
from services.yield_simulation_service import (
    list_simulations_for_parcel_season,
)


ALLOWED_STRATEGIES = {
    "conventional",
    "reduced",
}

ALLOWED_MACHINE_USE = {
    "low",
    "medium",
    "high",
}

ALLOWED_FERTILIZERS = {
    "AN",
    "urea",
    "CAN",
}


def validate_management_data(data):
    farming_strategy = data.get(
        "farming_strategy"
    )

    if (
        farming_strategy
        not in ALLOWED_STRATEGIES
    ):
        raise ValueError(
            "Invalid farming strategy."
        )

    is_irrigated = data.get(
        "is_irrigated"
    )

    if not isinstance(
        is_irrigated,
        bool,
    ):
        raise ValueError(
            "is_irrigated must be true or false."
        )

    machine_use = data.get(
        "machine_use"
    )

    if (
        machine_use is not None
        and machine_use
        not in ALLOWED_MACHINE_USE
    ):
        raise ValueError(
            "Invalid machine use."
        )

    fertilizer_type = data.get(
        "fertilizer_type"
    )

    if (
        fertilizer_type is not None
        and fertilizer_type
        not in ALLOWED_FERTILIZERS
    ):
        raise ValueError(
            "Invalid fertilizer type."
        )

    quantity = data.get(
        "fertilizer_quantity_kg_ha"
    )

    if quantity is not None:
        try:
            quantity = float(
                quantity
            )

        except (
            TypeError,
            ValueError,
        ) as error:
            raise ValueError(
                "Fertilizer quantity must be numeric."
            ) from error

        if quantity < 0:
            raise ValueError(
                "Fertilizer quantity cannot be negative."
            )

    if (
        fertilizer_type is None
        and quantity is not None
        and quantity > 0
    ):
        raise ValueError(
            "Fertilizer type is required when "
            "fertilizer quantity is provided."
        )

    if (
        fertilizer_type is not None
        and quantity is None
    ):
        raise ValueError(
            "Fertilizer quantity is required when "
            "a fertilizer type is selected."
        )

    if fertilizer_type is None:
        quantity = None

    return {
        "farming_strategy":
            farming_strategy,

        "is_irrigated":
            is_irrigated,

        "machine_use":
            machine_use,

        "fertilizer_type":
            fertilizer_type,

        "fertilizer_quantity_kg_ha":
            quantity,
    }


def validate_create_data(data):
    if "season_start_year" in data:
        raise ValueError(
            "Season start year is determined automatically."
        )

    crop = data.get(
        "crop"
    )

    if crop not in SUPPORTED_CROPS:
        raise ValueError(
            "Invalid crop."
        )

    management = (
        validate_management_data(
            data
        )
    )

    return {
        "crop": crop,
        **management,
    }


def season_row_to_dict(row):
    return {
        "id": row[0],
        "parcel_id": row[1],
        "season_start_year": row[2],
        "crop": row[3],
        "farming_strategy": row[4],
        "is_irrigated": row[5],
        "machine_use": row[6],
        "fertilizer_type": row[7],
        "fertilizer_quantity_kg_ha": (
            float(row[8])
            if row[8] is not None
            else None
        ),
        "created_at":
            row[9].isoformat(),
        "updated_at":
            row[10].isoformat(),
    }


def ensure_parcel_active(
    parcel_id,
):
    query = """
        SELECT is_active
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

    if not row[0]:
        raise ValueError(
            "Inactive parcels cannot be modified."
        )


def ensure_unique_season(
    parcel_id,
    season_start_year,
    crop,
):
    query = """
        SELECT id
        FROM app.parcel_season
        WHERE parcel_id = %s
          AND season_start_year = %s
          AND crop = %s
        LIMIT 1;
    """

    connection = get_connection()

    try:
        with connection.cursor() as cursor:
            cursor.execute(
                query,
                (
                    parcel_id,
                    season_start_year,
                    crop,
                ),
            )

            row = cursor.fetchone()

    finally:
        connection.close()

    if row is not None:
        raise ValueError(
            "A growing plan for the next "
            f"{crop} season already exists "
            "on this parcel."
        )


def create_parcel_season(
    parcel_id,
    user_id,
    data,
):
    ensure_parcel_owned_by_user(
        parcel_id,
        user_id,
    )

    ensure_parcel_active(
        parcel_id
    )

    validated = (
        validate_create_data(
            data
        )
    )

    crop = validated["crop"]

    target_season = (
        resolve_next_full_season(
            crop
        )
    )

    season_start_year = (
        target_season[
            "season_start_year"
        ]
    )

    ensure_unique_season(
        parcel_id,
        season_start_year,
        crop,
    )

    query = """
        INSERT INTO app.parcel_season (
            parcel_id,
            season_start_year,
            crop,
            farming_strategy,
            is_irrigated,
            machine_use,
            fertilizer_type,
            fertilizer_quantity_kg_ha
        )
        VALUES (
            %s,
            %s,
            %s,
            %s,
            %s,
            %s,
            %s,
            %s
        )
        RETURNING
            id,
            parcel_id,
            season_start_year,
            crop,
            farming_strategy,
            is_irrigated,
            machine_use,
            fertilizer_type,
            fertilizer_quantity_kg_ha,
            created_at,
            updated_at;
    """

    connection = get_connection()

    try:
        with connection.cursor() as cursor:
            cursor.execute(
                query,
                (
                    parcel_id,
                    season_start_year,
                    crop,
                    validated[
                        "farming_strategy"
                    ],
                    validated[
                        "is_irrigated"
                    ],
                    validated[
                        "machine_use"
                    ],
                    validated[
                        "fertilizer_type"
                    ],
                    validated[
                        "fertilizer_quantity_kg_ha"
                    ],
                ),
            )

            row = cursor.fetchone()

        connection.commit()

    except Exception:
        connection.rollback()
        raise

    finally:
        connection.close()

    return season_row_to_dict(
        row
    )


def get_parcel_season(
    parcel_season_id,
    user_id,
):
    ensure_parcel_season_owned_by_user(
        parcel_season_id,
        user_id,
    )

    query = """
        SELECT
            id,
            parcel_id,
            season_start_year,
            crop,
            farming_strategy,
            is_irrigated,
            machine_use,
            fertilizer_type,
            fertilizer_quantity_kg_ha,
            created_at,
            updated_at
        FROM app.parcel_season
        WHERE id = %s;
    """

    connection = get_connection()

    try:
        with connection.cursor() as cursor:
            cursor.execute(
                query,
                (
                    parcel_season_id,
                ),
            )

            row = cursor.fetchone()

    finally:
        connection.close()

    if row is None:
        raise ValueError(
            "Parcel season not found: "
            f"{parcel_season_id}"
        )

    return season_row_to_dict(
        row
    )


def list_parcel_seasons(
    parcel_id,
    user_id=None,
):
    if user_id is not None:
        ensure_parcel_owned_by_user(
            parcel_id,
            user_id,
        )

    query = """
        SELECT
            id,
            parcel_id,
            season_start_year,
            crop,
            farming_strategy,
            is_irrigated,
            machine_use,
            fertilizer_type,
            fertilizer_quantity_kg_ha,
            created_at,
            updated_at
        FROM app.parcel_season
        WHERE parcel_id = %s
        ORDER BY
            season_start_year DESC,
            crop;
    """

    connection = get_connection()

    try:
        with connection.cursor() as cursor:
            cursor.execute(
                query,
                (parcel_id,),
            )

            rows = cursor.fetchall()

    finally:
        connection.close()

    return [
        season_row_to_dict(
            row
        )
        for row in rows
    ]


def update_parcel_season(
    parcel_season_id,
    user_id,
    data,
):
    season = get_parcel_season(
        parcel_season_id,
        user_id,
    )

    ensure_parcel_active(
        season["parcel_id"]
    )

    management = (
        validate_management_data(
            data
        )
    )

    query = """
        UPDATE app.parcel_season
        SET
            farming_strategy = %s,
            is_irrigated = %s,
            machine_use = %s,
            fertilizer_type = %s,
            fertilizer_quantity_kg_ha = %s,
            updated_at = NOW()
        WHERE id = %s
        RETURNING
            id,
            parcel_id,
            season_start_year,
            crop,
            farming_strategy,
            is_irrigated,
            machine_use,
            fertilizer_type,
            fertilizer_quantity_kg_ha,
            created_at,
            updated_at;
    """

    connection = get_connection()

    try:
        with connection.cursor() as cursor:
            cursor.execute(
                query,
                (
                    management[
                        "farming_strategy"
                    ],
                    management[
                        "is_irrigated"
                    ],
                    management[
                        "machine_use"
                    ],
                    management[
                        "fertilizer_type"
                    ],
                    management[
                        "fertilizer_quantity_kg_ha"
                    ],
                    parcel_season_id,
                ),
            )

            row = cursor.fetchone()

        if row is None:
            connection.rollback()

            raise ValueError(
                "Parcel season not found: "
                f"{parcel_season_id}"
            )

        connection.commit()

    except Exception:
        connection.rollback()
        raise

    finally:
        connection.close()

    return season_row_to_dict(
        row
    )


def delete_parcel_season(
    parcel_season_id,
    user_id,
):
    season = get_parcel_season(
        parcel_season_id,
        user_id,
    )

    ensure_parcel_active(
        season["parcel_id"]
    )

    simulations = (
        list_simulations_for_parcel_season(
            parcel_season_id
        )
    )

    if simulations:
        raise ValueError(
            "This growing season has simulation "
            "history and cannot be deleted."
        )

    query = """
        DELETE FROM app.parcel_season
        WHERE id = %s
        RETURNING id;
    """

    connection = get_connection()

    try:
        with connection.cursor() as cursor:
            cursor.execute(
                query,
                (
                    parcel_season_id,
                ),
            )

            row = cursor.fetchone()

        if row is None:
            connection.rollback()

            raise ValueError(
                "Parcel season not found: "
                f"{parcel_season_id}"
            )

        connection.commit()

    except Exception:
        connection.rollback()
        raise

    finally:
        connection.close()


def get_parcel_season_details(
    parcel_season_id,
    user_id,
):
    season = get_parcel_season(
        parcel_season_id,
        user_id,
    )

    simulations = (
        list_simulations_for_parcel_season(
            parcel_season_id
        )
    )

    return {
        **season,
        "simulations":
            simulations,
    }