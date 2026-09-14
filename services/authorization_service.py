from ml.data_loader import get_connection


def ensure_parcel_owned_by_user(parcel_id, user_id):
    query = """
        SELECT 1
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
                (parcel_id, user_id),
            )
            row = cursor.fetchone()
    finally:
        connection.close()

    if row is None:
        raise ValueError(
            f"Parcel not found: {parcel_id}"
        )


def ensure_parcel_season_owned_by_user(
    parcel_season_id,
    user_id,
):
    query = """
        SELECT 1
        FROM app.parcel_season ps
        JOIN app.parcel p
          ON p.id = ps.parcel_id
        JOIN app.farm f
          ON f.id = p.farm_id
        WHERE ps.id = %s
          AND f.user_id = %s;
    """

    connection = get_connection()

    try:
        with connection.cursor() as cursor:
            cursor.execute(
                query,
                (
                    parcel_season_id,
                    user_id,
                ),
            )
            row = cursor.fetchone()
    finally:
        connection.close()

    if row is None:
        raise ValueError(
            f"Parcel season not found: {parcel_season_id}"
        )

def ensure_simulation_owned_by_user(
    simulation_id,
    user_id,
):
    query = """
        SELECT 1
        FROM app.yield_simulation ys
        JOIN app.parcel_season ps
          ON ps.id = ys.parcel_season_id
        JOIN app.parcel p
          ON p.id = ps.parcel_id
        JOIN app.farm f
          ON f.id = p.farm_id
        WHERE ys.id = %s
          AND f.user_id = %s;
    """

    connection = get_connection()

    try:
        with connection.cursor() as cursor:
            cursor.execute(
                query,
                (
                    simulation_id,
                    user_id,
                ),
            )
            row = cursor.fetchone()
    finally:
        connection.close()

    if row is None:
        raise ValueError(
            f"Simulation not found: {simulation_id}"
        )