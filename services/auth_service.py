from werkzeug.security import check_password_hash, generate_password_hash

from ml.data_loader import get_connection


def user_row_to_dict(row):
    return {
        "id": row[0],
        "first_name": row[1],
        "last_name": row[2],
        "email": row[3],
        "created_at": row[4].isoformat(),
    }


def register_user(data):
    first_name = data.get("first_name", "").strip()
    last_name = data.get("last_name", "").strip()
    email = data.get("email", "").strip().lower()
    password = data.get("password", "")

    if not first_name:
        raise ValueError("First name is required.")

    if not last_name:
        raise ValueError("Last name is required.")

    if not email:
        raise ValueError("Email is required.")

    if len(password) < 8:
        raise ValueError(
            "Password must contain at least 8 characters."
        )

    connection = get_connection()

    try:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT id
                FROM app.user
                WHERE email = %s;
                """,
                (email,),
            )

            if cursor.fetchone() is not None:
                raise ValueError(
                    "A user with this email already exists."
                )

            password_hash = generate_password_hash(password)

            cursor.execute(
                """
                INSERT INTO app.user (
                    first_name,
                    last_name,
                    email,
                    password_hash
                )
                VALUES (%s, %s, %s, %s)
                RETURNING
                    id,
                    first_name,
                    last_name,
                    email,
                    created_at;
                """,
                (
                    first_name,
                    last_name,
                    email,
                    password_hash,
                ),
            )

            row = cursor.fetchone()

        connection.commit()

    except Exception:
        connection.rollback()
        raise

    finally:
        connection.close()

    return user_row_to_dict(row)


def authenticate_user(email, password):
    email = email.strip().lower()

    query = """
        SELECT
            id,
            first_name,
            last_name,
            email,
            password_hash,
            created_at
        FROM app.user
        WHERE email = %s;
    """

    connection = get_connection()

    try:
        with connection.cursor() as cursor:
            cursor.execute(query, (email,))
            row = cursor.fetchone()
    finally:
        connection.close()

    if row is None:
        raise ValueError("Invalid email or password.")

    if not check_password_hash(row[4], password):
        raise ValueError("Invalid email or password.")

    return {
        "id": row[0],
        "first_name": row[1],
        "last_name": row[2],
        "email": row[3],
        "created_at": row[5].isoformat(),
    }