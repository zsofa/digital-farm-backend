from werkzeug.security import check_password_hash, generate_password_hash

from db.connection import get_connection


class EmailAlreadyRegistered(ValueError):
    pass


def user_row_to_dict(row):
    return {
        "id": row[0],
        "first_name": row[1],
        "last_name": row[2],
        "email": row[3],
        "created_at": row[4].isoformat(),
    }


def email_exists(email):
    with get_connection() as connection, connection.cursor() as cursor:
        cursor.execute(
            "SELECT EXISTS (SELECT 1 FROM app.user WHERE email = %s)",
            (email.strip().lower(),),
        )
        return cursor.fetchone()[0]


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
        raise ValueError("Password must contain at least 8 characters.")

    password_hash = generate_password_hash(password)

    with get_connection() as connection, connection.cursor() as cursor:
        cursor.execute(
            """
            INSERT INTO app.user (
                first_name,
                last_name,
                email,
                password_hash
            )
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (email) DO NOTHING
            RETURNING id, first_name, last_name, email, created_at;
            """,
            (first_name, last_name, email, password_hash),
        )

        row = cursor.fetchone()

        if row is None:
            raise EmailAlreadyRegistered(
                "A user with this email already exists."
            )

    return user_row_to_dict(row)


def authenticate_user(email, password):
    with get_connection() as connection, connection.cursor() as cursor:
        cursor.execute(
            """
            SELECT
                id,
                first_name,
                last_name,
                email,
                password_hash,
                created_at
            FROM app.user
            WHERE email = %s;
            """,
            (email.strip().lower(),),
        )
        row = cursor.fetchone()

    if row is None or not check_password_hash(row[4], password):
        raise ValueError("Invalid email or password.")

    return {
        "id": row[0],
        "first_name": row[1],
        "last_name": row[2],
        "email": row[3],
        "created_at": row[5].isoformat(),
    }