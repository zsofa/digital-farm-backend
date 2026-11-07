from flask import Blueprint, jsonify, request
from flask_jwt_extended import (
    create_access_token,
    get_jwt_identity,
    jwt_required,
)

from services.auth_service import (
    EmailAlreadyRegistered,
    authenticate_user,
    email_exists,
    register_user,
)


auth_bp = Blueprint("auth", __name__, url_prefix="/api/auth")


@auth_bp.post("/email-availability")
def email_availability():
    data = request.get_json(silent=True)
    email = data.get("email") if isinstance(data, dict) else None

    if not isinstance(email, str) or not email.strip():
        return jsonify({"error": "Email is required."}), 400

    return jsonify({"available": not email_exists(email)}), 200


@auth_bp.post("/register")
def register():
    data = request.get_json(silent=True)

    if not isinstance(data, dict):
        return jsonify({"error": "JSON body is required."}), 400

    try:
        user = register_user(data)
        return jsonify(user), 201

    except EmailAlreadyRegistered as error:
        return jsonify({
            "error": str(error),
            "code": "email_taken",
        }), 409

    except ValueError as error:
        return jsonify({"error": str(error)}), 400


@auth_bp.post("/login")
def login():
    data = request.get_json(silent=True)

    if not isinstance(data, dict):
        return jsonify({"error": "JSON body is required."}), 400

    try:
        user = authenticate_user(
            data.get("email", ""),
            data.get("password", ""),
        )

        access_token = create_access_token(identity=str(user["id"]))

        return jsonify({
            "access_token": access_token,
            "user": user,
        }), 200

    except ValueError as error:
        return jsonify({"error": str(error)}), 401


@auth_bp.get("/me")
@jwt_required()
def me():
    return jsonify({"user_id": int(get_jwt_identity())}), 200