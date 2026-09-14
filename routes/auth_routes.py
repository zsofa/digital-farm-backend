from flask import Blueprint, jsonify, request
from flask_jwt_extended import create_access_token, jwt_required, get_jwt_identity

from services.auth_service import (
    authenticate_user,
    register_user,
)


auth_bp = Blueprint(
    "auth",
    __name__,
    url_prefix="/api/auth",
)


@auth_bp.post("/register")
def register():
    try:
        data = request.get_json()

        if data is None:
            return jsonify({
                "error": "JSON body is required."
            }), 400

        user = register_user(data)

        return jsonify(user), 201

    except ValueError as error:
        return jsonify({
            "error": str(error)
        }), 400


@auth_bp.post("/login")
def login():
    try:
        data = request.get_json()

        if data is None:
            return jsonify({
                "error": "JSON body is required."
            }), 400

        email = data.get("email", "")
        password = data.get("password", "")

        user = authenticate_user(
            email,
            password,
        )

        access_token = create_access_token(
            identity=str(user["id"])
        )

        return jsonify({
            "access_token": access_token,
            "user": user,
        }), 200

    except ValueError as error:
        return jsonify({
            "error": str(error)
        }), 401


@auth_bp.get("/me")
@jwt_required()
def me():
    user_id = get_jwt_identity()

    return jsonify({
        "user_id": int(user_id)
    }), 200