from flask import Blueprint, jsonify, request
from flask_jwt_extended import get_jwt_identity, jwt_required

from services.farm_service import (
    create_farm,
    delete_farm,
    get_farm_details,
    list_farms,
    update_farm,
)


farm_bp = Blueprint(
    "farm",
    __name__,
    url_prefix="/api/farms",
)


@farm_bp.post("")
@jwt_required()
def create():
    try:
        user_id = int(get_jwt_identity())
        data = request.get_json()

        if data is None:
            return jsonify({
                "error": "JSON body is required."
            }), 400

        farm = create_farm(
            user_id,
            data,
        )

        return jsonify(farm), 201

    except ValueError as error:
        return jsonify({
            "error": str(error)
        }), 400


@farm_bp.get("")
@jwt_required()
def list_all():
    user_id = int(get_jwt_identity())

    farms = list_farms(user_id)

    return jsonify(farms), 200


@farm_bp.get("/<int:farm_id>")
@jwt_required()
def read(farm_id):
    try:
        user_id = int(get_jwt_identity())

        farm = get_farm_details(
            farm_id,
            user_id,
        )

        return jsonify(farm), 200

    except ValueError as error:
        return jsonify({
            "error": str(error)
        }), 404


@farm_bp.put("/<int:farm_id>")
@jwt_required()
def update(farm_id):
    try:
        user_id = int(get_jwt_identity())
        data = request.get_json()

        if data is None:
            return jsonify({
                "error": "JSON body is required."
            }), 400

        farm = update_farm(
            farm_id,
            user_id,
            data,
        )

        return jsonify(farm), 200

    except ValueError as error:
        return jsonify({
            "error": str(error)
        }), 400


@farm_bp.delete("/<int:farm_id>")
@jwt_required()
def delete(farm_id):
    try:
        user_id = int(get_jwt_identity())

        delete_farm(
            farm_id,
            user_id,
        )

        return "", 204

    except ValueError as error:
        return jsonify({
            "error": str(error)
        }), 404