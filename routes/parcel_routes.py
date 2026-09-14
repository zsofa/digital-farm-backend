from flask import Blueprint, jsonify, request
from flask_jwt_extended import (
    get_jwt_identity,
    jwt_required,
)

from services.parcel_service import (
    create_parcel,
    delete_parcel,
    get_parcel_details,
    list_parcels,
    set_parcel_active,
    update_parcel,
)


parcel_bp = Blueprint(
    "parcel",
    __name__,
)


@parcel_bp.post("/api/farms/<int:farm_id>/parcels")
@jwt_required()
def create(farm_id):
    try:
        user_id = int(get_jwt_identity())
        data = request.get_json()

        if data is None:
            return jsonify({
                "error": "JSON body is required."
            }), 400

        parcel = create_parcel(
            user_id,
            farm_id,
            data,
        )

        return jsonify(parcel), 201

    except ValueError as error:
        return jsonify({
            "error": str(error)
        }), 400


@parcel_bp.get("/api/farms/<int:farm_id>/parcels")
@jwt_required()
def list_all(farm_id):
    user_id = int(get_jwt_identity())

    parcels = list_parcels(
        farm_id,
        user_id,
    )

    return jsonify(parcels), 200


@parcel_bp.get("/api/parcels/<int:parcel_id>")
@jwt_required()
def read(parcel_id):
    try:
        user_id = int(get_jwt_identity())

        parcel = get_parcel_details(
            parcel_id,
            user_id,
        )

        return jsonify(parcel), 200

    except ValueError as error:
        return jsonify({
            "error": str(error)
        }), 404


@parcel_bp.put("/api/parcels/<int:parcel_id>")
@jwt_required()
def update(parcel_id):
    try:
        user_id = int(get_jwt_identity())
        data = request.get_json()

        if data is None:
            return jsonify({
                "error": "JSON body is required."
            }), 400

        parcel = update_parcel(
            parcel_id,
            user_id,
            data,
        )

        return jsonify(parcel), 200

    except ValueError as error:
        return jsonify({
            "error": str(error)
        }), 400


@parcel_bp.delete("/api/parcels/<int:parcel_id>")
@jwt_required()
def delete(parcel_id):
    try:
        user_id = int(get_jwt_identity())

        delete_parcel(
            parcel_id,
            user_id,
        )

        return "", 204

    except ValueError as error:
        return jsonify({
            "error": str(error)
        }), 404

@parcel_bp.patch("/api/parcels/<int:parcel_id>/status")
@jwt_required()
def update_status(parcel_id):
    try:
        user_id = int(
            get_jwt_identity()
        )

        data = request.get_json()

        if data is None:
            return jsonify({
                "error": "JSON body is required."
            }), 400

        if "is_active" not in data:
            return jsonify({
                "error": "is_active is required."
            }), 400

        if not isinstance(
            data["is_active"],
            bool,
        ):
            return jsonify({
                "error": "is_active must be a boolean."
            }), 400

        parcel = set_parcel_active(
            parcel_id,
            user_id,
            data["is_active"],
        )

        return jsonify(parcel), 200

    except ValueError as error:
        return jsonify({
            "error": str(error)
        }), 400