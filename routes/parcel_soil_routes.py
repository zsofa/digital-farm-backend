from flask import (
    Blueprint,
    jsonify,
    request,
)

from flask_jwt_extended import (
    get_jwt_identity,
    jwt_required,
)

from services.authorization_service import (
    ensure_parcel_owned_by_user,
)

from services.parcel_soil_service import (
    delete_user_soil,
    resolve_parcel_soil,
    save_user_soil,
)


parcel_soil_bp = Blueprint(
    "parcel_soil",
    __name__,
)


@parcel_soil_bp.get(
    "/api/parcels/<int:parcel_id>/soil"
)
@jwt_required()
def get_soil(
    parcel_id,
):
    try:
        user_id = int(
            get_jwt_identity()
        )

        ensure_parcel_owned_by_user(
            parcel_id,
            user_id,
        )

        soil = resolve_parcel_soil(
            parcel_id
        )

        return jsonify(
            soil
        ), 200

    except ValueError as error:
        return jsonify({
            "error": str(error)
        }), 404


@parcel_soil_bp.post(
    "/api/parcels/<int:parcel_id>/soil"
)
@jwt_required()
def save_soil_measurement(
    parcel_id,
):
    try:
        user_id = int(
            get_jwt_identity()
        )

        ensure_parcel_owned_by_user(
            parcel_id,
            user_id,
        )

        data = request.get_json()

        if data is None:
            return jsonify({
                "error":
                    "JSON body is required."
            }), 400

        soil = save_user_soil(
            parcel_id,
            data,
        )

        return jsonify(
            soil
        ), 200

    except ValueError as error:
        return jsonify({
            "error": str(error)
        }), 400


@parcel_soil_bp.delete(
    "/api/parcels/<int:parcel_id>/soil"
)
@jwt_required()
def delete_soil(
    parcel_id,
):
    try:
        user_id = int(
            get_jwt_identity()
        )

        ensure_parcel_owned_by_user(
            parcel_id,
            user_id,
        )

        delete_user_soil(
            parcel_id
        )

        soil = resolve_parcel_soil(
            parcel_id
        )

        return jsonify(
            soil
        ), 200

    except ValueError as error:
        return jsonify({
            "error": str(error)
        }), 404