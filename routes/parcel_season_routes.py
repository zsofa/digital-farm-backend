from flask import (
    Blueprint,
    jsonify,
    request,
)

from flask_jwt_extended import (
    get_jwt_identity,
    jwt_required,
)

from services.crop_season_service import (
    resolve_next_full_season,
)
from services.parcel_season_service import (
    create_parcel_season,
    delete_parcel_season,
    get_parcel_season_details,
    list_parcel_seasons,
    update_parcel_season,
)


parcel_season_bp = Blueprint(
    "parcel_season",
    __name__,
)


@parcel_season_bp.get(
    "/api/crop-seasons/next"
)
@jwt_required()
def get_next_crop_season():
    try:
        crop = request.args.get(
            "crop",
            "",
        ).lower()

        season = (
            resolve_next_full_season(
                crop
            )
        )

        return jsonify({
            "crop":
                season["crop"],

            "season_start_year":
                season[
                    "season_start_year"
                ],

            "start_date":
                season[
                    "start_date"
                ].isoformat(),

            "end_date":
                season[
                    "end_date"
                ].isoformat(),
        }), 200

    except ValueError as error:
        return jsonify({
            "error": str(error)
        }), 400


@parcel_season_bp.post(
    "/api/parcels/<int:parcel_id>/seasons"
)
@jwt_required()
def create(parcel_id):
    try:
        user_id = int(
            get_jwt_identity()
        )

        data = request.get_json()

        if data is None:
            return jsonify({
                "error":
                    "JSON body is required."
            }), 400

        season = create_parcel_season(
            parcel_id,
            user_id,
            data,
        )

        return jsonify(
            season
        ), 201

    except ValueError as error:
        return jsonify({
            "error": str(error)
        }), 400


@parcel_season_bp.get(
    "/api/parcels/<int:parcel_id>/seasons"
)
@jwt_required()
def list_all(parcel_id):
    try:
        user_id = int(
            get_jwt_identity()
        )

        seasons = (
            list_parcel_seasons(
                parcel_id,
                user_id,
            )
        )

        return jsonify(
            seasons
        ), 200

    except ValueError as error:
        return jsonify({
            "error": str(error)
        }), 404


@parcel_season_bp.get(
    "/api/parcel-seasons/"
    "<int:parcel_season_id>"
)
@jwt_required()
def read(parcel_season_id):
    try:
        user_id = int(
            get_jwt_identity()
        )

        season = (
            get_parcel_season_details(
                parcel_season_id,
                user_id,
            )
        )

        return jsonify(
            season
        ), 200

    except ValueError as error:
        return jsonify({
            "error": str(error)
        }), 404


@parcel_season_bp.put(
    "/api/parcel-seasons/"
    "<int:parcel_season_id>"
)
@jwt_required()
def update(parcel_season_id):
    try:
        user_id = int(
            get_jwt_identity()
        )

        data = request.get_json()

        if data is None:
            return jsonify({
                "error":
                    "JSON body is required."
            }), 400

        season = (
            update_parcel_season(
                parcel_season_id,
                user_id,
                data,
            )
        )

        return jsonify(
            season
        ), 200

    except ValueError as error:
        return jsonify({
            "error": str(error)
        }), 400


@parcel_season_bp.delete(
    "/api/parcel-seasons/"
    "<int:parcel_season_id>"
)
@jwt_required()
def delete(parcel_season_id):
    try:
        user_id = int(
            get_jwt_identity()
        )

        delete_parcel_season(
            parcel_season_id,
            user_id,
        )

        return "", 204

    except ValueError as error:
        return jsonify({
            "error": str(error)
        }), 400