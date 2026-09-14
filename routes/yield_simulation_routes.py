from flask import Blueprint, jsonify
from flask_jwt_extended import (
    get_jwt_identity,
    jwt_required,
)

from services.authorization_service import (
    ensure_parcel_season_owned_by_user,
    ensure_simulation_owned_by_user,
)
from services.yield_simulation_service import (
    create_yield_simulation,
    get_simulation,
    list_simulations_for_parcel_season,
)


yield_simulation_bp = Blueprint(
    "yield_simulation",
    __name__,
    url_prefix="/api/yield-simulations",
)


@yield_simulation_bp.post(
    "/parcel-seasons/<int:parcel_season_id>"
)
@jwt_required()
def create(parcel_season_id):
    try:
        user_id = int(get_jwt_identity())

        ensure_parcel_season_owned_by_user(
            parcel_season_id,
            user_id,
        )

        result = create_yield_simulation(
            parcel_season_id
        )

        return jsonify(result), 201

    except ValueError as error:
        return jsonify({
            "error": str(error)
        }), 400


@yield_simulation_bp.get(
    "/<int:simulation_id>"
)
@jwt_required()
def read(simulation_id):
    try:
        user_id = int(get_jwt_identity())

        ensure_simulation_owned_by_user(
            simulation_id,
            user_id,
        )

        simulation = get_simulation(
            simulation_id
        )

        return jsonify(simulation), 200

    except ValueError as error:
        return jsonify({
            "error": str(error)
        }), 404


@yield_simulation_bp.get(
    "/parcel-seasons/<int:parcel_season_id>"
)
@jwt_required()
def list_all(parcel_season_id):
    try:
        user_id = int(get_jwt_identity())

        ensure_parcel_season_owned_by_user(
            parcel_season_id,
            user_id,
        )

        simulations = (
            list_simulations_for_parcel_season(
                parcel_season_id
            )
        )

        return jsonify(simulations), 200

    except ValueError as error:
        return jsonify({
            "error": str(error)
        }), 404