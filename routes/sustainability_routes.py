from flask import (
    Blueprint,
    jsonify,
    request,
)
from flask_jwt_extended import (
    get_jwt_identity,
    jwt_required,
)

from services.farm_sustainability_service import (
    calculate_farm_sustainability,
)
from services.sustainability_service import (
    calculate_parcel_season_sustainability,
)


sustainability_bp = Blueprint(
    "sustainability",
    __name__,
)


@sustainability_bp.get(
    "/api/sustainability/parcel-seasons/"
    "<int:parcel_season_id>"
)
@jwt_required()
def get_parcel_season_sustainability(
    parcel_season_id,
):
    try:
        user_id = int(
            get_jwt_identity()
        )

        result = (
            calculate_parcel_season_sustainability(
                parcel_season_id,
                user_id,
            )
        )

        return jsonify(result), 200

    except ValueError as error:
        return jsonify({
            "error": str(error)
        }), 400


@sustainability_bp.post(
    "/api/sustainability/farms/"
    "<int:farm_id>/calculate"
)
@jwt_required()
def calculate_farm(
    farm_id,
):
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

        if "parcel_season_ids" not in data:
            return jsonify({
                "error":
                    "parcel_season_ids is required."
            }), 400

        result = (
            calculate_farm_sustainability(
                farm_id,
                user_id,
                data[
                    "parcel_season_ids"
                ],
            )
        )

        return jsonify(result), 200

    except ValueError as error:
        return jsonify({
            "error": str(error)
        }), 400