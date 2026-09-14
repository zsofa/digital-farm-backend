from services.parcel_service import (
    ensure_farm_owned_by_user,
    list_parcels,
)
from services.parcel_season_service import (
    list_parcel_seasons,
)
from services.sustainability_service import (
    calculate_parcel_season_sustainability,
)


def calculate_weighted_scores(
    complete_results,
):
    if not complete_results:
        return None

    total_area = sum(
        result["official_area_ha"]
        for result in complete_results
    )

    if total_area <= 0:
        return None

    sustainability_score = sum(
        result["sustainability_score"]
        * result["official_area_ha"]
        for result in complete_results
    ) / total_area

    nitrogen_score = sum(
        result["nitrogen_score"]
        * result["official_area_ha"]
        for result in complete_results
    ) / total_area

    soil_health_score = sum(
        result["soil_health_score"]
        * result["official_area_ha"]
        for result in complete_results
    ) / total_area

    ghg_score = sum(
        result["ghg_score"]
        * result["official_area_ha"]
        for result in complete_results
    ) / total_area

    return {
        "sustainability_score":
            sustainability_score,
        "sustainability_score_100":
            round(sustainability_score * 100),

        "nitrogen_score":
            nitrogen_score,
        "nitrogen_score_100":
            round(nitrogen_score * 100),

        "soil_health_score":
            soil_health_score,
        "soil_health_score_100":
            round(soil_health_score * 100),

        "ghg_score":
            ghg_score,
        "ghg_score_100":
            round(ghg_score * 100),
    }


def build_complete_season_result(
    sustainability_result,
):
    parcel = sustainability_result["parcel"]
    season = sustainability_result["season"]
    sustainability = (
        sustainability_result["sustainability"]
    )

    return {
        "parcel_season_id":
            sustainability_result[
                "parcel_season_id"
            ],

        "parcel_id":
            parcel["id"],

        "parcel_name":
            parcel["name"],

        "official_area_ha":
            parcel["official_area_ha"],

        "season_start_year":
            season["season_start_year"],

        "crop":
            season["crop"],

        "farming_strategy":
            season["farming_strategy"],

        "calculation_status":
            "complete",

        "yield_source":
            sustainability["yield_source"],

        "scores": {
            "sustainability_score":
                sustainability[
                    "sustainability_score"
                ],

            "sustainability_score_100":
                sustainability[
                    "sustainability_score_100"
                ],

            "nitrogen_score":
                sustainability[
                    "nitrogen"
                ]["nitrogen_score"],

            "nitrogen_score_100":
                round(
                    sustainability[
                        "nitrogen"
                    ]["nitrogen_score"] * 100
                ),

            "soil_health_score":
                sustainability[
                    "soil"
                ]["soil_health_score"],

            "soil_health_score_100":
                round(
                    sustainability[
                        "soil"
                    ]["soil_health_score"] * 100
                ),

            "ghg_score":
                sustainability[
                    "ghg"
                ]["ghg_score"],

            "ghg_score_100":
                round(
                    sustainability[
                        "ghg"
                    ]["ghg_score"] * 100
                ),
        },
    }


def build_incomplete_season_result(
    sustainability_result,
    parcel,
    season,
):
    return {
        "parcel_season_id":
            season["id"],

        "parcel_id":
            parcel["id"],

        "parcel_name":
            parcel["name"],

        "official_area_ha":
            parcel["official_area_ha"],

        "season_start_year":
            season["season_start_year"],

        "crop":
            season["crop"],

        "farming_strategy":
            season["farming_strategy"],

        "calculation_status":
            "incomplete",

        "missing_inputs":
            sustainability_result[
                "missing_inputs"
            ],

        "scores":
            None,
    }


def load_farm_plan_context(
    farm_id,
    user_id,
):
    parcels = list_parcels(
        farm_id,
        user_id,
    )

    active_parcels = [
        parcel
        for parcel in parcels
        if parcel["is_active"]
    ]

    parcels_by_id = {
        parcel["id"]: parcel
        for parcel in active_parcels
    }

    seasons_by_id = {}
    season_ids_by_parcel = {}

    for parcel in active_parcels:
        seasons = list_parcel_seasons(
            parcel["id"],
            user_id,
        )

        season_ids_by_parcel[
            parcel["id"]
        ] = []

        for season in seasons:
            seasons_by_id[
                season["id"]
            ] = season

            season_ids_by_parcel[
                parcel["id"]
            ].append(
                season["id"]
            )

    return {
        "parcels": active_parcels,
        "parcels_by_id": parcels_by_id,
        "seasons_by_id": seasons_by_id,
        "season_ids_by_parcel":
            season_ids_by_parcel,
    }


def validate_selected_seasons(
    parcel_season_ids,
    context,
):
    if not isinstance(
        parcel_season_ids,
        list,
    ):
        raise ValueError(
            "parcel_season_ids must be a list."
        )

    if len(parcel_season_ids) != len(
        set(parcel_season_ids)
    ):
        raise ValueError(
            "Duplicate parcel season IDs "
            "are not allowed."
        )

    selected_by_parcel = {}

    for parcel_season_id in (
        parcel_season_ids
    ):
        if not isinstance(
            parcel_season_id,
            int,
        ):
            raise ValueError(
                "Each parcel season ID "
                "must be an integer."
            )

        season = context[
            "seasons_by_id"
        ].get(
            parcel_season_id
        )

        if season is None:
            raise ValueError(
                "Parcel season does not belong "
                "to an active parcel of this farm: "
                f"{parcel_season_id}"
            )

        parcel_id = season["parcel_id"]

        if parcel_id in selected_by_parcel:
            raise ValueError(
                "Only one growing season can be "
                "selected per parcel."
            )

        selected_by_parcel[
            parcel_id
        ] = season

    return selected_by_parcel


def calculate_farm_sustainability(
    farm_id,
    user_id,
    parcel_season_ids,
):
    ensure_farm_owned_by_user(
        farm_id,
        user_id,
    )

    context = load_farm_plan_context(
        farm_id,
        user_id,
    )

    selected_by_parcel = (
        validate_selected_seasons(
            parcel_season_ids,
            context,
        )
    )

    parcel_season_results = []
    complete_results = []

    parcels_without_season = []
    parcels_without_selection = []

    incomplete_season_count = 0

    for parcel in context["parcels"]:
        parcel_id = parcel["id"]

        available_season_ids = (
            context[
                "season_ids_by_parcel"
            ].get(
                parcel_id,
                [],
            )
        )

        if not available_season_ids:
            parcels_without_season.append({
                "parcel_id":
                    parcel_id,

                "parcel_name":
                    parcel["name"],

                "official_area_ha":
                    parcel[
                        "official_area_ha"
                    ],
            })

            continue

        selected_season = (
            selected_by_parcel.get(
                parcel_id
            )
        )

        if selected_season is None:
            parcels_without_selection.append({
                "parcel_id":
                    parcel_id,

                "parcel_name":
                    parcel["name"],

                "official_area_ha":
                    parcel[
                        "official_area_ha"
                    ],

                "available_season_ids":
                    available_season_ids,
            })

            continue

        result = (
            calculate_parcel_season_sustainability(
                selected_season["id"],
                user_id,
            )
        )

        if (
            result["calculation_status"]
            == "incomplete"
        ):
            incomplete_season_count += 1

            parcel_season_results.append(
                build_incomplete_season_result(
                    result,
                    parcel,
                    selected_season,
                )
            )

            continue

        season_result = (
            build_complete_season_result(
                result
            )
        )

        parcel_season_results.append(
            season_result
        )

        complete_results.append({
            "official_area_ha":
                parcel[
                    "official_area_ha"
                ],

            "sustainability_score":
                season_result[
                    "scores"
                ][
                    "sustainability_score"
                ],

            "nitrogen_score":
                season_result[
                    "scores"
                ][
                    "nitrogen_score"
                ],

            "soil_health_score":
                season_result[
                    "scores"
                ][
                    "soil_health_score"
                ],

            "ghg_score":
                season_result[
                    "scores"
                ][
                    "ghg_score"
                ],
        })

    scores = calculate_weighted_scores(
        complete_results
    )

    if not complete_results:
        calculation_status = "unavailable"

    elif (
        incomplete_season_count > 0
        or parcels_without_selection
        or parcels_without_season
    ):
        calculation_status = "partial"

    else:
        calculation_status = "complete"

    weighted_area_sum_ha = sum(
        result["official_area_ha"]
        for result in complete_results
    )

    return {
        "calculation_status":
            calculation_status,

        "farm_id":
            farm_id,

        "aggregation_method":
            "official_area_weighted",

        "scores":
            scores,

        "summary": {
            "active_parcel_count":
                len(
                    context["parcels"]
                ),

            "selected_season_count":
                len(
                    selected_by_parcel
                ),

            "complete_season_count":
                len(
                    complete_results
                ),

            "incomplete_season_count":
                incomplete_season_count,

            "parcels_without_season_count":
                len(
                    parcels_without_season
                ),

            "parcels_without_selection_count":
                len(
                    parcels_without_selection
                ),

            "weighted_area_sum_ha":
                weighted_area_sum_ha,
        },

        "parcel_seasons":
            parcel_season_results,

        "parcels_without_season":
            parcels_without_season,

        "parcels_without_selection":
            parcels_without_selection,
    }