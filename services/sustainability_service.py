from ml.scenarios.scenario_service import (
    calculate_recent_yield,
)
from services.authorization_service import (
    ensure_parcel_season_owned_by_user,
)
from services.parcel_simulation_service import (
    load_parcel_season,
)
from services.parcel_soil_service import (
    resolve_parcel_soil,
)
from services.parcel_spatial_service import (
    resolve_parcel_county,
)
from services.sustainability import (
    calculate_sustainability,
)
from services.yield_simulation_service import (
    list_simulations_for_parcel_season,
)


def find_latest_current_simulation(
    parcel_season_id,
):
    simulations = (
        list_simulations_for_parcel_season(
            parcel_season_id
        )
    )

    for simulation in simulations:
        if (
            simulation["status"]
            == "current"
            and simulation[
                "expected_yield_t_ha"
            ]
            is not None
        ):
            return simulation

    return None


def resolve_sustainability_yield(
    parcel_season,
    county_name,
):
    parcel_season_id = (
        parcel_season[
            "parcel_season_id"
        ]
    )

    current_simulation = (
        find_latest_current_simulation(
            parcel_season_id
        )
    )

    if current_simulation is not None:
        return {
            "expected_yield_t_ha":
                current_simulation[
                    "expected_yield_t_ha"
                ],

            "yield_source":
                "current_simulation",

            "simulation_id":
                current_simulation[
                    "simulation_id"
                ],

            "recent_yield_years":
                None,
        }

    (
        recent_yield,
        recent_yield_years,
    ) = calculate_recent_yield(
        county_name=county_name,
        crop=parcel_season[
            "crop"
        ],
        target_season_start_year=(
            parcel_season[
                "season_start_year"
            ]
        ),
    )

    return {
        "expected_yield_t_ha":
            recent_yield,

        "yield_source":
            "regional_recent_yield",

        "simulation_id":
            None,

        "recent_yield_years":
            recent_yield_years,
    }


def get_missing_inputs(
    parcel_season,
    soil,
):
    missing_inputs = []

    if not parcel_season.get(
        "farming_strategy"
    ):
        missing_inputs.append(
            "farming_strategy"
        )

    if not parcel_season.get(
        "machine_use"
    ):
        missing_inputs.append(
            "machine_use"
        )

    if soil.get(
        "soil_ph"
    ) is None:
        missing_inputs.append(
            "soil_ph"
        )

    if (
        soil.get(
            "soil_soc_g_kg"
        )
        is None
    ):
        missing_inputs.append(
            "soil_soc_g_kg"
        )

    if (
        soil.get(
            "soil_clay_pct"
        )
        is None
    ):
        missing_inputs.append(
            "soil_clay_pct"
        )

    return missing_inputs


def build_incomplete_result(
    parcel_season,
    missing_inputs,
):
    return {
        "calculation_status":
            "incomplete",

        "parcel_season_id":
            parcel_season[
                "parcel_season_id"
            ],

        "parcel_id":
            parcel_season[
                "parcel_id"
            ],

        "plan_name":
            parcel_season[
                "plan_name"
            ],

        "season_start_year":
            parcel_season[
                "season_start_year"
            ],

        "crop":
            parcel_season[
                "crop"
            ],

        "missing_inputs":
            missing_inputs,

        "sustainability":
            None,
    }


def calculate_parcel_season_sustainability(
    parcel_season_id,
    user_id,
):
    ensure_parcel_season_owned_by_user(
        parcel_season_id,
        user_id,
    )

    parcel_season = (
        load_parcel_season(
            parcel_season_id
        )
    )

    parcel_id = (
        parcel_season[
            "parcel_id"
        ]
    )

    county_name = (
        resolve_parcel_county(
            parcel_id
        )
    )

    try:
        soil = (
            resolve_parcel_soil(
                parcel_id
            )
        )

    except ValueError:
        return build_incomplete_result(
            parcel_season,
            [
                "soil_ph",
                "soil_soc_g_kg",
                "soil_clay_pct",
            ],
        )

    missing_inputs = (
        get_missing_inputs(
            parcel_season,
            soil,
        )
    )

    if missing_inputs:
        return build_incomplete_result(
            parcel_season,
            missing_inputs,
        )

    yield_context = (
        resolve_sustainability_yield(
            parcel_season,
            county_name,
        )
    )

    result = calculate_sustainability(
        crop=(
            parcel_season[
                "crop"
            ]
        ),

        farming_strategy=(
            parcel_season[
                "farming_strategy"
            ]
        ),

        machine_use=(
            parcel_season[
                "machine_use"
            ]
        ),

        fertilizer_type=(
            parcel_season[
                "fertilizer_type"
            ]
        ),

        fertilizer_quantity_kg_ha=(
            parcel_season[
                "fertilizer_quantity_kg_ha"
            ]
        ),

        soil_ph=(
            soil[
                "soil_ph"
            ]
        ),

        soil_soc_g_kg=(
            soil[
                "soil_soc_g_kg"
            ]
        ),

        soil_clay_pct=(
            soil[
                "soil_clay_pct"
            ]
        ),

        expected_yield_t_ha=(
            yield_context[
                "expected_yield_t_ha"
            ]
        ),

        yield_source=(
            yield_context[
                "yield_source"
            ]
        ),
    )

    return {
        "calculation_status":
            "complete",

        "parcel_season_id":
            parcel_season_id,

        "parcel": {
            "id":
                parcel_id,

            "name":
                parcel_season[
                    "parcel_name"
                ],

            "official_area_ha":
                parcel_season[
                    "official_area_ha"
                ],

            "county_name":
                county_name,
        },

        "season": {
            "plan_name":
                parcel_season[
                    "plan_name"
                ],

            "season_start_year":
                parcel_season[
                    "season_start_year"
                ],

            "crop":
                parcel_season[
                    "crop"
                ],

            "farming_strategy":
                parcel_season[
                    "farming_strategy"
                ],

            "machine_use":
                parcel_season[
                    "machine_use"
                ],

            "fertilizer_type":
                parcel_season[
                    "fertilizer_type"
                ],

            "fertilizer_quantity_kg_ha":
                parcel_season[
                    "fertilizer_quantity_kg_ha"
                ],
        },

        "soil": {
            "source":
                soil[
                    "source"
                ],

            "soil_ph":
                soil[
                    "soil_ph"
                ],

            "soil_soc_g_kg":
                soil[
                    "soil_soc_g_kg"
                ],

            "soil_clay_pct":
                soil[
                    "soil_clay_pct"
                ],
        },

        "yield": {
            "source":
                yield_context[
                    "yield_source"
                ],

            "expected_yield_t_ha":
                yield_context[
                    "expected_yield_t_ha"
                ],

            "simulation_id":
                yield_context[
                    "simulation_id"
                ],

            "recent_yield_years":
                yield_context[
                    "recent_yield_years"
                ],
        },

        "sustainability":
            result,
    }