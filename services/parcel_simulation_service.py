from ml.data_loader import get_connection
from ml.scenarios.scenario_service import (
    simulate_yield_scenarios,
)
from services.parcel_soil_service import (
    resolve_parcel_soil,
)
from services.parcel_spatial_service import (
    resolve_parcel_county,
)


def load_parcel_season(
    parcel_season_id,
):
    query = """
        SELECT
            ps.id,
            ps.parcel_id,
            p.name AS parcel_name,
            p.official_area_ha,
            ps.plan_name,
            ps.season_start_year,
            ps.crop,
            ps.farming_strategy,
            ps.is_irrigated,
            ps.machine_use,
            ps.fertilizer_type,
            ps.fertilizer_quantity_kg_ha
        FROM app.parcel_season ps
        JOIN app.parcel p
          ON p.id = ps.parcel_id
        WHERE ps.id = %s;
    """

    connection = get_connection()

    try:
        with connection.cursor() as cursor:
            cursor.execute(
                query,
                (
                    parcel_season_id,
                ),
            )

            row = cursor.fetchone()

    finally:
        connection.close()

    if row is None:
        raise ValueError(
            "Parcel season not found: "
            f"{parcel_season_id}"
        )

    return {
        "parcel_season_id":
            row[0],

        "parcel_id":
            row[1],

        "parcel_name":
            row[2],

        "official_area_ha":
            float(row[3]),

        "plan_name":
            row[4],

        "season_start_year":
            row[5],

        "crop":
            row[6],

        "farming_strategy":
            row[7],

        "is_irrigated":
            row[8],

        "machine_use":
            row[9],

        "fertilizer_type":
            row[10],

        "fertilizer_quantity_kg_ha": (
            float(row[11])
            if row[11] is not None
            else None
        ),
    }


def simulate_parcel_season(
    parcel_season_id,
):
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

    resolved_soil = (
        resolve_parcel_soil(
            parcel_id
        )
    )

    soil = {
        "soil_ph":
            resolved_soil[
                "soil_ph"
            ],

        "soil_soc_g_kg":
            resolved_soil[
                "soil_soc_g_kg"
            ],

        "soil_clay_pct":
            resolved_soil[
                "soil_clay_pct"
            ],
    }

    simulation = (
        simulate_yield_scenarios(
            county_name=(
                county_name
            ),

            crop=(
                parcel_season[
                    "crop"
                ]
            ),

            target_season_start_year=(
                parcel_season[
                    "season_start_year"
                ]
            ),

            soil=
                soil,

            area_ha=(
                parcel_season[
                    "official_area_ha"
                ]
            ),
        )
    )

    return {
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
            "id":
                parcel_season[
                    "parcel_season_id"
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

            "farming_strategy":
                parcel_season[
                    "farming_strategy"
                ],

            "is_irrigated":
                parcel_season[
                    "is_irrigated"
                ],

            "machine_use":
                parcel_season[
                    "machine_use"
                ],

            "fertilizer_type":
                parcel_season[
                    "fertilizer_type"
                ],

            "fertilizer_quantity_kg_ha": (
                parcel_season[
                    "fertilizer_quantity_kg_ha"
                ]
            ),
        },

        "soil": {
            "source":
                resolved_soil[
                    "source"
                ],

            **soil,
        },

        "yield_simulation":
            simulation,
    }