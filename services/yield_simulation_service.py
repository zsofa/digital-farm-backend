from ml.data_loader import get_connection
from ml.scenarios.scenario_service import get_model_artifact
from services.parcel_simulation_service import simulate_parcel_season


def get_parcel_simulation_context(parcel_season_id):
    query = """
        SELECT
            p.id,
            p.revision,
            p.is_active
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
                (parcel_season_id,),
            )

            row = cursor.fetchone()

    finally:
        connection.close()

    if row is None:
        raise ValueError(
            f"Parcel season not found: {parcel_season_id}"
        )

    if not row[2]:
        raise ValueError(
            "Yield simulations cannot be created "
            "for an inactive parcel."
        )

    return {
        "parcel_id": row[0],
        "parcel_revision": row[1],
    }


def save_simulation_header(
    cursor,
    result,
    parcel_revision,
):
    yield_simulation = result["yield_simulation"]
    soil = result["soil"]

    artifact = get_model_artifact()
    model_type = artifact["model_type"]

    query = """
        INSERT INTO app.yield_simulation (
            parcel_season_id,
            county_name,
            soil_source,
            soil_ph,
            soil_soc_g_kg,
            soil_clay_pct,
            recent_yield_mean_t_ha,
            model_type,
            parcel_revision
        )
        VALUES (
            %s,
            %s,
            %s,
            %s,
            %s,
            %s,
            %s,
            %s,
            %s
        )
        RETURNING id;
    """

    cursor.execute(
        query,
        (
            result["season"]["id"],
            result["parcel"]["county_name"],
            soil["source"],
            soil["soil_ph"],
            soil["soil_soc_g_kg"],
            soil["soil_clay_pct"],
            yield_simulation[
                "recent_yield_mean_t_ha"
            ],
            model_type,
            parcel_revision,
        ),
    )

    return cursor.fetchone()[0]


def save_scenarios(
    cursor,
    simulation_id,
    result,
):
    scenarios = (
        result["yield_simulation"]["scenarios"]
    )

    query = """
        INSERT INTO app.yield_simulation_scenario (
            simulation_id,
            scenario,
            temperature_c,
            precipitation_mm,
            predicted_yield_t_ha,
            estimated_total_t
        )
        VALUES (
            %s,
            %s,
            %s,
            %s,
            %s,
            %s
        );
    """

    for scenario_name, scenario in scenarios.items():
        cursor.execute(
            query,
            (
                simulation_id,
                scenario_name,
                scenario["temperature_c"],
                scenario["precipitation_mm"],
                scenario["predicted_yield_t_ha"],
                scenario["estimated_total_t"],
            ),
        )


def create_yield_simulation(
    parcel_season_id,
):
    context = get_parcel_simulation_context(
        parcel_season_id
    )

    result = simulate_parcel_season(
        parcel_season_id
    )

    connection = get_connection()

    try:
        with connection.cursor() as cursor:
            simulation_id = save_simulation_header(
                cursor,
                result,
                context["parcel_revision"],
            )

            save_scenarios(
                cursor,
                simulation_id,
                result,
            )

        connection.commit()

    except Exception:
        connection.rollback()
        raise

    finally:
        connection.close()

    return {
        "simulation_id": simulation_id,
    }


def get_simulation(
    simulation_id,
):
    query = """
        SELECT
            ys.id,
            ys.parcel_season_id,
            ys.county_name,

            ys.soil_source,
            ys.soil_ph,
            ys.soil_soc_g_kg,
            ys.soil_clay_pct,

            ys.recent_yield_mean_t_ha,
            ys.model_type,
            ys.created_at,

            ys.parcel_revision,
            p.revision AS current_parcel_revision,

            ps.season_start_year,
            ps.crop,

            p.id AS parcel_id,
            p.name AS parcel_name,
            p.official_area_ha,

            CASE
                WHEN ys.parcel_revision = p.revision
                THEN 'current'
                ELSE 'outdated'
            END AS status

        FROM app.yield_simulation ys

        JOIN app.parcel_season ps
          ON ps.id = ys.parcel_season_id

        JOIN app.parcel p
          ON p.id = ps.parcel_id

        WHERE ys.id = %s;
    """

    scenario_query = """
        SELECT
            scenario,
            temperature_c,
            precipitation_mm,
            predicted_yield_t_ha,
            estimated_total_t
        FROM app.yield_simulation_scenario
        WHERE simulation_id = %s;
    """

    connection = get_connection()

    try:
        with connection.cursor() as cursor:
            cursor.execute(
                query,
                (simulation_id,),
            )

            row = cursor.fetchone()

            if row is None:
                raise ValueError(
                    "Yield simulation not found: "
                    f"{simulation_id}"
                )

            cursor.execute(
                scenario_query,
                (simulation_id,),
            )

            scenario_rows = cursor.fetchall()

    finally:
        connection.close()

    scenarios = {}

    for scenario in scenario_rows:
        scenarios[scenario[0]] = {
            "temperature_c":
                float(scenario[1]),

            "precipitation_mm":
                float(scenario[2]),

            "predicted_yield_t_ha":
                float(scenario[3]),

            "estimated_total_t":
                float(scenario[4]),
        }

    return {
        "simulation_id": row[0],
        "parcel_season_id": row[1],

        "county_name": row[2],

        "soil": {
            "source": row[3],
            "soil_ph": float(row[4]),
            "soil_soc_g_kg": float(row[5]),
            "soil_clay_pct": float(row[6]),
        },

        "recent_yield_mean_t_ha":
            float(row[7]),

        "model_type": row[8],

        "created_at":
            row[9].isoformat(),

        "parcel_revision_at_run":
            row[10],

        "current_parcel_revision":
            row[11],

        "status": row[17],

        "season": {
            "season_start_year":
                row[12],
            "crop":
                row[13],
        },

        "parcel": {
            "id": row[14],
            "name": row[15],
            "official_area_ha":
                float(row[16]),
        },

        "scenarios": scenarios,
    }


def list_simulations_for_parcel_season(
    parcel_season_id,
):
    query = """
        SELECT
            ys.id,
            ys.created_at,
            ys.recent_yield_mean_t_ha,
            ys.model_type,
            ys.parcel_revision,
            p.revision,

            expected.predicted_yield_t_ha,
            expected.estimated_total_t,

            CASE
                WHEN ys.parcel_revision = p.revision
                THEN 'current'
                ELSE 'outdated'
            END AS status

        FROM app.yield_simulation ys

        JOIN app.parcel_season ps
          ON ps.id = ys.parcel_season_id

        JOIN app.parcel p
          ON p.id = ps.parcel_id

        LEFT JOIN app.yield_simulation_scenario expected
          ON expected.simulation_id = ys.id
         AND expected.scenario = 'expected'

        WHERE ys.parcel_season_id = %s

        ORDER BY ys.created_at DESC;
    """

    connection = get_connection()

    try:
        with connection.cursor() as cursor:
            cursor.execute(
                query,
                (parcel_season_id,),
            )

            rows = cursor.fetchall()

    finally:
        connection.close()

    return [
        {
            "simulation_id":
                row[0],

            "created_at":
                row[1].isoformat(),

            "recent_yield_mean_t_ha":
                float(row[2]),

            "model_type":
                row[3],

            "parcel_revision_at_run":
                row[4],

            "current_parcel_revision":
                row[5],

            "expected_yield_t_ha": (
                float(row[6])
                if row[6] is not None
                else None
            ),

            "expected_total_t": (
                float(row[7])
                if row[7] is not None
                else None
            ),

            "status":
                row[8],
        }
        for row in rows
    ]