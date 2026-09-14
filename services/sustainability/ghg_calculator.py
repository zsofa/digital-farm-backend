import math

from .config import (
    FERTILIZER_EMISSION_FACTORS,
    FERTILIZER_EMISSION_MAX_KG_CO2E_HA,
    GHG_STRATEGY_FACTORS,
    MACHINERY_MIN_SCORE,
    MACHINE_USE_FACTORS,
)

from .nitrogen_calculator import (
    calculate_applied_nitrogen,
)


def calculate_fertilizer_emission(
    fertilizer_type: str | None,
    applied_nitrogen_kg_ha: float,
) -> float:
    if applied_nitrogen_kg_ha == 0:
        return 0.0

    if not fertilizer_type:
        raise ValueError(
            "Fertilizer type is required when applied nitrogen is greater than zero."
        )

    fertilizer_key = fertilizer_type.lower()

    if (
        fertilizer_key
        not in FERTILIZER_EMISSION_FACTORS
    ):
        raise ValueError(
            f"Unsupported fertilizer type: {fertilizer_type}"
        )

    emission_factor = (
        FERTILIZER_EMISSION_FACTORS[
            fertilizer_key
        ]
    )

    return (
        applied_nitrogen_kg_ha
        * emission_factor
    )


def calculate_fertilizer_emission_score(
    emission_kg_co2e_ha: float,
) -> float:
    normalized_emission = min(
        1.0,
        emission_kg_co2e_ha
        / FERTILIZER_EMISSION_MAX_KG_CO2E_HA,
    )

    return 1.0 - normalized_emission


def calculate_machinery_emission_score(
    machine_use: str,
    farming_strategy: str,
) -> tuple[float, float]:
    machine_key = machine_use.lower()
    strategy_key = farming_strategy.lower()

    if machine_key not in MACHINE_USE_FACTORS:
        raise ValueError(
            f"Unsupported machine use: {machine_use}"
        )

    if strategy_key not in GHG_STRATEGY_FACTORS:
        raise ValueError(
            f"Unsupported farming strategy: {farming_strategy}"
        )

    relative_emission = (
        MACHINE_USE_FACTORS[machine_key]
        * GHG_STRATEGY_FACTORS[strategy_key]
    )

    score = 1.0 - min(
        1.0,
        relative_emission,
    )

    score = max(
        MACHINERY_MIN_SCORE,
        score,
    )

    return score, relative_emission


def calculate_ghg_metrics(
    fertilizer_type: str | None,
    fertilizer_quantity_kg_ha: float | None,
    machine_use: str,
    farming_strategy: str,
) -> dict:
    applied_nitrogen = calculate_applied_nitrogen(
        fertilizer_type,
        fertilizer_quantity_kg_ha,
    )

    fertilizer_emission = (
        calculate_fertilizer_emission(
            fertilizer_type,
            applied_nitrogen,
        )
    )

    fertilizer_score = (
        calculate_fertilizer_emission_score(
            fertilizer_emission
        )
    )

    (
        machinery_score,
        machinery_relative_emission,
    ) = calculate_machinery_emission_score(
        machine_use,
        farming_strategy,
    )

    ghg_score = math.sqrt(
        fertilizer_score
        * machinery_score
    )

    return {
        "fertilizer_emission_kg_co2e_ha": (
            fertilizer_emission
        ),
        "fertilizer_emission_score": (
            fertilizer_score
        ),
        "machinery_relative_emission": (
            machinery_relative_emission
        ),
        "machinery_emission_score": (
            machinery_score
        ),
        "ghg_score": ghg_score,
    }