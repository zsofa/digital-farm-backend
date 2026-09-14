import math

from .ghg_calculator import (
    calculate_ghg_metrics,
)
from .nitrogen_calculator import (
    calculate_nitrogen_metrics,
)
from .soil_calculator import (
    calculate_soil_metrics,
)


def calculate_sustainability(
    *,
    crop: str,
    farming_strategy: str,
    machine_use: str,
    fertilizer_type: str | None,
    fertilizer_quantity_kg_ha: float | None,
    soil_ph: float,
    soil_soc_g_kg: float,
    soil_clay_pct: float,
    expected_yield_t_ha: float,
    yield_source: str,
) -> dict:
    nitrogen = calculate_nitrogen_metrics(
        fertilizer_type=(
            fertilizer_type
        ),
        fertilizer_quantity_kg_ha=(
            fertilizer_quantity_kg_ha
        ),
        crop=crop,
        expected_yield_t_ha=(
            expected_yield_t_ha
        ),
    )

    soil = calculate_soil_metrics(
        ph=soil_ph,
        soc_g_kg=soil_soc_g_kg,
        clay_pct=soil_clay_pct,
        farming_strategy=(
            farming_strategy
        ),
        crop=crop,
    )

    ghg = calculate_ghg_metrics(
        fertilizer_type=(
            fertilizer_type
        ),
        fertilizer_quantity_kg_ha=(
            fertilizer_quantity_kg_ha
        ),
        machine_use=machine_use,
        farming_strategy=(
            farming_strategy
        ),
    )

    sustainability_score = (
        nitrogen["nitrogen_score"]
        * soil["soil_health_score"]
        * ghg["ghg_score"]
    ) ** (1.0 / 3.0)

    return {
        "sustainability_score": (
            sustainability_score
        ),
        "sustainability_score_100": round(
            sustainability_score * 100
        ),
        "yield_source": yield_source,
        "expected_yield_t_ha": (
            expected_yield_t_ha
        ),
        "nitrogen": nitrogen,
        "soil": soil,
        "ghg": ghg,
    }