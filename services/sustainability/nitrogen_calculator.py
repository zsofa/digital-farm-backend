import math

from .config import (
    CROP_N_REMOVAL_FACTORS,
    FERTILIZER_N_FRACTIONS,
    N_DEFICIT_LIMIT_KG_HA,
    N_INPUT_LIMIT_KG_HA,
    N_SURPLUS_LIMIT_KG_HA,
)


def calculate_applied_nitrogen(
    fertilizer_type: str | None,
    fertilizer_quantity_kg_ha: float | None,
) -> float:
    quantity = fertilizer_quantity_kg_ha or 0.0

    if quantity < 0:
        raise ValueError(
            "Fertilizer quantity cannot be negative."
        )

    if quantity == 0:
        return 0.0

    if not fertilizer_type:
        raise ValueError(
            "Fertilizer type is required when fertilizer quantity is greater than zero."
        )

    fertilizer_key = fertilizer_type.lower()

    if fertilizer_key not in FERTILIZER_N_FRACTIONS:
        raise ValueError(
            f"Unsupported fertilizer type: {fertilizer_type}"
        )

    nitrogen_fraction = FERTILIZER_N_FRACTIONS[
        fertilizer_key
    ]

    return quantity * nitrogen_fraction


def calculate_nitrogen_input_score(
    applied_nitrogen_kg_ha: float,
) -> float:
    normalized_input = min(
        1.0,
        applied_nitrogen_kg_ha
        / N_INPUT_LIMIT_KG_HA,
    )

    return 1.0 - normalized_input


def calculate_nitrogen_removed(
    expected_yield_t_ha: float,
    crop: str,
) -> float:
    if expected_yield_t_ha < 0:
        raise ValueError(
            "Expected yield cannot be negative."
        )

    crop_key = crop.lower()

    if crop_key not in CROP_N_REMOVAL_FACTORS:
        raise ValueError(
            f"Unsupported crop: {crop}"
        )

    removal_factor = CROP_N_REMOVAL_FACTORS[
        crop_key
    ]

    return expected_yield_t_ha * removal_factor


def calculate_nitrogen_balance(
    applied_nitrogen_kg_ha: float,
    removed_nitrogen_kg_ha: float,
) -> float:
    return (
        applied_nitrogen_kg_ha
        - removed_nitrogen_kg_ha
    )


def calculate_nitrogen_balance_score(
    nitrogen_balance_kg_ha: float,
) -> float:
    if nitrogen_balance_kg_ha >= 0:
        risk = min(
            1.0,
            nitrogen_balance_kg_ha
            / N_SURPLUS_LIMIT_KG_HA,
        )
    else:
        risk = min(
            1.0,
            abs(nitrogen_balance_kg_ha)
            / N_DEFICIT_LIMIT_KG_HA,
        )

    return 1.0 - risk


def calculate_nitrogen_score(
    input_score: float,
    balance_score: float,
) -> float:
    return math.sqrt(
        input_score * balance_score
    )


def calculate_nitrogen_metrics(
    fertilizer_type: str | None,
    fertilizer_quantity_kg_ha: float | None,
    crop: str,
    expected_yield_t_ha: float,
) -> dict:
    applied_nitrogen = calculate_applied_nitrogen(
        fertilizer_type,
        fertilizer_quantity_kg_ha,
    )

    input_score = calculate_nitrogen_input_score(
        applied_nitrogen
    )

    removed_nitrogen = calculate_nitrogen_removed(
        expected_yield_t_ha,
        crop,
    )

    balance = calculate_nitrogen_balance(
        applied_nitrogen,
        removed_nitrogen,
    )

    balance_score = calculate_nitrogen_balance_score(
        balance
    )

    nitrogen_score = calculate_nitrogen_score(
        input_score,
        balance_score,
    )

    return {
        "applied_nitrogen_kg_ha": applied_nitrogen,
        "removed_nitrogen_kg_ha": removed_nitrogen,
        "nitrogen_balance_kg_ha": balance,
        "nitrogen_input_score": input_score,
        "nitrogen_balance_score": balance_score,
        "nitrogen_score": nitrogen_score,
    }