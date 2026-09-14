from .config import (
    CLAY_HIGH_PCT,
    CLAY_LOW_PCT,
    CLAY_MIN_SCORE,
    CLAY_OPTIMAL_MAX_PCT,
    CLAY_OPTIMAL_MIN_PCT,
    CROP_IMPACT_SCORES,
    PH_MAX,
    PH_MIN,
    PH_OPTIMAL_MAX,
    PH_OPTIMAL_MIN,
    SOC_REFERENCE_G_KG,
    SOIL_CONDITION_WEIGHTS,
    SOIL_HEALTH_WEIGHTS,
    SOIL_MANAGEMENT_WEIGHTS,
    SOIL_STRATEGY_SCORES,
)


def calculate_ph_score(ph: float) -> float:
    if ph <= PH_MIN or ph >= PH_MAX:
        return 0.0

    if PH_OPTIMAL_MIN <= ph <= PH_OPTIMAL_MAX:
        return 1.0

    if ph < PH_OPTIMAL_MIN:
        return (
            (ph - PH_MIN)
            / (PH_OPTIMAL_MIN - PH_MIN)
        )

    return (
        (PH_MAX - ph)
        / (PH_MAX - PH_OPTIMAL_MAX)
    )


def calculate_soc_score(
    soc_g_kg: float,
) -> float:
    if soc_g_kg < 0:
        raise ValueError(
            "SOC cannot be negative."
        )

    return min(
        1.0,
        soc_g_kg / SOC_REFERENCE_G_KG,
    )


def calculate_clay_score(
    clay_pct: float,
) -> float:
    if not 0 <= clay_pct <= 100:
        raise ValueError(
            "Clay percentage must be between 0 and 100."
        )

    if (
        CLAY_OPTIMAL_MIN_PCT
        <= clay_pct
        <= CLAY_OPTIMAL_MAX_PCT
    ):
        return 1.0

    if clay_pct < CLAY_OPTIMAL_MIN_PCT:
        if clay_pct <= CLAY_LOW_PCT:
            return CLAY_MIN_SCORE

        ratio = (
            clay_pct - CLAY_LOW_PCT
        ) / (
            CLAY_OPTIMAL_MIN_PCT
            - CLAY_LOW_PCT
        )

        return (
            CLAY_MIN_SCORE
            + ratio
            * (1.0 - CLAY_MIN_SCORE)
        )

    if clay_pct >= CLAY_HIGH_PCT:
        return CLAY_MIN_SCORE

    ratio = (
        clay_pct
        - CLAY_OPTIMAL_MAX_PCT
    ) / (
        CLAY_HIGH_PCT
        - CLAY_OPTIMAL_MAX_PCT
    )

    return (
        1.0
        - ratio
        * (1.0 - CLAY_MIN_SCORE)
    )


def calculate_soil_condition_score(
    ph: float,
    soc_g_kg: float,
    clay_pct: float,
) -> tuple[float, dict]:
    ph_score = calculate_ph_score(ph)
    soc_score = calculate_soc_score(soc_g_kg)
    clay_score = calculate_clay_score(clay_pct)

    condition_score = (
        SOIL_CONDITION_WEIGHTS["ph"]
        * ph_score
        + SOIL_CONDITION_WEIGHTS["soc"]
        * soc_score
        + SOIL_CONDITION_WEIGHTS["clay"]
        * clay_score
    )

    return condition_score, {
        "ph_score": ph_score,
        "soc_score": soc_score,
        "clay_score": clay_score,
    }


def calculate_soil_management_score(
    farming_strategy: str,
    crop: str,
) -> tuple[float, dict]:
    strategy_key = farming_strategy.lower()
    crop_key = crop.lower()

    if strategy_key not in SOIL_STRATEGY_SCORES:
        raise ValueError(
            f"Unsupported farming strategy: {farming_strategy}"
        )

    if crop_key not in CROP_IMPACT_SCORES:
        raise ValueError(
            f"Unsupported crop: {crop}"
        )

    strategy_score = SOIL_STRATEGY_SCORES[
        strategy_key
    ]

    crop_score = CROP_IMPACT_SCORES[
        crop_key
    ]

    management_score = (
        SOIL_MANAGEMENT_WEIGHTS["strategy"]
        * strategy_score
        + SOIL_MANAGEMENT_WEIGHTS["crop"]
        * crop_score
    )

    return management_score, {
        "strategy_score": strategy_score,
        "crop_impact_score": crop_score,
    }


def calculate_soil_metrics(
    ph: float,
    soc_g_kg: float,
    clay_pct: float,
    farming_strategy: str,
    crop: str,
) -> dict:
    condition_score, condition_parts = (
        calculate_soil_condition_score(
            ph,
            soc_g_kg,
            clay_pct,
        )
    )

    management_score, management_parts = (
        calculate_soil_management_score(
            farming_strategy,
            crop,
        )
    )

    soil_health_score = (
        SOIL_HEALTH_WEIGHTS["condition"]
        * condition_score
        + SOIL_HEALTH_WEIGHTS["management"]
        * management_score
    )

    return {
        **condition_parts,
        **management_parts,
        "soil_condition_score": condition_score,
        "soil_management_score": management_score,
        "soil_health_score": soil_health_score,
    }