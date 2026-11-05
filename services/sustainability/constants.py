# Nitrogen

FERTILIZER_N_FRACTIONS = {
    "an": 0.34,
    "urea": 0.46,
    "can": 0.26,
}

CROP_N_REMOVAL_FACTORS = {
    "wheat": 20.0,
    "barley": 18.0,
    "maize": 16.0,
}

N_INPUT_LIMIT_KG_HA = 300.0
N_SURPLUS_LIMIT_KG_HA = 100.0
N_DEFICIT_LIMIT_KG_HA = 150.0


# Soil

PH_OPTIMAL_MIN = 6.0
PH_OPTIMAL_MAX = 7.0
PH_MIN = 4.5
PH_MAX = 8.5

SOC_REFERENCE_G_KG = 20.0

CLAY_OPTIMAL_MIN_PCT = 15.0
CLAY_OPTIMAL_MAX_PCT = 35.0
CLAY_LOW_PCT = 5.0
CLAY_HIGH_PCT = 50.0
CLAY_MIN_SCORE = 0.60

SOIL_CONDITION_WEIGHTS = {
    "ph": 0.35,
    "soc": 0.50,
    "clay": 0.15,
}

SOIL_STRATEGY_SCORES = {
    "conventional": 0.50,
    "reduced": 0.90,
}

CROP_IMPACT_SCORES = {
    "wheat": 0.85,
    "barley": 0.85,
    "maize": 0.70,
}

SOIL_MANAGEMENT_WEIGHTS = {
    "strategy": 0.75,
    "crop": 0.25,
}

SOIL_HEALTH_WEIGHTS = {
    "condition": 0.70,
    "management": 0.30,
}


# GHG

FERTILIZER_EMISSION_FACTORS = {
    "an": 4.29,
    "can": 4.29,
    "urea": 5.88,
}

FERTILIZER_EMISSION_MAX_KG_CO2E_HA = 1764.0

MACHINE_USE_FACTORS = {
    "low": 0.25,
    "medium": 0.60,
    "high": 1.00,
}

MACHINERY_MIN_SCORE = 0.10

GHG_STRATEGY_FACTORS = {
    "conventional": 1.00,
    "reduced": 0.70,
}