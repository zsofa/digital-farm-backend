from pathlib import Path

import pandas as pd

from ml.data_loader import get_connection
from ml.scenarios.scenario_service import simulate_yield_scenarios


ML_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = ML_DIR / "results"

RESULTS_FILE = RESULTS_DIR / "scenario_sanity_check.csv"
WARNINGS_FILE = RESULTS_DIR / "scenario_sanity_warnings.csv"

SUPPORTED_CROPS = ("wheat", "barley", "maize")


def load_counties():
    query = """
        SELECT county_name
        FROM processed.soil
        ORDER BY county_name;
    """

    connection = get_connection()

    try:
        with connection.cursor() as cursor:
            cursor.execute(query)
            return [row[0] for row in cursor.fetchall()]
    finally:
        connection.close()


def load_historical_yield_ranges():
    query = """
        SELECT
            crop,
            MIN(average_yield_t_ha),
            MAX(average_yield_t_ha)
        FROM processed.crop_production
        WHERE average_yield_t_ha IS NOT NULL
        GROUP BY crop;
    """

    connection = get_connection()

    try:
        with connection.cursor() as cursor:
            cursor.execute(query)
            rows = cursor.fetchall()
    finally:
        connection.close()

    return {
        crop: {
            "min": float(min_yield),
            "max": float(max_yield),
        }
        for crop, min_yield, max_yield in rows
    }


def check_result(result, historical_range):
    warnings = []

    scenarios = result["scenarios"]
    reference = result["recent_yield_mean_t_ha"]

    dry = scenarios["dry_hot"]
    expected = scenarios["expected"]
    wet = scenarios["wet_cool"]

    yields = [
        dry["predicted_yield_t_ha"],
        expected["predicted_yield_t_ha"],
        wet["predicted_yield_t_ha"],
    ]

    # Hard plausibility check
    if any(value <= 0 for value in yields):
        warnings.append("non_positive_yield")

    # Compare with historical crop range, with 25% tolerance.
    lower_limit = max(0, historical_range["min"] * 0.75)
    upper_limit = historical_range["max"] * 1.25

    if any(value < lower_limit or value > upper_limit for value in yields):
        warnings.append("outside_historical_range")

    # Very large scenario response.
    scenario_spread = max(yields) - min(yields)

    if reference > 0 and scenario_spread / reference > 0.50:
        warnings.append("large_scenario_spread")

    # Expected scenario far away from recent regional yield.
    expected_difference = abs(expected["predicted_yield_t_ha"] - reference)

    if reference > 0 and expected_difference / reference > 0.50:
        warnings.append("expected_far_from_recent_yield")

    # Weather scenario construction should always follow this order.
    temperature_order_ok = (
        dry["temperature_c"]
        >= expected["temperature_c"]
        >= wet["temperature_c"]
    )

    precipitation_order_ok = (
        dry["precipitation_mm"]
        <= expected["precipitation_mm"]
        <= wet["precipitation_mm"]
    )

    if not temperature_order_ok:
        warnings.append("invalid_temperature_order")

    if not precipitation_order_ok:
        warnings.append("invalid_precipitation_order")

    # This is NOT necessarily an error, only something worth reviewing.
    if dry["predicted_yield_t_ha"] > wet["predicted_yield_t_ha"]:
        warnings.append("dry_hot_yield_above_wet_cool")

    return warnings


def flatten_result(result, warnings):
    scenarios = result["scenarios"]

    return {
        "county_name": result["county_name"],
        "crop": result["crop"],
        "target_season_start_year": result["target_season_start_year"],
        "recent_yield_mean_t_ha": result["recent_yield_mean_t_ha"],

        "dry_hot_temperature_c": scenarios["dry_hot"]["temperature_c"],
        "dry_hot_precipitation_mm": scenarios["dry_hot"]["precipitation_mm"],
        "dry_hot_yield_t_ha": scenarios["dry_hot"]["predicted_yield_t_ha"],

        "expected_temperature_c": scenarios["expected"]["temperature_c"],
        "expected_precipitation_mm": scenarios["expected"]["precipitation_mm"],
        "expected_yield_t_ha": scenarios["expected"]["predicted_yield_t_ha"],

        "wet_cool_temperature_c": scenarios["wet_cool"]["temperature_c"],
        "wet_cool_precipitation_mm": scenarios["wet_cool"]["precipitation_mm"],
        "wet_cool_yield_t_ha": scenarios["wet_cool"]["predicted_yield_t_ha"],

        "scenario_spread_t_ha": round(
            max(
                scenarios["dry_hot"]["predicted_yield_t_ha"],
                scenarios["expected"]["predicted_yield_t_ha"],
                scenarios["wet_cool"]["predicted_yield_t_ha"],
            )
            - min(
                scenarios["dry_hot"]["predicted_yield_t_ha"],
                scenarios["expected"]["predicted_yield_t_ha"],
                scenarios["wet_cool"]["predicted_yield_t_ha"],
            ),
            3,
        ),

        "warning_count": len(warnings),
        "warnings": "; ".join(warnings),
    }


def main():
    print("Loading counties and historical yield ranges...")

    counties = load_counties()
    historical_ranges = load_historical_yield_ranges()

    expected_combinations = len(counties) * len(SUPPORTED_CROPS)

    print(f"Counties: {len(counties)}")
    print(f"Crops: {len(SUPPORTED_CROPS)}")
    print(f"Combinations to test: {expected_combinations}")
    print("\nRunning scenario sanity check...")

    results = []
    failures = []

    counter = 0

    for county in counties:
        for crop in SUPPORTED_CROPS:
            counter += 1
            print(f"[{counter}/{expected_combinations}] {county} - {crop}")

            try:
                result = simulate_yield_scenarios(
                    county_name=county,
                    crop=crop,
                )

                warnings = check_result(
                    result,
                    historical_ranges[crop],
                )

                results.append(
                    flatten_result(result, warnings)
                )

            except Exception as error:
                failures.append({
                    "county_name": county,
                    "crop": crop,
                    "error": str(error),
                })

    dataframe = pd.DataFrame(results)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(RESULTS_FILE, index=False)

    warning_dataframe = dataframe[
        dataframe["warning_count"] > 0
    ].copy()

    warning_dataframe.to_csv(WARNINGS_FILE, index=False)

    print("\n" + "=" * 70)
    print("SCENARIO SANITY CHECK SUMMARY")
    print("=" * 70)

    print(f"Expected combinations: {expected_combinations}")
    print(f"Successful: {len(results)}")
    print(f"Failures: {len(failures)}")
    print(f"Combinations with warnings: {len(warning_dataframe)}")

    if failures:
        print("\nFAILURES")

        for failure in failures:
            print(
                f"- {failure['county_name']} / "
                f"{failure['crop']}: {failure['error']}"
            )

    if not warning_dataframe.empty:
        print("\nWARNINGS")

        print(
            warning_dataframe[
                ["county_name", "crop", "warnings"]
            ].to_string(index=False)
        )

    print("\n" + "=" * 70)
    print("PREDICTION RANGE BY CROP")
    print("=" * 70)

    for crop in SUPPORTED_CROPS:
        crop_rows = dataframe[dataframe["crop"] == crop]

        all_predictions = pd.concat([
            crop_rows["dry_hot_yield_t_ha"],
            crop_rows["expected_yield_t_ha"],
            crop_rows["wet_cool_yield_t_ha"],
        ])

        print(
            f"{crop:<8} "
            f"min={all_predictions.min():.3f} t/ha   "
            f"max={all_predictions.max():.3f} t/ha"
        )

    print(f"\nFull results saved to:\n{RESULTS_FILE}")
    print(f"\nWarnings saved to:\n{WARNINGS_FILE}")


if __name__ == "__main__":
    main()