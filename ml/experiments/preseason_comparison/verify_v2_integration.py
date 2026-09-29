from pathlib import Path
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd

from ml.scenarios import scenario_service
from ml.scenarios.features import SCENARIO_FEATURE_COLUMNS
from ml.scenarios.weather_scenarios import (
    create_weather_scenarios,
    get_next_season_start_year,
)
from ml.train.train_scenario_xgboost import load_dataset


ML_DIR = Path(__file__).resolve().parents[2]

V2_MODEL_FILE = (
    ML_DIR
    / "models"
    / "scenario_xgboost_v2"
    / "20260929T191958104775Z"
    / "scenario_xgboost_v2.joblib"
)

CURRENT_MODEL_FILE = Path(scenario_service.MODEL_FILE)

CROPS = ["barley", "maize", "wheat"]
SCENARIOS = ["dry_hot", "expected", "wet_cool"]
AREA_HA = 10.0

EXPECTED_V2_PARAMETERS = {
    "n_estimators": 200,
    "learning_rate": 0.05,
    "max_depth": 2,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "reg_lambda": 5.0,
}


def check(condition, message):
    if not condition:
        raise RuntimeError(message)


def load_and_validate_artifact(path, is_v2=False):
    check(path.is_file(), f"Model file not found: {path}")

    artifact = joblib.load(path)

    check(
        artifact["feature_columns"] == list(SCENARIO_FEATURE_COLUMNS),
        f"Feature names or order differ: {path}",
    )

    if is_v2:
        check(
            artifact.get("model_variant") == "v2",
            "The selected artifact is not labelled V2.",
        )

        actual_parameters = artifact["model"].named_steps[
            "model"
        ].get_params()

        for name, expected in EXPECTED_V2_PARAMETERS.items():
            check(
                actual_parameters.get(name) == expected,
                f"Unexpected V2 parameter: {name}",
            )

    return artifact


def prepare_cases():
    dataframe = load_dataset()
    counties = sorted(dataframe["county_name"].unique().tolist())

    check(len(counties) == 19, "Expected exactly 19 counties.")

    # Resolve target seasons once, before creating the cases.
    target_seasons = {
        crop: get_next_season_start_year(crop)
        for crop in CROPS
    }

    cases = []

    for county in counties:
        soil = scenario_service.load_county_soil(county)

        for crop in CROPS:
            target_season = target_seasons[crop]

            recent_yield, recent_years = (
                scenario_service.calculate_recent_yield(
                    county,
                    crop,
                    target_season,
                )
            )

            weather = create_weather_scenarios(
                county,
                crop,
                target_season,
            )

            rows = []

            for scenario in SCENARIOS:
                rows.append({
                    "crop": crop,
                    **soil,
                    "season_temperature_c": (
                        weather[scenario]["temperature_c"]
                    ),
                    "season_precipitation_mm": (
                        weather[scenario]["precipitation_mm"]
                    ),
                    "recent_yield_mean_t_ha": recent_yield,
                })

            inputs = pd.DataFrame(rows)[SCENARIO_FEATURE_COLUMNS]

            numeric = inputs.drop(columns="crop").to_numpy(
                dtype=float
            )
            check(
                np.isfinite(numeric).all(),
                f"Invalid inputs: {county}, {crop}",
            )

            cases.append({
                "county": county,
                "crop": crop,
                "target_season": target_season,
                "soil": dict(soil),
                "recent_yield": recent_yield,
                "recent_years": recent_years,
                "inputs": inputs,
            })

    print("Target season start years:")
    for crop, year in target_seasons.items():
        print(f"  {crop}: {year}")

    return cases


def evaluate_artifact(label, path, artifact, cases):
    records = []

    # These overrides exist only inside this Python process.
    # Both attributes are restored when the context exits.
    with (
        patch.object(scenario_service, "MODEL_FILE", path),
        patch.object(scenario_service, "_model_artifact", None),
    ):
        for case in cases:
            direct = np.asarray(
                artifact["model"].predict(case["inputs"]),
                dtype=float,
            )

            check(
                direct.shape == (len(SCENARIOS),),
                f"Unexpected prediction shape: {label}",
            )
            check(
                np.isfinite(direct).all(),
                f"Non-finite prediction: {label}, {case['county']}",
            )
            check(
                (direct >= 0).all(),
                f"Negative yield: {label}, "
                f"{case['county']}, {case['crop']}",
            )

            result = scenario_service.simulate_yield_scenarios(
                county_name=case["county"],
                crop=case["crop"],
                target_season_start_year=case["target_season"],
                soil=dict(case["soil"]),
                area_ha=AREA_HA,
            )

            check(
                result["county_name"] == case["county"]
                and result["crop"] == case["crop"]
                and result["target_season_start_year"]
                == case["target_season"],
                "Response identifiers do not match the request.",
            )
            check(
                result["soil"] == case["soil"],
                "Soil inputs changed unexpectedly.",
            )
            check(
                result["recent_yield_years"] == case["recent_years"],
                "Historical yield years differ.",
            )
            check(
                result["recent_yield_mean_t_ha"]
                == round(case["recent_yield"], 3),
                "Historical yield mean differs.",
            )
            check(
                set(result["scenarios"]) == set(SCENARIOS),
                "Unexpected scenario keys.",
            )

            for index, scenario in enumerate(SCENARIOS):
                returned = result["scenarios"][scenario]
                input_row = case["inputs"].iloc[index]

                expected_yield = round(float(direct[index]), 3)

                check(
                    returned["predicted_yield_t_ha"]
                    == expected_yield,
                    f"Direct/service prediction mismatch: "
                    f"{label}, {case['county']}, "
                    f"{case['crop']}, {scenario}",
                )
                check(
                    returned["temperature_c"]
                    == round(
                        float(input_row["season_temperature_c"]),
                        3,
                    ),
                    "Temperature response mismatch.",
                )
                check(
                    returned["precipitation_mm"]
                    == round(
                        float(input_row["season_precipitation_mm"]),
                        2,
                    ),
                    "Precipitation response mismatch.",
                )
                check(
                    returned["estimated_total_t"]
                    == round(expected_yield * AREA_HA, 3),
                    "Area-based total yield mismatch.",
                )

                records.append({
                    "county_name": case["county"],
                    "crop": case["crop"],
                    "season_start_year": case["target_season"],
                    "scenario": scenario,
                    "model": label,
                    "prediction_t_ha": expected_yield,
                })

    print(
        f"{label}: service checks passed "
        f"({len(cases)} requests, {len(records)} predictions)."
    )

    return records


def main():
    print("V2 APPLICATION SERVICE INTEGRATION CHECK")
    print(f"Current model: {CURRENT_MODEL_FILE}")
    print(f"V2 candidate: {V2_MODEL_FILE}\n")

    current = load_and_validate_artifact(CURRENT_MODEL_FILE)
    candidate = load_and_validate_artifact(
        V2_MODEL_FILE,
        is_v2=True,
    )

    cases = prepare_cases()
    records = []

    for label, path, artifact in [
        ("current", CURRENT_MODEL_FILE, current),
        ("v2", V2_MODEL_FILE, candidate),
    ]:
        records.extend(
            evaluate_artifact(label, path, artifact, cases)
        )

    predictions = pd.DataFrame(records)

    keys = [
        "county_name",
        "crop",
        "season_start_year",
        "scenario",
    ]

    comparison = predictions.pivot(
        index=keys,
        columns="model",
        values="prediction_t_ha",
    ).reset_index()
    comparison.columns.name = None

    comparison["v2_minus_current"] = (
        comparison["v2"] - comparison["current"]
    )
    comparison["absolute_change"] = (
        comparison["v2_minus_current"].abs()
    )

    summary = comparison.groupby(
        ["crop", "scenario"],
        as_index=False,
    ).agg(
        records=("v2", "size"),
        current_mean=("current", "mean"),
        v2_mean=("v2", "mean"),
        mean_change=("v2_minus_current", "mean"),
        mean_absolute_change=("absolute_change", "mean"),
        max_absolute_change=("absolute_change", "max"),
    )

    print("\nPREDICTION CHANGES BY CROP AND SCENARIO")
    print(summary.to_string(index=False, float_format="%.3f"))

    print("\n10 LARGEST PREDICTION CHANGES")
    print(
        comparison.nlargest(10, "absolute_change").to_string(
            index=False,
            float_format="%.3f",
        )
    )

    print("\nPASSED: model loading and service output checks.")
    print("- V2 parameters and feature order match expectations.")
    print("- All checked predictions are finite and non-negative.")
    print("- Direct and service predictions match after rounding.")
    print("- Total yield matches the existing area calculation.")
    print("- Prediction changes do NOT measure accuracy.")
    print("- This checks the Python service, not the HTTP/frontend layer.")
    print("- Custom parcel soil combinations were not tested.")
    print("- No files, database tables or application settings were changed.")


if __name__ == "__main__":
    main()