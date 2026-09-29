import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from ml.data_loader import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    load_training_dataset,
)
from ml.experiments.actual_weather_v2.evaluate_weather_scenarios_cv import (
    PRECIPITATION_COLUMN,
    TEMPERATURE_COLUMN,
    add_scenario_weather,
    calculate_metrics,
    load_weather_history,
)
from ml.experiments.actual_weather_v2.tune_xgboost import load_dataset
from ml.experiments.preseason_comparison.compare_preseason_models import (
    KEYS,
    print_table,
)
from ml.experiments.preseason_comparison.evaluate_feature_groups import (
    create_ridge,
)
from ml.train.train_xgboost import create_model


BASE_DIR = Path(__file__).resolve().parent
ML_DIR = Path(__file__).resolve().parents[2]

LEGACY_RESULTS = ML_DIR / "results" / "final_xgboost_test_predictions.csv"

TRAIN_YEARS = list(range(2005, 2023))
TEST_YEARS = [2023, 2024, 2025]

SOIL_COLUMNS = ["soil_ph", "soil_soc_g_kg", "soil_clay_pct"]
XGB_FEATURES = list(FEATURE_COLUMNS)
RIDGE_FEATURES = [
    column for column in FEATURE_COLUMNS
    if column not in SOIL_COLUMNS
]


def load_settings(run_name):
    reference_dir = BASE_DIR / "results" / run_name
    settings_file = reference_dir / "settings.json"

    if not settings_file.is_file():
        raise FileNotFoundError(settings_file)

    settings = json.loads(settings_file.read_text(encoding="utf-8"))

    if "xgboost_parameters" not in settings or "ridge_alpha" not in settings:
        raise ValueError("Use the previous feature_groups run.")

    if settings["feature_groups"]["full"] != XGB_FEATURES:
        raise ValueError("Reference XGBoost features differ.")

    if settings["feature_groups"]["base_weather"] != RIDGE_FEATURES:
        raise ValueError("Reference Ridge features differ.")

    return reference_dir, settings


def prepare_expected_dataset():
    actual = load_dataset().reset_index(drop=True)
    history = load_weather_history()
    reconstructed = add_scenario_weather(actual, history)

    expected = actual.copy()
    expected[TEMPERATURE_COLUMN] = reconstructed["expected_temperature"]
    expected[PRECIPITATION_COLUMN] = reconstructed["expected_precipitation"]

    stored = load_training_dataset()
    stored = stored[
        stored["yield_year"].isin(TRAIN_YEARS + TEST_YEARS)
    ].copy()

    numeric_columns = [
        column for column in FEATURE_COLUMNS if column != "crop"
    ] + [TARGET_COLUMN]

    compared = expected[KEYS + numeric_columns].merge(
        stored[KEYS + numeric_columns],
        on=KEYS,
        how="outer",
        suffixes=("_rebuilt", "_stored"),
        indicator=True,
        validate="one_to_one",
    )

    if len(compared) != 1197 or not compared["_merge"].eq("both").all():
        raise ValueError("Expected/stored dataset keys differ.")

    audit_rows = []

    for column in numeric_columns:
        rebuilt = compared[f"{column}_rebuilt"].to_numpy(dtype=float)
        saved = compared[f"{column}_stored"].to_numpy(dtype=float)

        if not np.isfinite(rebuilt).all() or not np.isfinite(saved).all():
            raise ValueError(f"Non-finite values: {column}")

        differences = np.abs(rebuilt - saved)

        audit_rows.append({
            "column": column,
            "rows": len(compared),
            "mismatches": int((differences > 1e-6).sum()),
            "max_difference": float(differences.max()),
        })

    audit = pd.DataFrame(audit_rows)
    print_table("STORED VS RECONSTRUCTED INPUT AUDIT", audit)

    if audit["mismatches"].sum():
        raise ValueError("Input audit failed. Inspect differences first.")

    return expected, audit


def check_legacy_predictions(test_predictions):
    if not LEGACY_RESULTS.is_file():
        print(
            "\nLegacy reproduction NOT CHECKED: file not found:\n"
            f"{LEGACY_RESULTS}"
        )
        return {"status": "not_checked", "reason": "file_missing"}

    legacy = pd.read_csv(LEGACY_RESULTS)

    required = KEYS + ["actual_yield_t_ha", "predicted_yield_t_ha"]
    missing = set(required) - set(legacy.columns)

    if missing:
        raise ValueError(f"Missing legacy CSV columns: {sorted(missing)}")

    legacy = legacy[required].rename(columns={
        "actual_yield_t_ha": "legacy_actual",
        "predicted_yield_t_ha": "legacy_prediction",
    })

    current = test_predictions[
        test_predictions["model"].eq("xgboost_full")
    ][KEYS + [TARGET_COLUMN, "prediction"]]

    compared = current.merge(
        legacy,
        on=KEYS,
        how="outer",
        indicator=True,
        validate="one_to_one",
    )

    if len(compared) != 171 or not compared["_merge"].eq("both").all():
        raise ValueError("Legacy/current test keys differ.")

    if not np.allclose(
        compared[TARGET_COLUMN],
        compared["legacy_actual"],
        atol=1e-6,
        rtol=0,
    ):
        raise ValueError("Legacy/current observed yields differ.")

    current_values = compared["prediction"].to_numpy(dtype=float)
    legacy_values = compared["legacy_prediction"].to_numpy(dtype=float)

    if not np.isfinite(legacy_values).all():
        raise ValueError("Non-finite legacy predictions.")

    difference = float(np.max(np.abs(current_values - legacy_values)))

    if not np.allclose(
        current_values, legacy_values, atol=1e-6, rtol=0
    ):
        raise ValueError(
            "Legacy XGBoost reproduction failed. "
            f"Maximum difference: {difference:.9f} t/ha. "
            "Check data, parameters and library versions."
        )

    print(
        "\nLegacy XGBoost predictions reproduced. "
        f"Maximum difference: {difference:.9f} t/ha"
    )

    return {
        "status": "passed",
        "max_absolute_difference": difference,
    }


def summarize(predictions, period):
    rows = []
    groups = [("ALL", predictions)]
    groups.extend(predictions.groupby("crop", sort=True))

    for crop, group in groups:
        for model_name, model_data in group.groupby("model", sort=True):
            rows.append({
                "period": period,
                "crop": crop,
                "model": model_name,
                "records": len(model_data),
                **calculate_metrics(
                    model_data[TARGET_COLUMN],
                    model_data["prediction"],
                ),
                "negative_predictions": int(
                    (model_data["prediction"] < 0).sum()
                ),
            })

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(
        description="Fixed preseason models: retrospective 2023-2025 test."
    )
    parser.add_argument(
        "--reference-run",
        required=True,
        help="Directory name of the previous feature_groups run.",
    )
    args = parser.parse_args()

    reference_dir, reference_settings = load_settings(args.reference_run)
    xgb_parameters = reference_settings["xgboost_parameters"]
    ridge_alpha = reference_settings["ridge_alpha"]

    print("FIXED PRESEASON MODELS: XGBOOST VS RIDGE VS BASELINE")
    print(f"Reference: {reference_dir}")
    print(f"XGBoost parameters: {xgb_parameters}")
    print(f"Ridge alpha: {ridge_alpha}")
    print("No tuning or crop-specific model selection is performed.")
    print("These test years were previously inspected.")
    print("Weather inputs are reconstructed from earlier seasons.")

    dataframe, audit = prepare_expected_dataset()

    train = dataframe[
        dataframe["yield_year"].isin(TRAIN_YEARS)
    ].copy()

    test = dataframe[
        dataframe["yield_year"].isin(TEST_YEARS)
    ].copy()

    if len(train) != 1026 or len(test) != 171:
        raise ValueError("Unexpected train/test row counts.")

    if sorted(train["yield_year"].unique()) != TRAIN_YEARS:
        raise ValueError("Unexpected training years.")

    if sorted(test["yield_year"].unique()) != TEST_YEARS:
        raise ValueError("Unexpected test years.")

    print(f"\nTrain: 2005-2022, {len(train)} rows")
    print(f"Test: {TEST_YEARS}, {len(test)} rows")

    template = test[
        KEYS + [TARGET_COLUMN, "recent_yield_mean_t_ha"]
    ].copy()

    baseline = template.copy()
    baseline["model"] = "baseline"
    baseline["prediction"] = baseline["recent_yield_mean_t_ha"]
    results = [baseline]

    candidates = [
        (
            "xgboost_full",
            create_model(**xgb_parameters),
            XGB_FEATURES,
        ),
        (
            "ridge_no_soil",
            create_ridge(RIDGE_FEATURES, ridge_alpha),
            RIDGE_FEATURES,
        ),
    ]

    for model_name, model, features in candidates:
        print(f"\nTraining {model_name}...", flush=True)

        model.fit(train[features], train[TARGET_COLUMN])
        predicted = model.predict(test[features])

        if not np.isfinite(predicted).all():
            raise ValueError(f"Non-finite predictions: {model_name}")

        result = template.copy()
        result["model"] = model_name
        result["prediction"] = predicted
        results.append(result)

    predictions = pd.concat(results, ignore_index=True)

    if predictions.duplicated(KEYS + ["model"]).any():
        raise ValueError("Duplicate test predictions.")

    legacy_check = check_legacy_predictions(predictions)

    predictions["error_t_ha"] = (
        predictions["prediction"] - predictions[TARGET_COLUMN]
    )
    predictions["absolute_error_t_ha"] = predictions["error_t_ha"].abs()

    pooled = summarize(predictions, "2023-2025")
    print_table("OVERALL AND BY CROP", pooled)

    metrics = [pooled]

    for year in TEST_YEARS:
        yearly = summarize(
            predictions[predictions["yield_year"].eq(year)],
            str(year),
        )
        metrics.append(yearly)
        print_table(f"YEAR {year}", yearly)

    for model_name in ["xgboost_full", "ridge_no_soil"]:
        largest_errors = predictions[
            predictions["model"].eq(model_name)
        ].nlargest(10, "absolute_error_t_ha")

        print_table(
            f"10 LARGEST ERRORS: {model_name}",
            largest_errors[
                KEYS + [
                    TARGET_COLUMN,
                    "prediction",
                    "recent_yield_mean_t_ha",
                    "error_t_ha",
                ]
            ],
        )

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output_dir = BASE_DIR / "results" / f"preseason_test_{run_id}"
    output_dir.mkdir(parents=True, exist_ok=False)

    audit.to_csv(output_dir / "input_audit.csv", index=False)
    dataframe.to_csv(output_dir / "input_snapshot.csv", index=False)
    predictions.to_csv(output_dir / "predictions.csv", index=False)
    pd.concat(metrics, ignore_index=True).to_csv(
        output_dir / "metrics.csv", index=False
    )

    settings = {
        "reference_run": str(reference_dir),
        "train_years": TRAIN_YEARS,
        "test_years": TEST_YEARS,
        "xgboost_parameters": xgb_parameters,
        "ridge_alpha": ridge_alpha,
        "xgboost_features": XGB_FEATURES,
        "ridge_features": RIDGE_FEATURES,
        "legacy_reproduction": legacy_check,
        "test_period_previously_inspected": True,
        "tuning_performed": False,
        "publication_delays_modelled": False,
        "models_updated_during_test": False,
        "historical_inputs_updated_each_year": True,
    }

    (output_dir / "settings.json").write_text(
        json.dumps(settings, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"\nResults saved to:\n{output_dir}")
    print("\nIMPORTANT:")
    print("- Both ML models were trained only on 2005-2022.")
    print("- Both use expected weather during training and prediction.")
    print("- Historical weather/yield inputs are updated for each test year.")
    print("- Models are not retrained during the test period.")
    print("- Positive bias means overestimation.")
    print("- Predictions were not clipped or corrected.")
    print("- Release-date availability is not verified.")
    print("- This is a previously inspected retrospective test period.")
    print("- Scenario ranges and parcel accuracy are not evaluated.")
    print("- No production models or database tables were modified.")


if __name__ == "__main__":
    main()