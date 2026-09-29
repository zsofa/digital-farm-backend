import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from ml.data_loader import FEATURE_COLUMNS, TARGET_COLUMN
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
from ml.train.train_xgboost import create_model


BASE_DIR = Path(__file__).resolve().parent

REFERENCE_FILE = (
    BASE_DIR.parent
    / "compare"
    / "results"
    / "combined_test_001"
    / "predictions.csv"
)

V2_PARAMETERS = {
    "n_estimators": 200,
    "learning_rate": 0.05,
    "max_depth": 2,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "reg_lambda": 5.0,
}

TRAIN_YEARS = list(range(2005, 2023))
TEST_YEARS = [2023, 2024, 2025]

PREDICTION_COLUMNS = {
    "baseline": "recent_yield_mean_t_ha",
    "v2_actual_weather": "actual_weather_prediction",
    "v2_expected_weather": "expected_weather_prediction",
}


def verify_actual_predictions(results):
    if not REFERENCE_FILE.is_file():
        raise FileNotFoundError(
            "The previous V2 test predictions are required:\n"
            f"{REFERENCE_FILE}\n"
            "If stored elsewhere, update REFERENCE_FILE."
        )

    reference = pd.read_csv(REFERENCE_FILE)

    required = KEYS + [TARGET_COLUMN, "v2_prediction"]
    missing = set(required) - set(reference.columns)

    if missing:
        raise ValueError(
            f"Missing reference columns: {sorted(missing)}"
        )

    reference = reference[required].rename(columns={
        TARGET_COLUMN: "reference_actual_yield",
        "v2_prediction": "reference_prediction",
    })

    compared = results[
        KEYS + [TARGET_COLUMN, "actual_weather_prediction"]
    ].merge(
        reference,
        on=KEYS,
        how="outer",
        indicator=True,
        validate="one_to_one",
    )

    if len(compared) != 171 or not compared["_merge"].eq("both").all():
        raise ValueError("Reference/current test keys differ.")

    if not np.allclose(
        compared[TARGET_COLUMN],
        compared["reference_actual_yield"],
        atol=1e-6,
        rtol=0,
    ):
        raise ValueError("Reference/current observed yields differ.")

    current = compared["actual_weather_prediction"].to_numpy(dtype=float)
    previous = compared["reference_prediction"].to_numpy(dtype=float)

    if not np.isfinite(previous).all():
        raise ValueError("Non-finite reference predictions.")

    max_difference = float(np.max(np.abs(current - previous)))

    if not np.allclose(current, previous, atol=1e-6, rtol=0):
        raise ValueError(
            "V2 actual-weather reproduction failed. "
            f"Maximum difference: {max_difference:.9f} t/ha."
        )

    print(
        "\nV2 actual-weather test predictions reproduced. "
        f"Maximum difference: {max_difference:.9f} t/ha"
    )

    return max_difference


def groups(dataframe):
    yield "ALL", dataframe

    for crop, group in dataframe.groupby("crop", sort=True):
        yield crop, group


def summarize(results, period):
    rows = []

    for crop, group in groups(results):
        for model_name, column in PREDICTION_COLUMNS.items():
            rows.append({
                "period": period,
                "crop": crop,
                "model": model_name,
                "records": len(group),
                **calculate_metrics(
                    group[TARGET_COLUMN], group[column]
                ),
            })

    return pd.DataFrame(rows)


def summarize_weather_change(results, period):
    rows = []

    for crop, group in groups(results):
        actual_mae = group["actual_weather_abs_error"].mean()
        expected_mae = group["expected_weather_abs_error"].mean()

        rows.append({
            "period": period,
            "crop": crop,
            "records": len(group),
            "actual_weather_mae": actual_mae,
            "expected_weather_mae": expected_mae,
            "mae_increase": expected_mae - actual_mae,
            "temperature_input_mae_c": (
                group["expected_temperature_c"]
                - group["actual_temperature_c"]
            ).abs().mean(),
            "precipitation_input_mae_mm": (
                group["expected_precipitation_mm"]
                - group["actual_precipitation_mm"]
            ).abs().mean(),
            "mean_abs_prediction_change": (
                group["expected_weather_prediction"]
                - group["actual_weather_prediction"]
            ).abs().mean(),
        })

    return pd.DataFrame(rows)


def main():
    if not REFERENCE_FILE.is_file():
        raise FileNotFoundError(REFERENCE_FILE)

    print("FIXED V2: ACTUAL VS EXPECTED WEATHER")
    print(f"Parameters: {V2_PARAMETERS}")
    print("One model fit; only weather inputs change at prediction time.")
    print("No tuning is performed.")
    print("The test period has already been inspected.")

    dataframe = load_dataset().reset_index(drop=True)

    train = dataframe[
        dataframe["yield_year"].isin(TRAIN_YEARS)
    ].copy()

    test = dataframe[
        dataframe["yield_year"].isin(TEST_YEARS)
    ].copy().reset_index(drop=True)

    if len(train) != 1026 or len(test) != 171:
        raise ValueError("Unexpected train/test row counts.")

    if sorted(train["yield_year"].unique()) != TRAIN_YEARS:
        raise ValueError("Unexpected training years.")

    if sorted(test["yield_year"].unique()) != TEST_YEARS:
        raise ValueError("Unexpected test years.")

    history = load_weather_history()
    scenarios = add_scenario_weather(test, history)

    actual_features = test[FEATURE_COLUMNS].copy()
    expected_features = actual_features.copy()

    expected_features[TEMPERATURE_COLUMN] = (
        scenarios["expected_temperature"]
    )
    expected_features[PRECIPITATION_COLUMN] = (
        scenarios["expected_precipitation"]
    )

    unchanged_columns = [
        column for column in FEATURE_COLUMNS
        if column not in [TEMPERATURE_COLUMN, PRECIPITATION_COLUMN]
    ]

    if not actual_features[unchanged_columns].equals(
        expected_features[unchanged_columns]
    ):
        raise ValueError("Non-weather inputs unexpectedly changed.")

    print(f"\nTrain: 2005-2022, {len(train)} rows")
    print(f"Test: {TEST_YEARS}, {len(test)} rows")
    print("Training V2 on actual historical weather...", flush=True)

    model = create_model(**V2_PARAMETERS)
    model.fit(train[FEATURE_COLUMNS], train[TARGET_COLUMN])

    results = test[
        KEYS + [TARGET_COLUMN, "recent_yield_mean_t_ha"]
    ].copy()

    results["actual_temperature_c"] = actual_features[TEMPERATURE_COLUMN]
    results["expected_temperature_c"] = expected_features[TEMPERATURE_COLUMN]
    results["actual_precipitation_mm"] = actual_features[PRECIPITATION_COLUMN]
    results["expected_precipitation_mm"] = expected_features[
        PRECIPITATION_COLUMN
    ]

    results["actual_weather_prediction"] = model.predict(actual_features)
    results["expected_weather_prediction"] = model.predict(expected_features)

    prediction_columns = [
        "actual_weather_prediction",
        "expected_weather_prediction",
    ]

    if not np.isfinite(
        results[prediction_columns].to_numpy(dtype=float)
    ).all():
        raise ValueError("Non-finite predictions.")

    max_difference = verify_actual_predictions(results)

    for name in ["actual_weather", "expected_weather"]:
        results[f"{name}_error"] = (
            results[f"{name}_prediction"] - results[TARGET_COLUMN]
        )
        results[f"{name}_abs_error"] = results[f"{name}_error"].abs()

    results["absolute_error_increase"] = (
        results["expected_weather_abs_error"]
        - results["actual_weather_abs_error"]
    )

    metrics = []
    changes = []

    periods = [("2023-2025", results)]
    periods.extend(
        (str(year), results[results["yield_year"].eq(year)])
        for year in TEST_YEARS
    )

    for period, period_data in periods:
        table = summarize(period_data, period)
        metrics.append(table)
        changes.append(summarize_weather_change(period_data, period))
        print_table(f"RESULTS: {period}", table)

    changes = pd.concat(changes, ignore_index=True)

    print_table(
        "WEATHER INPUT CHANGE: POSITIVE MAE INCREASE MEANS WORSE",
        changes,
    )

    print_table(
        "10 LARGEST INCREASES IN ABSOLUTE ERROR",
        results.nlargest(10, "absolute_error_increase")[
            KEYS + [
                TARGET_COLUMN,
                "actual_weather_prediction",
                "expected_weather_prediction",
                "absolute_error_increase",
            ]
        ],
    )

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output_dir = BASE_DIR / "results" / f"v2_weather_inputs_{run_id}"
    output_dir.mkdir(parents=True, exist_ok=False)

    results.to_csv(output_dir / "predictions.csv", index=False)
    pd.concat(metrics, ignore_index=True).to_csv(
        output_dir / "metrics.csv", index=False
    )
    changes.to_csv(output_dir / "weather_changes.csv", index=False)

    settings = {
        "v2_parameters": V2_PARAMETERS,
        "features": list(FEATURE_COLUMNS),
        "train_years": TRAIN_YEARS,
        "test_years": TEST_YEARS,
        "training_weather": "actual",
        "prediction_weather": ["actual", "expected"],
        "reference_file": str(REFERENCE_FILE),
        "reference_max_difference": max_difference,
        "test_period_previously_inspected": True,
        "tuning_performed": False,
        "publication_delays_modelled": False,
    }

    (output_dir / "settings.json").write_text(
        json.dumps(settings, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"\nResults saved to:\n{output_dir}")
    print("\nIMPORTANT:")
    print("- The exact same fitted model produced both ML predictions.")
    print("- Soil, crop and yield-history inputs were unchanged.")
    print("- Expected weather uses only seasons before the target season.")
    print("- Actual weather is an ex-post diagnostic input.")
    print("- Actual-weather accuracy is not guaranteed to be better.")
    print("- This measures the effect of replacing weather inputs.")
    print("- It does not establish causal crop responses.")
    print("- Publication delays and parcel accuracy are not evaluated.")
    print("- No production models or database tables were modified.")


if __name__ == "__main__":
    main()