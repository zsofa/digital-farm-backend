import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from ml.data_loader import FEATURE_COLUMNS, TARGET_COLUMN
from ml.experiments.actual_weather_v2.compare_models import load_best_params
from ml.experiments.actual_weather_v2.evaluate_weather_scenarios_cv import (
    FOLDS,
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
HISTORY_YEARS = 15

# Settings of the existing actual-weather scenario model.
CURRENT_PARAMETERS = {
    "n_estimators": 400,
    "learning_rate": 0.05,
    "max_depth": 3,
    "subsample": 0.8,
    "colsample_bytree": 1.0,
    "min_child_weight": 1,
    "reg_lambda": 5.0,
}

V2_PARAMETERS = {
    "n_estimators": 200,
    "learning_rate": 0.05,
    "max_depth": 2,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "reg_lambda": 5.0,
}

MODEL_PARAMETERS = {
    "current": CURRENT_PARAMETERS,
    "v2": V2_PARAMETERS,
}


def create_historical_inputs(validation, weather_history):
    records = []

    for row_id, row in validation.iterrows():
        county = row["county_name"]
        crop = row["crop"]
        target_year = int(row["season_start_year"])

        history = weather_history.get((county, crop), [])
        recent = [
            item for item in history
            if target_year - HISTORY_YEARS <= item["year"] < target_year
        ]

        expected_years = list(
            range(target_year - HISTORY_YEARS, target_year)
        )

        if [item["year"] for item in recent] != expected_years:
            raise ValueError(
                f"Incomplete scenario history: "
                f"{county}, {crop}, season {target_year}"
            )

        for weather in recent:
            record = {
                column: row[column] for column in FEATURE_COLUMNS
            }

            record[TEMPERATURE_COLUMN] = weather["temperature"]
            record[PRECIPITATION_COLUMN] = weather["precipitation"]

            record.update({
                "row_id": row_id,
                "yield_year": int(row["yield_year"]),
                "county_name": county,
                "source_season_start_year": int(weather["year"]),
            })

            records.append(record)

    scenarios = pd.DataFrame(records)

    if not scenarios.groupby("row_id").size().eq(HISTORY_YEARS).all():
        raise ValueError("Expected 15 scenarios per validation row.")

    return scenarios


def summarize_predictions(predictions, period):
    rows = []
    groups = [("ALL", predictions)]
    groups.extend(predictions.groupby("crop", sort=True))

    for crop, group in groups:
        for method, method_data in group.groupby("method", sort=True):
            rows.append({
                "period": period,
                "crop": crop,
                "method": method,
                "records": len(method_data),
                **calculate_metrics(
                    method_data[TARGET_COLUMN],
                    method_data["prediction"],
                ),
            })

    return pd.DataFrame(rows)


def summarize_ranges(ranges, period):
    rows = []
    groups = [("ALL", ranges)]
    groups.extend(ranges.groupby("crop", sort=True))

    for crop, group in groups:
        for model, model_data in group.groupby("model", sort=True):
            actual = model_data[TARGET_COLUMN]

            rows.append({
                "period": period,
                "crop": crop,
                "model": model,
                "records": len(model_data),
                "min_max_inside_pct": 100 * (
                    (actual >= model_data["minimum"])
                    & (actual <= model_data["maximum"])
                ).mean(),
                "min_max_mean_width": (
                    model_data["maximum"] - model_data["minimum"]
                ).mean(),
                "p10_p90_inside_pct": 100 * (
                    (actual >= model_data["p10"])
                    & (actual <= model_data["p90"])
                ).mean(),
                "p10_p90_mean_width": (
                    model_data["p90"] - model_data["p10"]
                ).mean(),
            })

    return pd.DataFrame(rows)


def main():
    saved_parameters, reference_v2_mae = load_best_params()

    if saved_parameters != V2_PARAMETERS:
        raise ValueError(
            "Saved V2 settings differ from the fixed comparison settings."
        )

    dataframe = load_dataset()
    dataframe = dataframe[
        dataframe["yield_year"].between(2005, 2022)
    ].copy()

    history = load_weather_history()

    print("CURRENT VS V2: HISTORICAL WEATHER SCENARIOS")
    print(f"Weather history: previous {HISTORY_YEARS} complete seasons")
    print("Temperature and precipitation remain paired by source season.")
    print("Central estimate: median of the 15 scenario predictions.")
    print("Historical weather is used without trend adjustment.")
    print("No tuning is performed.")
    print("Test years 2023-2025 are not evaluated.")
    print("Production model files are not loaded or overwritten.")

    all_predictions = []
    all_ranges = []
    all_scenarios = []
    v2_actual_maes = []

    for train_end, validation_start, validation_end in FOLDS:
        period = f"{validation_start}-{validation_end}"

        train = dataframe[
            dataframe["yield_year"].between(2005, train_end)
        ].copy()

        validation = dataframe[
            dataframe["yield_year"].between(
                validation_start, validation_end
            )
        ].copy().reset_index(drop=True)

        if len(train) != (train_end - 2005 + 1) * 57:
            raise ValueError("Unexpected training row count.")

        if len(validation) != 114:
            raise ValueError("Expected 114 validation rows.")

        print(
            f"\nTrain 2005-{train_end}; validation {period}",
            flush=True,
        )

        # Also checks crop/harvest-year alignment.
        expected_weather = add_scenario_weather(validation, history)

        expected_features = validation[FEATURE_COLUMNS].copy()
        expected_features[TEMPERATURE_COLUMN] = (
            expected_weather["expected_temperature"]
        )
        expected_features[PRECIPITATION_COLUMN] = (
            expected_weather["expected_precipitation"]
        )

        historical = create_historical_inputs(validation, history)

        template = validation[
            KEYS + [TARGET_COLUMN, "recent_yield_mean_t_ha"]
        ].copy()
        template["period"] = period

        baseline = template.copy()
        baseline["method"] = "baseline"
        baseline["prediction"] = baseline["recent_yield_mean_t_ha"]
        all_predictions.append(baseline)

        for model_name, parameters in MODEL_PARAMETERS.items():
            model = create_model(**parameters)
            model.fit(train[FEATURE_COLUMNS], train[TARGET_COLUMN])

            actual_prediction = model.predict(
                validation[FEATURE_COLUMNS]
            )
            expected_prediction = model.predict(expected_features)
            historical_prediction = model.predict(
                historical[FEATURE_COLUMNS]
            )

            for values in [
                actual_prediction,
                expected_prediction,
                historical_prediction,
            ]:
                if not np.isfinite(values).all():
                    raise ValueError("Non-finite model predictions.")

            scenario_results = historical.copy()
            scenario_results["prediction"] = historical_prediction
            scenario_results["model"] = model_name
            scenario_results["period"] = period
            all_scenarios.append(scenario_results)

            grouped = scenario_results.groupby("row_id")["prediction"]

            statistics = pd.DataFrame({
                "minimum": grouped.min(),
                "p10": grouped.quantile(0.10),
                "median": grouped.median(),
                "p90": grouped.quantile(0.90),
                "maximum": grouped.max(),
            }).reindex(validation.index)

            if statistics.isna().any().any():
                raise ValueError("Incomplete scenario statistics.")

            methods = {
                "actual_weather": actual_prediction,
                "expected_weather": expected_prediction,
                "historical_median": statistics["median"].to_numpy(),
            }

            for suffix, values in methods.items():
                result = template.copy()
                result["method"] = f"{model_name}_{suffix}"
                result["prediction"] = values
                all_predictions.append(result)

            range_result = template.copy()
            range_result["model"] = model_name

            for column in statistics.columns:
                range_result[column] = statistics[column].to_numpy()

            all_ranges.append(range_result)

            if model_name == "v2":
                v2_actual_maes.append(calculate_metrics(
                    validation[TARGET_COLUMN], actual_prediction
                )["mae"])

    reproduced_mae = float(np.mean(v2_actual_maes))

    if not np.isclose(
        reproduced_mae, reference_v2_mae, atol=1e-6, rtol=0
    ):
        raise ValueError(
            f"V2 reproduction failed: "
            f"{reproduced_mae:.9f} vs {reference_v2_mae:.9f}"
        )

    print(f"\nV2 actual-weather MAE reproduced: {reproduced_mae:.6f}")

    predictions = pd.concat(all_predictions, ignore_index=True)
    ranges = pd.concat(all_ranges, ignore_index=True)
    scenarios = pd.concat(all_scenarios, ignore_index=True)

    if predictions.duplicated(KEYS + ["method"]).any():
        raise ValueError("Duplicate validation predictions.")

    counts = predictions.groupby("method").size()
    if len(counts) != 7 or not counts.eq(456).all():
        raise ValueError("Expected seven methods with 456 predictions each.")

    pooled = summarize_predictions(predictions, "POOLED")
    range_metrics = summarize_ranges(ranges, "POOLED")

    print_table("POOLED VALIDATION RESULTS", pooled)
    print_table("DESCRIPTIVE SCENARIO RANGES", range_metrics)

    fold_metrics = []
    fold_ranges = []

    for period, period_data in predictions.groupby("period", sort=True):
        metrics = summarize_predictions(period_data, period)
        fold_metrics.append(metrics)

        fold_ranges.append(summarize_ranges(
            ranges[ranges["period"].eq(period)], period
        ))

        print_table(
            f"OVERALL VALIDATION {period}",
            metrics[metrics["crop"].eq("ALL")],
        )

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output_dir = BASE_DIR / "results" / f"historical_scenarios_{run_id}"
    output_dir.mkdir(parents=True, exist_ok=False)

    predictions.to_csv(output_dir / "predictions.csv", index=False)
    scenarios.to_csv(output_dir / "individual_scenarios.csv", index=False)
    ranges.to_csv(output_dir / "scenario_ranges.csv", index=False)
    pooled.to_csv(output_dir / "pooled_metrics.csv", index=False)
    range_metrics.to_csv(output_dir / "range_metrics.csv", index=False)

    pd.concat(fold_metrics, ignore_index=True).to_csv(
        output_dir / "fold_metrics.csv", index=False
    )
    pd.concat(fold_ranges, ignore_index=True).to_csv(
        output_dir / "fold_range_metrics.csv", index=False
    )

    settings = {
        "model_parameters": MODEL_PARAMETERS,
        "features": list(FEATURE_COLUMNS),
        "history_years": HISTORY_YEARS,
        "central_estimate": "median",
        "weather_trend_adjustment": False,
        "folds": FOLDS,
        "reserved_test_years": [2023, 2024, 2025],
        "tuning_performed": False,
        "independent_test": False,
        "publication_delays_modelled": False,
        "range_interpretation": (
            "Spread of model predictions over historical weather cases; "
            "not a calibrated prediction interval"
        ),
    }

    (output_dir / "settings.json").write_text(
        json.dumps(settings, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"\nResults saved to:\n{output_dir}")
    print("\nIMPORTANT:")
    print("- current uses the existing model's settings, retrained per fold.")
    print("- Both models are trained on actual historical weather.")
    print("- actual_weather is an ex-post diagnostic, not a preseason forecast.")
    print("- expected_weather uses the current central weather scenario.")
    print("- historical_median uses 15 paired historical weather cases.")
    print("- Scenario spread does not include the full model error.")
    print("- p10/p90 are scenario quantiles, not verified probability bounds.")
    print("- Wider ranges alone do not establish better performance.")
    print("- Past weather may not represent the target year's climate.")
    print("- V2 parameters were selected on these development folds.")
    print("- This is not an independent test or parcel-level validation.")
    print("- No production models or database tables were modified.")


if __name__ == "__main__":
    main()