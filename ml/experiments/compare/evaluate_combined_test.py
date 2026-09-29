import json
from pathlib import Path

import numpy as np
import pandas as pd

from ml.data_loader import FEATURE_COLUMNS, TARGET_COLUMN
from ml.evaluation.compare_final_to_baseline import calculate_metrics
from ml.experiments.actual_weather_v2.compare_models import (
    load_best_params,
)
from ml.experiments.actual_weather_v2.tune_xgboost import load_dataset
from ml.experiments.actual_weather_v4.compare_monthly_weather import (
    V4_FEATURES,
    add_monthly_features,
)
from ml.experiments.actual_weather_v4.tune_monthly_xgboost import (
    BEST_PARAMS_FILE,
)


OUTPUT_DIR = (
    Path(__file__).resolve().parent
    / "results"
    / "combined_test_001"
)

TRAIN_YEARS = list(range(2005, 2023))
TEST_YEARS = [2023, 2024, 2025]

PREDICTION_COLUMNS = {
    "baseline": "recent_yield_mean_t_ha",
    "seasonal_v2": "v2_prediction",
    "monthly_v4": "v4_prediction",
    "combined": "combined_prediction",
}


def summarize(dataframe, period):
    rows = []
    groups = [("ALL", dataframe)]
    groups.extend(dataframe.groupby("crop", sort=True))

    for crop, group in groups:
        actual = group[TARGET_COLUMN]

        for name, column in PREDICTION_COLUMNS.items():
            metrics = calculate_metrics(actual, group[column])

            rows.append({
                "period": period,
                "crop": crop,
                "model": name,
                "records": len(group),
                "mae": metrics["mae"],
                "rmse": metrics["rmse"],
                "r2": metrics["r2"],
                "bias": float((group[column] - actual).mean()),
            })

    return pd.DataFrame(rows)


def print_table(title, dataframe):
    print("\n" + "=" * 110)
    print(title)
    print("=" * 110)
    print(dataframe.round(3).to_string(index=False))


def main():
    if OUTPUT_DIR.exists():
        raise FileExistsError(
            f"Output directory already exists: {OUTPUT_DIR}. "
            "Use another run directory to preserve previous results."
        )

    v2_params, _ = load_best_params()

    with BEST_PARAMS_FILE.open("r", encoding="utf-8") as source:
        v4_params = json.load(source)

    dataframe = add_monthly_features(load_dataset())

    train = dataframe[
        dataframe["yield_year"].isin(TRAIN_YEARS)
    ].copy()

    test = dataframe[
        dataframe["yield_year"].isin(TEST_YEARS)
    ].copy()

    if len(train) != 1026 or len(test) != 171:
        raise ValueError(
            f"Unexpected row counts: train={len(train)}, test={len(test)}."
        )

    if set(train["yield_year"]) != set(TRAIN_YEARS):
        raise ValueError("Unexpected training years.")

    if set(test["yield_year"]) != set(TEST_YEARS):
        raise ValueError("Unexpected test years.")

    print("=" * 110)
    print("FIXED MODEL COMPARISON ON 2023-2025")
    print("=" * 110)
    print(f"Train: 2005-2022, {len(train)} rows")
    print(f"Test: {TEST_YEARS}, {len(test)} rows")
    print(f"V2 parameters: {v2_params}")
    print(f"V4 parameters: {v4_params}")
    print("Routing: wheat/barley -> V2; maize -> V4")
    print("These test years have already been inspected in earlier work.")

    for column, features, params in [
        ("v2_prediction", FEATURE_COLUMNS, v2_params),
        ("v4_prediction", V4_FEATURES, v4_params),
    ]:
        print(f"\nTraining {column}...", flush=True)

        model = create_model(**params)
        model.fit(train[features], train[TARGET_COLUMN])
        test[column] = model.predict(test[features])

    test["combined_prediction"] = np.where(
        test["crop"].eq("maize"),
        test["v4_prediction"],
        test["v2_prediction"],
    )

    if not np.isfinite(
        test[list(PREDICTION_COLUMNS.values())].to_numpy(dtype=float)
    ).all():
        raise ValueError("Invalid predictions.")

    summaries = [summarize(test, "2023-2025")]

    print_table("OVERALL AND BY CROP", summaries[0])

    for year, group in test.groupby("yield_year", sort=True):
        yearly_summary = summarize(group, str(year))
        summaries.append(yearly_summary)
        print_table(f"YEAR {year}: OVERALL AND BY CROP", yearly_summary)

    test["combined_error_t_ha"] = (
        test["combined_prediction"] - test[TARGET_COLUMN]
    )
    test["combined_absolute_error_t_ha"] = (
        test["combined_error_t_ha"].abs()
    )
    test["baseline_absolute_error_t_ha"] = (
        test["recent_yield_mean_t_ha"] - test[TARGET_COLUMN]
    ).abs()

    worst_columns = [
        "yield_year",
        "county_name",
        "crop",
        TARGET_COLUMN,
        "recent_yield_mean_t_ha",
        "combined_prediction",
        "combined_error_t_ha",
        "baseline_absolute_error_t_ha",
    ]

    print_table(
        "10 LARGEST COMBINED PREDICTION ERRORS",
        test.nlargest(10, "combined_absolute_error_t_ha")[worst_columns],
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=False)

    prediction_columns = [
        "yield_year",
        "county_name",
        "crop",
        TARGET_COLUMN,
        *PREDICTION_COLUMNS.values(),
        "combined_error_t_ha",
        "combined_absolute_error_t_ha",
        "baseline_absolute_error_t_ha",
    ]

    test[prediction_columns].to_csv(
        OUTPUT_DIR / "predictions.csv",
        index=False,
    )

    pd.concat(summaries, ignore_index=True).to_csv(
        OUTPUT_DIR / "metrics.csv",
        index=False,
    )

    settings = {
        "train_years": TRAIN_YEARS,
        "test_years": TEST_YEARS,
        "v2_params": v2_params,
        "v4_params": v4_params,
        "v2_features": FEATURE_COLUMNS,
        "v4_features": V4_FEATURES,
        "routing": {
            "barley": "seasonal_v2",
            "wheat": "seasonal_v2",
            "maize": "monthly_v4",
        },
        "test_period_previously_inspected": True,
    }

    with (OUTPUT_DIR / "settings.json").open(
        "x", encoding="utf-8"
    ) as output:
        json.dump(settings, output, indent=2, ensure_ascii=False)

    print(f"\nResults saved to:\n{OUTPUT_DIR}")
    print("No parameter tuning was performed.")
    print("Actual historical weather was used.")
    print("Weather scenarios and parcel-level accuracy were not evaluated.")
    print("No production models or database tables were modified.")


from ml.train.train_xgboost import create_model


if __name__ == "__main__":
    main()