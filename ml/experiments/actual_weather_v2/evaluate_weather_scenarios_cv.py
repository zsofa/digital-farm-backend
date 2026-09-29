import numpy as np
import pandas as pd

from ml.data_loader import FEATURE_COLUMNS, TARGET_COLUMN, get_connection
from ml.experiments.actual_weather_v2.compare_models import load_best_params
from ml.experiments.actual_weather_v2.tune_xgboost import load_dataset
from ml.scenarios.weather_scenarios import (
    PRECIPITATION_HISTORY_YEARS,
    TEMPERATURE_HISTORY_YEARS,
    calculate_precipitation_scenarios,
    calculate_temperature_scenarios,
)
from ml.train.train_xgboost import create_model


FOLDS = [
    (2014, 2015, 2016),
    (2016, 2017, 2018),
    (2018, 2019, 2020),
    (2020, 2021, 2022),
]

SCENARIOS = ("dry_hot", "expected", "wet_cool")
RESERVED_TEST_YEARS = [2023, 2024, 2025]

TEMPERATURE_COLUMN = "expected_season_temperature_c"
PRECIPITATION_COLUMN = "expected_season_precipitation_mm"

PREDICTION_COLUMNS = {
    "baseline": "recent_yield_mean_t_ha",
    "actual_weather": "actual_weather_prediction",
    "expected": "expected_prediction",
}


def load_weather_history():
    query = """
        SELECT
            county_name,
            crop,
            season_start_year,
            season_mean_temperature_c,
            season_precipitation_mm
        FROM processed.weather_seasonal
        ORDER BY county_name, crop, season_start_year;
    """

    connection = get_connection()

    try:
        with connection.cursor() as cursor:
            cursor.execute(query)
            rows = cursor.fetchall()
    finally:
        connection.close()

    dataframe = pd.DataFrame(
        rows,
        columns=[
            "county_name",
            "crop",
            "year",
            "temperature",
            "precipitation",
        ],
    )

    if dataframe.empty:
        raise ValueError("No seasonal weather records found.")

    keys = ["county_name", "crop", "year"]

    if dataframe[keys].isna().any().any():
        raise ValueError("Missing weather keys.")

    if dataframe.duplicated(keys).any():
        raise ValueError("Duplicate seasonal weather records.")

    numeric_columns = ["year", "temperature", "precipitation"]

    for column in numeric_columns:
        dataframe[column] = pd.to_numeric(
            dataframe[column], errors="raise"
        )

    if not np.isfinite(
        dataframe[numeric_columns].to_numpy(dtype=float)
    ).all():
        raise ValueError("Non-finite seasonal weather values.")

    if (dataframe["year"] % 1 != 0).any():
        raise ValueError("Non-integer season years.")

    if (dataframe["precipitation"] < 0).any():
        raise ValueError("Negative precipitation.")

    dataframe["year"] = dataframe["year"].astype(int)

    return {
        key: group.sort_values("year").to_dict("records")
        for key, group in dataframe.groupby(["county_name", "crop"])
    }


def weather_for_target(weather_history, county, crop, target_year):
    key = (county, crop)

    if key not in weather_history:
        raise ValueError(f"No weather history for {key}.")

    history = [
        row
        for row in weather_history[key]
        if row["year"] < target_year
    ]

    required_years = max(
        TEMPERATURE_HISTORY_YEARS,
        PRECIPITATION_HISTORY_YEARS,
    )

    expected_years = list(
        range(target_year - required_years, target_year)
    )
    available_years = [
        row["year"] for row in history[-required_years:]
    ]

    if available_years != expected_years:
        raise ValueError(
            f"Incomplete recent weather history for "
            f"{county}, {crop}, season {target_year}: "
            f"{available_years}"
        )

    temperatures = calculate_temperature_scenarios(
        history, target_year
    )
    precipitation = calculate_precipitation_scenarios(history)

    # Same rounding as the application's create_weather_scenarios().
    return {
        scenario: {
            "temperature": round(float(temperatures[scenario]), 3),
            "precipitation": round(
                float(precipitation[scenario]), 2
            ),
        }
        for scenario in SCENARIOS
    }


def add_scenario_weather(validation, weather_history):
    result = validation.copy()

    values = {
        scenario: {"temperature": [], "precipitation": []}
        for scenario in SCENARIOS
    }

    for row in result.itertuples(index=False):
        target_year = int(row.season_start_year)
        harvest_year = int(row.yield_year)

        expected_start_year = (
            harvest_year if row.crop == "maize"
            else harvest_year - 1
        )

        if target_year != expected_start_year:
            raise ValueError(
                f"Invalid season alignment: "
                f"{row.county_name}, {row.crop}, {harvest_year}"
            )

        scenarios = weather_for_target(
            weather_history,
            row.county_name,
            row.crop,
            target_year,
        )

        for scenario in SCENARIOS:
            for variable in ("temperature", "precipitation"):
                values[scenario][variable].append(
                    scenarios[scenario][variable]
                )

    for scenario in SCENARIOS:
        for variable in ("temperature", "precipitation"):
            result[f"{scenario}_{variable}"] = (
                values[scenario][variable]
            )

    return result


def calculate_metrics(actual, predicted):
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    if not np.isfinite(actual).all():
        raise ValueError("Non-finite observed yields.")

    if not np.isfinite(predicted).all():
        raise ValueError("Non-finite predictions.")

    errors = predicted - actual
    total_variation = np.sum((actual - actual.mean()) ** 2)

    r2 = (
        1 - np.sum(errors ** 2) / total_variation
        if total_variation > 0
        else np.nan
    )

    return {
        "mae": np.mean(np.abs(errors)),
        "rmse": np.sqrt(np.mean(errors ** 2)),
        "r2": r2,
        "bias": np.mean(errors),
    }


def iter_groups(dataframe):
    yield "ALL", dataframe

    for crop, group in dataframe.groupby("crop", sort=True):
        yield crop, group


def summarize(dataframe, period):
    rows = []

    for crop, group in iter_groups(dataframe):
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


def summarize_scenario_ranges(dataframe):
    rows = []
    scenario_columns = [
        f"{scenario}_prediction" for scenario in SCENARIOS
    ]

    for crop, group in iter_groups(dataframe):
        predictions = group[scenario_columns]
        lower = predictions.min(axis=1)
        upper = predictions.max(axis=1)
        actual = group[TARGET_COLUMN]

        expected_between = (
            (
                group["dry_hot_prediction"]
                <= group["expected_prediction"]
            )
            & (
                group["expected_prediction"]
                <= group["wet_cool_prediction"]
            )
        )

        rows.append({
            "crop": crop,
            "records": len(group),
            "observed_inside_pct": 100 * (
                (actual >= lower) & (actual <= upper)
            ).mean(),
            "mean_range_t_ha": (upper - lower).mean(),
            "dry_le_expected_le_wet_pct": (
                100 * expected_between.mean()
            ),
            "negative_prediction_rows": int(
                (predictions < 0).any(axis=1).sum()
            ),
        })

    return pd.DataFrame(rows)


def print_table(title, dataframe):
    print("\n" + "=" * 110)
    print(title)
    print("=" * 110)
    print(dataframe.to_string(
        index=False,
        float_format=lambda value: f"{value:.3f}",
    ))


def main():
    dataframe = load_dataset()
    parameters, reference_mae = load_best_params()
    weather_history = load_weather_history()

    print("V2: ACTUAL WEATHER VS APPLICATION WEATHER SCENARIOS")
    print(f"Parameters: {parameters}")
    print(f"Reserved test years: {RESERVED_TEST_YEARS}")
    print("Model parameters are fixed; no tuning is performed.")
    print("Weather scenarios use only earlier seasons.")
    print("Historical inputs are updated for each validation year.")

    all_predictions = []
    fold_maes = []

    for fold_number, (
        train_end, validation_start, validation_end
    ) in enumerate(FOLDS, start=1):
        train = dataframe[
            dataframe["yield_year"].between(2005, train_end)
        ].copy()

        validation = dataframe[
            dataframe["yield_year"].between(
                validation_start, validation_end
            )
        ].copy().reset_index(drop=True)

        expected_train_rows = (train_end - 2005 + 1) * 57

        if len(train) != expected_train_rows:
            raise ValueError("Unexpected training row count.")

        if len(validation) != 114:
            raise ValueError("Expected 114 validation rows.")

        if train["yield_year"].max() >= validation["yield_year"].min():
            raise ValueError("Training/validation years overlap.")

        period = f"{validation_start}-{validation_end}"

        print(
            f"\nFold {fold_number}: train 2005-{train_end}, "
            f"validation {period}"
        )

        model = create_model(**parameters)
        model.fit(train[FEATURE_COLUMNS], train[TARGET_COLUMN])

        validation = add_scenario_weather(
            validation, weather_history
        )

        validation["actual_weather_prediction"] = model.predict(
            validation[FEATURE_COLUMNS]
        )

        for scenario in SCENARIOS:
            features = validation[FEATURE_COLUMNS].copy()

            features[TEMPERATURE_COLUMN] = (
                validation[f"{scenario}_temperature"]
            )
            features[PRECIPITATION_COLUMN] = (
                validation[f"{scenario}_precipitation"]
            )

            validation[f"{scenario}_prediction"] = (
                model.predict(features)
            )

        prediction_columns = [
            "actual_weather_prediction",
            *[
                f"{scenario}_prediction"
                for scenario in SCENARIOS
            ],
        ]

        if not np.isfinite(
            validation[prediction_columns].to_numpy(dtype=float)
        ).all():
            raise ValueError("Non-finite model predictions.")

        fold_maes.append(calculate_metrics(
            validation[TARGET_COLUMN],
            validation["actual_weather_prediction"],
        )["mae"])

        validation["validation_period"] = period
        all_predictions.append(validation)

        print_table(
            f"VALIDATION {period}",
            summarize(validation, period),
        )

    reproduced_mae = float(np.mean(fold_maes))

    if not np.isclose(
        reproduced_mae, reference_mae, atol=1e-6, rtol=0
    ):
        raise ValueError(
            "V2 actual-weather MAE reproduction failed: "
            f"{reproduced_mae:.9f} vs {reference_mae:.9f}"
        )

    print(
        f"\nV2 actual-weather MAE reproduced: "
        f"{reproduced_mae:.6f} t/ha"
    )

    predictions = pd.concat(all_predictions, ignore_index=True)

    if predictions.duplicated(
        ["yield_year", "county_name", "crop"]
    ).any():
        raise ValueError("Duplicate validation predictions.")

    print_table(
        "POOLED VALIDATION RESULTS",
        summarize(predictions, "POOLED"),
    )

    print_table(
        "DESCRIPTIVE SCENARIO RANGE CHECK",
        summarize_scenario_ranges(predictions),
    )

    print("\nInterpretation:")
    print("- Smaller MAE/RMSE means smaller prediction error.")
    print("- Positive bias means overestimation.")
    print("- actual_weather uses observed seasonal weather.")
    print("- expected uses the application's central weather scenario.")
    print("- Scenario ranges use the minimum/maximum of all 3 predictions.")
    print("- Range inclusion is NOT a calibrated confidence level.")
    print("- Dry/hot weather is not assumed to always give lower yield.")
    print("- Publication delays of historical data are not modelled.")
    print("- V2 parameters were selected on these validation folds.")
    print("- This is a development evaluation, not an independent test.")
    print("- Reserved test years were not evaluated.")
    print("- No files, database tables or production models were modified.")


if __name__ == "__main__":
    main()