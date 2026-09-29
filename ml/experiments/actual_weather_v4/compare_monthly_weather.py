import numpy as np
import pandas as pd

from ml.data_loader import FEATURE_COLUMNS, TARGET_COLUMN
from ml.evaluation.compare_final_to_baseline import calculate_metrics
from ml.experiments.actual_weather_v2.compare_models import (
    load_best_params,
)
from ml.experiments.actual_weather_v2.tune_xgboost import (
    load_dataset,
    prepare_folds,
)
from ml.experiments.actual_weather_v4.validate_monthly_weather import (
    load_monthly,
    season_months,
)
from ml.train.train_xgboost import create_model


BASE_FEATURES = [
    "crop",
    "soil_ph",
    "soil_soc_g_kg",
    "soil_clay_pct",
    "recent_yield_mean_t_ha",
]

# Calendar months relative to the harvest year.
MONTH_SLOTS = (
    [f"prev_{month:02d}" for month in range(9, 13)]
    + [f"current_{month:02d}" for month in range(1, 11)]
)

MONTHLY_FEATURES = [
    f"{variable}_{slot}"
    for slot in MONTH_SLOTS
    for variable in ["temperature", "precipitation"]
]

V4_FEATURES = BASE_FEATURES + MONTHLY_FEATURES

PREDICTION_COLUMNS = {
    "baseline": "recent_yield_mean_t_ha",
    "seasonal_v2": "v2_prediction",
    "monthly_v4": "v4_prediction",
}


def add_monthly_features(dataframe):
    monthly = load_monthly()
    feature_rows = []

    for row in dataframe.itertuples(index=False):
        harvest_year = int(row.yield_year)
        start_year = int(row.season_start_year)

        expected_start = (
            harvest_year
            if row.crop == "maize"
            else harvest_year - 1
        )

        if start_year != expected_start:
            raise ValueError(
                f"Invalid season year: {row.county_name}, "
                f"{row.crop}, {harvest_year}."
            )

        # NaN means that the month is outside this crop's season.
        # Required in-season values must exist and be finite.
        features = {column: np.nan for column in MONTHLY_FEATURES}

        for year, month in season_months(row.crop, start_year):
            key = (year, month, row.county_name)

            if key not in monthly.index:
                raise ValueError(f"Missing monthly weather: {key}")

            weather = monthly.loc[key]

            temperature = float(weather["mean_temperature_c"])
            precipitation = float(weather["precipitation_mm"])

            if not np.isfinite([temperature, precipitation]).all():
                raise ValueError(f"Invalid monthly weather: {key}")

            if precipitation < 0:
                raise ValueError(f"Negative precipitation: {key}")

            if year == harvest_year - 1:
                slot = f"prev_{month:02d}"
            elif year == harvest_year:
                slot = f"current_{month:02d}"
            else:
                raise ValueError(f"Unexpected weather year: {key}")

            if slot not in MONTH_SLOTS:
                raise ValueError(f"Unexpected month slot: {slot}")

            features[f"temperature_{slot}"] = temperature
            features[f"precipitation_{slot}"] = precipitation

        expected_values = 16 if row.crop == "maize" else 22

        if sum(pd.notna(value) for value in features.values()) != expected_values:
            raise ValueError("Unexpected number of monthly inputs.")

        feature_rows.append(features)

    features = pd.DataFrame(feature_rows, index=dataframe.index)
    return pd.concat([dataframe, features], axis=1)


def summarize(dataframe, period):
    records = []
    groups = [("ALL", dataframe)]
    groups.extend(dataframe.groupby("crop", sort=True))

    for crop, group in groups:
        actual = group[TARGET_COLUMN]

        for name, column in PREDICTION_COLUMNS.items():
            metrics = calculate_metrics(actual, group[column])

            records.append({
                "period": period,
                "crop": crop,
                "model": name,
                "records": len(group),
                "mae": metrics["mae"],
                "rmse": metrics["rmse"],
                "r2": metrics["r2"],
                "bias": float((group[column] - actual).mean()),
            })

    return pd.DataFrame(records)


def print_table(title, dataframe):
    print("\n" + "=" * 110)
    print(title)
    print("=" * 110)
    print(dataframe.round(3).to_string(index=False))


def print_win_counts(metrics):
    records = []

    for crop, group in metrics.groupby("crop", sort=True):
        mae = group.pivot(
            index="period", columns="model", values="mae"
        )
        rmse = group.pivot(
            index="period", columns="model", values="rmse"
        )

        records.append({
            "crop": crop,
            "folds": len(mae),
            "v4_beats_baseline_mae": int(
                (mae["monthly_v4"] < mae["baseline"]).sum()
            ),
            "v4_beats_v2_mae": int(
                (mae["monthly_v4"] < mae["seasonal_v2"]).sum()
            ),
            "v4_beats_baseline_rmse": int(
                (rmse["monthly_v4"] < rmse["baseline"]).sum()
            ),
            "v4_beats_v2_rmse": int(
                (rmse["monthly_v4"] < rmse["seasonal_v2"]).sum()
            ),
        })

    print_table(
        "NUMBER OF VALIDATION PERIODS WHERE V4 IS BETTER",
        pd.DataFrame(records),
    )


def main():
    params, expected_v2_mae = load_best_params()

    dataframe = load_dataset()
    dataframe = add_monthly_features(dataframe)
    folds, test_years = prepare_folds(dataframe)

    print("=" * 110)
    print("BASELINE VS SEASONAL V2 VS MONTHLY V4")
    print("=" * 110)
    print(f"Shared parameters: {params}")
    print(f"Reserved test years: {test_years}")
    print(f"V2 input columns: {len(FEATURE_COLUMNS)}")
    print(f"V4 input columns: {len(V4_FEATURES)}")
    print("Months outside a crop's season are represented by NaN.")
    print("Missing in-season weather values are not allowed.")

    all_predictions = []
    all_metrics = []

    for index, (
        train,
        validation,
        train_years,
        validation_years,
    ) in enumerate(folds, start=1):
        period = f"{validation_years[0]}-{validation_years[-1]}"

        print(
            f"\nFold {index}: "
            f"train {train_years[0]}-{train_years[-1]}, "
            f"validation {period}",
            flush=True,
        )

        results = validation.copy()

        for prediction_column, features in [
            ("v2_prediction", FEATURE_COLUMNS),
            ("v4_prediction", V4_FEATURES),
        ]:
            model = create_model(**params)
            model.fit(
                train[features],
                train[TARGET_COLUMN],
            )

            results[prediction_column] = model.predict(
                validation[features]
            )

        if not np.isfinite(
            results[list(PREDICTION_COLUMNS.values())].to_numpy(
                dtype=float
            )
        ).all():
            raise ValueError("Invalid predictions.")

        metrics = summarize(results, period)
        all_predictions.append(results)
        all_metrics.append(metrics)

        print_table(f"VALIDATION {period}", metrics)

    combined = pd.concat(all_predictions, ignore_index=True)
    fold_metrics = pd.concat(all_metrics, ignore_index=True)

    if combined.duplicated(
        ["yield_year", "county_name", "crop"]
    ).any():
        raise ValueError("Repeated validation rows.")

    reproduced_mae = fold_metrics.loc[
        (fold_metrics["crop"] == "ALL")
        & (fold_metrics["model"] == "seasonal_v2"),
        "mae",
    ].mean()

    if not np.isclose(
        reproduced_mae,
        expected_v2_mae,
        rtol=0,
        atol=0.000001,
    ):
        raise ValueError(
            "V2 result could not be reproduced. "
            f"Expected {expected_v2_mae:.6f}, "
            f"received {reproduced_mae:.6f}."
        )

    print(f"\nV2 MAE reproduced: {reproduced_mae:.6f} t/ha")

    print_table(
        "POOLED VALIDATION RESULTS",
        summarize(combined, "POOLED"),
    )
    print_win_counts(fold_metrics)

    print("\nPositive bias means overestimation.")
    print("Both models use actual historical weather.")
    print("V4 replaces seasonal weather inputs with monthly inputs.")
    print("V4 has not been separately tuned.")
    print("This is a development comparison, not an independent test.")
    print("Reserved test years were not evaluated.")
    print("No files, database tables or production models were modified.")


if __name__ == "__main__":
    main()