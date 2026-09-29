from pathlib import Path
from statistics import mean, stdev

import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import ParameterGrid

from ml.data_loader import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    expanding_window_splits,
)
from ml.evaluation.evaluate_actual_weather_xg import (
    load_actual_weather_dataset,
)
from ml.train.train_xgboost import create_model


EXPERIMENT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = EXPERIMENT_DIR / "results" / "tuning_001"
RESULTS_FILE = RESULTS_DIR / "xgboost_tuning_results.csv"

PARAMETER_GRID = {
    "n_estimators": [200, 400],
    "learning_rate": [0.03, 0.05],
    "max_depth": [2, 3, 4],
    "subsample": [0.8, 1.0],
    "colsample_bytree": [0.8, 1.0],
    "min_child_weight": [1, 5],
    "reg_lambda": [1.0, 5.0],
}

EXPECTED_FOLDS = [
    (list(range(2005, 2015)), [2015, 2016]),
    (list(range(2005, 2017)), [2017, 2018]),
    (list(range(2005, 2019)), [2019, 2020]),
    (list(range(2005, 2021)), [2021, 2022]),
]

EXPECTED_TEST_YEARS = [2023, 2024, 2025]


def load_dataset():
    # Despite the "expected_" column names, this loader supplies
    # ACTUAL seasonal temperature and precipitation.
    dataframe = load_actual_weather_dataset()

    dataframe = dataframe[
        dataframe["yield_year"].between(2005, 2025)
    ].copy()

    keys = ["yield_year", "county_name", "crop"]

    if dataframe.duplicated(keys).any():
        raise ValueError("Duplicate county-crop-year rows.")

    if dataframe[keys].isna().any().any():
        raise ValueError("Missing year, county or crop.")

    if set(dataframe["crop"]) != {"wheat", "barley", "maize"}:
        raise ValueError("Unexpected crop set.")

    if dataframe["county_name"].nunique() != 19:
        raise ValueError("Expected 19 counties.")

    if set(dataframe["yield_year"]) != set(range(2005, 2026)):
        raise ValueError("Expected all years from 2005 through 2025.")

    records_per_year = dataframe.groupby("yield_year").size()

    if not records_per_year.eq(57).all():
        raise ValueError("Expected 57 county-crop rows in every year.")

    numeric_columns = [
        column for column in FEATURE_COLUMNS if column != "crop"
    ] + [TARGET_COLUMN]

    for column in numeric_columns:
        dataframe[column] = pd.to_numeric(
            dataframe[column]
        ).astype(float)

    if not np.isfinite(
        dataframe[numeric_columns].to_numpy()
    ).all():
        raise ValueError("Missing or non-finite numeric values.")

    return dataframe


def prepare_folds(dataframe):
    folds, test_years = expanding_window_splits(dataframe)

    actual_folds = [
        (train_years, validation_years)
        for _, _, train_years, validation_years in folds
    ]

    if test_years != EXPECTED_TEST_YEARS:
        raise ValueError(f"Unexpected test years: {test_years}")

    if actual_folds != EXPECTED_FOLDS:
        raise ValueError("Unexpected training/validation year splits.")

    for _, _, train_years, validation_years in folds:
        if max(train_years) >= min(validation_years):
            raise ValueError("Invalid chronological split.")

        used_years = set(train_years + validation_years)

        if used_years & set(test_years):
            raise ValueError("Test years entered tuning folds.")

    return folds, test_years


def calculate_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    r2 = r2_score(y_true, y_pred)

    return mae, rmse, r2


def evaluate_configuration(params, folds):
    fold_maes = []
    fold_rmses = []
    fold_r2s = []

    for train, validation, _, _ in folds:
        model = create_model(**params)

        model.fit(
            train[FEATURE_COLUMNS],
            train[TARGET_COLUMN],
        )

        predictions = model.predict(
            validation[FEATURE_COLUMNS]
        )

        mae, rmse, r2 = calculate_metrics(
            validation[TARGET_COLUMN],
            predictions,
        )

        fold_maes.append(mae)
        fold_rmses.append(rmse)
        fold_r2s.append(r2)

    return {
        **params,
        "mean_mae": mean(fold_maes),
        "std_mae": stdev(fold_maes),
        "mean_rmse": mean(fold_rmses),
        "mean_r2": mean(fold_r2s),
    }


def main():
    if RESULTS_FILE.exists():
        raise FileExistsError(
            f"Results already exist: {RESULTS_FILE}. "
            "Change the run directory to tuning_002 before rerunning."
        )

    print("Loading ACTUAL-weather dataset...")
    dataframe = load_dataset()

    folds, test_years = prepare_folds(dataframe)
    parameter_combinations = list(ParameterGrid(PARAMETER_GRID))

    print(f"Records loaded: {len(dataframe)}")
    print(f"Configurations: {len(parameter_combinations)}")
    print(f"Validation folds: {len(folds)}")
    print(f"Reserved test years: {test_years}")

    for index, (
        train,
        validation,
        train_years,
        validation_years,
    ) in enumerate(folds, start=1):
        print(
            f"Fold {index}: "
            f"train {train_years[0]}-{train_years[-1]} "
            f"({len(train)} rows), "
            f"validation {validation_years[0]}-{validation_years[-1]} "
            f"({len(validation)} rows)"
        )

    print("\nStarting actual-weather XGBoost tuning...")
    results = []

    for index, params in enumerate(parameter_combinations, start=1):
        print(
            f"\n[{index}/{len(parameter_combinations)}] {params}",
            flush=True,
        )

        result = evaluate_configuration(params, folds)
        results.append(result)

        print(
            f"Mean MAE: {result['mean_mae']:.3f} t/ha | "
            f"Mean RMSE: {result['mean_rmse']:.3f} t/ha | "
            f"Mean fold R²: {result['mean_r2']:.3f}",
            flush=True,
        )

    results_dataframe = (
        pd.DataFrame(results)
        .sort_values(
            by=["mean_mae", "mean_rmse", "std_mae"],
            ascending=True,
        )
        .reset_index(drop=True)
    )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Exclusive creation prevents accidental overwriting.
    with RESULTS_FILE.open(
        "x", encoding="utf-8", newline=""
    ) as output_file:
        results_dataframe.to_csv(output_file, index=False)

    print("\n" + "=" * 100)
    print("TOP 10 ACTUAL-WEATHER XGBOOST CONFIGURATIONS")
    print("=" * 100)
    print(
        results_dataframe.head(10)
        .round(4)
        .to_string(index=False)
    )

    best = results_dataframe.iloc[0]

    print("\n" + "=" * 70)
    print("BEST CONFIGURATION")
    print("=" * 70)

    integer_parameters = {
        "n_estimators",
        "max_depth",
        "min_child_weight",
    }

    for parameter in PARAMETER_GRID:
        value = best[parameter]

        if parameter in integer_parameters:
            value = int(value)
        else:
            value = float(value)

        print(f"{parameter}: {value}")

    print(f"\nMean validation MAE: {best['mean_mae']:.3f} t/ha")
    print(f"MAE standard deviation: {best['std_mae']:.3f} t/ha")
    print(f"Mean validation RMSE: {best['mean_rmse']:.3f} t/ha")
    print(f"Mean fold R²: {best['mean_r2']:.3f}")

    print(f"\nResults saved to:\n{RESULTS_FILE}")
    print(f"\nTest years {test_years} were NOT evaluated.")
    print("The production model was NOT modified.")
    print("Best means lowest validation error within this parameter grid.")
    print("Mean fold R² is not the R² of pooled validation predictions.")


if __name__ == "__main__":
    main()