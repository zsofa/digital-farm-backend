from pathlib import Path

import pandas as pd

from ml.evaluation.compare_final_to_baseline import (
    load_recent_yield,
    print_comparison,
)


ML_DIR = Path(__file__).resolve().parent.parent
PREDICTIONS_FILE = (
    ML_DIR / "results" / "actual_weather_xgboost_predictions.csv"
)
TARGET_YEAR = 2024


def load_results():
    predictions = pd.read_csv(PREDICTIONS_FILE)

    dataframe = predictions.merge(
        load_recent_yield(),
        on=["yield_year", "county_name", "crop"],
        how="left",
        validate="one_to_one",
    )

    dataframe = dataframe[
        dataframe["yield_year"] == TARGET_YEAR
    ].copy()

    if dataframe.empty:
        raise ValueError(f"No predictions found for {TARGET_YEAR}.")

    numeric_columns = [
        "actual_yield_t_ha",
        "predicted_yield_t_ha",
        "recent_yield_mean_t_ha",
    ]

    for column in numeric_columns:
        dataframe[column] = pd.to_numeric(dataframe[column])

    if dataframe[numeric_columns].isna().any().any():
        raise ValueError("Missing yield or baseline values.")

    dataframe["ml_error_t_ha"] = (
        dataframe["predicted_yield_t_ha"]
        - dataframe["actual_yield_t_ha"]
    )
    dataframe["baseline_error_t_ha"] = (
        dataframe["recent_yield_mean_t_ha"]
        - dataframe["actual_yield_t_ha"]
    )
    dataframe["ml_absolute_error_t_ha"] = (
        dataframe["ml_error_t_ha"].abs()
    )
    dataframe["baseline_absolute_error_t_ha"] = (
        dataframe["baseline_error_t_ha"].abs()
    )

    return dataframe


def print_bias(dataframe):
    baseline_bias = dataframe["baseline_error_t_ha"].mean()
    ml_bias = dataframe["ml_error_t_ha"].mean()

    print(f"Records: {len(dataframe)}")
    print(f"Baseline bias: {baseline_bias:+.3f} t/ha")
    print(f"XGBoost bias:  {ml_bias:+.3f} t/ha")


def main():
    dataframe = load_results()

    print("=" * 70)
    print(f"{TARGET_YEAR}: ACTUAL-WEATHER XGBOOST VS BASELINE")
    print("=" * 70)
    print("Positive bias = overestimation; negative bias = underestimation.")

    print_comparison("OVERALL", dataframe)
    print_bias(dataframe)

    for crop, group in dataframe.groupby("crop", sort=True):
        print_comparison(crop.upper(), group)
        print_bias(group)

    print("\n" + "=" * 70)
    print("10 LARGEST XGBOOST ABSOLUTE ERRORS")
    print("=" * 70)

    columns = [
        "county_name",
        "crop",
        "actual_yield_t_ha",
        "predicted_yield_t_ha",
        "recent_yield_mean_t_ha",
        "ml_error_t_ha",
        "ml_absolute_error_t_ha",
        "baseline_absolute_error_t_ha",
    ]

    worst = dataframe.nlargest(10, "ml_absolute_error_t_ha")
    print(worst[columns].round(3).to_string(index=False))


if __name__ == "__main__":
    main()