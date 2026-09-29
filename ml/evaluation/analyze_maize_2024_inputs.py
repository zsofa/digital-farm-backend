from pathlib import Path

import numpy as np
import pandas as pd

from ml.data_loader import get_connection


ML_DIR = Path(__file__).resolve().parent.parent
PREDICTIONS_FILE = (
    ML_DIR / "results" / "actual_weather_xgboost_predictions.csv"
)

TRAIN_START_YEAR = 2005
TRAIN_END_YEAR = 2022
TARGET_YEAR = 2024
TOP_COUNTIES = 5
SIMILAR_YEAR_COUNT = 3

FEATURES = [
    "temperature_c",
    "precipitation_mm",
    "recent_yield_mean_t_ha",
]

WEATHER_FEATURES = [
    "temperature_c",
    "precipitation_mm",
]


def load_inputs():
    query = """
        SELECT
            t.yield_year,
            t.county_name,
            t.crop,
            t.average_yield_t_ha,
            t.recent_yield_mean_t_ha,
            w.season_mean_temperature_c AS temperature_c,
            w.season_precipitation_mm AS precipitation_mm
        FROM ml.training_dataset t
        JOIN processed.weather_seasonal w
          ON w.season_start_year = t.season_start_year
         AND w.county_name = t.county_name
         AND w.crop = t.crop
        WHERE t.crop = 'maize'
        ORDER BY t.county_name, t.yield_year;
    """

    connection = get_connection()

    try:
        with connection.cursor() as cursor:
            cursor.execute(query)
            rows = cursor.fetchall()
            columns = [column.name for column in cursor.description]
    finally:
        connection.close()

    dataframe = pd.DataFrame(rows, columns=columns)

    for column in FEATURES + ["average_yield_t_ha"]:
        dataframe[column] = pd.to_numeric(
            dataframe[column]
        ).astype(float)

    if not np.isfinite(
        dataframe[FEATURES + ["average_yield_t_ha"]].to_numpy()
    ).all():
        raise ValueError("Missing or non-finite input values.")

    return dataframe


def load_predictions(inputs):
    predictions = pd.read_csv(PREDICTIONS_FILE)
    predictions = predictions[
        (predictions["crop"] == "maize")
        & (predictions["yield_year"] == TARGET_YEAR)
    ].copy()

    if len(predictions) != 19:
        raise ValueError(
            f"Expected 19 maize predictions, found {len(predictions)}."
        )

    dataframe = predictions.merge(
        inputs,
        on=["yield_year", "county_name", "crop"],
        how="left",
        validate="one_to_one",
        indicator=True,
    )

    if not dataframe["_merge"].eq("both").all():
        raise ValueError("Some predictions have no matching input row.")

    numeric_columns = [
        "actual_yield_t_ha",
        "predicted_yield_t_ha",
    ]

    for column in numeric_columns:
        dataframe[column] = pd.to_numeric(
            dataframe[column]
        ).astype(float)

    if not np.isfinite(dataframe[numeric_columns].to_numpy()).all():
        raise ValueError("Invalid prediction or actual yield values.")

    if not np.allclose(
        dataframe["actual_yield_t_ha"],
        dataframe["average_yield_t_ha"],
        rtol=0,
        atol=0.000001,
    ):
        raise ValueError(
            "CSV targets differ from current database targets. "
            "Check whether the data changed after evaluation."
        )

    dataframe["ml_error_t_ha"] = (
        dataframe["predicted_yield_t_ha"]
        - dataframe["actual_yield_t_ha"]
    )
    dataframe["absolute_error_t_ha"] = (
        dataframe["ml_error_t_ha"].abs()
    )
    dataframe["ml_minus_baseline_t_ha"] = (
        dataframe["predicted_yield_t_ha"]
        - dataframe["recent_yield_mean_t_ha"]
    )

    return dataframe.sort_values(
        "absolute_error_t_ha",
        ascending=False,
    )


def describe_range(feature, value, history, scope):
    values = history[feature]
    minimum = values.min()
    maximum = values.max()

    if value < minimum:
        position = "BELOW"
    elif value > maximum:
        position = "ABOVE"
    else:
        position = "INSIDE"

    return {
        "feature": feature,
        "scope": scope,
        "value_2024": value,
        "train_min": minimum,
        "train_mean": values.mean(),
        "train_max": maximum,
        "position": position,
    }


def find_similar_years(history, target):
    # Scale each weather feature by its within-county training SD.
    # Only training data are used to calculate the scales.
    scales = history[WEATHER_FEATURES].std(ddof=0)

    if (scales <= 0).any() or not np.isfinite(scales).all():
        raise ValueError(
            "Cannot calculate weather distance: invalid training SD."
        )

    target_weather = pd.Series({
        feature: float(target[feature])
        for feature in WEATHER_FEATURES
    })

    differences = (
        history[WEATHER_FEATURES] - target_weather
    ) / scales

    result = history.copy()
    result["weather_distance"] = np.sqrt(
        differences.pow(2).sum(axis=1)
    )

    return result.nsmallest(
        SIMILAR_YEAR_COUNT,
        "weather_distance",
    )


def print_county_analysis(target, train):
    county_name = target["county_name"]
    history = train[train["county_name"] == county_name].copy()

    expected_years = set(
        range(TRAIN_START_YEAR, TRAIN_END_YEAR + 1)
    )

    if (
        set(history["yield_year"]) != expected_years
        or len(history) != len(expected_years)
    ):
        raise ValueError(
            f"Incomplete or duplicated training history: {county_name}."
        )

    print("\n" + "=" * 100)
    print(county_name)
    print("=" * 100)
    print(f"Actual yield: {target['actual_yield_t_ha']:.3f} t/ha")
    print(f"ML prediction: {target['predicted_yield_t_ha']:.3f} t/ha")
    print(
        f"Five-year baseline: "
        f"{target['recent_yield_mean_t_ha']:.3f} t/ha"
    )
    print(f"ML signed error: {target['ml_error_t_ha']:+.3f} t/ha")
    print(
        f"ML minus baseline: "
        f"{target['ml_minus_baseline_t_ha']:+.3f} t/ha"
    )

    ranges = []

    for feature in FEATURES:
        for scope, reference in [
            ("same_county", history),
            ("all_counties_maize", train),
        ]:
            ranges.append(
                describe_range(
                    feature,
                    float(target[feature]),
                    reference,
                    scope,
                )
            )

    print("\nINPUT RANGES")
    print(pd.DataFrame(ranges).round(3).to_string(index=False))

    similar = find_similar_years(history, target)

    columns = [
        "yield_year",
        "temperature_c",
        "precipitation_mm",
        "recent_yield_mean_t_ha",
        "average_yield_t_ha",
        "weather_distance",
    ]

    print("\nMOST SIMILAR TRAINING YEARS BY WEATHER")
    print(similar[columns].round(3).to_string(index=False))


def main():
    inputs = load_inputs()

    train = inputs[
        inputs["yield_year"].between(
            TRAIN_START_YEAR,
            TRAIN_END_YEAR,
        )
    ].copy()

    predictions = load_predictions(inputs)

    print("=" * 100)
    print(f"MAIZE {TARGET_YEAR}: INPUT DIAGNOSTICS")
    print("=" * 100)
    print(f"Training reference: {TRAIN_START_YEAR}-{TRAIN_END_YEAR}")
    print(f"Training maize rows: {len(train)}")
    print(f"Target maize rows: {len(predictions)}")

    columns = [
        "county_name",
        "temperature_c",
        "precipitation_mm",
        "actual_yield_t_ha",
        "predicted_yield_t_ha",
        "recent_yield_mean_t_ha",
        "ml_error_t_ha",
        "ml_minus_baseline_t_ha",
    ]

    print("\nALL COUNTIES, SORTED BY ABSOLUTE ML ERROR")
    print(predictions[columns].round(3).to_string(index=False))

    for _, target in predictions.head(TOP_COUNTIES).iterrows():
        print_county_analysis(target, train)

    print("\nInterpretation:")
    print("- BELOW/ABOVE means outside the selected training range.")
    print("- INSIDE does not establish that the input combination is familiar.")
    print("- Weather distance uses temperature and precipitation only.")
    print("- Similar years are descriptive comparisons, not causal evidence.")
    print("- ML minus baseline is a prediction difference, not a feature effect.")


if __name__ == "__main__":
    main()