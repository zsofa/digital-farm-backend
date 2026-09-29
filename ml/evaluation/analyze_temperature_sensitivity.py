from pathlib import Path

import numpy as np
import pandas as pd

from ml.data_loader import FEATURE_COLUMNS, TARGET_COLUMN
from ml.evaluation.evaluate_actual_weather_xg import (
    load_actual_weather_dataset,
)
from ml.train.train_xgboost import create_model


ML_DIR = Path(__file__).resolve().parent.parent
PREDICTIONS_FILE = (
    ML_DIR / "results" / "actual_weather_xgboost_predictions.csv"
)

TRAIN_START_YEAR = 2005
TRAIN_END_YEAR = 2022
TARGET_YEAR = 2024

TEMPERATURE_COLUMN = "expected_season_temperature_c"
PRECIPITATION_COLUMN = "expected_season_precipitation_mm"

# The loader uses these column names for ACTUAL weather values.
TEMPERATURE_OFFSETS = [-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0]

COUNTIES = [
    "Vas",
    "Somogy",
    "Zala",
    "Komárom-Esztergom",
    "Pest",
    # Comparison cases with small original prediction errors:
    "Bács-Kiskun",
    "Hajdú-Bihar",
]


def train_test_model(dataframe):
    train = dataframe[
        dataframe["yield_year"].between(
            TRAIN_START_YEAR, TRAIN_END_YEAR
        )
    ].copy()

    if len(train) != 1026:
        raise ValueError(
            f"Expected 1026 training rows, found {len(train)}."
        )

    model = create_model(
        n_estimators=400,
        learning_rate=0.05,
        max_depth=3,
        subsample=0.8,
        colsample_bytree=1.0,
        min_child_weight=1,
        reg_lambda=5.0,
    )

    model.fit(train[FEATURE_COLUMNS], train[TARGET_COLUMN])
    return model, train


def verify_predictions(model, target):
    saved = pd.read_csv(PREDICTIONS_FILE)
    saved = saved[
        (saved["yield_year"] == TARGET_YEAR)
        & (saved["crop"] == "maize")
    ].copy()

    if len(saved) != 19:
        raise ValueError("Expected 19 saved maize predictions for 2024.")

    current = target[
        ["yield_year", "county_name", "crop", TARGET_COLUMN]
    ].copy()
    current["reproduced_prediction"] = model.predict(
        target[FEATURE_COLUMNS]
    )

    comparison = current.merge(
        saved,
        on=["yield_year", "county_name", "crop"],
        how="outer",
        validate="one_to_one",
        indicator=True,
    )

    if not comparison["_merge"].eq("both").all():
        raise ValueError("Saved and current prediction keys differ.")

    if not np.allclose(
        comparison[TARGET_COLUMN].to_numpy(dtype=float),
        comparison["actual_yield_t_ha"].to_numpy(dtype=float),
        rtol=0,
        atol=0.000001,
    ):
        raise ValueError("Actual yields differ from the saved results.")

    reproduced = comparison["reproduced_prediction"].to_numpy(
        dtype=float
    )
    original = comparison["predicted_yield_t_ha"].to_numpy(
        dtype=float
    )

    if not np.allclose(reproduced, original, rtol=0, atol=0.0001):
        raise ValueError(
            "Original predictions could not be reproduced. "
            "Check data, model code and package versions."
        )

    maximum_difference = np.max(np.abs(reproduced - original))
    print(f"Prediction reproduction passed: {maximum_difference:.6f} t/ha")


def analyze_county(model, train, target, county_name):
    county_target = target[
        target["county_name"] == county_name
    ].copy()

    if len(county_target) != 1:
        raise ValueError(f"Expected one target row for {county_name}.")

    row = county_target.iloc[0]

    history = train[
        (train["crop"] == "maize")
        & (train["county_name"] == county_name)
    ]

    local_min = float(history[TEMPERATURE_COLUMN].min())
    local_max = float(history[TEMPERATURE_COLUMN].max())

    maize_history = train[train["crop"] == "maize"]
    global_min = float(maize_history[TEMPERATURE_COLUMN].min())
    global_max = float(maize_history[TEMPERATURE_COLUMN].max())

    actual_temperature = float(row[TEMPERATURE_COLUMN])
    original_prediction = float(
        model.predict(county_target[FEATURE_COLUMNS])[0]
    )

    experiments = pd.concat(
        [county_target[FEATURE_COLUMNS]] * len(TEMPERATURE_OFFSETS),
        ignore_index=True,
    )
    temperatures = actual_temperature + np.array(
        TEMPERATURE_OFFSETS
    )
    experiments[TEMPERATURE_COLUMN] = temperatures

    predictions = model.predict(experiments)

    results = pd.DataFrame({
        "temperature_change_c": TEMPERATURE_OFFSETS,
        "temperature_c": temperatures,
        "predicted_yield_t_ha": predictions,
        "change_from_original_t_ha": predictions - original_prediction,
        "inside_local_temp_range": (
            (temperatures >= local_min)
            & (temperatures <= local_max)
        ),
        "inside_all_maize_temp_range": (
            (temperatures >= global_min)
            & (temperatures <= global_max)
        ),
    })

    print("\n" + "=" * 100)
    print(county_name)
    print("=" * 100)
    print(f"Actual temperature: {actual_temperature:.3f} C")
    print(
        f"Fixed precipitation: "
        f"{float(row[PRECIPITATION_COLUMN]):.2f} mm"
    )
    print(
        f"Fixed five-year yield mean: "
        f"{float(row['recent_yield_mean_t_ha']):.3f} t/ha"
    )
    print(f"Observed 2024 yield: {float(row[TARGET_COLUMN]):.3f} t/ha")
    print(f"Original ML prediction: {original_prediction:.3f} t/ha")
    print(f"Local training temperature range: {local_min:.3f}-{local_max:.3f} C")
    print(results.round(3).to_string(index=False))


def main():
    dataframe = load_actual_weather_dataset()

    numeric_columns = [
        column for column in FEATURE_COLUMNS if column != "crop"
    ] + [TARGET_COLUMN]

    if not np.isfinite(
        dataframe[numeric_columns].to_numpy(dtype=float)
    ).all():
        raise ValueError("Missing or non-finite numeric values.")

    target = dataframe[
        (dataframe["yield_year"] == TARGET_YEAR)
        & (dataframe["crop"] == "maize")
    ].copy()

    if len(target) != 19:
        raise ValueError("Expected 19 maize rows for 2024.")

    print("Training diagnostic model on 2005-2022, all three crops...")
    model, train = train_test_model(dataframe)

    verify_predictions(model, target)

    for county_name in COUNTIES:
        analyze_county(model, train, target, county_name)

    print("\nOnly temperature was changed; all other inputs were fixed.")
    print("Modified temperatures are hypothetical inputs, not observations.")
    print("Range checks concern temperature alone, not the full input combination.")
    print("These results describe model behaviour, not causal crop responses.")


if __name__ == "__main__":
    main()