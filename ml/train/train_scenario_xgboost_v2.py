import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
import xgboost

from ml.data_loader import TARGET_COLUMN
from ml.scenarios.features import SCENARIO_FEATURE_COLUMNS
from ml.train.train_scenario_xgboost import load_dataset
from ml.train.train_xgboost import create_model


ML_DIR = Path(__file__).resolve().parent.parent
MODEL_ROOT = ML_DIR / "models" / "scenario_xgboost_v2"

KEY_COLUMNS = ["yield_year", "county_name", "crop"]
EXPECTED_YEARS = list(range(2005, 2026))
EXPECTED_CROPS = {"barley", "maize", "wheat"}
EXPECTED_COUNTIES = 19

PARAMETERS = {
    "n_estimators": 200,
    "learning_rate": 0.05,
    "max_depth": 2,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "reg_lambda": 5.0,
}


def validate_dataset(dataframe):
    required = list(
        dict.fromkeys(
            KEY_COLUMNS + SCENARIO_FEATURE_COLUMNS + [TARGET_COLUMN]
        )
    )

    missing = set(required) - set(dataframe.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    if dataframe[required].isna().any().any():
        raise ValueError("Missing values found in the training dataset.")

    if dataframe.duplicated(KEY_COLUMNS).any():
        raise ValueError("Duplicate county/crop/year records found.")

    years = sorted(dataframe["yield_year"].unique().tolist())
    if years != EXPECTED_YEARS:
        raise ValueError(
            f"Expected years {EXPECTED_YEARS}; received {years}."
        )

    if set(dataframe["crop"]) != EXPECTED_CROPS:
        raise ValueError("Unexpected crop set.")

    counties = sorted(dataframe["county_name"].unique().tolist())
    if len(counties) != EXPECTED_COUNTIES:
        raise ValueError(
            f"Expected {EXPECTED_COUNTIES} counties; "
            f"received {len(counties)}."
        )

    expected_keys = pd.MultiIndex.from_product(
        [EXPECTED_YEARS, counties, sorted(EXPECTED_CROPS)],
        names=KEY_COLUMNS,
    )
    actual_keys = pd.MultiIndex.from_frame(dataframe[KEY_COLUMNS])

    if (
        len(actual_keys) != len(expected_keys)
        or len(expected_keys.difference(actual_keys)) > 0
        or len(actual_keys.difference(expected_keys)) > 0
    ):
        raise ValueError("The county/crop/year grid is incomplete.")

    numeric_columns = [
        column
        for column in SCENARIO_FEATURE_COLUMNS
        if column != "crop"
    ] + [TARGET_COLUMN]

    values = dataframe[numeric_columns].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Non-finite numeric values found.")

    if (dataframe["season_precipitation_mm"] < 0).any():
        raise ValueError("Negative precipitation found.")

    if (dataframe[TARGET_COLUMN] < 0).any():
        raise ValueError("Negative observed yield found.")

    return (
        dataframe.sort_values(KEY_COLUMNS)
        .reset_index(drop=True)
        .copy()
    )


def dataset_fingerprint(dataframe):
    columns = list(
        dict.fromkeys(
            KEY_COLUMNS + SCENARIO_FEATURE_COLUMNS + [TARGET_COLUMN]
        )
    )
    serialized = dataframe[columns].to_csv(
        index=False,
        float_format="%.12g",
        lineterminator="\n",
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def main():
    print("TRAIN VERSIONED V2 SCENARIO MODEL")
    print("Weather inputs: actual historical seasonal weather")
    print("Parameters are fixed; no tuning is performed.\n")

    dataframe = validate_dataset(load_dataset())

    print(f"Records: {len(dataframe)}")
    print(
        f"Training years: {dataframe['yield_year'].min()}-"
        f"{dataframe['yield_year'].max()}"
    )
    print(f"Counties: {dataframe['county_name'].nunique()}")
    print(f"Crops: {sorted(dataframe['crop'].unique())}")
    print(f"Parameters: {PARAMETERS}")

    X = dataframe[SCENARIO_FEATURE_COLUMNS]
    y = dataframe[TARGET_COLUMN]

    model = create_model(**PARAMETERS)

    print("\nTraining V2...")
    model.fit(X, y)

    # These predictions check serialization only.
    # They are NOT used to estimate predictive accuracy.
    predictions_before_save = model.predict(X)

    if not np.isfinite(predictions_before_save).all():
        raise ValueError("The trained model produced non-finite predictions.")

    created_at = datetime.now(timezone.utc)
    version = created_at.strftime("%Y%m%dT%H%M%S%fZ")

    output_directory = MODEL_ROOT / version
    output_directory.mkdir(parents=True, exist_ok=False)

    model_file = output_directory / "scenario_xgboost_v2.joblib"
    metadata_file = output_directory / "metadata.json"

    metadata = {
        "version": version,
        "created_at_utc": created_at.isoformat(),
        "model_type": "scenario_xgboost_actual_weather",
        "model_variant": "v2",
        "feature_columns": list(SCENARIO_FEATURE_COLUMNS),
        "target_column": TARGET_COLUMN,
        "training_year_min": int(dataframe["yield_year"].min()),
        "training_year_max": int(dataframe["yield_year"].max()),
        "training_records": len(dataframe),
        "county_count": int(dataframe["county_name"].nunique()),
        "crops": sorted(dataframe["crop"].unique().tolist()),
        "parameters": PARAMETERS,
        "random_state": 42,
        "objective": "reg:squarederror",
        "training_data_sha256": dataset_fingerprint(dataframe),
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "xgboost": xgboost.__version__,
            "joblib": joblib.__version__,
        },
        "intended_use": (
            "Conditional yield estimation for supplied weather scenarios."
        ),
        "limitations": [
            "This full-data fit is not an independent evaluation.",
            "Scenario outputs are not calibrated prediction intervals.",
            "County-level evaluation does not validate parcel accuracy.",
            "Historical input release dates have not been verified.",
        ],
    }

    # Preserve the artifact keys expected by the application.
    artifact = {
        "model": model,
        **metadata,
    }

    joblib.dump(artifact, model_file)

    reloaded = joblib.load(model_file)

    if reloaded["feature_columns"] != list(SCENARIO_FEATURE_COLUMNS):
        raise ValueError("Saved feature columns do not match the application.")

    predictions_after_load = reloaded["model"].predict(X)

    np.testing.assert_allclose(
        predictions_after_load,
        predictions_before_save,
        rtol=0,
        atol=1e-6,
    )

    maximum_difference = float(
        np.max(
            np.abs(
                predictions_after_load - predictions_before_save
            )
        )
    )

    metadata["serialization_check_max_difference_t_ha"] = maximum_difference
    metadata["artifact_sha256"] = hashlib.sha256(
        model_file.read_bytes()
    ).hexdigest()

    metadata_file.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\nTraining completed.")
    print(
        "Save/load verification passed. "
        f"Maximum difference: {maximum_difference:.9f} t/ha"
    )
    print(f"\nModel saved to:\n{model_file}")
    print(f"\nMetadata saved to:\n{metadata_file}")

    print("\nIMPORTANT:")
    print("- All 2005-2025 records were used for this deployment candidate.")
    print("- No new accuracy estimate was calculated.")
    print("- The application has NOT been switched to this model.")
    print("- The existing production model was NOT overwritten.")
    print("- No database tables were modified.")


if __name__ == "__main__":
    main()