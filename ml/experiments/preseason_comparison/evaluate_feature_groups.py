import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ml.data_loader import FEATURE_COLUMNS, TARGET_COLUMN
from ml.experiments.preseason_comparison.compare_preseason_models import (
    KEYS,
    build_folds,
    calculate_metrics,
    prepare_data,
    print_table,
)
from ml.train.train_xgboost import create_model


BASE_DIR = Path(__file__).resolve().parent

BASE_FEATURES = [
    "crop",
    "recent_yield_mean_t_ha",
]

SOIL_FEATURES = [
    "soil_ph",
    "soil_soc_g_kg",
    "soil_clay_pct",
]

WEATHER_FEATURES = [
    "expected_season_temperature_c",
    "expected_season_precipitation_mm",
]

FEATURE_GROUPS = {
    "base": BASE_FEATURES,
    "base_soil": BASE_FEATURES + SOIL_FEATURES,
    "base_weather": BASE_FEATURES + WEATHER_FEATURES,
    "full": BASE_FEATURES + SOIL_FEATURES + WEATHER_FEATURES,
}

# Preserve the original column order, including for full-model reproduction.
FEATURE_GROUPS = {
    name: [
        column for column in FEATURE_COLUMNS
        if column in selected_columns
    ]
    for name, selected_columns in FEATURE_GROUPS.items()
}

COMPARISONS = [
    ("soil_added_to_base", "base", "base_soil"),
    ("weather_added_to_base", "base", "base_weather"),
    ("soil_added_to_weather", "base_weather", "full"),
    ("weather_added_to_soil", "base_soil", "full"),
]


def create_ridge(features, alpha):
    numeric_features = [
        column for column in features if column != "crop"
    ]

    preprocessing = ColumnTransformer(
        transformers=[
            (
                "crop",
                OneHotEncoder(handle_unknown="ignore"),
                ["crop"],
            ),
            (
                "numeric",
                StandardScaler(),
                numeric_features,
            ),
        ],
        sparse_threshold=0,
    )

    return Pipeline([
        ("preprocessing", preprocessing),
        ("model", Ridge(alpha=alpha, solver="svd")),
    ])


def load_reference(run_name):
    run_dir = BASE_DIR / "results" / run_name
    settings_file = run_dir / "settings.json"
    predictions_file = run_dir / "predictions.csv"

    if not settings_file.is_file() or not predictions_file.is_file():
        raise FileNotFoundError(
            f"Missing reference settings or predictions in: {run_dir}"
        )

    settings = json.loads(settings_file.read_text(encoding="utf-8"))
    predictions = pd.read_csv(predictions_file)

    required_settings = [
        "best_expected_xgboost_parameters",
        "best_ridge_parameters",
        "features",
        "folds",
    ]

    for key in required_settings:
        if key not in settings:
            raise ValueError(f"Missing reference setting: {key}")

    if settings["features"] != list(FEATURE_COLUMNS):
        raise ValueError("Reference feature list has changed.")

    return run_dir, settings, predictions


def iter_groups(dataframe):
    yield "ALL", dataframe

    for crop, group in dataframe.groupby("crop", sort=True):
        yield crop, group


def summarize(predictions, pooled=False):
    rows = []

    periods = (
        [("POOLED", predictions)]
        if pooled
        else predictions.groupby("period", sort=True)
    )

    for period, period_data in periods:
        for crop, crop_data in iter_groups(period_data):
            for (model, features), group in crop_data.groupby(
                ["model", "features"], sort=True
            ):
                rows.append({
                    "period": period,
                    "crop": crop,
                    "model": model,
                    "features": features,
                    "records": len(group),
                    **calculate_metrics(
                        group[TARGET_COLUMN],
                        group["prediction"],
                    ),
                })

    return pd.DataFrame(rows)


def verify_full_models(predictions, reference_predictions):
    reference_names = {
        "xgboost": "xgb_expected_tuned",
        "ridge": "ridge_expected_tuned",
    }

    for model, reference_name in reference_names.items():
        current = predictions[
            predictions["model"].eq(model)
            & predictions["features"].eq("full")
        ]

        reference = reference_predictions[
            reference_predictions["model"].eq(reference_name)
        ]

        compared = current[
            KEYS + [TARGET_COLUMN, "prediction"]
        ].merge(
            reference[KEYS + [TARGET_COLUMN, "prediction"]],
            on=KEYS,
            how="outer",
            suffixes=("_current", "_reference"),
            indicator=True,
            validate="one_to_one",
        )

        if len(compared) != 456 or not compared["_merge"].eq("both").all():
            raise ValueError(f"{model}: reference keys differ.")

        for column in [TARGET_COLUMN, "prediction"]:
            current_values = compared[
                f"{column}_current"
            ].to_numpy(dtype=float)

            reference_values = compared[
                f"{column}_reference"
            ].to_numpy(dtype=float)

            if not np.allclose(
                current_values,
                reference_values,
                atol=1e-6,
                rtol=0,
            ):
                raise ValueError(
                    f"{model}: full-model reproduction failed "
                    f"for {column}."
                )

        metrics = calculate_metrics(
            current[TARGET_COLUMN],
            current["prediction"],
        )
        print(
            f"{model} full-model reproduction passed. "
            f"MAE: {metrics['mae']:.6f} t/ha"
        )


def compare_feature_groups(pooled_metrics, fold_metrics):
    rows = []

    for model in ["xgboost", "ridge"]:
        for crop in ["ALL", "barley", "maize", "wheat"]:
            pooled = pooled_metrics[
                pooled_metrics["model"].eq(model)
                & pooled_metrics["crop"].eq(crop)
            ].set_index("features")

            fold_data = fold_metrics[
                fold_metrics["model"].eq(model)
                & fold_metrics["crop"].eq(crop)
            ]

            fold_mae = fold_data.pivot(
                index="period",
                columns="features",
                values="mae",
            )

            for label, before, after in COMPARISONS:
                rows.append({
                    "model": model,
                    "crop": crop,
                    "comparison": label,
                    "mae_before": pooled.loc[before, "mae"],
                    "mae_after": pooled.loc[after, "mae"],
                    "mae_gain": (
                        pooled.loc[before, "mae"]
                        - pooled.loc[after, "mae"]
                    ),
                    "rmse_gain": (
                        pooled.loc[before, "rmse"]
                        - pooled.loc[after, "rmse"]
                    ),
                    "better_mae_folds": int((
                        fold_mae[after] < fold_mae[before] - 1e-9
                    ).sum()),
                    "folds": len(fold_mae),
                })

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(
        description="Fixed-parameter preseason feature-group comparison."
    )
    parser.add_argument(
        "--reference-run",
        required=True,
        help="Directory name of the previous preseason comparison run.",
    )
    args = parser.parse_args()

    run_dir, reference_settings, reference_predictions = load_reference(
        args.reference_run
    )

    xgb_parameters = reference_settings[
        "best_expected_xgboost_parameters"
    ]
    ridge_alpha = reference_settings["best_ridge_parameters"]["alpha"]

    print("PRESEASON FEATURE-GROUP COMPARISON")
    print(f"Reference: {run_dir}")
    print(f"XGBoost parameters: {xgb_parameters}")
    print(f"Ridge alpha: {ridge_alpha}")
    print("Parameters are fixed; no tuning is performed.")
    print("Both training and prediction use expected weather.")
    print("Reserved test years 2023-2025 are not evaluated.")

    for name, features in FEATURE_GROUPS.items():
        print(f"{name}: {features}")

    actual, expected, audit = prepare_data()
    folds = build_folds(actual, expected)

    current_fold_definitions = [
        [
            int(fold["expected_train"]["yield_year"].max()),
            int(fold["expected_validation"]["yield_year"].min()),
            int(fold["expected_validation"]["yield_year"].max()),
        ]
        for fold in folds
    ]

    if current_fold_definitions != reference_settings["folds"]:
        raise ValueError("Reference and current validation folds differ.")

    results = []

    for fold in folds:
        train = fold["expected_train"]
        validation = fold["expected_validation"]
        period = fold["period"]

        print(f"\nEvaluating {period}...", flush=True)

        base_result = validation[
            KEYS + [TARGET_COLUMN, "recent_yield_mean_t_ha"]
        ].copy()
        base_result["period"] = period

        baseline = base_result.copy()
        baseline["model"] = "baseline"
        baseline["features"] = "five_year_mean"
        baseline["prediction"] = baseline["recent_yield_mean_t_ha"]
        results.append(baseline)

        for group_name, features in FEATURE_GROUPS.items():
            for model_name in ["xgboost", "ridge"]:
                if model_name == "xgboost":
                    model = create_model(**xgb_parameters)
                else:
                    model = create_ridge(features, ridge_alpha)

                model.fit(train[features], train[TARGET_COLUMN])
                predicted = model.predict(validation[features])

                if not np.isfinite(predicted).all():
                    raise ValueError(
                        f"Non-finite predictions: "
                        f"{period}, {model_name}, {group_name}"
                    )

                result = base_result.copy()
                result["model"] = model_name
                result["features"] = group_name
                result["prediction"] = predicted
                results.append(result)

    predictions = pd.concat(results, ignore_index=True)

    if predictions.duplicated(
        KEYS + ["model", "features"]
    ).any():
        raise ValueError("Duplicate validation predictions.")

    counts = predictions.groupby(["model", "features"]).size()

    if len(counts) != 9 or not counts.eq(456).all():
        raise ValueError("Expected 9 variants with 456 predictions each.")

    verify_full_models(predictions, reference_predictions)

    fold_metrics = summarize(predictions)
    pooled_metrics = summarize(predictions, pooled=True)
    contributions = compare_feature_groups(
        pooled_metrics, fold_metrics
    )

    print_table("POOLED VALIDATION RESULTS", pooled_metrics)

    print_table(
        "FEATURE-GROUP CONTRIBUTIONS: POSITIVE GAIN MEANS IMPROVEMENT",
        contributions,
    )

    overall_fold_metrics = fold_metrics[
        fold_metrics["crop"].eq("ALL")
    ]

    print_table("OVERALL RESULTS BY VALIDATION PERIOD",
                overall_fold_metrics)

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output_dir = BASE_DIR / "results" / f"feature_groups_{run_id}"
    output_dir.mkdir(parents=True, exist_ok=False)

    audit.to_csv(output_dir / "input_audit.csv", index=False)
    predictions.to_csv(output_dir / "predictions.csv", index=False)
    fold_metrics.to_csv(output_dir / "fold_metrics.csv", index=False)
    pooled_metrics.to_csv(output_dir / "pooled_metrics.csv", index=False)
    contributions.to_csv(
        output_dir / "feature_contributions.csv",
        index=False,
    )

    settings = {
        "reference_run": str(run_dir),
        "feature_groups": FEATURE_GROUPS,
        "xgboost_parameters": xgb_parameters,
        "ridge_alpha": ridge_alpha,
        "folds": current_fold_definitions,
        "reserved_test_years": [2023, 2024, 2025],
        "retuned_per_feature_group": False,
        "independent_test": False,
        "publication_delays_modelled": False,
    }

    (output_dir / "settings.json").write_text(
        json.dumps(settings, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"\nResults saved to:\n{output_dir}")
    print("\nInterpretation:")
    print("- Positive mae_gain/rmse_gain means added inputs reduced error.")
    print("- Each variant is retrained using only its selected inputs.")
    print("- Parameters were selected previously with the full feature set.")
    print("- Reduced feature sets were not separately tuned.")
    print("- Results describe predictive value under these fixed settings.")
    print("- Static county soil may also act as a location proxy.")
    print("- Soil contribution does not establish a causal soil effect.")
    print("- County-level results do not validate parcel-level accuracy.")
    print("- This is a development comparison, not an independent test.")
    print("- No production models or database tables were modified.")


if __name__ == "__main__":
    main()