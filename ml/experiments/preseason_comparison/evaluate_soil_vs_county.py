import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

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

SOIL_COLUMNS = [
    "soil_ph",
    "soil_soc_g_kg",
    "soil_clay_pct",
]

FEATURE_SETS = {
    "no_soil": [
        column for column in FEATURE_COLUMNS
        if column not in SOIL_COLUMNS
    ],
    "soil": list(FEATURE_COLUMNS),
    "county": [
        column for column in FEATURE_COLUMNS
        if column not in SOIL_COLUMNS
    ] + ["county_name"],
}


def load_reference(run_name):
    run_dir = BASE_DIR / "results" / run_name
    settings_file = run_dir / "settings.json"
    predictions_file = run_dir / "predictions.csv"

    if not settings_file.is_file() or not predictions_file.is_file():
        raise FileNotFoundError(
            f"Missing reference files in: {run_dir}"
        )

    settings = json.loads(settings_file.read_text(encoding="utf-8"))
    predictions = pd.read_csv(predictions_file)

    if "xgboost_parameters" not in settings:
        raise ValueError(
            "Use the feature_groups run as the reference."
        )

    expected_groups = {
        "base_weather": FEATURE_SETS["no_soil"],
        "full": FEATURE_SETS["soil"],
    }

    for name, features in expected_groups.items():
        if settings.get("feature_groups", {}).get(name) != features:
            raise ValueError(
                f"Reference features differ for {name}."
            )

    return run_dir, settings, predictions


def create_variant(parameters, variant):
    model = create_model(**parameters)

    if variant == "county":
        preprocessing = ColumnTransformer(
            transformers=[
                (
                    "categorical_encoder",
                    OneHotEncoder(handle_unknown="ignore"),
                    ["crop", "county_name"],
                ),
            ],
            remainder="passthrough",
        )

        model.set_params(preprocessor=preprocessing)

    return model


def verify_reference(predictions, reference):
    mappings = {
        "no_soil": "base_weather",
        "soil": "full",
    }

    for variant, feature_group in mappings.items():
        current = predictions[predictions["variant"].eq(variant)]

        previous = reference[
            reference["model"].eq("xgboost")
            & reference["features"].eq(feature_group)
        ]

        merged = current[
            KEYS + [TARGET_COLUMN, "prediction"]
        ].merge(
            previous[KEYS + [TARGET_COLUMN, "prediction"]],
            on=KEYS,
            how="outer",
            suffixes=("_current", "_reference"),
            indicator=True,
            validate="one_to_one",
        )

        if len(merged) != 456 or not merged["_merge"].eq("both").all():
            raise ValueError(
                f"Reference keys differ for {variant}."
            )

        for column in [TARGET_COLUMN, "prediction"]:
            current_values = merged[
                f"{column}_current"
            ].to_numpy(dtype=float)

            previous_values = merged[
                f"{column}_reference"
            ].to_numpy(dtype=float)

            if not np.allclose(
                current_values,
                previous_values,
                atol=1e-6,
                rtol=0,
            ):
                raise ValueError(
                    f"Reference reproduction failed: "
                    f"{variant}, {column}"
                )

        print(f"{variant}: reference predictions reproduced.")


def summarize(predictions, pooled=False):
    rows = []

    periods = (
        [("POOLED", predictions)]
        if pooled
        else predictions.groupby("period", sort=True)
    )

    for period, period_data in periods:
        groups = [("ALL", period_data)]
        groups.extend(period_data.groupby("crop", sort=True))

        for crop, group in groups:
            for variant, variant_data in group.groupby(
                "variant", sort=True
            ):
                rows.append({
                    "period": period,
                    "crop": crop,
                    "variant": variant,
                    "records": len(variant_data),
                    **calculate_metrics(
                        variant_data[TARGET_COLUMN],
                        variant_data["prediction"],
                    ),
                })

    return pd.DataFrame(rows)


def compare_variants(pooled_metrics, fold_metrics):
    comparisons = [
        ("soil_vs_no_soil", "no_soil", "soil"),
        ("county_vs_no_soil", "no_soil", "county"),
        ("soil_vs_county", "county", "soil"),
    ]

    rows = []

    for crop in ["ALL", "barley", "maize", "wheat"]:
        pooled = pooled_metrics[
            pooled_metrics["crop"].eq(crop)
        ].set_index("variant")

        fold_mae = fold_metrics[
            fold_metrics["crop"].eq(crop)
        ].pivot(
            index="period",
            columns="variant",
            values="mae",
        )

        for comparison, before, after in comparisons:
            rows.append({
                "crop": crop,
                "comparison": comparison,
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
        description="Compare soil features with county identification."
    )
    parser.add_argument(
        "--reference-run",
        required=True,
        help="Directory name of the previous feature_groups run.",
    )
    args = parser.parse_args()

    run_dir, reference_settings, reference_predictions = load_reference(
        args.reference_run
    )
    parameters = reference_settings["xgboost_parameters"]

    print("PRESEASON XGBOOST: SOIL VS COUNTY")
    print(f"Reference: {run_dir}")
    print(f"Fixed parameters: {parameters}")
    print("Reserved test years: 2023-2025; not evaluated.")
    print("Both training and prediction use expected weather.")

    for variant, features in FEATURE_SETS.items():
        print(f"{variant}: {features}")

    actual, expected, audit = prepare_data()
    folds = build_folds(actual, expected)

    fold_definitions = [
        [
            int(fold["expected_train"]["yield_year"].max()),
            int(fold["expected_validation"]["yield_year"].min()),
            int(fold["expected_validation"]["yield_year"].max()),
        ]
        for fold in folds
    ]

    if fold_definitions != reference_settings["folds"]:
        raise ValueError("Reference validation folds differ.")

    results = []

    for fold in folds:
        train = fold["expected_train"]
        validation = fold["expected_validation"]
        period = fold["period"]

        if set(train["county_name"]) != set(validation["county_name"]):
            raise ValueError(
                "This comparison expects the same counties "
                "in training and validation."
            )

        print(f"\nEvaluating {period}...", flush=True)

        result_columns = KEYS + [
            TARGET_COLUMN,
            "recent_yield_mean_t_ha",
        ]
        template = validation[result_columns].copy()
        template["period"] = period

        baseline = template.copy()
        baseline["variant"] = "baseline"
        baseline["prediction"] = baseline["recent_yield_mean_t_ha"]
        results.append(baseline)

        for variant, features in FEATURE_SETS.items():
            model = create_variant(parameters, variant)
            model.fit(train[features], train[TARGET_COLUMN])

            predicted = model.predict(validation[features])

            if not np.isfinite(predicted).all():
                raise ValueError(
                    f"Non-finite predictions: {period}, {variant}"
                )

            result = template.copy()
            result["variant"] = variant
            result["prediction"] = predicted
            results.append(result)

    predictions = pd.concat(results, ignore_index=True)

    if predictions.duplicated(KEYS + ["variant"]).any():
        raise ValueError("Duplicate predictions.")

    counts = predictions.groupby("variant").size()

    if len(counts) != 4 or not counts.eq(456).all():
        raise ValueError(
            "Expected four variants with 456 predictions each."
        )

    verify_reference(predictions, reference_predictions)

    pooled_metrics = summarize(predictions, pooled=True)
    fold_metrics = summarize(predictions)
    comparisons = compare_variants(pooled_metrics, fold_metrics)

    print_table("POOLED VALIDATION RESULTS", pooled_metrics)
    print_table(
        "COMPARISONS: POSITIVE GAIN MEANS IMPROVEMENT",
        comparisons,
    )
    print_table(
        "OVERALL RESULTS BY VALIDATION PERIOD",
        fold_metrics[fold_metrics["crop"].eq("ALL")],
    )

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output_dir = BASE_DIR / "results" / f"soil_vs_county_{run_id}"
    output_dir.mkdir(parents=True, exist_ok=False)

    audit.to_csv(output_dir / "input_audit.csv", index=False)
    predictions.to_csv(output_dir / "predictions.csv", index=False)
    pooled_metrics.to_csv(
        output_dir / "pooled_metrics.csv", index=False
    )
    fold_metrics.to_csv(
        output_dir / "fold_metrics.csv", index=False
    )
    comparisons.to_csv(
        output_dir / "comparisons.csv", index=False
    )

    settings = {
        "reference_run": str(run_dir),
        "feature_sets": FEATURE_SETS,
        "xgboost_parameters": parameters,
        "folds": fold_definitions,
        "reserved_test_years": [2023, 2024, 2025],
        "county_encoding": "one-hot, fitted on training data",
        "retuned_per_variant": False,
        "independent_test": False,
        "publication_delays_modelled": False,
    }

    (output_dir / "settings.json").write_text(
        json.dumps(settings, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"\nResults saved to:\n{output_dir}")
    print("\nInterpretation:")
    print("- Positive gain means the after variant has a smaller error.")
    print("- soil_vs_county: positive gain means soil beats county.")
    print("- County categories are one-hot encoded, not numeric labels.")
    print("- Parameters were originally selected with soil features.")
    print("- The county variant was not separately tuned.")
    print("- This evaluates later years for already observed counties.")
    print("- This does not evaluate unseen counties or parcels.")
    print("- Similar performance cannot prove identical model mechanisms.")
    print("- Better soil performance does not establish a causal effect.")
    print("- Soil and county use different feature representations.")
    print("- Publication delays are not modelled.")
    print("- No production models or database tables were modified.")


if __name__ == "__main__":
    main()