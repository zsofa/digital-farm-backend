from pathlib import Path
from statistics import mean, stdev

import numpy as np
import pandas as pd
from sklearn.model_selection import ParameterGrid

from ml.data_loader import TARGET_COLUMN
from ml.evaluation.compare_final_to_baseline import calculate_metrics
from ml.experiments.actual_weather_v2.tune_xgboost import (
    PARAMETER_GRID,
    load_dataset,
    prepare_folds,
)
from ml.experiments.actual_weather_v4.compare_monthly_weather import (
    V4_FEATURES,
    add_monthly_features,
)
from ml.train.train_xgboost import create_model


EXPERIMENT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = EXPERIMENT_DIR / "results" / "tuning_001"

SUMMARY_FILE = RESULTS_DIR / "tuning_summary.csv"
FOLD_METRICS_FILE = RESULTS_DIR / "fold_metrics.csv"
BEST_PARAMS_FILE = RESULTS_DIR / "best_params.json"


def evaluate_configuration(configuration_id, params, folds):
    fold_maes = []
    fold_rmses = []
    fold_r2s = []
    detailed_results = []

    for fold_number, (
        train,
        validation,
        train_years,
        validation_years,
    ) in enumerate(folds, start=1):
        model = create_model(**params)

        model.fit(
            train[V4_FEATURES],
            train[TARGET_COLUMN],
        )

        results = validation.copy()
        results["prediction"] = model.predict(
            validation[V4_FEATURES]
        )

        if not np.isfinite(results["prediction"]).all():
            raise ValueError("Non-finite predictions.")

        groups = [("ALL", results)]
        groups.extend(results.groupby("crop", sort=True))

        for crop, group in groups:
            actual = group[TARGET_COLUMN]
            metrics = calculate_metrics(actual, group["prediction"])
            baseline = calculate_metrics(
                actual,
                group["recent_yield_mean_t_ha"],
            )

            detailed_results.append({
                "configuration_id": configuration_id,
                "fold": fold_number,
                "train_start": train_years[0],
                "train_end": train_years[-1],
                "validation_start": validation_years[0],
                "validation_end": validation_years[-1],
                "crop": crop,
                "records": len(group),
                "mae": metrics["mae"],
                "rmse": metrics["rmse"],
                "r2": metrics["r2"],
                "bias": float((group["prediction"] - actual).mean()),
                "baseline_mae": baseline["mae"],
                "baseline_rmse": baseline["rmse"],
            })

            if crop == "ALL":
                fold_maes.append(metrics["mae"])
                fold_rmses.append(metrics["rmse"])
                fold_r2s.append(metrics["r2"])

    summary = {
        "configuration_id": configuration_id,
        **params,
        "mean_mae": mean(fold_maes),
        "std_mae": stdev(fold_maes),
        "mean_rmse": mean(fold_rmses),
        "mean_r2": mean(fold_r2s),
    }

    return summary, detailed_results


def main():
    import json

    print("Loading actual monthly weather inputs...")
    dataframe = add_monthly_features(load_dataset())
    folds, test_years = prepare_folds(dataframe)

    combinations = list(ParameterGrid(PARAMETER_GRID))

    # A fresh directory prevents overwriting an earlier run.
    RESULTS_DIR.mkdir(parents=True, exist_ok=False)

    settings = {
        "feature_columns": V4_FEATURES,
        "parameter_grid": PARAMETER_GRID,
        "reserved_test_years": test_years,
        "selection_order": ["mean_mae", "mean_rmse", "std_mae"],
        "folds": [
            {
                "train_years": train_years,
                "validation_years": validation_years,
            }
            for _, _, train_years, validation_years in folds
        ],
    }

    with (RESULTS_DIR / "settings.json").open(
        "x", encoding="utf-8"
    ) as output:
        json.dump(settings, output, indent=2, ensure_ascii=False)

    print("=" * 90)
    print("MONTHLY V4 XGBOOST TUNING")
    print("=" * 90)
    print(f"Configurations: {len(combinations)}")
    print(f"Validation folds: {len(folds)}")
    print(f"Input columns: {len(V4_FEATURES)}")
    print(f"Reserved test years: {test_years}")
    print("In-season missing values are rejected during data preparation.")
    print("Out-of-season months remain NaN.")

    summaries = []

    for index, params in enumerate(combinations, start=1):
        print(
            f"\n[{index}/{len(combinations)}] {params}",
            flush=True,
        )

        summary, details = evaluate_configuration(index, params, folds)
        summaries.append(summary)

        # Save each completed configuration so partial results survive
        # an interrupted run.
        pd.DataFrame([summary]).to_csv(
            SUMMARY_FILE,
            mode="a",
            header=index == 1,
            index=False,
        )
        pd.DataFrame(details).to_csv(
            FOLD_METRICS_FILE,
            mode="a",
            header=index == 1,
            index=False,
        )

        print(
            f"Mean MAE: {summary['mean_mae']:.4f} t/ha | "
            f"Mean RMSE: {summary['mean_rmse']:.4f} t/ha | "
            f"Mean fold R²: {summary['mean_r2']:.4f}",
            flush=True,
        )

    ranked = (
        pd.DataFrame(summaries)
        .sort_values(
            ["mean_mae", "mean_rmse", "std_mae"],
            kind="stable",
        )
        .reset_index(drop=True)
    )

    ranked.to_csv(SUMMARY_FILE, index=False)
    best = ranked.iloc[0]

    integer_parameters = {
        "n_estimators",
        "max_depth",
        "min_child_weight",
    }

    best_params = {
        name: (
            int(best[name])
            if name in integer_parameters
            else float(best[name])
        )
        for name in PARAMETER_GRID
    }

    with BEST_PARAMS_FILE.open("x", encoding="utf-8") as output:
        json.dump(best_params, output, indent=2)

    print("\n" + "=" * 110)
    print("TOP 10 MONTHLY V4 CONFIGURATIONS")
    print("=" * 110)
    print(ranked.head(10).round(4).to_string(index=False))

    print("\nBEST PARAMETERS")
    for name, value in best_params.items():
        print(f"{name}: {value}")

    print(f"\nMean validation MAE: {best['mean_mae']:.6f} t/ha")
    print(f"MAE standard deviation: {best['std_mae']:.6f} t/ha")
    print(f"Mean validation RMSE: {best['mean_rmse']:.6f} t/ha")
    print(f"Mean fold R²: {best['mean_r2']:.6f}")

    details = pd.read_csv(FOLD_METRICS_FILE)
    best_details = details[
        details["configuration_id"] == int(best["configuration_id"])
    ]

    print("\nBEST CONFIGURATION: METRICS BY VALIDATION PERIOD AND CROP")
    columns = [
        "validation_start",
        "validation_end",
        "crop",
        "mae",
        "baseline_mae",
        "rmse",
        "baseline_rmse",
        "r2",
        "bias",
    ]
    print(best_details[columns].round(3).to_string(index=False))

    print(f"\nResults saved to:\n{RESULTS_DIR}")
    print("Tuning completed successfully.")
    print("The selected result is not an independent test estimate.")
    print("Mean fold RMSE and R² are not pooled metrics.")
    print("Reserved test years were not evaluated.")
    print("The production model was not modified.")


if __name__ == "__main__":
    main()