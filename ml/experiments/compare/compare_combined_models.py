import json

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
from ml.experiments.actual_weather_v4.compare_monthly_weather import (
    V4_FEATURES,
    add_monthly_features,
)
from ml.experiments.actual_weather_v4.tune_monthly_xgboost import (
    BEST_PARAMS_FILE,
    SUMMARY_FILE,
)
from ml.train.train_xgboost import create_model


PREDICTION_COLUMNS = {
    "baseline": "recent_yield_mean_t_ha",
    "seasonal_v2": "v2_prediction",
    "monthly_v4": "v4_prediction",
    "combined_v5": "v5_prediction",
}


def load_v4_params():
    with BEST_PARAMS_FILE.open("r", encoding="utf-8") as source:
        params = json.load(source)

    summary = pd.read_csv(SUMMARY_FILE)

    if summary.empty:
        raise ValueError("V4 tuning results are empty.")

    best = summary.sort_values(
        ["mean_mae", "mean_rmse", "std_mae"],
        kind="stable",
    ).iloc[0]

    for name, value in params.items():
        if not np.isclose(float(value), float(best[name])):
            raise ValueError(
                f"V4 parameter file and tuning summary disagree: {name}"
            )

    return params, float(best["mean_mae"])


def summarize(dataframe, period):
    rows = []
    groups = [("ALL", dataframe)]
    groups.extend(dataframe.groupby("crop", sort=True))

    for crop, group in groups:
        actual = group[TARGET_COLUMN]

        for model_name, column in PREDICTION_COLUMNS.items():
            metrics = calculate_metrics(actual, group[column])

            rows.append({
                "period": period,
                "crop": crop,
                "model": model_name,
                "records": len(group),
                "mae": metrics["mae"],
                "rmse": metrics["rmse"],
                "r2": metrics["r2"],
                "bias": float((group[column] - actual).mean()),
            })

    return pd.DataFrame(rows)


def print_table(title, dataframe):
    print("\n" + "=" * 110)
    print(title)
    print("=" * 110)
    print(dataframe.round(3).to_string(index=False))


def predict_fold(train, validation, v2_params, v4_params):
    results = validation.copy()

    for column, features, params in [
        ("v2_prediction", FEATURE_COLUMNS, v2_params),
        ("v4_prediction", V4_FEATURES, v4_params),
    ]:
        # Both models are trained on all three crops.
        model = create_model(**params)
        model.fit(train[features], train[TARGET_COLUMN])
        results[column] = model.predict(validation[features])

    # Fixed routing rule, not a per-row choice based on actual yield.
    results["v5_prediction"] = np.where(
        results["crop"].eq("maize"),
        results["v4_prediction"],
        results["v2_prediction"],
    )

    columns = list(PREDICTION_COLUMNS.values())

    if not np.isfinite(
        results[columns].to_numpy(dtype=float)
    ).all():
        raise ValueError("Non-finite predictions.")

    return results


def verify_reproduction(fold_metrics, model_name, expected_mae):
    actual_mae = fold_metrics.loc[
        fold_metrics["crop"].eq("ALL")
        & fold_metrics["model"].eq(model_name),
        "mae",
    ].mean()

    if not np.isclose(actual_mae, expected_mae, rtol=0, atol=0.000001):
        raise ValueError(
            f"{model_name} MAE could not be reproduced. "
            f"Expected {expected_mae:.6f}, received {actual_mae:.6f}."
        )

    print(f"{model_name} MAE reproduced: {actual_mae:.6f} t/ha")


def print_win_counts(fold_metrics):
    rows = []

    for crop, group in fold_metrics.groupby("crop", sort=True):
        mae = group.pivot(
            index="period", columns="model", values="mae"
        )
        rmse = group.pivot(
            index="period", columns="model", values="rmse"
        )

        rows.append({
            "crop": crop,
            "folds": len(mae),
            "v5_beats_baseline_mae": int(
                (mae["combined_v5"] < mae["baseline"]).sum()
            ),
            "v5_beats_v2_mae": int(
                (mae["combined_v5"] < mae["seasonal_v2"]).sum()
            ),
            "v5_ties_v2_mae": int(
                np.isclose(
                    mae["combined_v5"],
                    mae["seasonal_v2"],
                    rtol=0,
                    atol=1e-12,
                ).sum()
            ),
            "v5_beats_baseline_rmse": int(
                (rmse["combined_v5"] < rmse["baseline"]).sum()
            ),
            "v5_beats_v2_rmse": int(
                (rmse["combined_v5"] < rmse["seasonal_v2"]).sum()
            ),
        })

    print_table(
        "NUMBER OF VALIDATION PERIODS WHERE V5 IS BETTER",
        pd.DataFrame(rows),
    )


def main():
    v2_params, expected_v2_mae = load_best_params()
    v4_params, expected_v4_mae = load_v4_params()

    dataframe = add_monthly_features(load_dataset())
    folds, test_years = prepare_folds(dataframe)

    print("=" * 110)
    print("V5: SEASONAL V2 FOR WHEAT/BARLEY, MONTHLY V4 FOR MAIZE")
    print("=" * 110)
    print(f"V2 parameters: {v2_params}")
    print(f"V4 parameters: {v4_params}")
    print(f"Reserved test years: {test_years}")

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

        results = predict_fold(
            train,
            validation,
            v2_params,
            v4_params,
        )
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

    verify_reproduction(fold_metrics, "seasonal_v2", expected_v2_mae)
    verify_reproduction(fold_metrics, "monthly_v4", expected_v4_mae)

    print_table(
        "POOLED VALIDATION RESULTS",
        summarize(combined, "POOLED"),
    )
    print_win_counts(fold_metrics)

    print("\nPositive bias means overestimation.")
    print("V5 uses V2 for wheat/barley and V4 for maize.")
    print("For wheat/barley, V5 and V2 predictions are identical.")
    print("Both parameters and routing were selected using validation results.")
    print("This is a development comparison, NOT an independent test.")
    print("Reserved test years were not evaluated.")
    print("No files, database tables or production models were modified.")


if __name__ == "__main__":
    main()