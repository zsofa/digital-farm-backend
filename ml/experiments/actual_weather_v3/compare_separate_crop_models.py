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
from ml.train.train_xgboost import create_model


CROPS = ["barley", "maize", "wheat"]

PREDICTION_COLUMNS = {
    "baseline": "recent_yield_mean_t_ha",
    "shared_v2": "shared_prediction",
    "separate_v3": "separate_prediction",
}


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


def predict_fold(train, validation, params):
    results = validation.copy()

    shared_model = create_model(**params)
    shared_model.fit(
        train[FEATURE_COLUMNS],
        train[TARGET_COLUMN],
    )

    results["shared_prediction"] = shared_model.predict(
        validation[FEATURE_COLUMNS]
    )
    results["separate_prediction"] = np.nan

    for crop in CROPS:
        crop_train = train[train["crop"] == crop]
        crop_mask = results["crop"] == crop
        crop_validation = results.loc[crop_mask]

        if crop_train.empty or crop_validation.empty:
            raise ValueError(f"Missing training or validation data: {crop}")

        separate_model = create_model(**params)
        separate_model.fit(
            crop_train[FEATURE_COLUMNS],
            crop_train[TARGET_COLUMN],
        )

        results.loc[crop_mask, "separate_prediction"] = (
            separate_model.predict(crop_validation[FEATURE_COLUMNS])
        )

    prediction_columns = list(PREDICTION_COLUMNS.values())

    if not np.isfinite(
        results[prediction_columns].to_numpy(dtype=float)
    ).all():
        raise ValueError("Missing or non-finite predictions.")

    return results


def print_win_counts(fold_metrics):
    rows = []

    for crop, group in fold_metrics.groupby("crop", sort=True):
        mae = group.pivot(
            index="period",
            columns="model",
            values="mae",
        )
        rmse = group.pivot(
            index="period",
            columns="model",
            values="rmse",
        )

        rows.append({
            "crop": crop,
            "folds": len(mae),
            "v3_beats_baseline_mae": int(
                (mae["separate_v3"] < mae["baseline"]).sum()
            ),
            "v3_beats_v2_mae": int(
                (mae["separate_v3"] < mae["shared_v2"]).sum()
            ),
            "v3_beats_baseline_rmse": int(
                (rmse["separate_v3"] < rmse["baseline"]).sum()
            ),
            "v3_beats_v2_rmse": int(
                (rmse["separate_v3"] < rmse["shared_v2"]).sum()
            ),
        })

    print_table(
        "NUMBER OF VALIDATION PERIODS WHERE V3 IS BETTER",
        pd.DataFrame(rows),
    )


def main():
    params, expected_v2_mae = load_best_params()
    dataframe = load_dataset()
    folds, test_years = prepare_folds(dataframe)

    print("=" * 110)
    print("BASELINE VS SHARED V2 VS SEPARATE CROP V3")
    print("=" * 110)
    print(f"Parameters used for both ML approaches: {params}")
    print(f"Reserved test years: {test_years}")
    print("Separate crop models are not individually tuned.")

    all_predictions = []
    all_fold_metrics = []

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
        print(
            f"Shared training rows: {len(train)}; "
            f"per-crop training rows: "
            f"{train.groupby('crop').size().to_dict()}",
            flush=True,
        )

        results = predict_fold(train, validation, params)
        metrics = summarize(results, period)

        all_predictions.append(results)
        all_fold_metrics.append(metrics)

        print_table(f"VALIDATION {period}", metrics)

    combined = pd.concat(all_predictions, ignore_index=True)
    fold_metrics = pd.concat(all_fold_metrics, ignore_index=True)

    keys = ["yield_year", "county_name", "crop"]

    if combined.duplicated(keys).any():
        raise ValueError("Repeated validation rows across folds.")

    reproduced_v2_mae = fold_metrics.loc[
        (fold_metrics["crop"] == "ALL")
        & (fold_metrics["model"] == "shared_v2"),
        "mae",
    ].mean()

    if not np.isclose(
        reproduced_v2_mae,
        expected_v2_mae,
        rtol=0,
        atol=0.000001,
    ):
        raise ValueError(
            "The v2 tuning MAE could not be reproduced. "
            f"Saved: {expected_v2_mae:.6f}; "
            f"current: {reproduced_v2_mae:.6f}. "
            "Check data, code and package versions."
        )

    print(f"\nV2 MAE reproduced: {reproduced_v2_mae:.6f} t/ha")

    print_table(
        "POOLED VALIDATION RESULTS",
        summarize(combined, "POOLED"),
    )

    print_win_counts(fold_metrics)

    print("\nPositive bias means overestimation.")
    print("Pooled metrics combine predictions from all validation folds.")
    print("V2 parameters were selected using these same validation folds.")
    print("V3 uses those parameters without separate crop tuning.")
    print("This is a development comparison, NOT an independent test.")
    print("Reserved test years were not evaluated.")
    print("No files or production models were overwritten.")


if __name__ == "__main__":
    main()