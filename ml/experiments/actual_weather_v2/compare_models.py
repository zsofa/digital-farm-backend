import numpy as np
import pandas as pd

from ml.data_loader import FEATURE_COLUMNS, TARGET_COLUMN
from ml.evaluation.compare_final_to_baseline import calculate_metrics
from ml.experiments.actual_weather_v2.tune_xgboost import (
    RESULTS_FILE,
    load_dataset,
    prepare_folds,
)
from ml.train.train_xgboost import create_model


OLD_PARAMS = {
    "n_estimators": 400,
    "learning_rate": 0.05,
    "max_depth": 3,
    "subsample": 0.8,
    "colsample_bytree": 1.0,
    "min_child_weight": 1,
    "reg_lambda": 5.0,
}

PREDICTION_COLUMNS = {
    "baseline": "recent_yield_mean_t_ha",
    "old_ml": "old_prediction",
    "new_ml": "new_prediction",
}


def load_best_params():
    results = pd.read_csv(RESULTS_FILE)

    if results.empty:
        raise ValueError("The tuning results file is empty.")

    best = results.sort_values(
        ["mean_mae", "mean_rmse", "std_mae"]
    ).iloc[0]

    integer_parameters = {
        "n_estimators",
        "max_depth",
        "min_child_weight",
    }

    params = {}

    for name in OLD_PARAMS:
        value = best[name]
        params[name] = (
            int(value)
            if name in integer_parameters
            else float(value)
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
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)
    print(dataframe.round(3).to_string(index=False))


def print_win_counts(fold_metrics):
    rows = []

    for crop, group in fold_metrics.groupby("crop", sort=True):
        mae = group.pivot(index="period", columns="model", values="mae")
        rmse = group.pivot(
            index="period", columns="model", values="rmse"
        )

        rows.append({
            "crop": crop,
            "folds": len(mae),
            "new_beats_baseline_mae": int(
                (mae["new_ml"] < mae["baseline"]).sum()
            ),
            "new_beats_old_mae": int(
                (mae["new_ml"] < mae["old_ml"]).sum()
            ),
            "new_beats_baseline_rmse": int(
                (rmse["new_ml"] < rmse["baseline"]).sum()
            ),
            "new_beats_old_rmse": int(
                (rmse["new_ml"] < rmse["old_ml"]).sum()
            ),
        })

    print_table(
        "NUMBER OF VALIDATION PERIODS WHERE THE NEW MODEL IS BETTER",
        pd.DataFrame(rows),
    )


def main():
    new_params, expected_mae = load_best_params()
    dataframe = load_dataset()
    folds, test_years = prepare_folds(dataframe)

    print("Comparing baseline, old ML and tuned ML.")
    print(f"Old parameters: {OLD_PARAMS}")
    print(f"New parameters: {new_params}")
    print(f"Reserved test years: {test_years}")

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

        predictions = validation.copy()

        for model_name, params in [
            ("old", OLD_PARAMS),
            ("new", new_params),
        ]:
            model = create_model(**params)

            model.fit(
                train[FEATURE_COLUMNS],
                train[TARGET_COLUMN],
            )

            predictions[f"{model_name}_prediction"] = model.predict(
                validation[FEATURE_COLUMNS]
            )

        fold_metrics = summarize(predictions, period)

        all_predictions.append(predictions)
        all_fold_metrics.append(fold_metrics)

        print_table(f"VALIDATION {period}", fold_metrics)

    combined = pd.concat(all_predictions, ignore_index=True)
    fold_metrics = pd.concat(all_fold_metrics, ignore_index=True)

    keys = ["yield_year", "county_name", "crop"]

    if combined.duplicated(keys).any():
        raise ValueError("Repeated validation rows across folds.")

    reproduced_mae = fold_metrics.loc[
        (fold_metrics["crop"] == "ALL")
        & (fold_metrics["model"] == "new_ml"),
        "mae",
    ].mean()

    if not np.isclose(
        reproduced_mae,
        expected_mae,
        rtol=0,
        atol=0.000001,
    ):
        raise ValueError(
            "The tuning MAE could not be reproduced. "
            f"Saved: {expected_mae:.6f}; "
            f"current: {reproduced_mae:.6f}. "
            "Check whether data, code or package versions changed."
        )

    print(f"\nTuning MAE reproduced: {reproduced_mae:.6f} t/ha")

    print_table(
        "POOLED VALIDATION RESULTS",
        summarize(combined, "POOLED"),
    )

    print_win_counts(fold_metrics)

    print("\nPositive bias means overestimation.")
    print("Pooled metrics use all validation predictions together.")
    print("The new parameters were selected using these validation folds.")
    print("This comparison is NOT an independent test.")
    print("Reserved test years were not evaluated.")
    print("No files or production models were overwritten.")


if __name__ == "__main__":
    main()