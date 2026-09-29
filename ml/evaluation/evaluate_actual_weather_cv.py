import numpy as np
import pandas as pd

from ml.data_loader import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    expanding_window_splits,
)
from ml.evaluation.evaluate_actual_weather_xg import (
    load_actual_weather_dataset,
)
from ml.evaluation.compare_final_to_baseline import calculate_metrics
from ml.train.train_xgboost import create_model


def evaluate_group(dataframe, period, crop):
    actual = dataframe[TARGET_COLUMN]

    baseline = calculate_metrics(
        actual,
        dataframe["recent_yield_mean_t_ha"],
    )
    ml = calculate_metrics(
        actual,
        dataframe["predicted_yield_t_ha"],
    )

    return {
        "period": period,
        "crop": crop,
        "records": len(dataframe),
        "baseline_mae": baseline["mae"],
        "ml_mae": ml["mae"],
        "mae_gain": baseline["mae"] - ml["mae"],
        "baseline_rmse": baseline["rmse"],
        "ml_rmse": ml["rmse"],
        "baseline_r2": baseline["r2"],
        "ml_r2": ml["r2"],
    }


def summarize_predictions(dataframe, period):
    rows = [evaluate_group(dataframe, period, "ALL")]

    for crop, group in dataframe.groupby("crop", sort=True):
        rows.append(evaluate_group(group, period, crop))

    return rows


def main():
    dataframe = load_actual_weather_dataset()

    numeric_columns = [
        column for column in FEATURE_COLUMNS if column != "crop"
    ] + [TARGET_COLUMN]

    for column in numeric_columns:
        dataframe[column] = pd.to_numeric(
            dataframe[column]
        ).astype(float)

    if not np.isfinite(
        dataframe[numeric_columns].to_numpy()
    ).all():
        raise ValueError("Missing or non-finite numeric values.")

    keys = ["yield_year", "county_name", "crop"]
    if dataframe.duplicated(keys).any():
        raise ValueError("Duplicate county-crop-year rows.")

    folds, test_years = expanding_window_splits(dataframe)

    print("=" * 100)
    print("ACTUAL-WEATHER CROSS-VALIDATION: ML VS BASELINE")
    print("=" * 100)
    print(f"Reserved test years: {test_years}")
    print(f"Validation folds: {len(folds)}")
    print("Positive MAE gain means the ML has a smaller error.")

    comparison_rows = []
    validation_predictions = []

    for index, (
        train,
        validation,
        train_years,
        validation_years,
    ) in enumerate(folds, start=1):
        if max(train_years) >= min(validation_years):
            raise ValueError("Invalid chronological split.")

        if set(train_years + validation_years) & set(test_years):
            raise ValueError("Reserved test years entered validation.")

        period = f"{validation_years[0]}-{validation_years[-1]}"

        print(f"\nFold {index}")
        print(f"Train: {train_years[0]}-{train_years[-1]}")
        print(f"Validation: {period}")
        print(f"Rows: {len(train)} train, {len(validation)} validation")

        model = create_model(
            n_estimators=400,
            learning_rate=0.05,
            max_depth=3,
            subsample=0.8,
            colsample_bytree=1.0,
            min_child_weight=1,
            reg_lambda=5.0,
        )

        model.fit(
            train[FEATURE_COLUMNS],
            train[TARGET_COLUMN],
        )

        results = validation.copy()
        results["predicted_yield_t_ha"] = model.predict(
            results[FEATURE_COLUMNS]
        )

        fold_rows = summarize_predictions(results, period)
        comparison_rows.extend(fold_rows)
        validation_predictions.append(results)

        print(
            pd.DataFrame(fold_rows)
            .round(3)
            .to_string(index=False)
        )

    combined = pd.concat(validation_predictions, ignore_index=True)

    if combined.duplicated(keys).any():
        raise ValueError("Validation rows occur in multiple folds.")

    print("\n" + "=" * 100)
    print("POOLED VALIDATION RESULTS")
    print("=" * 100)
    print("Each prediction comes from a model trained on earlier years.")

    pooled_rows = summarize_predictions(combined, "POOLED")
    print(
        pd.DataFrame(pooled_rows)
        .round(3)
        .to_string(index=False)
    )

    comparisons = pd.DataFrame(comparison_rows)
    summary_rows = []

    for crop, group in comparisons.groupby("crop", sort=True):
        summary_rows.append({
            "crop": crop,
            "folds": len(group),
            "ml_better_mae_folds": int(
                (group["ml_mae"] < group["baseline_mae"]).sum()
            ),
            "ml_better_rmse_folds": int(
                (group["ml_rmse"] < group["baseline_rmse"]).sum()
            ),
        })

    print("\n" + "=" * 100)
    print("NUMBER OF VALIDATION PERIODS WHERE ML BEATS BASELINE")
    print("=" * 100)
    print(pd.DataFrame(summary_rows).to_string(index=False))

    print("\nReserved test years were not evaluated.")
    print("No model or result files were overwritten.")


if __name__ == "__main__":
    main()