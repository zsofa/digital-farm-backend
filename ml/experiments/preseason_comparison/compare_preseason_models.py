import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.model_selection import ParameterGrid
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ml.data_loader import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    load_training_dataset,
)
from ml.experiments.actual_weather_v2.compare_models import load_best_params
from ml.experiments.actual_weather_v2.evaluate_weather_scenarios_cv import (
    FOLDS,
    PRECIPITATION_COLUMN,
    TEMPERATURE_COLUMN,
    add_scenario_weather,
    calculate_metrics,
    load_weather_history,
)
from ml.experiments.actual_weather_v2.tune_xgboost import (
    PARAMETER_GRID,
    load_dataset,
)
from ml.train.train_xgboost import create_model


BASE_DIR = Path(__file__).resolve().parent
KEYS = ["yield_year", "county_name", "crop"]
RIDGE_ALPHAS = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
TEST_YEARS = [2023, 2024, 2025]


def print_table(title, dataframe):
    print("\n" + "=" * 110)
    print(title)
    print("=" * 110)
    print(dataframe.to_string(
        index=False,
        float_format=lambda value: f"{value:.3f}",
    ))


def create_ridge(alpha):
    numeric_columns = [
        column for column in FEATURE_COLUMNS if column != "crop"
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
                numeric_columns,
            ),
        ],
        sparse_threshold=0,
    )

    return Pipeline([
        ("preprocessing", preprocessing),
        ("model", Ridge(alpha=alpha, solver="svd")),
    ])


def prepare_data():
    actual = load_dataset()
    actual = actual[
        actual["yield_year"].between(2005, 2022)
    ].copy().reset_index(drop=True)

    if len(actual) != 1026:
        raise ValueError("Expected 1026 development rows.")

    history = load_weather_history()
    reconstructed = add_scenario_weather(actual, history)

    expected = actual.copy()
    expected[TEMPERATURE_COLUMN] = reconstructed[
        "expected_temperature"
    ]
    expected[PRECIPITATION_COLUMN] = reconstructed[
        "expected_precipitation"
    ]

    stored = load_training_dataset()
    stored = stored[
        stored["yield_year"].between(2005, 2022)
    ].copy()

    if len(stored) != len(expected) or stored.duplicated(KEYS).any():
        raise ValueError("Unexpected stored training dataset keys.")

    numeric_columns = [
        column for column in FEATURE_COLUMNS if column != "crop"
    ] + [TARGET_COLUMN]

    audit = expected[KEYS + numeric_columns].merge(
        stored[KEYS + numeric_columns],
        on=KEYS,
        how="outer",
        suffixes=("_rebuilt", "_stored"),
        indicator=True,
        validate="one_to_one",
    )

    if not audit["_merge"].eq("both").all():
        raise ValueError("Stored/reconstructed row keys differ.")

    audit_rows = []

    for column in numeric_columns:
        rebuilt = audit[f"{column}_rebuilt"].to_numpy(dtype=float)
        saved = audit[f"{column}_stored"].to_numpy(dtype=float)

        if not np.isfinite(rebuilt).all() or not np.isfinite(saved).all():
            raise ValueError(f"Non-finite values in {column}.")

        differences = np.abs(rebuilt - saved)

        audit_rows.append({
            "column": column,
            "rows": len(audit),
            "mismatches": int((differences > 1e-6).sum()),
            "max_difference": float(differences.max()),
        })

    audit_results = pd.DataFrame(audit_rows)
    print_table("STORED VS RECONSTRUCTED INPUT AUDIT", audit_results)

    if audit_results["mismatches"].sum() != 0:
        raise ValueError(
            "Stored and reconstructed inputs differ. "
            "Inspect the audit before comparing models."
        )

    return actual, expected, audit_results


def build_folds(actual, expected):
    folds = []

    if not actual[KEYS].equals(expected[KEYS]):
        raise ValueError("Actual/expected dataset alignment failed.")

    for train_end, validation_start, validation_end in FOLDS:
        train_mask = actual["yield_year"].between(2005, train_end)
        validation_mask = actual["yield_year"].between(
            validation_start, validation_end
        )

        actual_train = actual.loc[train_mask].copy()
        expected_train = expected.loc[train_mask].copy()
        actual_validation = actual.loc[validation_mask].copy()
        expected_validation = expected.loc[validation_mask].copy()

        if len(actual_train) != (train_end - 2005 + 1) * 57:
            raise ValueError("Unexpected training row count.")

        if len(actual_validation) != 114:
            raise ValueError("Expected 114 validation rows.")

        if actual_train["yield_year"].max() >= validation_start:
            raise ValueError("Training/validation overlap.")

        folds.append({
            "period": f"{validation_start}-{validation_end}",
            "actual_train": actual_train,
            "expected_train": expected_train,
            "actual_validation": actual_validation,
            "expected_validation": expected_validation,
        })

    return folds


def evaluate_configuration(folds, factory, actual_training=False):
    predictions = []
    fold_metrics = []

    for fold in folds:
        train_key = (
            "actual_train" if actual_training else "expected_train"
        )
        train = fold[train_key]
        validation = fold["expected_validation"]

        model = factory()
        model.fit(train[FEATURE_COLUMNS], train[TARGET_COLUMN])

        predicted = model.predict(validation[FEATURE_COLUMNS])
        metrics = calculate_metrics(validation[TARGET_COLUMN], predicted)
        fold_metrics.append(metrics)

        result = validation[
            KEYS + [TARGET_COLUMN, "recent_yield_mean_t_ha"]
        ].copy()
        result["period"] = fold["period"]
        result["prediction"] = predicted
        predictions.append(result)

    summary = {
        "mean_mae": float(np.mean([m["mae"] for m in fold_metrics])),
        "std_mae": float(np.std([m["mae"] for m in fold_metrics])),
        "mean_rmse": float(np.mean([m["rmse"] for m in fold_metrics])),
    }

    return summary, pd.concat(predictions, ignore_index=True)


def tune(folds, name, configurations, factory):
    records = []
    best_score = None
    best_parameters = None
    best_predictions = None

    for number, parameters in enumerate(configurations, start=1):
        summary, predictions = evaluate_configuration(
            folds,
            lambda: factory(parameters),
        )

        records.append({
            "configuration_id": number,
            **parameters,
            **summary,
        })

        score = (
            summary["mean_mae"],
            summary["mean_rmse"],
            summary["std_mae"],
        )

        if best_score is None or score < best_score:
            best_score = score
            best_parameters = dict(parameters)
            best_predictions = predictions

        if number == 1 or number % 16 == 0 or number == len(configurations):
            print(
                f"{name}: {number}/{len(configurations)} configurations; "
                f"best mean MAE = {best_score[0]:.6f} t/ha",
                flush=True,
            )

    tuning = pd.DataFrame(records).sort_values(
        ["mean_mae", "mean_rmse", "std_mae", "configuration_id"],
        kind="stable",
    )

    return best_parameters, best_predictions, tuning


def summarize(predictions, pooled=False):
    rows = []

    if pooled:
        periods = [("POOLED", predictions)]
    else:
        periods = predictions.groupby("period", sort=True)

    for period, period_data in periods:
        groups = [("ALL", period_data)]
        groups.extend(period_data.groupby("crop", sort=True))

        for crop, group in groups:
            for model_name, model_data in group.groupby("model", sort=True):
                rows.append({
                    "period": period,
                    "crop": crop,
                    "model": model_name,
                    "records": len(model_data),
                    **calculate_metrics(
                        model_data[TARGET_COLUMN],
                        model_data["prediction"],
                    ),
                })

    return pd.DataFrame(rows)


def label_predictions(dataframe, model_name):
    result = dataframe.copy()
    result["model"] = model_name
    return result


def verify_v2_actual_weather(folds, parameters, reference_mae):
    maes = []

    for fold in folds:
        train = fold["actual_train"]
        validation = fold["actual_validation"]

        model = create_model(**parameters)
        model.fit(train[FEATURE_COLUMNS], train[TARGET_COLUMN])

        predicted = model.predict(validation[FEATURE_COLUMNS])
        maes.append(calculate_metrics(
            validation[TARGET_COLUMN], predicted
        )["mae"])

    reproduced = float(np.mean(maes))

    if not np.isclose(reproduced, reference_mae, atol=1e-6, rtol=0):
        raise ValueError(
            f"V2 reproduction failed: "
            f"{reproduced:.9f} vs {reference_mae:.9f}"
        )

    print(f"\nV2 actual-weather MAE reproduced: {reproduced:.6f}")


def main():
    print("PRESEASON MODEL COMPARISON")
    print("Development years: 2005-2022")
    print(f"Reserved test years: {TEST_YEARS}")
    print("All evaluated predictions use historical expected weather.")
    print("Tuning and comparison use the same development folds.")

    actual, expected, audit = prepare_data()
    folds = build_folds(actual, expected)
    v2_parameters, reference_mae = load_best_params()

    verify_v2_actual_weather(folds, v2_parameters, reference_mae)

    print("\nEvaluating V2 trained on actual weather...")
    _, v2_predictions = evaluate_configuration(
        folds,
        lambda: create_model(**v2_parameters),
        actual_training=True,
    )

    print("Evaluating expected-weather training with fixed V2 parameters...")
    _, fixed_predictions = evaluate_configuration(
        folds,
        lambda: create_model(**v2_parameters),
    )

    xgb_configurations = list(ParameterGrid(PARAMETER_GRID))

    xgb_parameters, xgb_predictions, xgb_tuning = tune(
        folds,
        "Expected-weather XGBoost",
        xgb_configurations,
        lambda parameters: create_model(**parameters),
    )

    ridge_configurations = [
        {"alpha": alpha} for alpha in RIDGE_ALPHAS
    ]

    ridge_parameters, ridge_predictions, ridge_tuning = tune(
        folds,
        "Expected-weather Ridge",
        ridge_configurations,
        lambda parameters: create_ridge(parameters["alpha"]),
    )

    baseline = v2_predictions.copy()
    baseline["prediction"] = baseline["recent_yield_mean_t_ha"]

    predictions = pd.concat([
        label_predictions(baseline, "baseline"),
        label_predictions(v2_predictions, "v2_actual_train"),
        label_predictions(fixed_predictions, "xgb_expected_fixed"),
        label_predictions(xgb_predictions, "xgb_expected_tuned"),
        label_predictions(ridge_predictions, "ridge_expected_tuned"),
    ], ignore_index=True)

    if predictions.duplicated(KEYS + ["model"]).any():
        raise ValueError("Duplicate model predictions.")

    counts = predictions.groupby("model").size()
    if not counts.eq(456).all():
        raise ValueError("Expected 456 predictions per model.")

    fold_results = summarize(predictions)
    pooled_results = summarize(predictions, pooled=True)

    print_table("TOP 10 EXPECTED-WEATHER XGBOOST CONFIGURATIONS",
                xgb_tuning.head(10))
    print_table("RIDGE CONFIGURATIONS", ridge_tuning)

    print("\nBest expected-weather XGBoost parameters:")
    print(json.dumps(xgb_parameters, indent=2))
    print(f"Best Ridge alpha: {ridge_parameters['alpha']}")

    for period in fold_results["period"].unique():
        print_table(
            f"VALIDATION {period}",
            fold_results[fold_results["period"] == period],
        )

    print_table("POOLED VALIDATION RESULTS", pooled_results)

    pivot = fold_results.pivot(
        index=["period", "crop"],
        columns="model",
        values="mae",
    )

    win_rows = []
    model_names = [
        name for name in pivot.columns if name != "baseline"
    ]

    for crop in ["ALL", "barley", "maize", "wheat"]:
        crop_results = pivot.xs(crop, level="crop")

        for model_name in model_names:
            win_rows.append({
                "crop": crop,
                "model": model_name,
                "folds": len(crop_results),
                "beats_baseline_mae": int((
                    crop_results[model_name]
                    < crop_results["baseline"] - 1e-9
                ).sum()),
                "beats_v2_mae": int((
                    crop_results[model_name]
                    < crop_results["v2_actual_train"] - 1e-9
                ).sum()),
            })

    wins = pd.DataFrame(win_rows)
    print_table("VALIDATION PERIOD WIN COUNTS", wins)

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output_dir = BASE_DIR / "results" / run_id
    output_dir.mkdir(parents=True, exist_ok=False)

    audit.to_csv(output_dir / "input_audit.csv", index=False)
    xgb_tuning.to_csv(output_dir / "xgboost_tuning.csv", index=False)
    ridge_tuning.to_csv(output_dir / "ridge_tuning.csv", index=False)
    predictions.to_csv(output_dir / "predictions.csv", index=False)
    fold_results.to_csv(output_dir / "fold_metrics.csv", index=False)
    pooled_results.to_csv(output_dir / "pooled_metrics.csv", index=False)
    wins.to_csv(output_dir / "win_counts.csv", index=False)

    settings = {
        "features": list(FEATURE_COLUMNS),
        "target": TARGET_COLUMN,
        "folds": FOLDS,
        "reserved_test_years": TEST_YEARS,
        "v2_parameters": v2_parameters,
        "best_expected_xgboost_parameters": xgb_parameters,
        "best_ridge_parameters": ridge_parameters,
        "xgboost_grid": PARAMETER_GRID,
        "ridge_alphas": RIDGE_ALPHAS,
        "selection_metric": "mean validation MAE",
        "tie_breakers": ["mean validation RMSE", "MAE std"],
        "independent_test": False,
        "publication_delays_modelled": False,
        "weather_source": (
            "Historical application expected scenario, "
            "reconstructed separately for each target season"
        ),
    }

    (output_dir / "settings.json").write_text(
        json.dumps(settings, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"\nResults saved to:\n{output_dir}")
    print("\nIMPORTANT:")
    print("- All comparison rows use expected weather at prediction time.")
    print("- v2_actual_train is trained on actual historical weather.")
    print("- xgb_expected_fixed changes training weather only.")
    print("- xgb_expected_tuned also selects new XGBoost parameters.")
    print("- Ridge uses scaled numeric features and one-hot crop encoding.")
    print("- Scaling is fitted only on each fold's training rows.")
    print("- Tuning scores are not independent test results.")
    print("- Earlier harvest values are filtered by year, not release date.")
    print("- Static soil data are used, as in previous experiments.")
    print("- Scenario responses and parcel-level accuracy are not evaluated.")
    print("- Reserved test years were not evaluated.")
    print("- No production model or database table was modified.")


if __name__ == "__main__":
    main()