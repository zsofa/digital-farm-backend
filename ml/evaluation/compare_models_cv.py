# compare all the models in the same expanding window validation
# baseline, linear regression, random forest, xgboost

from statistics import mean, stdev

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from ml.data_loader import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    expanding_window_splits,
    load_training_dataset
)

from ml.train.train_linear_regression import (create_model as create_linear_regression)

from ml.train.train_random_forest import (create_model as create_random_forest)

from ml.train.train_xgboost import (create_model as create_xgboost)

# Calculates the mae, rmse, r2 together
def calculate_metrics(y_true,y_pred):
    mae = mean_absolute_error(y_true,y_pred)

    rmse = mean_squared_error(y_true,y_pred) ** 0.5

    r2 = r2_score(y_true,y_pred)

    return {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
    }


def evaluate_baseline(validation):

    y_true = validation[TARGET_COLUMN]
    y_pred = validation["recent_yield_mean_t_ha"]

    return calculate_metrics(y_true,y_pred)

# on every fold: new modell -> fit(train) -> predict(validation) -> metrics
def evaluate_ml_model(model,train,validation):
    X_train = train[FEATURE_COLUMNS]
    y_train = train[TARGET_COLUMN]
    X_validation = validation[FEATURE_COLUMNS]
    y_validation = validation[TARGET_COLUMN]

    model.fit(X_train,y_train)

    y_pred = model.predict(X_validation)

    return calculate_metrics(y_validation, y_pred)


def print_fold_result(model_name, metrics):
    print(
        f"{model_name:<20} "
        f"MAE: {metrics['mae']:.3f}   "
        f"RMSE: {metrics['rmse']:.3f}   "
        f"R²: {metrics['r2']:.3f}"
    )


def print_summary(model_name, results,):
    mae_values = [result["mae"] for result in results]
    rmse_values = [result["rmse"]for result in results]

    r2_values = [result["r2"] for result in results]

    print(f"\n{model_name}")

    print(
        f"Mean MAE:  "
        f"{mean(mae_values):.3f} t/ha"
    )

    print(
        f"MAE std:   "
        f"{stdev(mae_values):.3f}"
    )

    print(
        f"Mean RMSE: "
        f"{mean(rmse_values):.3f} t/ha"
    )

    print(
        f"RMSE std:  "
        f"{stdev(rmse_values):.3f}"
    )

    print(
        f"Mean R²:   "
        f"{mean(r2_values):.3f}"
    )


def main():
    print( "Loading ML training dataset...")

    dataframe = (load_training_dataset())

    folds, test_years = (
        expanding_window_splits(
            dataframe
        )
    )

    print(
        f"\nTotal records: "
        f"{len(dataframe)}"
    )

    print(
        f"Final test years reserved: "
        f"{test_years}"
    )

    print(
        f"Validation folds: "
        f"{len(folds)}"
    )

    results = {
        "Baseline": [],
        "Linear Regression": [],
        "Random Forest": [],
        "XGBoost": [],
    }

    for fold_number, (
        train,
        validation,
        train_years,
        validation_years,
    ) in enumerate(
        folds,
        start=1,
    ):
        print(
            "\n"
            + "=" * 75
        )

        print(
            f"FOLD {fold_number}"
        )

        print(
            f"Train: "
            f"{train_years[0]}"
            f"-"
            f"{train_years[-1]} "
            f"({len(train)} records)"
        )

        print(
            f"Validation: "
            f"{validation_years[0]}"
            f"-"
            f"{validation_years[-1]} "
            f"({len(validation)} records)"
        )

        print("-" * 75)

        # -------------------------
        # Baseline
        # -------------------------

        baseline_metrics = (evaluate_baseline(validation))

        results["Baseline"].append(baseline_metrics)
        print_fold_result("Baseline", baseline_metrics)

        # -------------------------
        # Linear Regression
        # -------------------------

        linear_model = (create_linear_regression())

        linear_metrics = (
            evaluate_ml_model(
                linear_model,
                train,
                validation,
            )
        )

        results[
            "Linear Regression"
        ].append(
            linear_metrics
        )

        print_fold_result("Linear Regression",linear_metrics)

        # -------------------------
        # Random Forest
        # -------------------------

        random_forest_model = (create_random_forest())

        rf_metrics = (
            evaluate_ml_model(
                random_forest_model,
                train,
                validation,
            )
        )

        results["Random Forest"].append(rf_metrics)

        print_fold_result("Random Forest",rf_metrics)

        # -------------------------
        # XGBoost
        # -------------------------

        xgboost_model = (create_xgboost())

        xgb_metrics = (
            evaluate_ml_model(
                xgboost_model,
                train,
                validation,
            )
        )

        results["XGBoost"].append(xgb_metrics)

        print_fold_result("XGBoost",xgb_metrics)

    # -------------------------
    # Summary
    # -------------------------

    print("\n\n"+ "=" * 75)

    print(
        "EXPANDING-WINDOW "
        "VALIDATION SUMMARY"
    )

    print("=" * 75)

    for (
        model_name,
        model_results,
    ) in results.items():

        print_summary(
            model_name,
            model_results,
        )

    print(
        "\nFinal test years "
        f"{test_years} "
        "were NOT evaluated."
    )


if __name__ == "__main__":
    main()