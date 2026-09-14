# 192 configuration

from pathlib import Path
from statistics import mean, stdev

import pandas as pd

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import ParameterGrid

from ml.data_loader import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    expanding_window_splits,
    load_training_dataset,
)

from ml.train.train_xgboost import (
    create_model,
)


BASE_DIR = Path(__file__).resolve().parent.parent

RESULTS_DIR = BASE_DIR / "results"

RESULTS_FILE = (
    RESULTS_DIR
    / "xgboost_tuning_results.csv"
)


PARAMETER_GRID = {
    "n_estimators": [
        200,
        400,
    ],

    "learning_rate": [
        0.03,
        0.05,
    ],

    "max_depth": [
        2,
        3,
        4,
    ],

    "subsample": [
        0.8,
        1.0,
    ],

    "colsample_bytree": [
        0.8,
        1.0,
    ],

    "min_child_weight": [
        1,
        5,
    ],

    "reg_lambda": [
        1.0,
        5.0,
    ],
}


def calculate_metrics(
    y_true,
    y_pred,
):
    mae = mean_absolute_error(
        y_true,
        y_pred,
    )

    rmse = (
        mean_squared_error(
            y_true,
            y_pred,
        )
        ** 0.5
    )

    r2 = r2_score(
        y_true,
        y_pred,
    )

    return mae, rmse, r2


def evaluate_configuration(
    params,
    folds,
):
    fold_maes = []
    fold_rmses = []
    fold_r2s = []

    for (
        train,
        validation,
        _,
        _,
    ) in folds:

        X_train = train[
            FEATURE_COLUMNS
        ]

        y_train = train[
            TARGET_COLUMN
        ]

        X_validation = validation[
            FEATURE_COLUMNS
        ]

        y_validation = validation[
            TARGET_COLUMN
        ]

        model = create_model(
            **params
        )

        model.fit(
            X_train,
            y_train,
        )

        predictions = model.predict(
            X_validation
        )

        mae, rmse, r2 = (
            calculate_metrics(
                y_validation,
                predictions,
            )
        )

        fold_maes.append(
            mae
        )

        fold_rmses.append(
            rmse
        )

        fold_r2s.append(
            r2
        )

    return {
        **params,

        "mean_mae": mean(
            fold_maes
        ),

        "std_mae": stdev(
            fold_maes
        ),

        "mean_rmse": mean(
            fold_rmses
        ),

        "mean_r2": mean(
            fold_r2s
        ),
    }


def main():
    print(
        "Loading ML training dataset..."
    )

    dataframe = (
        load_training_dataset()
    )

    folds, test_years = (
        expanding_window_splits(
            dataframe
        )
    )

    parameter_combinations = list(
        ParameterGrid(
            PARAMETER_GRID
        )
    )

    print(
        f"\nConfigurations to test: "
        f"{len(parameter_combinations)}"
    )

    print(
        f"Validation folds: "
        f"{len(folds)}"
    )

    print(
        f"Final test years reserved: "
        f"{test_years}"
    )

    print(
        "\nStarting XGBoost tuning..."
    )

    results = []

    for index, params in enumerate(
        parameter_combinations,
        start=1,
    ):
        print(
            f"[{index}/"
            f"{len(parameter_combinations)}] "
            f"{params}"
        )

        result = (
            evaluate_configuration(
                params,
                folds,
            )
        )

        results.append(
            result
        )

    results_dataframe = pd.DataFrame(
        results
    )

    results_dataframe = (
        results_dataframe
        .sort_values(
            by=[
                "mean_mae",
                "mean_rmse",
                "std_mae",
            ],
            ascending=True,
        )
        .reset_index(
            drop=True
        )
    )

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    results_dataframe.to_csv(
        RESULTS_FILE,
        index=False,
    )

    print(
        "\n"
        + "=" * 75
    )

    print(
        "TOP 10 XGBOOST CONFIGURATIONS"
    )

    print(
        "=" * 75
    )

    columns_to_show = [
        "n_estimators",
        "learning_rate",
        "max_depth",
        "subsample",
        "colsample_bytree",
        "min_child_weight",
        "reg_lambda",
        "mean_mae",
        "std_mae",
        "mean_rmse",
        "mean_r2",
    ]

    print(
        results_dataframe[
            columns_to_show
        ]
        .head(10)
        .to_string(
            index=False
        )
    )

    best = (
        results_dataframe
        .iloc[0]
    )

    print(
        "\n"
        + "=" * 75
    )

    print(
        "BEST XGBOOST CONFIGURATION"
    )

    print(
        "=" * 75
    )

    print(
        f"n_estimators: "
        f"{int(best['n_estimators'])}"
    )

    print(
        f"learning_rate: "
        f"{best['learning_rate']}"
    )

    print(
        f"max_depth: "
        f"{int(best['max_depth'])}"
    )

    print(
        f"subsample: "
        f"{best['subsample']}"
    )

    print(
        f"colsample_bytree: "
        f"{best['colsample_bytree']}"
    )

    print(
        f"min_child_weight: "
        f"{best['min_child_weight']}"
    )

    print(
        f"reg_lambda: "
        f"{best['reg_lambda']}"
    )

    print(
        f"\nMean MAE: "
        f"{best['mean_mae']:.3f} t/ha"
    )

    print(
        f"MAE std: "
        f"{best['std_mae']:.3f}"
    )

    print(
        f"Mean RMSE: "
        f"{best['mean_rmse']:.3f} t/ha"
    )

    print(
        f"Mean R²: "
        f"{best['mean_r2']:.3f}"
    )

    print(
        f"\nFull results saved to:\n"
        f"{RESULTS_FILE}"
    )

    print(
        "\nFinal test years "
        f"{test_years} "
        "were NOT evaluated."
    )

"""
best:
n_estimators = 400
learning_rate = 0.05
max_depth = 3
subsample = 0.8
colsample_bytree = 1.0
min_child_weight = 1
reg_lambda = 5

expanding window:
MAE = 0.884
RMSE = 1.174
R² = 0.208
"""


if __name__ == "__main__":
    main()