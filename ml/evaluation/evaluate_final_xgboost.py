# last forecasting test
# worst then baseline; maize has poblems

from pathlib import Path

import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from ml.data_loader import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    get_complete_years,
    load_training_dataset,
)
from ml.train.train_xgboost import create_model


BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "results"
PREDICTIONS_FILE = RESULTS_DIR / "final_xgboost_test_predictions.csv"

TEST_YEAR_COUNT = 3


def calculate_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    r2 = r2_score(y_true, y_pred)

    return mae, rmse, r2


def print_metrics(title, dataframe):
    mae, rmse, r2 = calculate_metrics(
        dataframe["actual_yield_t_ha"],
        dataframe["predicted_yield_t_ha"],
    )

    print(f"\n{title}")
    print(f"Records: {len(dataframe)}")
    print(f"MAE:  {mae:.3f} t/ha")
    print(f"RMSE: {rmse:.3f} t/ha")
    print(f"R²:   {r2:.3f}")


def main():
    print("Loading ML training dataset...")

    dataframe = load_training_dataset()
    complete_years = get_complete_years(dataframe)

    test_years = complete_years[-TEST_YEAR_COUNT:]
    train_years = complete_years[:-TEST_YEAR_COUNT]

    train = dataframe[dataframe["yield_year"].isin(train_years)].copy()
    test = dataframe[dataframe["yield_year"].isin(test_years)].copy()

    print(f"\nFinal train years: {train_years[0]}-{train_years[-1]}")
    print(f"Final test years:  {test_years}")
    print(f"Train records: {len(train)}")
    print(f"Test records:  {len(test)}")

    X_train = train[FEATURE_COLUMNS]
    y_train = train[TARGET_COLUMN]
    X_test = test[FEATURE_COLUMNS]

    print("\nTraining final XGBoost model...")

    model = create_model(
        n_estimators=400,
        learning_rate=0.05,
        max_depth=3,
        subsample=0.8,
        colsample_bytree=1.0,
        min_child_weight=1,
        reg_lambda=5.0,
    )

    model.fit(X_train, y_train)

    print("Training completed.")
    print("Evaluating FINAL TEST...")

    predictions = model.predict(X_test)

    results = test[
        ["yield_year", "county_name", "crop", TARGET_COLUMN]
    ].copy()

    results = results.rename(
        columns={TARGET_COLUMN: "actual_yield_t_ha"}
    )

    results["predicted_yield_t_ha"] = predictions
    results["absolute_error_t_ha"] = (
        results["actual_yield_t_ha"] - results["predicted_yield_t_ha"]
    ).abs()

    print("\n" + "=" * 65)
    print("FINAL XGBOOST TEST RESULT")
    print("=" * 65)

    print_metrics("OVERALL", results)

    print("\n" + "=" * 65)
    print("RESULTS BY CROP")
    print("=" * 65)

    for crop in sorted(results["crop"].unique()):
        crop_results = results[results["crop"] == crop]
        print_metrics(crop.upper(), crop_results)

    print("\n" + "=" * 65)
    print("RESULTS BY YEAR")
    print("=" * 65)

    for year in sorted(results["yield_year"].unique()):
        year_results = results[results["yield_year"] == year]
        print_metrics(str(year), year_results)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results.to_csv(PREDICTIONS_FILE, index=False)

    print(f"\nDetailed predictions saved to:\n{PREDICTIONS_FILE}")
    print("\nIMPORTANT: Do not tune the model based on these test results.")


if __name__ == "__main__":
    main()