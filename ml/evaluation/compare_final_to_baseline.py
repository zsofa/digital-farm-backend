# baseline turned out better then all ml

from pathlib import Path

import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from ml.data_loader import get_connection


BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_FILE = BASE_DIR / "results" / "final_xgboost_test_predictions.csv"


def calculate_metrics(y_true, y_pred):
    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": mean_squared_error(y_true, y_pred) ** 0.5,
        "r2": r2_score(y_true, y_pred),
    }


def load_recent_yield():
    query = """
        SELECT yield_year, county_name, crop, recent_yield_mean_t_ha
        FROM ml.training_dataset;
    """

    connection = get_connection()

    try:
        with connection.cursor() as cursor:
            cursor.execute(query)
            rows = cursor.fetchall()
            columns = [description.name for description in cursor.description]

        return pd.DataFrame(rows, columns=columns)

    finally:
        connection.close()


def print_comparison(title, dataframe):
    actual = dataframe["actual_yield_t_ha"]

    baseline = calculate_metrics(actual, dataframe["recent_yield_mean_t_ha"])
    xgboost = calculate_metrics(actual, dataframe["predicted_yield_t_ha"])

    print(f"\n{title}")
    print(f"{'Model':<15} {'MAE':>8} {'RMSE':>8} {'R²':>8}")
    print("-" * 42)
    print(f"{'Baseline':<15} {baseline['mae']:>8.3f} {baseline['rmse']:>8.3f} {baseline['r2']:>8.3f}")
    print(f"{'XGBoost':<15} {xgboost['mae']:>8.3f} {xgboost['rmse']:>8.3f} {xgboost['r2']:>8.3f}")


def main():
    predictions = pd.read_csv(RESULTS_FILE)
    recent_yield = load_recent_yield()

    dataframe = predictions.merge(
        recent_yield,
        on=["yield_year", "county_name", "crop"],
        how="left",
    )

    dataframe["recent_yield_mean_t_ha"] = pd.to_numeric(
        dataframe["recent_yield_mean_t_ha"]
    )

    print("=" * 60)
    print("FINAL TEST: BASELINE VS XGBOOST")
    print("=" * 60)

    print_comparison("OVERALL", dataframe)

    print("\n" + "=" * 60)
    print("BY CROP")
    print("=" * 60)

    for crop in sorted(dataframe["crop"].unique()):
        print_comparison(
            crop.upper(),
            dataframe[dataframe["crop"] == crop],
        )

    print("\n" + "=" * 60)
    print("BY YEAR")
    print("=" * 60)

    for year in sorted(dataframe["yield_year"].unique()):
        print_comparison(
            str(year),
            dataframe[dataframe["yield_year"] == year],
        )


if __name__ == "__main__":
    main()