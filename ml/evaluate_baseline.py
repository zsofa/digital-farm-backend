# Baseline: Next yield should be equal to recent_yield_mean_t_ha
# ML should be better, this id a reference
# important benchmark

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from ml.data_loader import (
    load_training_dataset,
    split_dataset,
)

def evaluate_baseline(dataframe,dataset_name):
    actual = dataframe["average_yield_t_ha"]

    predicted = dataframe["recent_yield_mean_t_ha"]

    mae = mean_absolute_error(actual,predicted)

    rmse = mean_squared_error(actual,predicted) ** 0.5

    r2 = r2_score(actual,predicted)

    print(f"\n{dataset_name}")
    print(f"Records: {len(dataframe)}")
    print(f"MAE:  {mae:.3f} t/ha")
    print(f"RMSE: {rmse:.3f} t/ha")
    print(f"R²:   {r2:.3f}")

def main():
    print("Loading ML training dataset...")

    dataframe = (load_training_dataset())

    print(
        f"Total records: "
        f"{len(dataframe)}"
    )

    print(
        f"Years: "
        f"{dataframe['yield_year'].min()}"
        f"-"
        f"{dataframe['yield_year'].max()}"
    )

    train, validation, test = (split_dataset(dataframe))

    print("\nTime-based split:")

    print(
        f"Train:      "
        f"{len(train)} records "
        f"({train['yield_year'].min()}"
        f"-"
        f"{train['yield_year'].max()})"
    )

    print(
        f"Validation: "
        f"{len(validation)} records "
        f"({validation['yield_year'].min()}"
        f"-"
        f"{validation['yield_year'].max()})"
    )

    print(
        f"Test:       "
        f"{len(test)} records "
        f"({test['yield_year'].min()}"
        f"-"
        f"{test['yield_year'].max()})"
    )

    print("\nRecent yield baseline:")

    evaluate_baseline(train,"TRAIN")
    evaluate_baseline(validation, "VALIDATION")
    evaluate_baseline(test,"TEST")


if __name__ == "__main__":
    main()