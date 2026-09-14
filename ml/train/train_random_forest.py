from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from ml.data_loader import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    load_training_dataset,
    split_dataset,
)

# params are important fot the tuning
#OneHotEncoder -> RandomForestRegressor
def create_model(
    n_estimators=300,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features=1.0,
):

    #Create the preprocessing + Random Forest pipeline

    categorical_features = [
        "crop",
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "crop_encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                ),
                categorical_features,
            ),
        ],
        remainder="passthrough",
    )

    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        random_state=42,
        n_jobs=-1,
    )

    pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor,
            ),
            (
                "model",
                model,
            ),
        ]
    )

    return pipeline


def evaluate_model(
    model,
    dataframe,
    dataset_name,
):
    X = dataframe[FEATURE_COLUMNS]

    y_true = dataframe[TARGET_COLUMN]

    y_pred = model.predict(X)

    mae = mean_absolute_error(y_true,y_pred)

    rmse = mean_squared_error( y_true, y_pred) ** 0.5

    r2 = r2_score(y_true,y_pred)

    print(f"\n{dataset_name}")

    print(f"Records: {len(dataframe)}")

    print( f"MAE:  {mae:.3f} t/ha")

    print(f"RMSE: {rmse:.3f} t/ha")

    print(f"R²:   {r2:.3f}")


def main():

    dataframe = (load_training_dataset())

    train, validation, test = (
        split_dataset(
            dataframe
        )
    )

    print(
        f"Train records: "
        f"{len(train)}"
    )

    print(
        f"Validation records: "
        f"{len(validation)}"
    )

    print(
        f"Test records reserved: "
        f"{len(test)}"
    )

    # -------------------------
    # Features and target
    # -------------------------

    X_train = train[FEATURE_COLUMNS]

    y_train = train[TARGET_COLUMN]

    print("\nFeatures:")

    for feature in FEATURE_COLUMNS:
        print(f"- {feature}")

    print(
        f"\nTarget: "
        f"{TARGET_COLUMN}"
    )

    # -------------------------
    # Create and train model
    # -------------------------

    print("\nTraining Random Forest...")

    model = create_model()

    model.fit(X_train,y_train)

    print("Training completed.")

    # -------------------------
    # Evaluation
    # -------------------------

    print("\nRandom Forest results:")

    evaluate_model(model,train,"TRAIN")
    evaluate_model(model,validation,"VALIDATION")

    print("\nTest dataset was NOT evaluated.")


if __name__ == "__main__":
    main()