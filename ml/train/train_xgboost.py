from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor

from ml.data_loader import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    load_training_dataset,
    split_dataset,
)


def create_model(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=4,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=1,
    reg_lambda=1.0,
):

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

    model = XGBRegressor(
        objective="reg:squarederror",
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        min_child_weight=min_child_weight,
        reg_lambda=reg_lambda,
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


def evaluate_model(model,dataframe,dataset_name):
    X = dataframe[FEATURE_COLUMNS]

    y_true = dataframe[TARGET_COLUMN]

    y_pred = model.predict(X)

    mae = mean_absolute_error(y_true,y_pred)

    rmse = mean_squared_error(y_true,y_pred) ** 0.5

    r2 = r2_score(y_true,y_pred)

    print(f"\n{dataset_name}")

    print(f"Records: {len(dataframe)}")

    print(f"MAE:  {mae:.3f} t/ha")

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

    X_train = train[FEATURE_COLUMNS]

    y_train = train[TARGET_COLUMN]

    print("\nFeatures:")

    for feature in FEATURE_COLUMNS:
        print(f"- {feature}")

    print(
        f"\nTarget: "
        f"{TARGET_COLUMN}"
    )

    print("\nTraining XGBoost...")

    model = create_model()

    model.fit(X_train,y_train)

    print("Training completed.")

    print("\nXGBoost results:")

    evaluate_model(model,train,"TRAIN")

    evaluate_model(model,validation,"VALIDATION")

    print( "\nTest dataset was NOT evaluated.")


if __name__ == "__main__":
    main()